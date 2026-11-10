"""Independent noisy ranking and coefficient projections for SO(5) checkpoints.

Uses common fresh task instances and probes across every listed condition.
The original frozen-T2 metric is read from the matching logged checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--source', default=str(Path(__file__).resolve().parents[2]))
p.add_argument('--cases-json', required=True)
p.add_argument('--out', required=True)
p.add_argument('--n-probe', type=int, default=8192)
p.add_argument('--n-eval', type=int, default=8192)
p.add_argument('--analysis-seed', type=int, default=92371)
p.add_argument('--batch', type=int, default=256)
p.add_argument('--device', default='cuda')
args = p.parse_args()
source = Path(args.source).resolve()
sys.path.insert(0, str(source))

import numpy as np
import torch
from core.spin5 import get_spin5_encoding, haar_sp2, rep_vector, rep_adjoint
from core.so5_conditioning import FrozenCoordinateEncoding
from core.su2_model import TripleScorer
from experiments.spin5_train import make_pair_set, ranking_accuracy

cases = json.loads(Path(args.cases_json).read_text())
if not cases:
    p.error('The case list is empty')
rng = np.random.default_rng(args.analysis_seed)
probe = haar_sp2(args.n_probe, rng)
dataset = make_pair_set(haar_sp2(args.n_eval, rng), haar_sp2(args.n_eval, rng),
                        64, .15, .7, 'so5', rng)
bases = {}
for name, c in {
    'vector': rep_vector(probe).reshape(len(probe), 25),
    'adjoint': rep_adjoint(probe).reshape(len(probe), 100),
}.items():
    u, s, _ = np.linalg.svd(c - c.mean(0), full_matrices=False)
    if (s > s[0] * 1e-9).sum() != c.shape[1]:
        raise ValueError('Insufficient probe rank for ' + name)
    bases[name] = u

records = []
for case in cases:
    run = source / case['run']
    ck = run / case.get('checkpoint', 'ckpt_100000.pt')
    state = torch.load(ck, map_location='cpu', weights_only=False)
    cfg = {**json.loads((run / 'config.json').read_text()), **state['config']}
    assert state['step'] == 100000 and cfg['variant'] == 'so5' and cfg['frac'] == .03
    assert cfg['K'] == 64 and cfg['sigma'] == .15 and cfg['excl_radius'] == .7
    encoding = cfg.get('encoding', 'landmark')
    mode = 'intrinsic' if encoding == 'so5rf' else cfg.get('conditioning', 'identity')
    assert case.get('mode', mode) == mode
    assert case.get('seed', cfg['seed']) == cfg['seed']
    if mode in ['rotate', 'whiten']:
        enc = FrozenCoordinateEncoding(mode=mode, m=cfg['m'], alpha=cfg['alpha'],
            enc_seed=cfg['enc_seed'], probe_seed=cfg['conditioning_seed'],
            n_probe=cfg['conditioning_n'])
        transform = run / 'observation_transform.npz'
        assert hashlib.sha256(transform.read_bytes()).hexdigest() == cfg['conditioning_transform_sha256']
        saved = np.load(transform)['matrix']
        assert np.allclose(saved, enc.W, atol=1e-8, rtol=1e-8)
        enc.W = saved  # Evaluate the exact transformation used during training.
    else:
        enc = get_spin5_encoding(encoding, 'so5', m=cfg['m'], alpha=cfg['alpha'], seed=cfg['enc_seed'])
    weights = state['model']
    model = TripleScorer(m=weights['enc.0.weight'].shape[1],
        d_enc=weights['enc.2.weight'].shape[0],
        enc_hidden=weights['enc.0.weight'].shape[0],
        mlp_hidden=weights['interact.0.weight'].shape[0]).to(args.device).eval()
    model.load_state_dict(weights)
    with torch.no_grad():
        z = np.concatenate([model.enc(torch.tensor(enc(probe[i:i+1024]),
            device=args.device)).cpu().numpy() for i in range(0, len(probe), 1024)])
    z = z.astype(float) - z.mean(0)
    total = np.sum(z*z)
    fve = {name: float(np.sum((q.T @ z)**2) / total) for name, q in bases.items()}
    accuracy = ranking_accuracy(model, enc.torch_fn(args.device), dataset, args.device, batch=args.batch)
    logged = [json.loads(line) for line in (run / 'log.jsonl').read_text().splitlines()]
    final = next(row for row in reversed(logged) if row['step'] == state['step'])
    records.append({'condition': case['condition'], 'mode': mode, 'seed': cfg['seed'],
        'run': case['run'], 'status': 'complete', 'checkpoint': ck.name,
        'checkpoint_sha256': hashlib.sha256(ck.read_bytes()).hexdigest(),
        'frozen_noisy_T2': final['acc_T2'], 'independent_noisy_T2': accuracy,
        'projection_FVE': fve,
        'parameter_norm': float(sum(v.double().square().sum().item()
                                  for v in model.parameters())**.5)})
    print(mode, cfg['seed'], 'T2', accuracy, 'FVE', fve, flush=True)
    del model, state, weights
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

paired = []
for seed in sorted({r['seed'] for r in records}):
    pair = {r['mode']: r for r in records if r['seed'] == seed}
    if 'identity' in pair and 'whiten' in pair:
        paired.append({'seed': seed,
            'primary_frozen_T2_change': pair['whiten']['frozen_noisy_T2'] - pair['identity']['frozen_noisy_T2'],
            'independent_T2_change': pair['whiten']['independent_noisy_T2'] - pair['identity']['independent_noisy_T2']})
report = {'records': records, 'paired_changes': paired,
    'analysis': {'seed': args.analysis_seed, 'n_probe': args.n_probe, 'n_eval': args.n_eval,
        'FVE': 'Separate SVD projections; sectors are not an orthogonal partition',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + '\n')
