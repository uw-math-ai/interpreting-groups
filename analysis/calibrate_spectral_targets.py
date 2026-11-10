"""Recover exact training-target scales for interpretable Haar coefficients.

Uses the original trainer's RNG prefix and raw target generator. No model
training, archived artifacts, or analysis outputs are modified.
"""
import json
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parents[1]
SRC = BASE
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from core.su2 import haar, wigner_character
from experiments.su2_train import make_hk_set

cache = {}
records = []
patterns = ['su2bl_j1j2_f0.01_s{}', 'su2bl_j1j2_wd0.1_f0.01_s{}']
patterns += [f'su2hk_t{t}_N2048_f0.01_s{{}}' for t in ['0.3','0.1','0.05']]
for pattern in patterns:
    for seed in range(5):
        name = pattern.format(seed)
        run = SRC / 'data/su2' / name
        cfg = json.loads((run / 'config.json').read_text())
        recipe = {k:cfg.get(k) for k in ['N','frac','n_eval','K','loss','hk_t','bl_js','bl_coefs','seed']}
        # band-limited targets do not depend on the unused heat-kernel t.
        key = json.dumps(recipe, sort_keys=True)
        if key not in cache:
            rng = np.random.default_rng(cfg['seed'])
            pool = haar(cfg['N'], rng)
            n = int(round(cfg['frac'] * cfg['N']**2))
            ids = rng.choice(cfg['N']**2, size=n+cfg['n_eval'], replace=False)[:n]
            target = None
            if cfg['loss'] == 'bandlim':
                js,cs = cfg['bl_js'],cfg['bl_coefs']
                target = lambda rel:sum(c*wigner_character(j,rel) for j,c in zip(js,cs))
            raw = make_hk_set(pool[ids//cfg['N']], pool[ids%cfg['N']],
                              cfg['K'], cfg['hk_t'], rng, target)
            y = raw[3]
            mean,sd = float(y.mean()),float(y.std()+1e-8)
            normalized = ((y-mean)/sd).astype(np.float32)
            cache[key] = {'mean':mean,'sd':sd,
                          'standardized_mean':float(normalized.mean()),
                          'standardized_sd':float(normalized.std()),
                          'target_count':int(y.size)}
            del raw,y,normalized
        scale = cache[key]
        path = OUT/'selected_checkpoints/data/su2'/name/'step_60000/hk_haar_ckpt_60000.json'
        d = json.loads(path.read_text())['results']
        ratios = {}
        for j,a in d['estimate'].items():
            t = d['target_coefficients'][j]
            if j != '0.0' and abs(t)>1e-6:
                factor = scale['sd']/t
                ratios[j] = {'ratio':a['a_hat']*factor,
                             'ci95':[x*factor for x in a['ci95']]}
        records.append({'run':name,'recipe':recipe,'target_scale':scale,
                        'coefficient_recovery':ratios,
                        'source':str(path.relative_to(BASE))})
        print(name, 'target sd',scale['sd'],'ratios',
              {j:round(x['ratio'],4) for j,x in ratios.items()},flush=True)
(OUT/'spectral_target_calibration.json').write_text(json.dumps({
    'normalization':'estimated signed Haar coefficient times original training-target sd divided by unstandardized target coefficient; j=0 excluded',
    'records':records},indent=2))
