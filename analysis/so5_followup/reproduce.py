"""Retrain a declared SO(5) preprocessing condition into a fresh directory."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['identity', 'whiten', 'rotate', 'intrinsic'], required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--device', default='cuda')
p.add_argument('--steps', type=int, default=100000)
p.add_argument('--output-root', default='runs/so5_followup')
p.add_argument('--dry-run', action='store_true')
args = p.parse_args()
root = Path(__file__).resolve().parents[2]
run = Path(args.output_root) / f'{args.mode}_f0.03_s{args.seed}'
if (root / run).exists():
    p.error(f'Refusing to append to or overwrite {run}; choose a fresh output root')
module = 'so5_conditioning_train' if args.mode in ['whiten', 'rotate'] else 'spin5_train'
cmd = [sys.executable, '-m', 'experiments.' + module,
       '--variant', 'so5', '--encoding', 'so5rf' if args.mode == 'intrinsic' else 'landmark',
       '--N', '2048', '--frac', '0.03', '--K', '64', '--sigma', '0.15',
       '--excl_radius', '0.7', '--m', '48', '--alpha', '2.0', '--enc_seed', '0',
       '--steps', str(args.steps), '--batch', '2048', '--lr', '0.001', '--wd', '1.0',
       '--n_eval', '4096', '--eval_every', '1000', '--spec_every', '5000',
       '--ckpt_every', '25000', '--device', args.device, '--seed', str(args.seed),
       '--save_dir', str(run)]
if module == 'so5_conditioning_train':
    cmd += ['--conditioning', args.mode]
print(shlex.join(cmd), flush=True)
if not args.dry_run:
    subprocess.run(cmd, cwd=root, check=True)
