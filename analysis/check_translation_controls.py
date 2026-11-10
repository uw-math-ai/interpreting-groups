"""Compare trained and untrained controls using matched translation protocols."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--out',required=True)
p.add_argument('--device',default='cuda');p.add_argument('--include-exact-controls',action='store_true')
args=p.parse_args();work=Path(args.work).resolve();sys.path.insert(0,str(work))
from core.su2 import haar
from core.su2_model import TripleScorer
from experiments.su2_causal import load_model
from experiments.su2_mechanism import PCAFrame,embed,spin_block_frame
from experiments.twosided import analyze,rot_matrix

all_results={}
for seed in range(5):
    run=work/f'analysis/selected_checkpoints/data/su2/so3_N2048_f0.01_s{seed}/step_100000'
    model,enc,cfg,step=load_model(run,device=args.device)
    rng=np.random.default_rng(29)
    g=haar(6144,rng);tr,te=g[:5120],g[5120:];ref=haar(4096,rng);ks=haar(12,rng)
    torch.manual_seed(seed)
    default=TripleScorer(m=cfg['m']).to(args.device).eval()
    normal=TripleScorer(m=cfg['m']).to(args.device).eval()
    torch.manual_seed(seed+999)
    for weight in normal.parameters():torch.nn.init.normal_(weight,std=0.2)
    systems={};info={}
    for label,net in [('trained',model),('default_initialization',default),('normal_std02_initialization',normal)]:
        z=embed(net,enc,ref,args.device);mu=z.mean(0);B=spin_block_frame(z,ref,1.0)
        fr=PCAFrame(z,cap=128)
        systems[label+'_selected_block']=lambda q,net=net,mu=mu,B=B:(embed(net,enc,q,args.device)-mu)@B
        systems[label+'_full_PCA999']=lambda q,net=net,fr=fr:fr(embed(net,enc,q,args.device))
        C=rot_matrix(ref).reshape(len(ref),9);C-=C.mean(0)
        Q=np.linalg.qr(C)[0];zc=z-z.mean(0)
        info[label]={'selected_dim':B.shape[1],'PCA999_dim':fr.V.shape[1],
                     'vector_FVE':float(np.sum((Q.T@zc)**2)/np.sum(zc**2))}
    capped=PCAFrame(embed(normal,enc,ref,args.device),cap=16)
    systems['native_normal_std02_cap16']=lambda q:capped(embed(normal,enc,q,args.device))
    systems['raw_observation']=lambda q:enc(q).astype(float)
    if args.include_exact_controls:
        systems['true_block']=lambda q:rot_matrix(q).reshape(len(q),9)
        systems['frame']=lambda q:rot_matrix(q)[:,:,:2].reshape(len(q),6)
        proj=np.linalg.qr(rng.standard_normal((9,6)))[0]
        systems['rand_trunc']=lambda q:rot_matrix(q).reshape(len(q),9)@proj
    rows={}
    for label,fn in systems.items():
        rows[label]=analyze(fn,tr,te,ks,np.random.default_rng(123))
        print('seed',seed,label,rows[label],flush=True)
    all_results[str(seed)]={'feature_information':info,'translation':rows}

report={'seeds':all_results,'protocol':{
    'training_fraction':0.01,'checkpoint_step':100000,'n_train':5120,'n_test':1024,
    'n_reference':4096,'n_translations':12,'probe_seed':29,'commutator_pair_seed':123,
    'ridge':0.01,'selected_block_energy_retained':0.99,'global_PCA_variance_retained':0.999,
    'device':args.device,'include_exact_controls':args.include_exact_controls,
    'purpose':'test sensitivity of the initialization comparison to PCA cap and initialization distribution, holding data and affine-fit protocol fixed',
    'dependency_sha256':{str(f.relative_to(work)):hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [work/'experiments/twosided.py',work/'experiments/su2_mechanism.py',work/'core/su2_model.py']}}}
Path(args.out).write_text(json.dumps(report,indent=2))
