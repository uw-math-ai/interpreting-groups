"""Random argument-encoder ablations at the selected sector's actual rank.

The legacy suite used the spin-half rank (4) for both groups. SO(3)'s
selected spin-one space instead has rank 9. Reuse its exact evaluation
construction while matching the control rank to each selected sector.
"""
import gc
import json
from pathlib import Path
import sys
import numpy as np
import torch

BASE=Path(__file__).resolve().parents[1]
VER=Path(__file__).resolve().parent
WORK=BASE
sys.path.insert(0,str(WORK))
from core.su2 import haar,qmul
from experiments.su2_causal import load_model,embed_np,score_with,Intervention

rows=[]
for variant,rank in [('su2',4),('so3',9)]:
    for seed in range(5):
        run=VER/f'selected_checkpoints/data/su2/{variant}_N2048_f0.01_s{seed}/step_100000'
        model,enc,cfg,step=load_model(run,device='cuda')
        fit_rng=np.random.default_rng(123)
        probe=haar(4096,fit_rng)
        z=embed_np(model,enc,probe,'cuda')
        mu=torch.tensor(z.mean(0),device='cuda')
        basis=np.linalg.qr(fit_rng.standard_normal((z.shape[1],rank)))[0]
        B=torch.tensor(basis,dtype=torch.float32,device='cuda')
        def random_ablation(model,enc,q,device):
            with torch.no_grad():
                Z=model.embed(torch.tensor(enc(q),device=device))
                return Z-(Z-mu)@B@B.T
        iv=Intervention('ablate_random_selected_rank',random_ablation)
        rng=np.random.default_rng(7);n=8192
        qa,qb=haar(n,rng),haar(n,rng);prod=qmul(qa,qb)
        neg=haar(n*63,rng).reshape(n,63,4)
        for _ in range(50):
            dot=np.sum(neg*prod[:,None,:],axis=-1)
            if variant=='so3':dot=np.abs(dot)
            bad=np.arccos(np.clip(dot,-1,1))<cfg['excl_radius']
            if not bad.any():break
            neg[bad]=haar(int(bad.sum()),rng)
        cands=np.concatenate([prod[:,None,:],neg],axis=1).astype(np.float32)
        scores=score_with(model,enc,iv,qa,qb,cands,'cuda',batch=256)
        center=np.stack([prod,-prod],axis=1).astype(np.float32)
        sc=score_with(model,enc,iv,qa,qb,center,'cuda',batch=256)
        row={'variant':variant,'seed':seed,'step':step,'rank':rank,'n':n,
             'task_acc':float((scores.argmax(1)==0).mean()),
             'center_disc':float(((sc[:,0]>sc[:,1])+.5*(sc[:,0]==sc[:,1])).mean())}
        rows.append(row);print(row,flush=True)
        del model,B,mu,z,scores,sc
        gc.collect();torch.cuda.empty_cache()
(VER/'rank_matched_ablation.json').write_text(json.dumps({
    'records':rows,'protocol':{'training_fraction':.01,'step':100000,
    'fit_probe_seed':123,'fit_n':4096,'evaluation_seed':7,'evaluation_n':8192,
    'positive':'exact ab','negative':'63 Haar negatives with original exclusion',
    'locus':'both argument encoders','checkpoint_source':'original transferred weights',
    'rank':'4 for SU(2), 9 for SO(3)','inference_batch':256}},indent=2))
