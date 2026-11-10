"""Cross-fitted functional ablations of SO(5) vector and adjoint content.

Uses independent fit and evaluation elements, a joint coefficient fit to
avoid assigning finite-sample cross-correlations twice, both encoder loci,
the same noisy/exact candidates for every intervention, and a shuffled-fit
control. Does not train models or modify original evidence.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--cases-json',required=True)
p.add_argument('--verification',default='.')
p.add_argument('--out',required=True)
p.add_argument('--include-fresh',action='store_true')
p.add_argument('--fresh-only',action='store_true')
p.add_argument('--limit',type=int,default=None)
p.add_argument('--n-fit',type=int,default=16384)
p.add_argument('--n-eval',type=int,default=4096)
p.add_argument('--batch',type=int,default=128)
p.add_argument('--device',default='cuda')
args=p.parse_args()
source=Path(args.source).resolve();verification=Path(args.verification).resolve()
sys.path.insert(0,str(source))
from core.spin5 import (get_spin5_encoding,haar_sp2,qmat_mul,rep_vector,rep_adjoint)
from core.su2_model import TripleScorer
from experiments.spin5_train import make_pair_set

def features(g):
    return np.c_[rep_vector(g).reshape(len(g),25),rep_adjoint(g).reshape(len(g),100)]

def embed(model,enc,g,candidate=False):
    net=model.enc_c if candidate else model.enc
    result=[]
    with torch.no_grad():
        for start in range(0,len(g),2048):
            result.append(net(torch.tensor(enc(g[start:start+2048]),device=args.device)).cpu().numpy())
    return np.concatenate(result).astype(float)

cases=[(item['condition'],source/item['run']) for item in json.loads(Path(args.cases_json).read_text())]
if args.limit is not None:cases=cases[:args.limit]

# All models see the exact same independently generated task instances.
rng=np.random.default_rng(92211)
a,b=haar_sp2(args.n_eval,rng),haar_sp2(args.n_eval,rng)
dataset=make_pair_set(a,b,64,.15,.7,'so5',rng)
ga,gb,candidates=dataset
product=qmat_mul(ga,gb)
fa,fb,fp=features(ga),features(gb),features(product)
probe=haar_sp2(args.n_fit,np.random.default_rng(92212))
fit_features=features(probe);fm=fit_features.mean(0);F=fit_features-fm
U,S,Vt=np.linalg.svd(F,full_matrices=False)
assert len(S)==125 and S[-1]>S[0]*1e-8
pinv=(Vt.T/S)@U.T
shuffled=np.random.default_rng(92213).permutation(args.n_fit)
modes=['none','no_vector','no_adjoint','no_both','sham']
iv_cases={'none':('none','none'),
    'argument_no_vector':('no_vector','none'),
    'argument_no_adjoint':('no_adjoint','none'),
    'argument_no_both':('no_both','none'),
    'candidate_no_vector':('none','no_vector'),
    'candidate_no_adjoint':('none','no_adjoint'),
    'candidate_no_both':('none','no_both'),
    'both_no_vector':('no_vector','no_vector'),
    'both_no_adjoint':('no_adjoint','no_adjoint'),
    'both_no_both':('no_both','no_both'),
    'sham_both':('sham','sham')}
records=[]
for label,run in cases:
    started=time.time();checkpoint=run/'ckpt_100000.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['step']==100000
    cfg={**json.loads((run/'config.json').read_text()),**state.get('config',{})}
    assert cfg['variant']=='so5' and cfg['K']==64 and cfg['sigma']==.15 and cfg['excl_radius']==.7
    model=TripleScorer(m=state['model']['enc.0.weight'].shape[1]).to(args.device).eval()
    model.load_state_dict(state['model'])
    if cfg.get('conditioning','identity') != 'identity':
        from core.so5_conditioning import FrozenCoordinateEncoding
        enc=FrozenCoordinateEncoding(mode=cfg['conditioning'],m=cfg['m'],alpha=cfg['alpha'],
            enc_seed=cfg['enc_seed'],probe_seed=cfg['conditioning_seed'],n_probe=cfg['conditioning_n'])
    else:
        enc=get_spin5_encoding(cfg.get('encoding','landmark'),'so5',m=cfg['m'],alpha=cfg['alpha'],seed=cfg['enc_seed'])
    Z=np.c_[embed(model,enc,probe),embed(model,enc,probe,candidate=True)]
    W=pinv@(Z-Z.mean(0));Ws=pinv@(Z[shuffled]-Z.mean(0))
    def parts(gfeat,w,ws):
        x=gfeat-fm
        return {'no_vector':x[:,:25]@w[:25],
                'no_adjoint':x[:,25:]@w[25:],
                'no_both':x@w,'sham':x@ws}
    za,zb=embed(model,enc,ga),embed(model,enc,gb)
    pa,pb=parts(fa,W[:,:128],Ws[:,:128]),parts(fb,W[:,:128],Ws[:,:128])
    feature_energy={}
    for locus,original,part in [('argument',za,pa)]:
        total=np.sum((original-original.mean(0))**2)
        feature_energy[locus]={k:float(np.sum(v*v)/total) for k,v in part.items()}
    hs={}
    with torch.no_grad():
        for mode in modes:
            aa=za if mode=='none' else za-pa[mode]
            bb=zb if mode=='none' else zb-pb[mode]
            hs[mode]=model.interact(torch.tensor(np.c_[aa,bb],dtype=torch.float32,device=args.device))
    successes={k:{kind:[] for kind in ['noisy','exact']} for k in iv_cases}
    margins={k:{kind:[] for kind in ['noisy','exact']} for k in iv_cases}
    wc=torch.tensor(W[:,128:],dtype=torch.float32,device=args.device)
    wcs=torch.tensor(Ws[:,128:],dtype=torch.float32,device=args.device)
    fmean=torch.tensor(fm,dtype=torch.float32,device=args.device)
    def candidate_variants(g,f):
        z=model.enc_c(torch.tensor(enc(g),device=args.device))
        ff=torch.tensor(f,dtype=torch.float32,device=args.device)-fmean
        return {'none':z,'no_vector':z-ff[:,:25]@wc[:25],
                'no_adjoint':z-ff[:,25:]@wc[25:],'no_both':z-ff@wc,'sham':z-ff@wcs}
    with torch.no_grad():
        for start in range(0,args.n_eval,args.batch):
            end=min(start+args.batch,args.n_eval);n=end-start
            c=candidates[start:end].reshape(-1,2,2,4)
            zc=candidate_variants(c,features(c))
            ze=candidate_variants(product[start:end],fp[start:end])
            for key,(argmode,candmode) in iv_cases.items():
                h=hs[argmode][start:end]
                scores=torch.einsum('bd,bkd->bk',h,zc[candmode].reshape(n,64,128))
                neg=scores[:,1:].max(1).values
                exact=(h*ze[candmode]).sum(-1)
                for kind,positive in [('noisy',scores[:,0]),('exact',exact)]:
                    margin=(positive-neg).cpu().numpy()
                    successes[key][kind].append(margin>=0)
                    margins[key][kind].append(margin)
    results={}
    for key in iv_cases:
        results[key]={}
        for kind in ['noisy','exact']:
            s=np.concatenate(successes[key][kind]);s0=np.concatenate(successes['none'][kind])
            delta=s.astype(float)-s0.astype(float)
            results[key][kind]={'accuracy':float(s.mean()),
                'mean_margin':float(np.concatenate(margins[key][kind]).mean()),
                'paired_accuracy_change':float(delta.mean()),
                'paired_change_se':float(delta.std(ddof=1)/np.sqrt(len(delta)))}
    row={'condition':label,'run':str(run),'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
         'training_seed':cfg['seed'],'training_fraction':cfg['frac'],
         'step':100000,'encoding':cfg.get('encoding','landmark'),
         'argument_functional_energy':feature_energy['argument'],
         'interventions':results,'elapsed_s':time.time()-started}
    records.append(row)
    print(label,'seed',cfg['seed'],'noisy',
          {k:round(results[k]['noisy']['accuracy'],4) for k in iv_cases},flush=True)
    report={'records':records,'protocol':{'fit_n':args.n_fit,'evaluation_n':args.n_eval,
        'evaluation_seed':92211,'fit_seed':92212,'shuffle_seed':92213,
        'coefficient_dimensions':{'vector':25,'adjoint':100},'fit':'joint centered OLS, independent Haar fit elements',
        'loci':'both argument encoders, candidate encoder, and both together',
        'controls':'same examples and excluded negatives; exact and noisy positives; shuffled-fit sham',
        'parameter_change':'none','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(out)
    del model,state,Z,W,Ws,hs,wc,wcs
    gc.collect()
    if args.device=='cuda':torch.cuda.empty_cache()
