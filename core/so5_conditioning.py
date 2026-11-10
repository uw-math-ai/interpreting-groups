"""Frozen, invertible coordinate controls for the averaged SO(5) map."""
import numpy as np
from core.spin5 import get_spin5_encoding, haar_sp2


class FrozenCoordinateEncoding:
    def __init__(self, *, mode, m=48, alpha=2.0, enc_seed=0,
                 probe_seed=2001, n_probe=65536):
        if mode not in ('identity','rotate','whiten'):raise ValueError(mode)
        self.mode=mode;self.m=m;self.alpha=alpha
        self.base=get_spin5_encoding('landmark','so5',m=m,alpha=alpha,seed=enc_seed)
        g=haar_sp2(n_probe,np.random.default_rng(probe_seed))
        x=self.base(g).astype(float);xc=x-x.mean(0)
        cov=xc.T@xc/(len(x)-1)
        eig,Q=np.linalg.eigh(cov)
        if eig[0]<=eig[-1]*1e-8:raise RuntimeError('Covariance is numerically singular')
        if mode=='identity':W=np.eye(m)
        elif mode=='rotate':
            W,R=np.linalg.qr(np.random.default_rng(probe_seed+1).standard_normal((m,m)))
            W*=np.sign(np.diag(R))[None,:]
        else:W=(Q/np.sqrt(eig))@Q.T
        self.W=W
        post=np.linalg.eigvalsh(W.T@cov@W)
        self.diagnostics={'mode':mode,'n_probe':n_probe,'probe_seed':probe_seed,
            'map':'base observation right-multiplied by a fixed full-rank matrix; no extra centering',
            'pre_cov_condition':float(eig[-1]/eig[0]),
            'post_cov_condition':float(post[-1]/post[0]),
            'pre_cov_eigenvalues':eig.tolist(),'post_cov_eigenvalues':post.tolist(),
            'transform_condition':float(np.linalg.cond(W)),'transform_rank':int(np.linalg.matrix_rank(W)),
            'analytic_representation_features_used':False}

    def __call__(self,g):
        x=self.base(g)
        if self.mode=='identity':return x
        return (x.astype(float)@self.W).astype(np.float32)

    def torch_fn(self,device):
        import torch
        base=self.base.torch_fn(device)
        if self.mode=='identity':return base
        W=torch.tensor(self.W,dtype=torch.float32,device=device)
        return lambda g:base(g)@W
