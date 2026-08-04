"""Encoder-factorized triple scorer for continuous group composition.

    z_a = E(Psi(a)),  z_b = E(Psi(b))          (shared argument encoder E)
    h   = MLP([z_a, z_b])                       (product locus)
    s(a, b, c) = <h, E_c(Psi(c))>               (candidate encoder E_c)

Designated analysis loci: E outputs (embedding-table analog, Peter-Weyl /
spin-FVE spectroscopy), h (composition test), scores (character fit).
"""
import torch
import torch.nn as nn


def _mlp(dims, act):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class Quadratic(nn.Module):
    def forward(self, x):
        return x * x


class TripleScorer(nn.Module):
    def __init__(self, m=32, d_enc=128, enc_hidden=256, mlp_hidden=512,
                 activation="gelu"):
        super().__init__()
        act = {"gelu": nn.GELU, "quadratic": Quadratic}[activation]
        self.enc = _mlp([m, enc_hidden, d_enc], act)          # E (shared for a, b)
        self.enc_c = _mlp([m, enc_hidden, d_enc], act)        # E_c (candidates)
        self.interact = _mlp([2 * d_enc, mlp_hidden, d_enc], act)

    def embed(self, psi):
        return self.enc(psi)

    def product_state(self, psi_a, psi_b):
        za, zb = self.enc(psi_a), self.enc(psi_b)
        return self.interact(torch.cat([za, zb], dim=-1))

    def forward(self, psi_a, psi_b, psi_cands):
        """psi_a, psi_b: [B, m]; psi_cands: [B, K, m] -> logits [B, K]."""
        h = self.product_state(psi_a, psi_b)                  # [B, d]
        zc = self.enc_c(psi_cands)                            # [B, K, d]
        return torch.einsum("bd,bkd->bk", h, zc)
