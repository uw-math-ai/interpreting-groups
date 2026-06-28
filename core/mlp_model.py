"""MLP-only model for finite-group composition (no attention).

A clean feed-forward control for the transformer (``core.model.ModMultDecoderOnly``):
the two operands a, b are embedded (shared table) and their embeddings are
**concatenated** (so operand order is preserved — required for non-abelian groups,
the MLP analog of the transformer's positional embeddings), then an MLP predicts
the product c.

  embed(a), embed(b)  ->  concat [2d]  ->  Linear -> ReLU -> Linear -> ReLU  ->  unembed[n]

Exposes ``.embed`` (n x d, the group-element embedding analysed for geometry) and
``.unembed`` so the existing analysis pipeline works unchanged.
"""
import torch
import torch.nn as nn


class MLPGroup(nn.Module):
    def __init__(self, P: int, d_model: int = 128, mlp_hidden: int = 512,
                 n_layers: int = 2):
        super().__init__()
        self.P = P
        self.d_model = d_model
        self.mlp_hidden = mlp_hidden
        self.embed = nn.Embedding(P, d_model)          # group elements 0..P-1
        layers = [nn.Linear(2 * d_model, mlp_hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU()]
        self.mlp = nn.Sequential(*layers)
        self.unembed = nn.Linear(mlp_hidden, P)        # logits over the P classes

    def forward(self, x, **_):
        """x: [B, >=2] token ids; uses operands x[:,0]=a, x[:,1]=b (ignores the rest)."""
        a = self.embed(x[:, 0])
        b = self.embed(x[:, 1])
        return self.unembed(self.mlp(torch.cat([a, b], dim=-1)))
