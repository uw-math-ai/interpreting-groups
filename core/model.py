import torch
import torch.nn as nn
import torch.nn.functional as F


class ModMultDecoderOnly(nn.Module):
    """Decoder-only transformer for modular multiplication: a * b = c (mod P).

    Input is a sequence of 3 tokens [a, b, P] where token 2 carries the value P
    itself (acting as the "=" separator). The embedding table has P+1 entries so
    that token index P is valid. The model predicts c = (a * b) % P.

    Single-layer architecture: embedding -> causal multihead self-attention -> MLP -> unembed.

    Args:
        P: modulus (required, no default — forces explicit choice)
        d_model: embedding / hidden dimension
        n_heads: number of attention heads
        mlp_hidden: MLP hidden layer width
        use_pos_embed: add learned positional embeddings for the 3 sequence
            positions. Off by default (the modular-multiplication task is
            commutative, a*b=b*a, so operand order is irrelevant and the original
            paper used no positions). Required for NON-ABELIAN group tasks, where
            the readout must distinguish operand a (pos 0) from b (pos 1) —
            otherwise the "=" readout is a symmetric function of {a, b} and cannot
            represent a non-commutative product.
    """

    def __init__(self, P: int, d_model: int = 128, n_heads: int = 4, mlp_hidden: int = 512,
                 use_pos_embed: bool = False):
        super().__init__()

        self.P = P
        self.d_model = d_model
        self.n_heads = n_heads
        self.mlp_hidden = mlp_hidden
        self.use_pos_embed = use_pos_embed
        # Token embedding: P values + 1 extra token for the "=" position
        self.embed = nn.Embedding(P + 1, d_model)
        # Optional learned positional embedding for the 3 positions [a, b, "="]
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, 3, d_model))

        # Multihead self-attention (causal)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Feedforward MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, d_model),
        )

        # Output projection to predict one of P residue classes
        self.unembed = nn.Linear(d_model, P)

        # Causal mask: prevent future tokens from being attended to
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(3, 3), diagonal=1).bool(),
        )

    def forward(self, x, return_attn=False, return_mlp=False):
        """
        Args:
            x: [B, 3] tensor of token ids [a, b, P]
            return_attn: if True, return (logits, attn_weights) instead of just logits
            return_mlp: if True, store pre-ReLU MLP activations in self.mlp_activations

        Returns:
            logits: [B, P] prediction over residue classes
            attn_weights: [B, 3, 3] head-averaged attention weights (only when return_attn=True)

        Side effects:
            When return_mlp=True, sets self.mlp_activations to the pre-ReLU linear output
            of shape [B, 3, mlp_hidden] (covering all 3 sequence positions).
        """
        h = self.embed(x)  # [B, 3, d_model]
        if self.use_pos_embed:
            h = h + self.pos_embed

        attn_out, attn_weights = self.attn(
            h, h, h,
            attn_mask=self.causal_mask
        )

        # Residual connections
        h = h + attn_out

        if return_mlp:
            pre_relu = self.mlp[0](h)
            post_relu = F.relu(pre_relu)
            mlp_out = self.mlp[2](post_relu)
            h = h + mlp_out
            self.mlp_activations = pre_relu
        else:
            h = h + self.mlp(h)

        # Readout from the "=" position (last token)
        logits = self.unembed(h[:, -1, :])  # [B, P]

        if return_attn:
            return logits, attn_weights
        return logits
