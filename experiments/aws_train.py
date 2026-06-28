"""Lean, fast CUDA training driver for the icosahedral grokking run.

Same experiment as ``core.trainer`` (identical model, seeding, and data split so
checkpoints are interchangeable with the analysis pipeline), but tuned for a long
detached GPU run:

  * fused AdamW + TF32 + optional torch.compile (kernel fusion)
  * per-epoch loss/accuracy history kept ON-GPU and copied to host only at eval /
    checkpoint boundaries — avoids ~N host syncs from ``.item()`` every epoch,
    which is the dominant overhead for this tiny launch-bound model
  * writes the same checkpoint dict as ``core.checkpoint.save_checkpoint`` plus a
    ``*_best.pt`` at every new eval-accuracy high (captures the grok moment)

Checkpoints are 100% compatible with every ``experiments/*`` analysis script.
"""
import argparse
import os
import random
import time

import torch
import torch.nn as nn

from core.model import ModMultDecoderOnly
from core.data import generate_group_dataset, split_dataset, get_device
from core.checkpoint import checkpoint_path, save_checkpoint
from core.groups import get_group, GROUP_TASKS


def main():
    p = argparse.ArgumentParser(description="Fast CUDA trainer for finite-group grokking")
    p.add_argument("--task", default="icosahedral", choices=GROUP_TASKS)
    p.add_argument("--epochs", type=int, default=500000)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--mlp_dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.98])
    p.add_argument("--train_frac", type=float, default=0.5)
    p.add_argument("--eval_frac", type=float, default=0.2)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--snapshot_every", type=int, default=0,
                   help="if >0 (multiple of eval_every), save the embedding at each such "
                        "epoch to a *_snapshots.npz time series for grokking animations")
    p.add_argument("--snapshot_dense_until", type=int, default=0,
                   help="snapshot every eval_every epoch up to here (dense early-learning capture)")
    p.add_argument("--grok_thresh", type=float, default=0.95,
                   help="eval acc that counts as 'grokked'")
    p.add_argument("--extra_after_grok", type=int, default=10000,
                   help="after grok is first detected, save a purified *_grok10k.pt this many "
                        "epochs later (post-grok representation cleanup)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save_dir", default="data/transformer")
    p.add_argument("--device", default=None)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"CUDA: {torch.cuda.get_device_name(0)}, TF32 ON", flush=True)

    # --- task / data ---
    G = get_group(args.task)
    P = G["n"]
    inputs, targets = generate_group_dataset(G["table"], P)
    use_pos_embed = True   # all these groups are non-abelian -> operand order matters

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = checkpoint_path(args.save_dir, P, args.embed_dim, args.n_heads,
                                args.mlp_dim, args.seed, args.task)
    best_path = save_path.replace(".pt", "_best.pt")
    if os.path.exists(save_path) and not args.force:
        print(f"SKIP (exists): {save_path}", flush=True)
        return

    # --- seeding & split (match core.trainer ordering for reproducibility) ---
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    splits = split_dataset(inputs, targets, args.train_frac, args.eval_frac,
                           seed=args.seed, device=device)
    xtr, ytr = splits["train_inputs"], splits["train_targets"]
    xev, yev = splits["eval_inputs"], splits["eval_targets"]
    print(f"task={args.task} P={P} pos_embed={use_pos_embed} "
          f"train={xtr.shape[0]} eval={xev.shape[0]} "
          f"tf={args.train_frac} ef={args.eval_frac} seed={args.seed}", flush=True)

    model = ModMultDecoderOnly(P, args.embed_dim, args.n_heads, args.mlp_dim,
                               use_pos_embed=use_pos_embed).to(device)
    net = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=tuple(args.betas),
                            weight_decay=args.weight_decay,
                            fused=(device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss()

    E = args.epochs
    # Per-epoch history lives on-GPU; copied to host only when we checkpoint.
    loss_hist = torch.zeros(E, device=device)
    tracc_hist = torch.zeros(E, device=device)
    eval_accs, eval_epochs = [], []
    best_eval = -1.0

    def history(n):
        """Host-side per-epoch lists up to epoch n (bulk transfer, one sync)."""
        return loss_hist[:n].tolist(), tracc_hist[:n].tolist()

    def write(path, n):
        losses, tras = history(n)
        run = {"model": model, "losses": losses, "train_accs": tras,
               "eval_accs": list(eval_accs), "eval_epochs": list(eval_epochs)}
        save_checkpoint(run, P, args.embed_dim, args.n_heads, args.mlp_dim,
                        args.seed, path, args.task, use_pos_embed)

    grok_path = save_path.replace(".pt", "_grok10k.pt")
    grok_epoch = -1          # first epoch eval >= grok_thresh
    grok10k_saved = False

    import numpy as np
    snap_path = save_path.replace(".pt", "_snapshots.npz")
    snap_ep, snap_ev, snap_emb = [], [], []
    def save_snaps():
        if snap_emb:
            np.savez(snap_path, epochs=np.array(snap_ep), evals=np.array(snap_ev),
                     embeds=np.stack(snap_emb))

    t0 = time.time()
    for epoch in range(E):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = net(xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
        # record on-GPU, no host sync
        loss_hist[epoch] = loss.detach()
        tracc_hist[epoch] = (logits.detach().argmax(-1) == ytr).float().mean()

        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                ev = (net(xev).argmax(-1) == yev).float().mean().item()
            tr = tracc_hist[epoch].item()
            ls = loss_hist[epoch].item()
            eval_accs.append(ev)
            eval_epochs.append(epoch + 1)
            e1 = epoch + 1
            if args.snapshot_every and ((args.snapshot_dense_until and e1 <= args.snapshot_dense_until)
                                        or e1 % args.snapshot_every == 0):
                snap_ep.append(e1); snap_ev.append(ev)
                snap_emb.append(model.embed.weight[:P].detach().cpu().numpy().copy())
            rate = (epoch + 1) / (time.time() - t0)
            print(f"[s{args.seed} tf{args.train_frac}] ep {epoch+1} | loss {ls:.4f} "
                  f"| train {tr:.3f} | eval {ev:.3f} | best {max(best_eval,ev):.3f} "
                  f"| {rate:.0f} eps/s", flush=True)
            if ev > best_eval:
                best_eval = ev
                write(best_path, epoch + 1)
            # grok detection + purified grok+extra checkpoint
            if grok_epoch < 0 and ev >= args.grok_thresh:
                grok_epoch = epoch + 1
                print(f"  >> GROK at ep {grok_epoch} (eval {ev:.3f}); will save purified "
                      f"checkpoint at ep {grok_epoch + args.extra_after_grok}", flush=True)
            if grok_epoch > 0 and not grok10k_saved and (epoch + 1) >= grok_epoch + args.extra_after_grok:
                write(grok_path, epoch + 1)
                grok10k_saved = True
                print(f"  >> saved grok+{args.extra_after_grok} purified checkpoint -> {grok_path}", flush=True)
            if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
                write(save_path, epoch + 1)
                save_snaps()

    write(save_path, E)
    save_snaps()
    print(f"DONE seed={args.seed} tf={args.train_frac} best_eval={best_eval:.4f} "
          f"-> {best_path}", flush=True)


if __name__ == "__main__":
    main()
