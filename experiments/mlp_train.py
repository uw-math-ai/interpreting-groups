"""MLP-only grokking trainer for finite groups (no attention).

Parallel to experiments/aws_train.py but trains core.mlp_model.MLPGroup. Same lean
loop (per-epoch history on-GPU, fused AdamW, TF32, compile), and the same
grok-detection: saves a purified *_grok20k.pt `extra_after_grok` epochs after grok.
Checkpoints carry config['arch']='mlp' so core.checkpoint.load_model rebuilds the
MLP and every analysis script works unchanged.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from core.mlp_model import MLPGroup
from core.data import generate_group_dataset, split_dataset, get_device
from core.checkpoint import checkpoint_path
from core.groups import get_group, GROUP_TASKS


def save_ckpt(path, model, P, d, hid, seed, task, losses, tras, evac, evep):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"P": P, "embed_dim": d, "num_heads": 4, "mlp_dim": hid,
                   "task": task, "arch": "mlp"},
        "seed": seed, "losses": losses, "train_accs": tras,
        "eval_accs": evac, "eval_epochs": evep,
    }, path)
    print(f"Saved -> {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description="MLP-only finite-group grokking trainer")
    p.add_argument("--task", default="icosahedral", choices=GROUP_TASKS)
    p.add_argument("--epochs", type=int, default=800000)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--mlp_dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1.0)
    p.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.98])
    p.add_argument("--train_frac", type=float, default=0.3)
    p.add_argument("--eval_frac", type=float, default=0.7)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--checkpoint_every", type=int, default=10000)
    p.add_argument("--grok_thresh", type=float, default=0.95)
    p.add_argument("--extra_after_grok", type=int, default=20000)
    p.add_argument("--snapshot_every", type=int, default=0,
                   help="if >0 (multiple of eval_every), save embedding to *_snapshots.npz")
    p.add_argument("--snapshot_dense_until", type=int, default=0,
                   help="snapshot every eval_every epoch up to here (dense early-learning capture)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save_dir", default="data/mlp")
    p.add_argument("--device", default=None)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"CUDA: {torch.cuda.get_device_name(0)}, TF32 ON", flush=True)

    G = get_group(args.task)
    P = G["n"]
    inputs, targets = generate_group_dataset(G["table"], P)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = checkpoint_path(args.save_dir, P, args.embed_dim, 4, args.mlp_dim, args.seed, args.task)
    best_path = save_path.replace(".pt", "_best.pt")
    grok_path = save_path.replace(".pt", "_grok20k.pt")
    if os.path.exists(save_path) and not args.force:
        print(f"SKIP (exists): {save_path}", flush=True)
        return

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    sp = split_dataset(inputs, targets, args.train_frac, args.eval_frac, seed=args.seed, device=device)
    xtr, ytr = sp["train_inputs"], sp["train_targets"]
    xev, yev = sp["eval_inputs"], sp["eval_targets"]
    print(f"task={args.task} P={P} ARCH=MLP train={xtr.shape[0]} eval={xev.shape[0]} "
          f"tf={args.train_frac} ef={args.eval_frac} seed={args.seed}", flush=True)

    model = MLPGroup(P, args.embed_dim, args.mlp_dim).to(device)
    net = torch.compile(model) if args.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=tuple(args.betas),
                            weight_decay=args.weight_decay, fused=(device.type == "cuda"))
    loss_fn = nn.CrossEntropyLoss()

    E = args.epochs
    loss_hist = torch.zeros(E, device=device)
    tracc_hist = torch.zeros(E, device=device)
    eval_accs, eval_epochs = [], []
    best_eval = -1.0
    grok_epoch = -1
    grok_saved = False
    snap_path = save_path.replace(".pt", "_snapshots.npz")
    snap_ep, snap_ev, snap_emb = [], [], []
    def save_snaps():
        if snap_emb:
            np.savez(snap_path, epochs=np.array(snap_ep), evals=np.array(snap_ev),
                     embeds=np.stack(snap_emb))

    def hist(n):
        return loss_hist[:n].tolist(), tracc_hist[:n].tolist()

    t0 = time.time()
    for epoch in range(E):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = net(xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()
        loss_hist[epoch] = loss.detach()
        tracc_hist[epoch] = (logits.detach().argmax(-1) == ytr).float().mean()

        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                ev = (net(xev).argmax(-1) == yev).float().mean().item()
            eval_accs.append(ev); eval_epochs.append(epoch + 1)
            e1 = epoch + 1
            if args.snapshot_every and ((args.snapshot_dense_until and e1 <= args.snapshot_dense_until)
                                        or e1 % args.snapshot_every == 0):
                snap_ep.append(e1); snap_ev.append(ev)
                snap_emb.append(model.embed.weight[:P].detach().cpu().numpy().copy())
            rate = (epoch + 1) / (time.time() - t0)
            print(f"[MLP {args.task} tf{args.train_frac}] ep {epoch+1} | loss {loss_hist[epoch].item():.4f} "
                  f"| train {tracc_hist[epoch].item():.3f} | eval {ev:.3f} | best {max(best_eval,ev):.3f} "
                  f"| {rate:.0f} eps/s", flush=True)
            ls, tras = None, None
            if ev > best_eval:
                best_eval = ev
                ls, tras = hist(epoch + 1)
                save_ckpt(best_path, model, P, args.embed_dim, args.mlp_dim, args.seed, args.task, ls, tras, list(eval_accs), list(eval_epochs))
            if grok_epoch < 0 and ev >= args.grok_thresh:
                grok_epoch = epoch + 1
                print(f"  >> GROK at ep {grok_epoch}; purified checkpoint at ep {grok_epoch+args.extra_after_grok}", flush=True)
            if grok_epoch > 0 and not grok_saved and (epoch + 1) >= grok_epoch + args.extra_after_grok:
                ls, tras = hist(epoch + 1)
                save_ckpt(grok_path, model, P, args.embed_dim, args.mlp_dim, args.seed, args.task, ls, tras, list(eval_accs), list(eval_epochs))
                grok_saved = True
            if args.checkpoint_every and (epoch + 1) % args.checkpoint_every == 0:
                ls, tras = hist(epoch + 1)
                save_ckpt(save_path, model, P, args.embed_dim, args.mlp_dim, args.seed, args.task, ls, tras, list(eval_accs), list(eval_epochs))
                save_snaps()

    save_snaps()
    ls, tras = hist(E)
    save_ckpt(save_path, model, P, args.embed_dim, args.mlp_dim, args.seed, args.task, ls, tras, list(eval_accs), list(eval_epochs))
    print(f"DONE MLP {args.task} tf{args.train_frac} best_eval={best_eval:.4f}", flush=True)


if __name__ == "__main__":
    main()
