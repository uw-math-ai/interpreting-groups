import argparse
import os
import random

import torch
import torch.nn as nn

from core.model import ModMultDecoderOnly
from core.data import generate_mod_mult_dataset, generate_group_dataset, split_dataset, get_device
from core.checkpoint import checkpoint_path, save_checkpoint


def train_single_run(P, embed_dim, n_heads, mlp_dim, epochs, lr, weight_decay, betas,
                     train_frac, eval_frac, eval_every, seed, device, dataset=None,
                     use_pos_embed=False, compile_model=False, progress_callback=None):
    """Train a single model with full determinism.

    Args:
        dataset: optional precomputed (inputs, targets) tensors. When None, the
            modular-multiplication dataset for modulus P is generated. Used to
            train on an arbitrary finite-group Cayley table (P is then the group
            order, which sizes the embedding/unembed).

    Returns:
        dict with:
            model: trained ModMultDecoderOnly
            losses: list of length `epochs` — per-epoch training cross-entropy loss
            train_accs: list of length `epochs` — per-epoch training accuracy
            eval_accs: list of length `epochs // eval_every` — eval accuracy sampled every eval_every epochs
            eval_epochs: list of length `epochs // eval_every` — epoch numbers matching eval_accs
    """
    torch.manual_seed(seed)
    random.seed(seed)

    inputs, targets = dataset if dataset is not None else generate_mod_mult_dataset(P)
    splits = split_dataset(inputs, targets, train_frac, eval_frac, seed=seed, device=device)

    model = ModMultDecoderOnly(P, embed_dim, n_heads, mlp_dim, use_pos_embed=use_pos_embed).to(device)
    train_model = torch.compile(model) if compile_model else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    loss_fn = nn.CrossEntropyLoss()

    losses, train_accs, eval_accs, eval_epochs = [], [], [], []

    for epoch in range(epochs):
        train_model.train()
        optimizer.zero_grad()

        logits = train_model(splits["train_inputs"])
        loss = loss_fn(logits, splits["train_targets"])

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        train_accs.append((logits.argmax(dim=-1) == splits["train_targets"]).float().mean().item())

        if (epoch + 1) % eval_every == 0:
            train_model.eval()
            with torch.no_grad():
                eval_logits = train_model(splits["eval_inputs"])
                acc = (eval_logits.argmax(dim=-1) == splits["eval_targets"]).float().mean().item()
            eval_accs.append(acc)
            eval_epochs.append(epoch + 1)
            print(
                f"[Seed {seed}] Epoch {epoch+1} | "
                f"Loss {loss.item():.4f} | "
                f"Train acc {train_accs[-1]:.3f} | "
                f"Eval acc {acc:.3f}"
            )
            if progress_callback is not None:
                # Pass the underlying (uncompiled) model so checkpoints are clean.
                progress_callback(epoch + 1, model, {
                    "model": model, "losses": losses, "train_accs": train_accs,
                    "eval_accs": eval_accs, "eval_epochs": eval_epochs,
                })

    return {
        "model": model,
        "losses": losses,
        "train_accs": train_accs,
        "eval_accs": eval_accs,
        "eval_epochs": eval_epochs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ModMultDecoderOnly on modular multiplication or a finite group")
    parser.add_argument("--task", type=str, default="modmult", choices=["modmult", "icosahedral"],
                        help="modmult: a*b mod P. icosahedral: Cayley table of I_h = A5 x C2 (order 120)")
    parser.add_argument("--P", type=int, default=None,
                        help="Modulus for modmult (required for modmult; ignored/forced to the order for groups)")
    parser.add_argument("--epochs", type=int, default=25000)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.98])
    parser.add_argument("--train_frac", type=float, default=0.3)
    parser.add_argument("--eval_frac", type=float, default=0.3)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1, help="Base seed")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs (seeds: seed..seed+N-1)")
    parser.add_argument("--save_dir", type=str, default="experiments")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if omitted)")
    parser.add_argument("--force", action="store_true", help="Retrain even if matching checkpoint exists")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the model (recommended on CUDA for long runs)")
    parser.add_argument("--checkpoint_every", type=int, default=0,
                        help="Save a checkpoint every N epochs (0=only at end). Also keeps a "
                             "*_best.pt at the highest eval accuracy seen — captures the grok moment")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    # CUDA throughput: TF32 matmuls are a large speedup for this tiny model and
    # numerically harmless for a grokking study.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"CUDA: {torch.cuda.get_device_name(0)}, TF32 enabled")

    os.makedirs(args.save_dir, exist_ok=True)

    # Resolve the task: build a group Cayley-table dataset, or use modular mult.
    # Non-abelian groups need positional embeddings (operand order matters).
    dataset = None
    use_pos_embed = False
    if args.task == "icosahedral":
        from core.groups import icosahedral_group
        from core.data import generate_group_dataset
        G = icosahedral_group()
        P = G["n"]
        dataset = generate_group_dataset(G["table"], P)
        use_pos_embed = True
        print(f"Task: icosahedral I_h = A5 x C2 (order {P}), positional embeddings ON")
    else:
        if args.P is None:
            parser.error("--P is required for --task modmult")
        P = args.P

    for run in range(args.num_runs):
        seed = args.seed + run
        save_path = checkpoint_path(args.save_dir, P, args.embed_dim, args.n_heads, args.mlp_dim, seed, args.task)

        if not args.force and os.path.exists(save_path):
            print(f"\n=== Run {run + 1}/{args.num_runs} (seed={seed}) — SKIPPED (checkpoint exists: {save_path}) ===")
            continue

        print(f"\n=== Run {run + 1}/{args.num_runs} (seed={seed}) ===")

        # Periodic + best-eval checkpointing so a long grokking run captures the
        # transition (and survives interruption) rather than only the final state.
        best_path = save_path.replace(".pt", "_best.pt")
        state = {"best_eval": -1.0}

        def progress_callback(epoch, model, partial, _best=best_path, _save=save_path, _st=state):
            if args.checkpoint_every and epoch % args.checkpoint_every == 0:
                save_checkpoint(partial, P, args.embed_dim, args.n_heads, args.mlp_dim,
                                seed, _save, args.task, use_pos_embed)
            if partial["eval_accs"] and partial["eval_accs"][-1] > _st["best_eval"]:
                _st["best_eval"] = partial["eval_accs"][-1]
                if args.checkpoint_every:
                    save_checkpoint(partial, P, args.embed_dim, args.n_heads, args.mlp_dim,
                                    seed, _best, args.task, use_pos_embed)

        result = train_single_run(
            P=P, embed_dim=args.embed_dim, n_heads=args.n_heads, mlp_dim=args.mlp_dim,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas),
            train_frac=args.train_frac, eval_frac=args.eval_frac, eval_every=args.eval_every,
            seed=seed, device=device, dataset=dataset, use_pos_embed=use_pos_embed,
            compile_model=args.compile, progress_callback=progress_callback,
        )

        save_checkpoint(result, P, args.embed_dim, args.n_heads, args.mlp_dim, seed, save_path,
                        args.task, use_pos_embed)
        print(f"Best eval accuracy this run: {state['best_eval']:.4f}"
              + (f" (best checkpoint: {best_path})" if args.checkpoint_every else ""))


if __name__ == "__main__":
    main()
