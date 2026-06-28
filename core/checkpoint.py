import os

import torch

from core.model import ModMultDecoderOnly


def experiment_name(P, embed_dim, n_heads, mlp_dim, seed, task="modmult"):
    """Build the experiment identifier.

    For the default modular-multiplication task the name is unchanged
    (P{P}_d..._s{seed}) for back-compat; other tasks are prefixed with the task
    name (e.g. icosahedral_P120_d128_h4_mlp512_s1) so checkpoints don't collide.
    """
    base = f"P{P}_d{embed_dim}_h{n_heads}_mlp{mlp_dim}_s{seed}"
    return base if task == "modmult" else f"{task}_{base}"


def checkpoint_path(save_dir, P, embed_dim, n_heads, mlp_dim, seed, task="modmult"):
    """Build checkpoint filepath."""
    name = experiment_name(P, embed_dim, n_heads, mlp_dim, seed, task)
    return os.path.join(save_dir, f"{name}.pt")


def image_dir(P, embed_dim, n_heads, mlp_dim, seed, base_dir="images", task="modmult"):
    """Build image output directory for an experiment."""
    name = experiment_name(P, embed_dim, n_heads, mlp_dim, seed, task)
    path = os.path.join(base_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_checkpoint(run_result, P, embed_dim, n_heads, mlp_dim, seed, save_path, task="modmult",
                    use_pos_embed=False):
    """Save model checkpoint with config and training history.

    Checkpoint keys: model_state_dict, seed, losses, train_accs, eval_accs, eval_epochs,
    and config dict with keys: P, embed_dim, num_heads, mlp_dim.
    Note: config uses 'embed_dim'/'num_heads'/'mlp_dim' which map to the constructor
    params d_model/n_heads/mlp_hidden.
    """
    checkpoint = {
        "model_state_dict": run_result["model"].state_dict(),
        "config": {
            "P": P,
            "embed_dim": embed_dim,
            "num_heads": n_heads,
            "mlp_dim": mlp_dim,
            "task": task,
            "use_pos_embed": use_pos_embed,
        },
        "seed": seed,
        "losses": run_result["losses"],
        "train_accs": run_result["train_accs"],
        "eval_accs": run_result["eval_accs"],
        "eval_epochs": run_result["eval_epochs"],
    }
    torch.save(checkpoint, save_path)
    print(f"Saved -> {save_path}")


def load_model(path, device):
    """Load a model from a checkpoint file.

    Returns:
        (model, checkpoint) where model is in eval mode and checkpoint is the full
        dict containing losses, train_accs, eval_accs, eval_epochs, seed, and config.
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]

    if config.get("arch") == "mlp":
        from core.mlp_model import MLPGroup
        model = MLPGroup(config["P"], config["embed_dim"], config["mlp_dim"]).to(device)
    else:
        model = ModMultDecoderOnly(
            config["P"], config["embed_dim"], config["num_heads"], config["mlp_dim"],
            use_pos_embed=config.get("use_pos_embed", False),
        ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint
