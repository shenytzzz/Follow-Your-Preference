import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Import after DPO_CONFIG is set by the caller so train_dpo resolves the smoke config.
import train_dpo  # noqa: E402


def main():
    cfg_path = os.environ.get("DPO_CONFIG")
    if not cfg_path:
        raise RuntimeError("DPO_CONFIG must point to the CaEN smoke config")

    with open(train_dpo.args.train_json_dir, "r") as f:
        examples = json.load(f)

    collate = train_dpo.MyDataset(
        resolution=train_dpo.args.resolution,
        random_mask=train_dpo.args.random_mask,
        model_type=train_dpo.model_type,
        tokenizer=None,
        score_file=train_dpo.args.metrics.score_file,
        metrics_enable=train_dpo.args.metrics.enable,
    )
    batch = collate(examples[:1])

    assert batch["pixel_values"].shape[0] == 1
    assert batch["inpainting_pixel_values"].shape[0] == 1
    assert batch["masks"].shape[0] == 1
    assert "win_scores" in batch and "lose_scores" in batch
    assert torch.isfinite(batch["win_scores"]).all()
    assert torch.isfinite(batch["lose_scores"]).all()

    score_delta = batch["win_scores"] - batch["lose_scores"]
    assert (score_delta > 0).all(), score_delta

    # Tiny differentiable stand-in for the train loop after model_pred/target.
    # This exercises the exact CaPO objective shape: model_diff/ref_diff -> inside_term -> loss -> backward.
    model_pred = torch.nn.Parameter(torch.randn(2, 4, 8, 8) * 0.01)
    target = torch.zeros_like(model_pred)
    ref_pred = torch.zeros_like(model_pred)
    loss_masks = torch.ones_like(model_pred)

    model_losses = ((model_pred - target) * loss_masks).pow(2).mean(dim=[1, 2, 3])
    model_losses_w, model_losses_l = model_losses.chunk(2)
    model_diff = model_losses_w - model_losses_l

    with torch.no_grad():
        ref_losses = ((ref_pred - target) * loss_masks).pow(2).mean(dim=[1, 2, 3])
        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
        ref_diff = ref_losses_w - ref_losses_l

    inside_term = -0.5 * train_dpo.args.beta_dpo * (model_diff - ref_diff)
    capo_loss = (score_delta.to(dtype=inside_term.dtype) - inside_term).pow(2).mean()
    capo_loss.backward()

    grad = model_pred.grad
    if grad is None:
        raise RuntimeError("No gradient was produced")
    if not torch.isfinite(grad).all():
        raise RuntimeError("Gradient contains non-finite values")
    grad_norm = grad.float().norm().item()
    if grad_norm <= 0:
        raise RuntimeError("Gradient norm is zero")

    print("CaEN/CaPO core train smoke passed")
    print(f"image_id={examples[0]['image_id']}")
    print(f"win_score={batch['win_scores'][0].item():.6f}")
    print(f"lose_score={batch['lose_scores'][0].item():.6f}")
    print(f"capo_loss={capo_loss.item():.6f}")
    print(f"grad_norm={grad_norm:.6e}")


if __name__ == "__main__":
    main()
