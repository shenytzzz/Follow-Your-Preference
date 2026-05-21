# CaEN Smoke Dataset

Tiny CaEN subset for checking `scripts/train_dpo.py` CaPO/CaEN training wiring.

Files:
- `annotations/results.json`: 2 synthetic annotation rows with repository-relative image paths.
- `scores/caen_scores.json`: CaEN scores for the selected `image_id`s only.
- `batch_1/local_seed_*`: local copies of the seed images needed by max/min CaEN selection.
- `gt_images`: copied GT images for completeness.
- `configs_dpo_caen_smoke.yaml`: one-step smoke config.
- `configs_dpo_caen_smoke_cpu.yaml`: CPU-oriented one-step smoke config.
- `configs_dpo_caen_smoke_cpu_tiny.yaml`: tiny CPU smoke config validated with `conda run -n flux`.

Run example:

```bash
CUDA_VISIBLE_DEVICES="" DPO_CONFIG=smoke_data/caen_train/configs_dpo_caen_smoke_cpu_tiny.yaml conda run -n flux python scripts/train_dpo.py
```
