"""
tool.py -- Score JSON processing toolkit

Usage:
    python tool.py <command> [options]

Commands:
    merge-annotations       Merge per-image annotation JSONs from a single seed into one results.json
    format-vllm-scores      Parse vllm JSONL result files and extract per-image per-seed scores
    find-vllm-resamples     Scan vllm result files for failed inference records to resample
    merge-scores            Merge JSON files across multiple score types into one combined file
    add-extra-scores        Inject a new score type from a separate directory into an existing scores file
    create-ensemble         Rank each score type per seed, sum weighted ranks, add min/max for ensemble
    create-scaling          For scale factors [2,4,8], find best/worst seeds within top-N candidates
    calculate-capo          Compute CaPO reward via sigmoid pairwise comparison across score types
    calculate-caen          Compute CaEN reward via step-function pairwise comparison with per-type weights
    generate-user-study     Sample annotation items and produce an image-pair JSONL for human evaluation

Run `python tool.py <command> --help` for per-command options.
"""

import argparse
import copy
import json
import os
import random
from collections import defaultdict, Counter

import numpy as np


DEFAULT_CAEN_WEIGHTS = {
    "hps": 5 / 7,
    "clip_score": 2,
    "pick_score": 5 / 7,
    "aesthetic_score": 5 / 7,
    "image_reward": 5 / 7,
    "unified_reward": 5 / 7,
    "pe_score": 2,
    "hpsv3": 5 / 7,
    "vqa_score": 5 / 7,
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _add_min_max(data: dict, score_type_filter: str = None) -> dict:
    """Return a deep copy with notebook-style max/min annotations.

    The original notebook uses values.index(max/min), so ties are resolved by
    taking the first seed encountered in JSON order. That also keeps all-equal
    score types as draw=True with max_seed == min_seed.
    """
    new_data = copy.deepcopy(data)
    for image_id, score_types in new_data.items():
        for score_type, seed_scores in score_types.items():
            if score_type_filter is not None and score_type != score_type_filter:
                continue
            pairs = [
                (s, v) for s, v in seed_scores.items()
                if s.startswith("seed_") and isinstance(v, (int, float))
            ]
            if not pairs:
                continue
            seeds, values = zip(*pairs)
            max_idx = values.index(max(values))
            min_idx = values.index(min(values))
            seed_scores["draw"] = (max_idx == min_idx)
            seed_scores["max_value"] = values[max_idx]
            seed_scores["max_seed"] = seeds[max_idx]
            seed_scores["min_value"] = values[min_idx]
            seed_scores["min_seed"] = seeds[min_idx]
    return new_data


def _add_min_max_ensemble(data: dict, random_seed: int = 42) -> dict:
    """Variant for ensemble: the min-rank seed is actually the winner (lowest rank = best)."""
    rng = random.Random(random_seed)
    new_data = copy.deepcopy(data)
    for image_id, score_types in new_data.items():
        for score_type, seed_scores in score_types.items():
            if score_type != "ensemble":
                continue
            pairs = [
                (s, v) for s, v in seed_scores.items()
                if s.startswith("seed_") and isinstance(v, (int, float))
            ]
            if not pairs:
                continue
            seeds, values = zip(*pairs)
            min_val, max_val = min(values), max(values)
            min_idxs = [i for i, v in enumerate(values) if v == min_val]
            max_idxs = [i for i, v in enumerate(values) if v == max_val]
            max_idx = rng.choice(min_idxs)  # winner = lowest rank value
            min_idx = rng.choice(max_idxs)  # loser  = highest rank value
            seed_scores["draw"] = (max_idx == min_idx)
            seed_scores["max_value"] = values[max_idx]
            seed_scores["max_seed"] = seeds[max_idx]
            seed_scores["min_value"] = values[min_idx]
            seed_scores["min_seed"] = seeds[min_idx]
    return new_data


def _reorder_scores(original: dict) -> dict:
    """Reshape {image_id: {seed_id: {score_type: val}}} -> {image_id: {score_type: {seed_id: val}}}."""
    new = {}
    for image_id, seeds in original.items():
        new[image_id] = {}
        for seed_id, score_dict in seeds.items():
            for score_type, score_value in score_dict.items():
                new[image_id].setdefault(score_type, {})[seed_id] = score_value
    return new


def _merge_score_dicts(dicts: list) -> dict:
    merged = {}
    for d in dicts:
        for image_id, score_types in d.items():
            merged.setdefault(image_id, {})
            for score_type, seed_scores in score_types.items():
                merged[image_id].setdefault(score_type, {}).update(seed_scores)
    return merged


def _rank_scores(scores: dict, descending: bool = True) -> dict:
    vals = list(scores.values())
    counts = Counter(vals)
    uniq_vals = sorted(counts.keys(), reverse=descending)
    value_to_rank = {v: i + 1 for i, v in enumerate(uniq_vals)}
    return {k: value_to_rank[v] for k, v in scores.items()}


def _validate_seed_count(data: dict, num_seeds: int) -> None:
    for image_id, seed_scores in data.items():
        seed_keys = [k for k in seed_scores if k.startswith("seed_")]
        if len(seed_keys) != num_seeds:
            raise AssertionError(
                f"Image {image_id} does not have scores for all seeds: "
                f"{len(seed_keys)}/{num_seeds}"
            )


def _validate_scores_complete(data: dict, score_types: list, num_seeds: int) -> None:
    expected_seeds = {f"seed_{i}" for i in range(num_seeds)}
    for image_id, scores in data.items():
        missing_types = [score_type for score_type in score_types if score_type not in scores]
        if missing_types:
            raise AssertionError(
                f"Image {image_id} does not have all scores, missing {missing_types}"
            )
        for score_type in score_types:
            seed_keys = {k for k in scores[score_type] if k.startswith("seed_")}
            missing_seeds = sorted(expected_seeds - seed_keys)
            if len(seed_keys) != num_seeds or missing_seeds:
                raise AssertionError(
                    f"Image {image_id} does not have all seeds for score {score_type}: "
                    f"{len(seed_keys)}/{num_seeds}, missing {missing_seeds}"
                )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_merge_annotations(args):
    """
    Merge per-image annotation JSONs in <seed_dir>/annotations/ into a single results.json.

    Options:
        --seed-dir   Path to local_seed_X directory (must contain an 'annotations' sub-folder)
        --out-dir    Directory where results.json is written
    """
    annotations_dir = os.path.join(args.seed_dir, "annotations")
    results = []
    for fname in os.listdir(annotations_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(annotations_dir, fname)
        print(f"Processing {fpath}")
        with open(fpath) as f:
            data = json.load(f)
        for k, v in data.items():
            v["image_id"] = k
            results.append(v)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} annotations -> {out_path}")


def cmd_format_vllm_scores(args):
    """
    Parse vllm JSONL result files across multiple seeds and save per-image per-seed scores.

    The input layout is expected to be:
        <score-dir>/local_seed_<N>/unified_reward/vllm_results_0.json

    Options:
        --score-dir   Root directory containing local_seed_* sub-folders
        --out-dir     Directory where scores_0.json is written
        --num-seeds   Number of seeds (default: 16)
        --score-file  Name of the vllm result file inside unified_reward/ (default: vllm_results_0.json)
    """
    from collections import defaultdict
    results = defaultdict(dict)

    def process_file(fpath):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                image_id = record.get("image_id", "")
                if image_id.startswith("images_"):
                    image_id = image_id[len("images_"):]
                if not image_id.endswith(".jpg"):
                    image_id += ".jpg"
                final_score = record.get("final_score")
                for path in record.get("image_path", []):
                    if "local_seed_" in path:
                        for part in path.split("/"):
                            if part.startswith("local_seed_"):
                                seed_key = "seed_" + part.replace("local_seed_", "")
                                results[image_id][seed_key] = {"unified_reward": final_score}
                                break

    seeds = [f"local_seed_{i}" for i in range(args.num_seeds)]
    for seed in seeds:
        fpath = os.path.join(args.score_dir, seed, "unified_reward", args.score_file)
        if not os.path.exists(fpath):
            print(f"  Warning: {fpath} not found, skipping")
            continue
        print(f"Processing {fpath}")
        process_file(fpath)

    results = dict(results)
    _validate_seed_count(results, args.num_seeds)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "scores_0.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved scores for {len(results)} images -> {out_path}")


def cmd_find_vllm_resamples(args):
    """
    Scan vllm JSONL result files across seeds and collect failed inference records for resampling.

    Options:
        --score-dirs       One or more root score directories
        --annotations-file Path to the existing annotations JSON list (for captions)
        --out-dir          Directory where resamples.json is written
        --num-seeds        Number of seeds (default: 16)
    """
    with open(args.annotations_file) as f:
        ann_list = json.load(f)
    all_annotations = {item["image_id"]: item for item in ann_list}

    resamples = []
    for base_dir in args.score_dirs:
        for seed in range(args.num_seeds):
            seed_path = os.path.join(base_dir, f"local_seed_{seed}", "unified_reward")
            if not os.path.isdir(seed_path):
                continue
            score_files = sorted(os.listdir(seed_path))
            for score_file in score_files:
                if not (score_file.startswith("vllm_results") and score_file.endswith(".json")):
                    continue
                fpath = os.path.join(seed_path, score_file)
                print(f"Processing {fpath}")
                with open(fpath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        if not record.get("success") and \
                                "HTTPConnectionPool" not in record.get("error", "") and \
                                "Server Error" not in record.get("error", ""):
                            img_id = record["image_id"]
                            caption = all_annotations.get(img_id, {}).get("caption", "")
                            resamples.append({
                                "image_id": img_id,
                                "image_path": record.get("image_path", [None])[0],
                                "caption": caption,
                            })

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "resamples.json")
    with open(out_path, "w") as f:
        json.dump(resamples, f, indent=2)
    print(f"Found {len(resamples)} failed records -> {out_path}")


def cmd_merge_scores(args):
    """
    Merge JSON files across multiple score types into a single combined file,
    then annotate each score with max/min seed info.

    The expected layout is:
        <score-base>/<score-type>/*.json
    where each JSON has the structure {image_id: {seed_id: {score_type: value}}}.

    Options:
        --score-base   Base directory containing one sub-folder per score type
        --score-types  Space-separated list of score type names (sub-folder names)
        --out-dir      Directory where new_results.json is written
        --num-seeds    Number of seeds expected per image (default: 16)
    """
    score_levels = []
    for score_type in args.score_types:
        score_level = {}
        type_dir = os.path.join(args.score_base, score_type)
        for fname in os.listdir(type_dir):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(type_dir, fname)
            print(f"Processing {fpath}")
            with open(fpath) as f:
                score_level.update(json.load(f))
        score_levels.append(_reorder_scores(score_level))

    merged = _merge_score_dicts(score_levels)
    _validate_scores_complete(merged, args.score_types, args.num_seeds)
    merged_with_mm = _add_min_max(merged)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "new_results.json")
    with open(out_path, "w") as f:
        json.dump(merged_with_mm, f, indent=2)
    print(f"Merged {len(merged_with_mm)} images -> {out_path}")


def cmd_add_extra_scores(args):
    """
    Inject a new score type from a separate directory into an existing scores file.

    The new score directory should contain JSON files with structure:
        {image_id: {seed_id: {<score-type>: value}}}

    Options:
        --new-scores-dir   Directory containing new score JSON files
        --cur-scores-file  Existing merged scores JSON file to update
        --score-type       Name of the new score type to inject (default: hpsv3)
        --num-seeds        Number of seeds to expect per image (default: 16)
        --out-file         Output path for updated scores JSON
    """
    new_scores = {}
    for fname in os.listdir(args.new_scores_dir):
        if fname.endswith(".json"):
            with open(os.path.join(args.new_scores_dir, fname)) as f:
                new_scores.update(json.load(f))

    with open(args.cur_scores_file) as f:
        cur_scores = json.load(f)

    for image_id in cur_scores:
        if image_id not in new_scores:
            raise ValueError(f"Image ID {image_id} not found in new scores")
        score_by_seed = {}
        for seed in range(args.num_seeds):
            score_by_seed[f"seed_{seed}"] = new_scores[image_id][f"seed_{seed}"][args.score_type]
        cur_scores[image_id][args.score_type] = score_by_seed

    updated = _add_min_max(cur_scores, score_type_filter=args.score_type)

    out_file = args.out_file or args.cur_scores_file.replace(".json", "_updated.json")
    with open(out_file, "w") as f:
        json.dump(updated, f, indent=2)
    print(f"Updated scores -> {out_file}")


def cmd_create_ensemble(args):
    """
    Build an ensemble score by ranking seeds within each score type, optionally applying
    per-model accuracy weights, then summing ranks across score types.

    Supported weighting modes:
        --weight-mode none       Raw rank sum (default)
        --weight-mode softmax    Scale ranks by (1 - softmax weight)
        --weight-mode linear     Scale ranks by (1 - linear-normalized weight)
        --weight-mode custom     Scale ranks by (1 - custom linear weight)

    Options:
        --input-file    Path to merged scores JSON (output of merge-scores)
        --score-types   Score type names to include in ensemble
        --weight-mode   Weighting strategy: none | softmax | linear | custom (default: none)
        --weights       JSON string mapping score_type -> weight value (used for all modes except none)
        --out-file      Output path for ensemble JSON
    """
    with open(args.input_file) as f:
        data = json.load(f)

    _validate_scores_complete(data, args.score_types, args.num_seeds)

    weights_map = {}
    if args.weight_mode != "none":
        if args.weights:
            raw_weights = json.loads(args.weights)
        else:
            raw_weights = {st: 1.0 for st in args.score_types}

        if args.weight_mode == "softmax":
            vals = np.array([raw_weights[k] for k in args.score_types])
            exp_vals = np.exp(vals - vals.max())
            softmax_vals = exp_vals / exp_vals.sum()
            weights_map = dict(zip(args.score_types, softmax_vals.tolist()))
        else:
            total = sum(raw_weights[k] for k in args.score_types)
            weights_map = {k: raw_weights[k] / total for k in args.score_types}

    for image_id, scores in data.items():
        for score_type in args.score_types:
            seed_vals = {k: v for k, v in scores[score_type].items() if k.startswith("seed_")}
            rank = _rank_scores(seed_vals)
            if weights_map:
                w = 1.0 - weights_map.get(score_type, 0.0)
                rank = {k: v * w for k, v in rank.items()}
            if scores[score_type].get("draw"):
                rank = {k: 0 for k in rank}
            scores[score_type]["rank"] = rank

    for image_id, scores in data.items():
        ensemble = {}
        for seed_idx in range(args.num_seeds):
            seed_key = f"seed_{seed_idx}"
            ensemble[seed_key] = sum(
                scores[st]["rank"].get(seed_key, 0) for st in args.score_types
            )
        scores["ensemble"] = ensemble

    result = _add_min_max_ensemble(data)

    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Ensemble scores -> {args.out_file}")


def cmd_create_scaling(args):
    """
    For each image, find the best and worst seed within the first N candidates
    for multiple scaling factors.

    Options:
        --input-file     Path to scores JSON with ensemble field
        --scalings       Space-separated scaling factors, e.g. 2 4 8 (default: 2 4 8)
        --num-seeds      Total number of seeds (default: 16)
        --random-seed    Random seed for tie-breaking (default: 42)
        --out-file       Output path
    """
    with open(args.input_file) as f:
        scores = json.load(f)

    rng = random.Random(args.random_seed)
    seeds_list = [f"seed_{i}" for i in range(args.num_seeds)]
    new_results = {}

    for key, value in scores.items():
        new_results[key] = {}
        ensemble = copy.deepcopy(value["ensemble"])
        for scaling in args.scalings:
            subset = {s: ensemble[s] for s in seeds_list[:scaling]}
            min_val = min(subset.values())
            max_val = max(subset.values())
            min_candidates = [s for s, v in subset.items() if v == max_val]  # highest rank = loser
            max_candidates = [s for s, v in subset.items() if v == min_val]  # lowest rank = winner
            max_seed = rng.choice(max_candidates)
            min_seed = rng.choice(min_candidates)
            while min_seed == max_seed and len(min_candidates) > 1:
                min_seed = rng.choice(min_candidates)
            ensemble[f"max_seed_{scaling}"] = max_seed
            ensemble[f"max_value_{scaling}"] = subset[max_seed]
            ensemble[f"min_seed_{scaling}"] = min_seed
            ensemble[f"min_value_{scaling}"] = subset[min_seed]
        new_results[key]["ensemble"] = ensemble

    with open(args.out_file, "w") as f:
        json.dump(new_results, f, indent=2)
    print(f"Scaling results -> {args.out_file}")


def cmd_calculate_capo(args):
    """
    Compute CaPO reward: for each score type, apply sigmoid pairwise comparison between seeds,
    then average the win-rates across all score types to produce a CaPO score per seed.

    Options:
        --input-file    Path to merged scores JSON
        --score-types   Score type names to include
        --num-seeds     Number of seeds (default: 16)
        --out-file      Output path
    """
    from scipy.special import expit

    with open(args.input_file) as f:
        scores = json.load(f)

    _validate_scores_complete(scores, args.score_types, args.num_seeds)

    new_dict = {}
    for image_id, score_types in scores.items():
        new_dict[image_id] = {}
        for score_type, score_values in score_types.items():
            if score_type not in args.score_types:
                continue
            arr = np.array([score_values[f"seed_{i}"] for i in range(args.num_seeds)])
            diff = arr[:, None] - arr[None, :]
            mat = expit(diff)
            np.fill_diagonal(mat, 0.0)
            sums = mat.sum(axis=1) / (args.num_seeds - 1)
            new_dict[image_id][score_type] = {f"seed_{i}": float(sums[i]) for i in range(args.num_seeds)}

    score_num = len(args.score_types)
    for image_id, score_types in new_dict.items():
        capo = {}
        for i in range(args.num_seeds):
            capo[f"seed_{i}"] = sum(
                score_types[st][f"seed_{i}"] for st in args.score_types if st in score_types
            ) / score_num
        new_dict[image_id]["CaPO"] = capo

    result = _add_min_max(new_dict)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"CaPO scores -> {args.out_file}")


def cmd_calculate_caen(args):
    """
    Compute CaEN (Calibrated Ensemble) reward: for each score type, use a step-function
    pairwise comparison (win=1, lose=0), then compute a weighted average across score types.

    Options:
        --input-file    Path to merged scores JSON
        --score-types   Score type names to include
        --weights       JSON string mapping score_type -> weight (default: notebook CaEN weights)
        --num-seeds     Number of seeds (default: 16)
        --out-file      Output path
    """
    with open(args.input_file) as f:
        scores = json.load(f)

    _validate_scores_complete(scores, args.score_types, args.num_seeds)

    if args.weights:
        weights = json.loads(args.weights)
    else:
        weights = DEFAULT_CAEN_WEIGHTS

    new_dict = {}
    for image_id, score_types in scores.items():
        new_dict[image_id] = {}
        for score_type, score_values in score_types.items():
            if score_type not in args.score_types:
                continue
            arr = np.array([score_values[f"seed_{i}"] for i in range(args.num_seeds)])
            diff = arr[:, None] - arr[None, :]
            mat = (diff > 0).astype(float)
            np.fill_diagonal(mat, 0.0)
            sums = mat.sum(axis=1) / (args.num_seeds - 1)
            new_dict[image_id][score_type] = {f"seed_{i}": float(sums[i]) for i in range(args.num_seeds)}

    score_num = len(args.score_types)
    for image_id, score_types in new_dict.items():
        caen = {}
        for i in range(args.num_seeds):
            caen[f"seed_{i}"] = sum(
                score_types.get(st, {}).get(f"seed_{i}", 0.0) * weights.get(st, 1.0)
                for st in args.score_types
            ) / score_num
        new_dict[image_id]["CaEN"] = caen

    result = _add_min_max(new_dict)
    with open(args.out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"CaEN scores -> {args.out_file}")


def cmd_generate_user_study(args):
    """
    Sample annotation items and build an image-pair comparison JSON for human evaluation.
    For each sampled item, the best and worst seeds (according to ensemble) are paired.

    Options:
        --scores-file       Path to merged scores JSON with ensemble field
        --annotations-file  Path to annotations list JSON
        --out-file          Output path for user-study JSON
        --num-samples       Number of items to sample (default: 100)
        --random-seed       Random seed (default: 1235)
        --reward-models     Space-separated reward model names to include in choices
        --mask-out-dir      Directory to save processed mask images (optional)
        --num-seeds         Total seeds per image (default: 16)
    """
    import cv2
    from PIL import Image

    def rle2mask(mask_rle, shape):
        mask_rle = np.array(mask_rle)
        starts, lengths = (
            np.asarray(x, dtype=int)
            for x in (mask_rle[0:][::2], mask_rle[1:][::2])
        )
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape, order="F")

    with open(args.scores_file) as f:
        scores = json.load(f)
    with open(args.annotations_file) as f:
        annotations = json.load(f)

    random.seed(args.random_seed)
    items = random.sample(annotations, args.num_samples)
    results = []

    for item in items:
        image_id = item["image_id"]

        if args.mask_out_dir:
            height, width = item["height"], item["width"]
            mask_rle = list(map(int, item["mask"].split(",")))
            mask = rle2mask(mask_rle, (height, width))[:, :, np.newaxis]
            w, h, _ = mask.shape
            scale = 512 / h if w > h else 512 / w
            w_new, h_new = int(np.ceil(w * scale)), int(np.ceil(h * scale))
            crop = item["crop"]
            mask = cv2.resize(mask, (h_new, w_new), interpolation=cv2.INTER_CUBIC)[:, :, np.newaxis]
            mask = mask[crop[0]:crop[0] + 512, crop[1]:crop[1] + 512, :]
            mask = (1 - mask) * 255
            os.makedirs(args.mask_out_dir, exist_ok=True)
            Image.fromarray(mask.squeeze(), mode="L").save(
                os.path.join(args.mask_out_dir, f"{image_id}.jpg")
            )

        seed_pair = [scores[image_id]["ensemble"]["max_seed"], scores[image_id]["ensemble"]["min_seed"]]
        random.shuffle(seed_pair)
        seed_1, seed_2 = seed_pair

        image_1 = item["image_path"].replace("local_seed_0", f"local_{seed_1}")
        image_2 = item["image_path"].replace("local_seed_0", f"local_{seed_2}")

        choice = {
            "image_id": image_id,
            "mask": os.path.join(args.mask_out_dir, f"{image_id}.jpg") if args.mask_out_dir else "",
            "caption": item["caption"],
            "figure_1": image_1,
            "figure_2": image_2,
            "ensemble": 1 if seed_1 == scores[image_id]["ensemble"]["max_seed"] else 2,
        }
        for model in args.reward_models:
            s1 = scores[image_id][model].get(seed_1, 0)
            s2 = scores[image_id][model].get(seed_2, 0)
            choice[model] = 1 if s1 > s2 else (2 if s1 < s2 else 0)
        results.append(choice)

    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"User study ({len(results)} items) -> {args.out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # merge-annotations
    p = sub.add_parser("merge-annotations", help=cmd_merge_annotations.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_merge_annotations.__doc__)
    p.add_argument("--seed-dir", required=True, help="Path to local_seed_X directory")
    p.add_argument("--out-dir", required=True, help="Output directory for results.json")

    # format-vllm-scores
    p = sub.add_parser("format-vllm-scores", help=cmd_format_vllm_scores.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_format_vllm_scores.__doc__)
    p.add_argument("--score-dir", required=True, help="Root directory with local_seed_* sub-folders")
    p.add_argument("--out-dir", required=True, help="Output directory for scores_0.json")
    p.add_argument("--num-seeds", type=int, default=16, help="Number of seeds (default: 16)")
    p.add_argument("--score-file", default="vllm_results_0.json",
                   help="Filename inside unified_reward/ (default: vllm_results_0.json)")

    # find-vllm-resamples
    p = sub.add_parser("find-vllm-resamples", help=cmd_find_vllm_resamples.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_find_vllm_resamples.__doc__)
    p.add_argument("--score-dirs", nargs="+", required=True, help="One or more root score directories")
    p.add_argument("--annotations-file", required=True, help="Existing annotations JSON list")
    p.add_argument("--out-dir", required=True, help="Output directory for resamples.json")
    p.add_argument("--num-seeds", type=int, default=16, help="Number of seeds (default: 16)")

    # merge-scores
    p = sub.add_parser("merge-scores", help=cmd_merge_scores.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_merge_scores.__doc__)
    p.add_argument("--score-base", required=True, help="Base directory with one sub-folder per score type")
    p.add_argument("--score-types", nargs="+", required=True, help="Score type names (sub-folder names)")
    p.add_argument("--out-dir", required=True, help="Output directory for new_results.json")
    p.add_argument("--num-seeds", type=int, default=16, help="Seeds per image (default: 16)")

    # add-extra-scores
    p = sub.add_parser("add-extra-scores", help=cmd_add_extra_scores.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_add_extra_scores.__doc__)
    p.add_argument("--new-scores-dir", required=True, help="Directory with new score JSON files")
    p.add_argument("--cur-scores-file", required=True, help="Existing merged scores JSON to update")
    p.add_argument("--score-type", default="hpsv3", help="Name of new score type (default: hpsv3)")
    p.add_argument("--num-seeds", type=int, default=16, help="Seeds per image (default: 16)")
    p.add_argument("--out-file", default=None, help="Output file (default: <cur>_updated.json)")

    # create-ensemble
    p = sub.add_parser("create-ensemble", help=cmd_create_ensemble.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_create_ensemble.__doc__)
    p.add_argument("--input-file", required=True, help="Merged scores JSON")
    p.add_argument("--score-types", nargs="+", required=True, help="Score types to include")
    p.add_argument("--weight-mode", choices=["none", "softmax", "linear", "custom"], default="none",
                   help="Weighting strategy (default: none)")
    p.add_argument("--weights", default=None,
                   help='JSON string: {"score_type": weight_value, ...}')
    p.add_argument("--num-seeds", type=int, default=16, help="Number of seeds (default: 16)")
    p.add_argument("--out-file", required=True, help="Output file")

    # create-scaling
    p = sub.add_parser("create-scaling", help=cmd_create_scaling.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_create_scaling.__doc__)
    p.add_argument("--input-file", required=True, help="Scores JSON with ensemble field")
    p.add_argument("--scalings", nargs="+", type=int, default=[2, 4, 8],
                   help="Scaling factors (default: 2 4 8)")
    p.add_argument("--num-seeds", type=int, default=16, help="Total seeds (default: 16)")
    p.add_argument("--random-seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--out-file", required=True, help="Output file")

    # calculate-capo
    p = sub.add_parser("calculate-capo", help=cmd_calculate_capo.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_calculate_capo.__doc__)
    p.add_argument("--input-file", required=True, help="Merged scores JSON")
    p.add_argument("--score-types", nargs="+", required=True, help="Score types to include")
    p.add_argument("--num-seeds", type=int, default=16, help="Number of seeds (default: 16)")
    p.add_argument("--out-file", required=True, help="Output file")

    # calculate-caen
    p = sub.add_parser("calculate-caen", help=cmd_calculate_caen.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_calculate_caen.__doc__)
    p.add_argument("--input-file", required=True, help="Merged scores JSON")
    p.add_argument("--score-types", nargs="+", required=True, help="Score types to include")
    p.add_argument("--weights", default=None,
                   help='JSON string: {"score_type": weight, ...} (default: notebook CaEN weights)')
    p.add_argument("--num-seeds", type=int, default=16, help="Number of seeds (default: 16)")
    p.add_argument("--out-file", required=True, help="Output file")

    # generate-user-study
    p = sub.add_parser("generate-user-study", help=cmd_generate_user_study.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=cmd_generate_user_study.__doc__)
    p.add_argument("--scores-file", required=True, help="Merged scores JSON with ensemble field")
    p.add_argument("--annotations-file", required=True, help="Annotations list JSON")
    p.add_argument("--out-file", required=True, help="Output JSON file")
    p.add_argument("--num-samples", type=int, default=100, help="Items to sample (default: 100)")
    p.add_argument("--random-seed", type=int, default=1235, help="Random seed (default: 1235)")
    p.add_argument("--reward-models", nargs="+",
                   default=["hps", "vqa_score", "clip_score", "pick_score",
                            "aesthetic_score", "image_reward", "unified_reward", "pe_score", "hpsv3"],
                   help="Reward model names to record per-model choices")
    p.add_argument("--mask-out-dir", default=None, help="Directory to save processed mask images")
    p.add_argument("--num-seeds", type=int, default=16, help="Total seeds per image (default: 16)")

    args = parser.parse_args()

    dispatch = {
        "merge-annotations": cmd_merge_annotations,
        "format-vllm-scores": cmd_format_vllm_scores,
        "find-vllm-resamples": cmd_find_vllm_resamples,
        "merge-scores": cmd_merge_scores,
        "add-extra-scores": cmd_add_extra_scores,
        "create-ensemble": cmd_create_ensemble,
        "create-scaling": cmd_create_scaling,
        "calculate-capo": cmd_calculate_capo,
        "calculate-caen": cmd_calculate_caen,
        "generate-user-study": cmd_generate_user_study,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
