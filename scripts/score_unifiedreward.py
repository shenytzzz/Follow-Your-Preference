"""Batch UnifiedReward scoring via vLLM server.

Streams prompts+image paths to a running UnifiedReward model; writes raw JSON
outputs. Use --resample to generate an alternate set under *_resampled.json.
"""

import argparse
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import os
from pathlib import Path
import numpy as np
import torch
import cv2
import json
import logging
from datasets import load_dataset
from PIL import Image
import time

import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a PowerPoint presentation from images."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="unified_reward",
        help="Metric to use for scoring images. Should be in [clip_score, hps, pick_score, image_reward, vqa_score , unified_reward].",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to save the scoring results."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument("--seed", type=int, help="local seed")
    parser.add_argument(
        "--annotations_path",
        type=str,
        required=True,
        default="annotations.json",
        help="Path to the annotations file.",
    )
    parser.add_argument(
        "--port", type=str, default="8080", help="Port to use for the VLLM server."
    )
    parser.add_argument("--start_step", type=int, default=0, help="")
    parser.add_argument("--resample", action="store_true")
    return parser.parse_args()


args = parse_args()

if args.seed is None and not args.resample:
    raise ValueError(
        "Please specify a seed using --seed or use --resample to resample the score."
    )


logging_dir = os.path.join(args.output_dir, f"{args.metric}", "logs")
if not os.path.exists(logging_dir):
    os.makedirs(logging_dir, exist_ok=True)
accelerator_project_config = ProjectConfiguration(
    project_dir=args.output_dir, logging_dir=logging_dir
)
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    project_config=accelerator_project_config,
)


logging.basicConfig(
    filename=f"{logging_dir}/log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


class MetricsCalculator:
    def __init__(self, metric, device="cuda"):
        self.device = device
        self.metric = metric
        if self.metric not in [
            "clip_score",
            "hps",
            "pick_score",
            "aesthetic_score",
            "image_reward",
            "vqa_score",
            "unified_reward",
        ]:
            raise ValueError(
                f"Unsupported metric: {self.metric}, should be one of [clip_score, hps, pick_score, aesthetic_score, image_reward, vqa_score , unified_reward]."
            )

        elif self.metric == "unified_reward":
            self.api_url = f"http://127.0.0.1:{args.port}"
            self.max_workers = 16
            self.vllm_output_path = (
                os.path.join(
                    args.output_dir,
                    f"{args.metric}",
                    f"vllm_results_{accelerator.process_index}_resampled.json",
                )
                if args.resample
                else os.path.join(
                    args.output_dir,
                    f"{args.metric}",
                    f"vllm_results_{accelerator.process_index}.json",
                )
            )
            open(self.vllm_output_path, "w").close()

    def calculate_unified_reward(self, image_paths: list[str], prompts: list[str]):
        from vllm_client_score import gen_results  # Wait for the VLLM server to start

        gen_results(
            image_paths=image_paths,
            prompts=prompts,
            output_path=self.vllm_output_path,
            api=self.api_url,
            max_workers=self.max_workers,
        )


def main():
    logging.info(accelerator.state)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    with open(args.annotations_path, "r") as f:
        annotations = json.load(f)

    metrics_calculator = MetricsCalculator(
        metric=args.metric, device=accelerator.device
    )

    results = {}
    start_step = args.start_step
    global_step = 0
    bs = 32
    length_of_annotations = len(annotations)
    total_step = (
        length_of_annotations // bs
        if length_of_annotations % bs == 0
        else (length_of_annotations // bs) + 1
    )

    while True:
        images = []
        captions = []
        i = 0
        start = bs * (global_step + start_step)

        logging.info(f"Processing item {global_step + start_step} / {total_step}...")

        while i < bs:

            item = annotations[start + i]
            images.append(
                item["image_path"].replace("local_seed_0", f"local_seed_{args.seed}")
                if not args.resample
                else item["image_path"]
            )
            captions.append(item["caption"])
            i += 1
            if start + i >= length_of_annotations:
                break

        if len(images) == 0:
            logging.error(f"No more items to process. Exiting...")
            break

        with torch.no_grad():
            if args.metric == "unified_reward":
                metrics_calculator.calculate_unified_reward(
                    image_paths=[
                        f"{os.path.join('/root/BrushNet', img)}" for img in images
                    ],
                    prompts=captions,
                )
            else:
                raise ValueError(
                    f"Unsupported metric: {args.metric}. This script is exclusive for UnifiedReward scoring."
                )

        global_step += 1
        if global_step + start_step >= total_step:
            break
        time.sleep(0.01)

    if args.metric != "unified_reward":
        with open(
            os.path.join(
                args.output_dir,
                f"{args.metric}",
                f"scores_{accelerator.process_index}.json",
            ),
            "w",
        ) as f:
            json.dump(results, f)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
