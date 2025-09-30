"""Generate inpainted images for BrushBench / EditBench.

Supports FLUX Fill and BrushNet models (baseline or fine-tuned). Optionally
creates a blended result by pasting generated region onto original with a
blurred mask for smoother boundaries.
"""

from diffusers import (
    FluxFillPipeline,
    FluxTransformer2DModel,
    StableDiffusionBrushNetPipeline,
    BrushNetModel,
    UniPCMultistepScheduler,
)
import torch
import cv2
import json
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
import torch
from torchvision.transforms import Resize
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from urllib.request import urlretrieve
from PIL import Image
import os
from contextlib import nullcontext
from typing import Optional
import math
from transformers import AutoProcessor, AutoModel
from accelerate import Accelerator
from torch.distributed import destroy_process_group
from diffusers.utils import load_image

# for vllm
import argparse
import concurrent.futures
import json
from tqdm import tqdm
from functools import partial

import requests
import base64
import os
import json
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import time

from accelerate.logging import get_logger
import logging
from copy import deepcopy


logger = get_logger(__name__)


def rle2mask(mask_rle, shape):  # height, width
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument(
        "--base_model_path", type=str, default="black-forest-labs/FLUX.1-Fill-dev"
    )
    parser.add_argument("--image_save_path", type=str, default="")
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="/root/data/BrushBench/mapping_file_list.json",
    )
    parser.add_argument("--base_dir", type=str, default="/root/data/BrushBench")
    parser.add_argument("--mask_key", type=str, default="inpainting_mask")
    parser.add_argument("--use_blended", action="store_true")
    parser.add_argument("--paintingnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--delete_images", action="store_true")
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--model", type=str, default="flux", required=True)
    parser.add_argument("--benchmark", type=str, default="brushbench")

    args = parser.parse_args()
    args.blended = args.use_blended

    accelerator = Accelerator()
    rank = accelerator.process_index
    num_processes = accelerator.num_processes
    device = accelerator.device

    base_model_path = args.base_model_path
    ckpt_path = args.ckpt_path

    logging_dir = os.path.join(args.image_save_path, "logs")
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(logging_dir, "log.txt"),
    )

    with open(args.mapping_file, "r") as f:
        mapping_file = json.load(f)

    if not args.baseline:

        if args.model == "flux":
            print(f"[{rank}] loading transformer from {ckpt_path} ...")
            transformer = FluxTransformer2DModel.from_pretrained(
                ckpt_path, torch_dtype=torch.bfloat16
            ).to(device)

            pipe = FluxFillPipeline.from_pretrained(
                base_model_path, torch_dtype=torch.bfloat16, transformer=transformer
            ).to(device)
        elif args.model == "brushnet":
            brushnet = BrushNetModel.from_pretrained(
                ckpt_path, torch_dtype=torch.bfloat16
            ).to(device)
            pipe = StableDiffusionBrushNetPipeline.from_pretrained(
                base_model_path,
                brushnet=brushnet,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            ).to(device)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    else:
        print(f"evaluating baseline ...")
        if args.model == "flux":
            pipe = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                torch_dtype=torch.bfloat16,
            ).to(device)
        elif args.model == "brushnet":
            brushnet = BrushNetModel.from_pretrained(
                "/root/BrushNet/data/ckpt/segmentation_mask_brushnet_ckpt/checkpoint-1000",
                torch_dtype=torch.bfloat16,
            ).to(device)
            pipe = StableDiffusionBrushNetPipeline.from_pretrained(
                base_model_path,
                brushnet=brushnet,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            ).to(device)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.set_progress_bar_config(disable=True)

    accelerator.wait_for_everyone()
    for i, item in enumerate(mapping_file):
        if i % num_processes != rank:
            continue

        if args.benchmark == "brushbench":
            key = item["image_id"]
            image_path = item["image"]
            mask = item[args.mask_key]
            caption = item["caption"]

            init_image = cv2.imread(
                os.path.join(args.base_dir, image_path), cv2.IMREAD_COLOR
            )[:, :, ::-1]

            mask_image = rle2mask(mask, (512, 512))[:, :, np.newaxis]
            masked_image = init_image * (1 - mask_image)

            masked_image = Image.fromarray(masked_image).convert("RGB")
            init_image = Image.fromarray(init_image).convert("RGB")
            mask_image = Image.fromarray(mask_image.repeat(3, -1) * 255).convert("RGB")

            save_path = os.path.join(args.image_save_path, image_path)
            masked_image_save_path = save_path.replace(".jpg", "_masked.jpg")
            if args.blended:
                blended_image_save_path = save_path.replace(".jpg", "_blended.jpg")

        elif args.benchmark == "editbench":
            key = item["aos"]
            image_path = item["image_path"]
            save_path = f"images/{key}_{item['type']}.jpg"
            masked_image_path = item["masked_image_path"]
            caption = item["caption"]
            print(f"[{rank}] generating image {image_path} ...")
            logger.info(
                f"[{rank}] generating image {image_path} ...", main_process_only=False
            )

            init_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
            init_image = cv2.resize(
                init_image, (512, 512), interpolation=cv2.INTER_CUBIC
            )
            init_img = deepcopy(init_image)

            masked_image = cv2.imread(masked_image_path, cv2.IMREAD_COLOR)
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            masked_image = cv2.resize(
                masked_image, (512, 512), interpolation=cv2.INTER_CUBIC
            )
            image_diff = masked_image.sum(axis=2, keepdims=True)
            mask = np.zeros_like(image_diff)
            mask[image_diff < 15] = 1  # hacking for no masks are provided
            mask = mask.astype(np.uint8)

            masked_image = Image.fromarray(masked_image)
            init_image = Image.fromarray(init_image)
            mask_image = Image.fromarray(
                ((mask) * 255).squeeze().astype(np.uint8), mode="L"
            )

            save_path = os.path.join(args.image_save_path, save_path)
            masked_image_save_path = save_path.replace(".jpg", "_masked.jpg")
            gt_save_path = save_path.replace(".jpg", "_gt.jpg")
            if args.blended:
                blended_image_save_path = save_path.replace(".jpg", "_blended.jpg")

        print(f"[{rank}] generating image {key} ...")
        with torch.no_grad():
            if args.model == "flux":
                image = pipe(
                    prompt=caption,
                    image=init_image,
                    mask_image=mask_image,
                    height=512,
                    width=512,
                    num_inference_steps=args.num_steps,
                    max_sequence_length=512,
                    guidance_scale=30,
                    generator=torch.Generator("cuda").manual_seed(42),
                ).images[0]

            elif args.model == "brushnet":
                image = pipe(
                    caption,
                    masked_image,
                    mask_image,
                    num_inference_steps=args.num_steps,
                    generator=torch.Generator("cuda").manual_seed(42),
                    paintingnet_conditioning_scale=args.paintingnet_conditioning_scale,
                ).images[0]

        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        if args.blended:
            mask_np = rle2mask(mask, (512, 512))[:, :, np.newaxis]
            image_np = np.array(image)
            init_image_np = cv2.imread(os.path.join(args.base_dir, image_path))[
                :, :, ::-1
            ]

            # blur
            mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
            mask_blurred = mask_blurred[:, :, np.newaxis]
            mask_np = 1 - (1 - mask_np) * (1 - mask_blurred)

            image_pasted = init_image_np * (1 - mask_np) + image_np * mask_np
            image_pasted = image_pasted.astype(image_np.dtype)
            blended_image = Image.fromarray(image_pasted)

        image.save(save_path)
        if args.blended:
            blended_image.save(blended_image_save_path)
        masked_image.save(masked_image_save_path)

    print(f"[{rank}] The generated images are saved in {args.image_save_path}")
    accelerator.wait_for_everyone()
    destroy_process_group()
