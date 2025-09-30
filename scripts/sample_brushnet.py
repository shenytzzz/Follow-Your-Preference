#!/usr/bin/env python
# coding=utf-8
"""Sample BrushNet outputs for a subset of training data.

Performs on-the-fly mask selection & cropping to ensure non-trivial masked
regions, then runs the BrushNet pipeline to save images plus metadata.
"""

import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import json
import cv2

# import imgaug.augmenters as iaa

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DistributedSampler
from torch.distributed import destroy_process_group
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    BrushNetModel,
    DDPMScheduler,
    StableDiffusionBrushNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from itertools import islice

from omegaconf import OmegaConf

import datetime


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.27.0.dev0")

full_args = OmegaConf.load("/root/BrushNet/configs/configs.yaml")
args = full_args.sample
logging_dir = Path(args.output_dir, args.logging_dir)
accelerator_project_config = ProjectConfiguration(
    project_dir=args.output_dir, logging_dir=logging_dir
)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
)
logger = get_logger(__name__, log_file=os.path.join(args.output_dir, "log.txt"))

selected_files = None


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


class MyDataset:
    """Collate function: decodes RLE masks, rescales, crops, filters masks."""
    def __init__(self, resolution=512, rank=-1):
        self.resolution = resolution
        self.rank = rank
        self.mask_area_threshold = 50

    def rle2mask(self, mask_rle, shape):  # height width
        # Decode rle encoded mask.
        mask_rle = np.array(mask_rle)
        starts, lengths = [
            np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])
        ]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape, order="F")

    def __call__(self, examples):
        pixel_values = []
        masks = []
        input_ids = []
        urls = []
        heights = []
        widths = []
        gt_image_paths = []
        crops = []
        mask_strs = []
        choices = []
        # image_ids=[]

        for example in examples:
            caption = example["caption"]
            height = int(example["height"])
            width = int(example["width"])
            image_id = f"{example["image_id"]}"
            gt_image_path = example["gt_image_path"]
            image = cv2.imread(gt_image_path, cv2.IMREAD_COLOR)
            segmentation = json.loads(example["segmentation"])

            if len(segmentation["mask"]) > 0:
                choice = random.randint(0, len(segmentation["mask"]) - 1)
                mask = self.rle2mask(segmentation["mask"][choice], (height, width))[
                    :, :, np.newaxis
                ]
                mask = 1 - mask

            else:  # some images do not have segmentation mask, we use full image as inpainting input
                choice = -1
                mask = np.ones_like(image)[:, :, [0]]
                mask = 1 - mask

            inpainting_mask_choice = choice
            if inpainting_mask_choice == -1:
                logger.info(
                    f"[{self.rank}] {image_id} weird inpainting mask choice: {inpainting_mask_choice}"
                )
                continue

            w, h, c = mask.shape  # here w means height, h means width, sorry for that
            if w > h:
                scale = self.resolution / h
            else:
                scale = self.resolution / w
            w_new = int(np.ceil(w * scale))
            h_new = int(np.ceil(h * scale))

            image = cv2.resize(image, (h_new, w_new), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (h_new, w_new), interpolation=cv2.INTER_CUBIC)[
                :, :, np.newaxis
            ]

            random_crop = (
                random.randint(0, w_new - self.resolution),
                random.randint(0, h_new - self.resolution),
            )
            cur_mask = mask[
                random_crop[0] : random_crop[0] + self.resolution,
                random_crop[1] : random_crop[1] + self.resolution,
                :,
            ]
            new_random_crop = None
            # We want to avoid the mask being all 0 or all 1
            if (
                cur_mask.sum() <= self.mask_area_threshold
                or cur_mask.sum() >= self.resolution**2 - self.mask_area_threshold
            ):
                logger.info(
                    f"[{accelerator.process_index}] resample {image_id} for improper mask\ncurrent mask: {random_crop}, mask sum: {cur_mask.sum()}",
                    main_process_only=False,
                )
                # if mask is all 0 or all 1, we need to resample the crop
                if random_crop[0] == 0 and h_new > w_new:
                    # put the center of crop to the place where the mask area is the smallest
                    mask_2d = np.squeeze(mask)
                    mask_2d_sum = mask_2d.sum(axis=0)
                    new_crop = np.argmin(mask_2d_sum).item()
                    logger.info(
                        f"[{accelerator.process_index}] h_new > w_new, {image_id} find min at {new_crop}"
                    )
                    new_crop = new_crop - self.resolution // 2
                    if new_crop < 0:
                        new_crop = 0
                    elif new_crop + 512 > h_new:
                        new_crop = h_new - 512
                    new_random_crop = (0, new_crop)
                    random_crop = new_random_crop

                elif random_crop[1] == 0 and w_new > h_new:
                    mask_2d = np.squeeze(mask)
                    mask_2d_sum = mask_2d.sum(axis=1)
                    new_crop = np.argmin(mask_2d_sum).item()
                    logger.info(
                        f"[{accelerator.process_index}] w_new > h_new, {image_id} find min at {new_crop}"
                    )
                    new_crop = new_crop - self.resolution // 2
                    if new_crop < 0:
                        new_crop = 0
                    elif new_crop + 512 > w_new:
                        new_crop = w_new - 512

                    new_random_crop = (new_crop, 0)
                    random_crop = new_random_crop

                else:
                    logger.info(
                        f"[{accelerator.process_index}] resample {image_id} weird\nrandom_crop: {random_crop}\noriginal resolution: {w}x{h}\nnew resolution: {w_new}x{h_new}",
                        main_process_only=False,
                    )
                    continue

                if new_random_crop is None:
                    logger.info(
                        f"[{accelerator.process_index}] resample {image_id} failed\nrandom_crop: {random_crop}\noriginal resolution: {w}x{h}\nnew resolution: {w_new}x{h_new}",
                        main_process_only=False,
                    )
                    continue

            image = image[
                random_crop[0] : random_crop[0] + self.resolution,
                random_crop[1] : random_crop[1] + self.resolution,
                :,
            ]
            mask = mask[
                random_crop[0] : random_crop[0] + self.resolution,
                random_crop[1] : random_crop[1] + self.resolution,
                :,
            ]
            image = image * mask  # here's slightly different from FLUX

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = (image.astype(np.float32) / 127.5) - 1.0
            pixel_values.append(image)
            urls.append((example["id"].split("_")[0], example["id"].split("_")[1]))
            masks.append(mask)
            input_ids.append(caption)
            heights.append(height)
            widths.append(width)
            gt_image_paths.append(gt_image_path)
            crops.append(random_crop)
            mask_strs.append(example["mask"])
            choices.append(choice)

        return {
            "pixel_values": pixel_values,
            "masks": masks,
            "input_ids": input_ids,
            "urls": urls,
            "heights": heights,
            "widths": widths,
            "gt_image_paths": gt_image_paths,
            "crops": crops,
            "mask_strs": mask_strs,
            "choices": choices,
        }


def main():
    """Build pipeline, iterate data loader, save generated samples."""
    base_model_path = args.base_model_path
    brushnet_path = args.pretrained_model_path

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        logger.info(f"[{accelerator.process_index}] set seed to {args.seed}")
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    brushnet_conditioning_scale = 1.0

    brushnet = BrushNetModel.from_pretrained(brushnet_path, torch_dtype=torch.float16)
    pipe = StableDiffusionBrushNetPipeline.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=False,
    ).to(accelerator.device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if args.enable_xformers_memory_efficient_attention:
        logger.info(
            f"[{accelerator.process_index}] enabled xformers memory efficient attention"
        )
        pipe.enable_xformers_memory_efficient_attention()

    pipe.set_progress_bar_config(disable=True)

    train_dataset = load_dataset(
        "json",
        # data_files={"train": "/root/BrushNet/runs/samples/brushnet_segmentationmask_10steps/20250608_0031/annotations/pairs.json"},
        data_files={"train": args.train_json},
        split="train",
    )

    train_dataset_len = len(train_dataset)  # type: ignore

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=args.seed,
    )

    assert args.seed is not None, "seed must be set in our experiment"

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        # shuffle=True,
        collate_fn=MyDataset(
            resolution=args.resolution, rank=accelerator.process_index
        ),
        batch_size=args.sample_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    global_step = 0
    check_first = True
    if args.skip_batches is not None:
        logger.info(
            f"[{accelerator.process_index}] skipping first {args.skip_batches} batches"
        )
        train_dataloader_iter = iter(train_dataloader)
        train_dataloader_iter = islice(train_dataloader_iter, args.skip_batches, None)
        train_dataloader = train_dataloader_iter
        global_step += args.skip_batches

    train_dataloader_len = train_dataset_len // args.sample_batch_size
    max_steps = train_dataloader_len // accelerator.num_processes + 1

    result = dict()

    img_output_dir = Path(args.output_dir, "images")
    for step, batch in enumerate(train_dataloader):

        accelerator.wait_for_everyone()
        logger.info(f"step: {global_step} / {max_steps}", main_process_only=True)
        (
            init_images,
            mask_images,
            captions,
            urls,
            heights,
            widths,
            gt_image_paths,
            crops,
            mask_strs,
            choices,
        ) = (
            batch["pixel_values"],
            batch["masks"],
            batch["input_ids"],
            batch["urls"],
            batch["heights"],
            batch["widths"],
            batch["gt_image_paths"],
            batch["crops"],
            batch["mask_strs"],
            batch["choices"],
        )
        # images, urls = batch["pixel_values"], batch["urls"]
        # if len(images) == 0:
        #     continue

        with torch.no_grad():
            with accelerator.autocast():
                images = pipe(
                    captions,
                    init_images,
                    mask_images,
                    num_inference_steps=10,
                    generator=torch.Generator("cuda").manual_seed(args.generator_seed),
                    brushnet_conditioning_scale=brushnet_conditioning_scale,
                    rank=accelerator.process_index,
                ).images

        def check_and_save(_image):
            i, image = _image
            if not os.path.exists(
                Path(img_output_dir, urls[i][0].split("/")[-1].split(".")[0])
            ):
                os.makedirs(
                    Path(img_output_dir, urls[i][0].split("/")[-1].split(".")[0]),
                    exist_ok=True,
                )

            if check_first:
                logger.info(
                    f"[{accelerator.process_index}] check first for {urls[i][0].split('/')[-1].split('.')[0]}_{urls[i][1]}"
                )
                if os.path.exists(
                    f"{Path(img_output_dir, urls[i][0].split('/')[-1].split('.')[0], f'{urls[i][1]}.png')}"
                ):
                    raise ValueError(
                        f"[{accelerator.process_index}] {Path(img_output_dir, urls[i][0].split('/')[-1].split('.')[0], f'{urls[i][1]}.png')} already exists"
                    )
            image.save(
                f"{Path(img_output_dir, urls[i][0].split('/')[-1].split('.')[0], f'{urls[i][1]}.png')}"
            )
            # Image.fromarray((255 * (1 - mask_images[i])).squeeze().astype(np.uint8), mode="L").save(f"{Path(img_output_dir, urls[i][0].split('/')[-1].split('.')[0], f'{urls[i][1]}_mask.png')}")
            result[f"{urls[i][0].split('/')[-1].split('.')[0]}_{urls[i][1]}"] = {
                "gt_image_path": gt_image_paths[i],
                "crop": crops[i],
                "image_path": f"{Path(img_output_dir, urls[i][0].split('/')[-1].split('.')[0], f'{urls[i][1]}.png')}",
                "height": heights[i],
                "width": widths[i],
                "caption": captions[i],
                "mask": mask_strs[i],
                "choice": choices[i],
            }

        _ = list(map(check_and_save, enumerate(images)))
        # if step % args.save_step == 0:
        #     with open(os.path.join(args.output_dir, f"result_{global_step}_{accelerator.process_index}.json"), "w") as f:
        #         json.dump(result, f)
        #     result = dict()

        global_step += 1
        check_first = False

        if args.max_steps is not None:
            if global_step == args.max_steps:
                break

    if not os.path.exists(os.path.join(args.output_dir, "annotations")):
        os.makedirs(os.path.join(args.output_dir, "annotations"), exist_ok=True)

    with open(
        os.path.join(
            args.output_dir, "annotations", f"result_{accelerator.process_index}.json"
        ),
        "w",
    ) as f:
        json.dump(result, f, indent=2)

    accelerator.wait_for_everyone()
    destroy_process_group()


if __name__ == "__main__":
    main()
