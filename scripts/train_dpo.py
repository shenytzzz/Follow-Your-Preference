#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#################################################################################################
# Unified DPO training: Flux Fill transformer OR BrushNet (SD1.5 UNet adapter).
# Selected at runtime via the `model_type` field of configs/configs_dpo.yaml.
#################################################################################################

import argparse
import contextlib
import copy
import gc
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from tqdm import tqdm
import json
import cv2

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
    DistributedType,
    DeepSpeedPlugin,
)
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from packaging import version
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    PretrainedConfig,
    T5EncoderModel,
    T5TokenizerFast,
)
from einops import rearrange, repeat

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    FlowMatchEulerDiscreteScheduler,
    FluxFillPipeline,
    FluxTransformer2DModel,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

try:
    from diffusers import BrushNetModel, StableDiffusionBrushNetPipeline
except ImportError:
    BrushNetModel = None
    StableDiffusionBrushNetPipeline = None
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from datasets import load_dataset
from deepspeed.runtime.engine import DeepSpeedEngine

from train_utils import (
    prepare_fill_with_mask,
    prepare_latents,
    encode_images_to_latents,
)

from omegaconf import OmegaConf

if is_wandb_available():
    import wandb


def resolve_config(full_cfg, section, model_type):
    """Merge `full_cfg[section][model_type]` overrides onto the shared defaults
    under `full_cfg[section]`, and strip the other model's overrides.
    """
    section_cfg = OmegaConf.to_container(full_cfg[section], resolve=True)
    overrides = section_cfg.pop(model_type)
    section_cfg.pop("flux", None)
    section_cfg.pop("brushnet", None)
    return OmegaConf.merge(OmegaConf.create(section_cfg), OmegaConf.create(overrides))


def check_args(args, model_type):
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError(
            "`--validation_image` must be set if `--validation_prompt` is set"
        )

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError(
            "`--validation_prompt` must be set if `--validation_image` is set"
        )

    if args.train_json_dir is None:
        raise ValueError(
            "`--train_json_dir` must be set to the path of the training json file."
        )

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the brushnet encoder."
        )

    if args.metrics.enable and args.metrics.metric not in [
        "clip_score",
        "hps",
        "pick_score",
        "aesthetic_score",
        "image_reward",
        "vqa_score",
        "unified_reward",
        "pe_score",
        "ensemble",
        "random",
        "hpsv3",
    ]:
        raise ValueError(
            f"Unsupported metric: {args.metrics.metric}, should be one of [clip_score, hps, pick_score, aesthetic_score, image_reward, vqa_score , unified_reward, pe_score, ensemble, random]."
        )


# ---------------------------------------------------------------------------
# Config load + accelerator setup (must run at import time to match originals)
# ---------------------------------------------------------------------------

config_path = os.environ.get("DPO_CONFIG", "/root/test_env/configs/configs_dpo.yaml")
full_args = OmegaConf.load(config_path)
model_type = full_args.model_type
assert model_type in {"flux", "brushnet"}, (
    f"model_type must be 'flux' or 'brushnet', got {model_type!r}"
)
args = resolve_config(full_args, "train", model_type)
check_args(args, model_type)
args.output_dir = (
    os.path.join(args.output_dir, args.metrics.metric)
    if args.metrics.enable
    else args.output_dir
)
os.makedirs(args.output_dir, exist_ok=True)
logging_dir = os.path.join(args.output_dir, args.logging_dir)
os.makedirs(logging_dir, exist_ok=True)
accelerator_project_config = ProjectConfiguration(
    project_dir=args.output_dir, logging_dir=logging_dir
)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
)
logger = get_logger(__name__)
with open(os.path.join(logging_dir, "config.yaml"), "w") as f:
    OmegaConf.save(args, f)


# ---------------------------------------------------------------------------
# Flux-only helpers (text encoders & prompt encoding)
# ---------------------------------------------------------------------------

def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    return text_encoder_one, text_encoder_two


def import_model_class_from_model_name_or_path_flux(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def import_model_class_from_model_name_or_path_brushnet(
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


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype
    device = device if device is not None else text_encoders[1].device
    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


# ---------------------------------------------------------------------------
# Validation loggers (one per model type)
# ---------------------------------------------------------------------------

def log_validation_flux(
    pipeline,
    args,
    accelerator,
    epoch,
    step,
    tag=None,
    is_final_validation=False,
):
    logger.info(f"Running validation... \n ")

    prompts = "a red pepper on a white background"

    image = load_image("/root/flux/examples/flux/src/test_image.jpg")
    mask = load_image("/root/flux/examples/flux/src/test_mask.jpg")
    pipeline.set_progress_bar_config(disable=True)

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed
        else None
    )
    inference_ctx = (
        contextlib.nullcontext()
        if is_final_validation
        else torch.autocast("cuda", dtype=torch.bfloat16)
    )

    image_logs = []
    images = []
    num_validation_images = args.num_validation_images
    for _ in range(num_validation_images):
        with inference_ctx:
            image = pipeline(
                prompt=prompts,
                image=image,
                mask_image=mask,
                height=512,
                width=512,
                num_inference_steps=20,
                max_sequence_length=512,
            ).images[0]
        images.append(image)

    image_logs.append(
        {
            "validation_image": image,
            "mask": mask,
            "images": images,
            "validation_prompt": prompts,
        }
    )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    validation_prompt, formatted_images, step, dataformats="NHWC"
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                masks = log["mask"]

                formatted_images.append(
                    wandb.Image(validation_image, caption="BrushNet conditioning")
                )
                formatted_images.append(wandb.Image(masks, caption="Mask"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    return image_logs


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation_brushnet(
    vae,
    text_encoder,
    tokenizer,
    unet,
    brushnet,
    args,
    accelerator,
    weight_dtype,
    step,
    is_final_validation=False,
):
    logger.info(f"[{accelerator.process_index}] Running validation... ")

    if not is_final_validation:
        brushnet = accelerator.unwrap_model(brushnet)
    else:
        brushnet = BrushNetModel.from_pretrained(
            args.output_dir, torch_dtype=weight_dtype
        )

    pipeline = StableDiffusionBrushNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        brushnet=brushnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt) and len(
        args.validation_image
    ) == len(args.validation_mask):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
        validation_masks = args.validation_mask
    else:
        raise ValueError(
            "number of `args.validation_image`, `args.validation_mask`, and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    inference_ctx = (
        contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")
    )

    for validation_prompt, validation_image, validation_mask in zip(
        validation_prompts, validation_images, validation_masks
    ):
        init_image = cv2.imread(validation_image)[:, :, ::-1]
        mask_image = 1.0 * (cv2.imread(validation_mask).sum(-1) > 255)[:, :, np.newaxis]
        log_mask_image = 1 - mask_image
        init_image = init_image * (1 - mask_image)
        init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
        mask_image = Image.fromarray(
            mask_image.astype(np.uint8).repeat(3, -1) * 255
        ).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    validation_prompt,
                    init_image,
                    mask_image,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]

            images.append(image)

        image_logs.append(
            {
                "validation_image": init_image,
                "mask": log_mask_image,
                "images": images,
                "validation_prompt": validation_prompt,
            }
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(
                    validation_prompt, formatted_images, step, dataformats="NHWC"
                )
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                masks = log["mask"]

                formatted_images.append(
                    wandb.Image(validation_image, caption="BrushNet conditioning")
                )
                formatted_images.append(wandb.Image(masks, caption="Mask"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs


def save_model_card_brushnet(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(
                os.path.join(repo_folder, f"images_{i}.png")
            )
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# brushnet-{repo_id}

These are brushnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "brushnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


# ---------------------------------------------------------------------------
# Shared dataset collate (model_type-aware)
# ---------------------------------------------------------------------------

class MyDataset:
    """Collate: produce (win, loss) image tensors + mask for DPO.

    For BrushNet it additionally produces masked conditioning tensors and
    tokenizes captions with the CLIP tokenizer; for Flux it passes captions
    through as raw strings (tokenization happens in the training loop).
    """

    def __init__(
        self,
        resolution,
        random_mask,
        model_type,
        tokenizer=None,
        score_file=None,
        metrics_enable=False,
    ):
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.random_mask = random_mask
        self.model_type = model_type
        if score_file is not None:
            with open(score_file, "r") as f:
                self.scores = json.load(f)
        elif metrics_enable and model_type == "flux":
            raise ValueError(
                "`score_file` must be provided to load the scoring results."
            )
        else:
            self.scores = None

    def rle2mask(self, mask_rle, shape):  # height width
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

    def tokenize_captions(self, caption, is_train=True):
        if random.random() < args.proportion_empty_prompts:
            caption = ""
        elif isinstance(caption, str):
            caption = caption
        elif isinstance(caption, (list, np.ndarray)):
            caption = random.choice(caption) if is_train else caption[0]
        else:
            raise ValueError(
                f"Caption column should contain either strings or lists of strings."
            )
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def __call__(self, examples):
        pixel_values = []
        masks = []
        input_ids = []
        inpainting_pixel_values = []
        conditioning_pixel_values = []
        conditioning_inpainting_pixel_values = []

        for example in examples:
            caption = example["caption"]
            image_id = example["image_id"]
            if args.metrics.enable:
                if args.metrics.metric == "random":
                    random_seed = random.sample(range(0, args.metrics.scaling), k=2)
                    win_seed = f"local_seed_{random_seed[0]}"
                    lose_seed = f"local_seed_{random_seed[1]}"
                elif not self.scores[image_id][args.metrics.metric]["draw"]:
                    if args.metrics.scaling == 16:
                        win_seed = f"local_{self.scores[image_id][args.metrics.metric]["max_seed"]}"
                        lose_seed = f"local_{self.scores[image_id][args.metrics.metric]["min_seed"]}"
                    else:
                        win_seed = f"local_{self.scores[image_id][args.metrics.metric][f"max_seed_{args.metrics.scaling}"]}"
                        lose_seed = f"local_{self.scores[image_id][args.metrics.metric][f"min_seed_{args.metrics.scaling}"]}"
                else:  # random sample if draw
                    random_seed = random.sample(range(0, args.metrics.scaling), k=2)
                    win_seed = f"local_seed_{random_seed[0]}"
                    lose_seed = f"local_seed_{random_seed[1]}"

                win_image = cv2.imread(
                    example["image_path"].replace("local_seed_0", win_seed),
                    cv2.IMREAD_COLOR,
                )
                lose_image = cv2.imread(
                    example["image_path"].replace("local_seed_0", lose_seed),
                    cv2.IMREAD_COLOR,
                )
            else:
                win_image = cv2.imread(example["gt_image_path"], cv2.IMREAD_COLOR)
                lose_image = cv2.imread(example["image_path"], cv2.IMREAD_COLOR)

            inpainting_crop = example["crop"]
            height = example["height"]
            width = example["width"]

            mask = list(map(int, example["mask"].split(",")))
            mask = self.rle2mask(mask, (height, width))[:, :, np.newaxis]
            if self.model_type == "brushnet":
                mask = 1 - mask

            w, h, _ = mask.shape
            if w > h:
                scale = self.resolution / h
            else:
                scale = self.resolution / w
            w_new = int(np.ceil(w * scale))
            h_new = int(np.ceil(h * scale))

            if self.model_type == "brushnet" and random.random() < 0.3:
                kernel = np.ones((8, 8), np.uint8)
                mask_erosion = cv2.erode(mask, kernel, iterations=1)
                mask_dilation = cv2.dilate(mask_erosion, kernel, iterations=1)
                mask = 1 * (mask_dilation > 0)[:, :, np.newaxis]
                mask = mask.astype(np.uint8)

            random_crop = inpainting_crop
            if not args.metrics.enable:
                win_image = cv2.resize(
                    win_image, (w_new, h_new), interpolation=cv2.INTER_CUBIC
                )
                lose_image = cv2.resize(
                    lose_image, (w_new, h_new), interpolation=cv2.INTER_CUBIC
                )
                win_image = win_image[
                    random_crop[0] : random_crop[0] + self.resolution,
                    random_crop[1] : random_crop[1] + self.resolution,
                    :,
                ]
                lose_image = lose_image[
                    random_crop[0] : random_crop[0] + self.resolution,
                    random_crop[1] : random_crop[1] + self.resolution,
                    :,
                ]
            mask = cv2.resize(mask, (h_new, w_new), interpolation=cv2.INTER_CUBIC)[
                :, :, np.newaxis
            ]
            mask = mask[
                random_crop[0] : random_crop[0] + self.resolution,
                random_crop[1] : random_crop[1] + self.resolution,
                :,
            ]

            if self.model_type == "flux":
                win_image = cv2.cvtColor(win_image, cv2.COLOR_BGR2RGB)
                lose_image = cv2.cvtColor(lose_image, cv2.COLOR_BGR2RGB)
                win_image = (win_image.astype(np.float32) / 127.5) - 1.0
                lose_image = (lose_image.astype(np.float32) / 127.5) - 1.0

                mask = mask.astype(np.float32)

                pixel_values.append(torch.tensor(win_image).permute(2, 0, 1))
                inpainting_pixel_values.append(
                    torch.tensor(lose_image).permute(2, 0, 1)
                )
                masks.append(torch.tensor(mask).permute(2, 0, 1))
                input_ids.append(caption)
            else:  # brushnet
                image = win_image
                inpainting_image = lose_image
                masked_image = win_image * mask
                masked_inpainting_image = inpainting_image * mask

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
                inpainting_image = cv2.cvtColor(inpainting_image, cv2.COLOR_BGR2RGB)
                masked_inpainting_image = cv2.cvtColor(
                    masked_inpainting_image, cv2.COLOR_BGR2RGB
                )
                image = (image.astype(np.float32) / 127.5) - 1.0
                masked_image = (masked_image.astype(np.float32) / 127.5) - 1.0
                inpainting_image = (inpainting_image.astype(np.float32) / 127.5) - 1.0
                masked_inpainting_image = (
                    masked_inpainting_image.astype(np.float32) / 127.5
                ) - 1.0

                mask = mask.astype(np.float32)

                pixel_values.append(torch.tensor(image).permute(2, 0, 1))
                inpainting_pixel_values.append(
                    torch.tensor(inpainting_image).permute(2, 0, 1)
                )
                conditioning_pixel_values.append(
                    torch.tensor(masked_image).permute(2, 0, 1)
                )
                conditioning_inpainting_pixel_values.append(
                    torch.tensor(masked_inpainting_image).permute(2, 0, 1)
                )
                masks.append(torch.tensor(mask).permute(2, 0, 1))
                input_ids.append(self.tokenize_captions(caption)[0])

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        inpainting_pixel_values = torch.stack(inpainting_pixel_values)
        inpainting_pixel_values = inpainting_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        masks = torch.stack(masks)
        masks = masks.to(memory_format=torch.contiguous_format).float()

        batch = {
            "pixel_values": pixel_values,
            "inpainting_pixel_values": inpainting_pixel_values,
            "masks": masks,
            "input_ids": input_ids,
        }

        if self.model_type == "brushnet":
            conditioning_pixel_values = torch.stack(conditioning_pixel_values)
            conditioning_pixel_values = conditioning_pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
            conditioning_inpainting_pixel_values = torch.stack(
                conditioning_inpainting_pixel_values
            )
            conditioning_inpainting_pixel_values = conditioning_inpainting_pixel_values.to(
                memory_format=torch.contiguous_format
            ).float()
            batch["conditioning_pixel_values"] = conditioning_pixel_values
            batch["conditioning_inpainting_pixel_values"] = (
                conditioning_inpainting_pixel_values
            )
            batch["input_ids"] = torch.stack(batch["input_ids"])

        return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args, model_type):
    """Full DPO training loop (data, forward, DPO loss, checkpoints) for
    either Flux or BrushNet, dispatched on `model_type`.
    """
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(logging_dir, "log.txt"),
    )

    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # -----------------------------------------------------------------------
    # Model / tokenizer / scheduler loading (per model_type)
    # -----------------------------------------------------------------------

    if model_type == "flux":
        tokenizer_one = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
        )
        tokenizer_two = T5TokenizerFast.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
        )

        text_encoder_cls_one = import_model_class_from_model_name_or_path_flux(
            args.pretrained_model_name_or_path, args.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path_flux(
            args.pretrained_model_name_or_path,
            args.revision,
            subfolder="text_encoder_2",
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )

        text_encoder_one, text_encoder_two = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two
        )

        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )

        vae_scale_factor = (
            2 ** (len(vae.config.block_out_channels) - 1) if vae is not None else 8
        )

        image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor * 2,
            vae_latent_channels=vae.config.latent_channels,
        )
        mask_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor * 2,
            vae_latent_channels=vae.config.latent_channels,
            do_convert_grayscale=True,
            do_normalize=False,
            do_binarize=True,
        )

        ref_transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=args.revision,
            variant=args.variant,
        )

        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            revision=args.revision,
            variant=args.variant,
        )

        ref_transformer.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        transformer.requires_grad_(False)
        transformer.train()

        grad_params = [
            "transformer_blocks.0.",
            "transformer_blocks.1.",
            "transformer_blocks.2.",
            "transformer_blocks.3.",
            "transformer_blocks.4.",
            "transformer_blocks.5.",
            "transformer_blocks.6.",
            "transformer_blocks.7.",
            "transformer_blocks.8.",
            "transformer_blocks.9.",
            "transformer_blocks.10.",
            "transformer_blocks.11.",
            "transformer_blocks.12.",
            "transformer_blocks.13.",
            "transformer_blocks.14.",
            "transformer_blocks.15.",
            "transformer_blocks.16.",
            "transformer_blocks.17.",
            "transformer_blocks.18.",
            "single_transformer_blocks.0.",
            "single_transformer_blocks.1.",
            "single_transformer_blocks.2.",
            "single_transformer_blocks.3.",
            "single_transformer_blocks.4.",
            "single_transformer_blocks.5.",
            "single_transformer_blocks.6.",
            "single_transformer_blocks.7.",
            "single_transformer_blocks.8.",
            "single_transformer_blocks.9.",
            "single_transformer_blocks.10.",
            "single_transformer_blocks.13.",
            "single_transformer_blocks.14.",
            "single_transformer_blocks.15.",
            "single_transformer_blocks.16.",
            "single_transformer_blocks.17.",
            "single_transformer_blocks.18.",
            "single_transformer_blocks.19.",
            "single_transformer_blocks.20.",
            "single_transformer_blocks.21.",
            "single_transformer_blocks.22.",
            "single_transformer_blocks.23.",
            "single_transformer_blocks.24.",
            "single_transformer_blocks.25.",
            "single_transformer_blocks.26.",
            "single_transformer_blocks.27.",
            "single_transformer_blocks.28.",
            "single_transformer_blocks.29.",
            "single_transformer_blocks.30.",
            "single_transformer_blocks.31.",
            "single_transformer_blocks.32.",
            "single_transformer_blocks.33.",
            "single_transformer_blocks.34.",
            "single_transformer_blocks.35.",
            "single_transformer_blocks.36.",
            "single_transformer_blocks.37.",
        ]

        if args.train_base_model:
            transformer.requires_grad_(False)

            for name, param in transformer.named_parameters():
                if any(grad_param in name for grad_param in grad_params):
                    if "attn" in name:
                        param.requires_grad = True
                        logger.info(
                            f"[{accelerator.process_index}] Enabling gradients for: {name}",
                            main_process_only=True,
                        )
        else:
            transformer.requires_grad_(False)

        if args.do_test:
            trainable_params = [
                (name, param)
                for name, param in transformer.named_parameters()
                if param.requires_grad
            ]
            if not trainable_params:
                raise RuntimeError(
                    "Smoke test expected at least one trainable Flux parameter."
                )
            keep_name, _ = trainable_params[-1]
            for name, param in trainable_params[:-1]:
                param.requires_grad = False
            logger.info(
                f"Smoke test mode: only keeping {keep_name} trainable."
            )

        logger.info(
            f"[{accelerator.process_index}] transformer parameters: {sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000}"
        )

    else:  # brushnet
        if BrushNetModel is None or StableDiffusionBrushNetPipeline is None:
            raise ImportError(
                "BrushNet training requires a diffusers build that provides "
                "BrushNetModel and StableDiffusionBrushNetPipeline."
            )

        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_name, revision=args.revision, use_fast=False
            )
        elif args.pretrained_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
                use_fast=False,
            )

        text_encoder_cls = import_model_class_from_model_name_or_path_brushnet(
            args.pretrained_model_name_or_path, args.revision
        )

        noise_scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        text_encoder = text_encoder_cls.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            variant=args.variant,
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            variant=args.variant,
        )

        if args.brushnet_model_name_or_path and args.resume_from_checkpoint is None:
            logger.info(
                f"[{accelerator.process_index}] Loading existing brushnet weights from {args.brushnet_model_name_or_path}"
            )
            brushnet = BrushNetModel.from_pretrained(args.brushnet_model_name_or_path)
            ref_brushnet = BrushNetModel.from_pretrained(
                args.brushnet_model_name_or_path
            )
        else:
            logger.info(
                f"[{accelerator.process_index}] Initializing brushnet weights from unet"
            )
            brushnet = BrushNetModel.from_unet(unet)
            ref_brushnet = BrushNetModel.from_unet(unet)

        vae.requires_grad_(False)
        unet.requires_grad_(False)
        text_encoder.requires_grad_(False)
        ref_brushnet.requires_grad_(False)
        brushnet.train()

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                unet.enable_xformers_memory_efficient_attention()
                brushnet.enable_xformers_memory_efficient_attention()
                ref_brushnet.enable_xformers_memory_efficient_attention()
                logger.info(
                    f"[{accelerator.process_index}] xformers memory efficient attention enabled"
                )
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if unwrap_model(brushnet).dtype != torch.float32:
            raise ValueError(
                f"BrushNet loaded as datatype {unwrap_model(brushnet).dtype}. Please make sure to always have all model weights in full float32 precision when starting training - even if doing mixed precision training, copy of the weights should still be float32."
            )

    # Mixed precision dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if model_type == "flux":
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
        ref_transformer.to(accelerator.device, dtype=weight_dtype)

        if args.gradient_checkpointing:
            if args.train_base_model:
                transformer.enable_gradient_checkpointing()
    else:
        if args.gradient_checkpointing:
            brushnet.enable_gradient_checkpointing()

    # -----------------------------------------------------------------------
    # Save / load state hooks
    # -----------------------------------------------------------------------

    if model_type == "flux":
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if isinstance(model, DeepSpeedEngine):
                        model = model.module
                    if isinstance(unwrap_model(model), FluxTransformer2DModel):
                        logger.info(
                            f"[{accelerator.process_index}] saving transformer to {os.path.join(output_dir, 'transformer')}",
                            main_process_only=False,
                        )
                        unwrap_model(model).save_pretrained(
                            os.path.join(output_dir, "transformer")
                        )
                    elif isinstance(
                        unwrap_model(model),
                        (CLIPTextModelWithProjection, T5EncoderModel),
                    ):
                        if isinstance(unwrap_model(model), CLIPTextModelWithProjection):
                            unwrap_model(model).save_pretrained(
                                os.path.join(output_dir, "text_encoder")
                            )
                        else:
                            unwrap_model(model).save_pretrained(
                                os.path.join(output_dir, "text_encoder_2")
                            )
                    else:
                        raise ValueError(f"Wrong model supplied: {type(model)=}.")

                    if weights:
                        weights.pop()
                    else:
                        logger.info(
                            f"[{accelerator.process_index}] no weights",
                            main_process_only=False,
                        )

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                model = models.pop()

                if isinstance(unwrap_model(model), FluxTransformer2DModel):
                    load_model = FluxTransformer2DModel.from_pretrained(
                        input_dir, subfolder="transformer"
                    )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                elif isinstance(
                    unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)
                ):
                    try:
                        load_model = CLIPTextModelWithProjection.from_pretrained(
                            input_dir, subfolder="text_encoder"
                        )
                        model(**load_model.config)
                        model.load_state_dict(load_model.state_dict())
                    except Exception:
                        try:
                            load_model = T5EncoderModel.from_pretrained(
                                input_dir, subfolder="text_encoder_2"
                            )
                            model(**load_model.config)
                            model.load_state_dict(load_model.state_dict())
                        except Exception:
                            raise ValueError(
                                f"Couldn't load the model of type: ({type(model)})."
                            )
                else:
                    raise ValueError(f"Unsupported model found: {type(model)=}")

                del load_model
    else:  # brushnet
        if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    i = len(weights) - 1
                    while len(weights) > 0:
                        weights.pop()
                        model = models[i]
                        sub_dir = "brushnet"
                        model.save_pretrained(os.path.join(output_dir, sub_dir))
                        i -= 1

            def load_model_hook(models, input_dir):
                while len(models) > 0:
                    model = models.pop()
                    load_model = BrushNetModel.from_pretrained(
                        input_dir, subfolder="brushnet"
                    )
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # TF32
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------

    if model_type == "flux":
        if args.train_base_model:
            transformer_parameters_with_lr = {
                "params": transformer.parameters(),
                "lr": args.learning_rate,
            }
        params_to_optimize = [transformer_parameters_with_lr]

        if not (
            args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"
        ):
            logger.warning(
                f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
                "Defaulting to adamW"
            )
            args.optimizer = "adamw"

        if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
            logger.warning(
                f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
                f"set to {args.optimizer.lower()}"
            )

        if args.optimizer.lower() == "adamw":
            if args.use_8bit_adam:
                try:
                    import bitsandbytes as bnb
                except ImportError:
                    raise ImportError(
                        "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                    )
                optimizer_class = bnb.optim.AdamW8bit
            else:
                optimizer_class = torch.optim.AdamW

            optimizer = optimizer_class(
                params_to_optimize,
                betas=(args.adam_beta1, args.adam_beta2),
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
            )

        if args.optimizer.lower() == "prodigy":
            try:
                import prodigyopt
            except ImportError:
                raise ImportError(
                    "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
                )
            optimizer_class = prodigyopt.Prodigy

            if args.learning_rate <= 0.1:
                logger.warning(
                    "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
                )

            optimizer = optimizer_class(
                params_to_optimize,
                lr=args.learning_rate,
                betas=(args.adam_beta1, args.adam_beta2),
                beta3=args.prodigy_beta3,
                weight_decay=args.adam_weight_decay,
                eps=args.adam_epsilon,
                decouple=args.prodigy_decouple,
                use_bias_correction=args.prodigy_use_bias_correction,
                safeguard_warmup=args.prodigy_safeguard_warmup,
            )
    else:  # brushnet
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = brushnet.parameters()
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    train_dataset = load_dataset(
        "json", data_files={"train": args.train_json_dir}, split="train"
    )
    train_dataset_len = len(train_dataset)

    if model_type == "flux":
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]
        collate_tokenizer = None
    else:
        collate_tokenizer = tokenizer

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=MyDataset(
            resolution=args.resolution,
            random_mask=args.random_mask,
            model_type=model_type,
            tokenizer=collate_tokenizer,
            score_file=args.metrics.score_file if args.metrics.enable else None,
            metrics_enable=args.metrics.enable,
        ),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    train_dataloader_len = train_dataset_len // args.train_batch_size

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # -----------------------------------------------------------------------
    # Scheduler + accelerator.prepare
    # -----------------------------------------------------------------------

    overrode_max_train_steps = False
    if model_type == "flux":
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
    else:
        num_update_steps_per_epoch = math.ceil(
            train_dataloader_len / args.gradient_accumulation_steps
        )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    if model_type == "flux":
        if args.train_base_model:
            transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                transformer, optimizer, train_dataloader, lr_scheduler
            )
    else:
        brushnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            brushnet, optimizer, train_dataloader, lr_scheduler
        )
        vae.to(accelerator.device, dtype=weight_dtype)
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        ref_brushnet.to(accelerator.device, dtype=weight_dtype)

    if model_type == "flux":
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
    else:
        num_update_steps_per_epoch = math.ceil(
            train_dataloader_len / args.gradient_accumulation_steps
        )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # -----------------------------------------------------------------------
    # Tracker init + banner logs
    # -----------------------------------------------------------------------

    if accelerator.is_main_process:
        if model_type == "flux":
            tracker_name = "flux-inpainting-dpo"
            accelerator.init_trackers(
                tracker_name,
                config=vars(args),
                init_kwargs={"wandb": {"settings": wandb.Settings(code_dir=".")}},
            )
        else:
            tracker_config = dict(vars(args))
            accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"[{accelerator.process_index}]  Num examples = {train_dataset_len}")
    if model_type == "flux":
        logger.info(f"[{accelerator.process_index}]  Num batches each epoch = {len(train_dataloader)}")
    else:
        logger.info(f"[{accelerator.process_index}]  Num batches each epoch = {train_dataloader_len}")
    logger.info(f"[{accelerator.process_index}]  Num Epochs = {args.num_train_epochs}")
    logger.info(f"[{accelerator.process_index}]  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"[{accelerator.process_index}]  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"[{accelerator.process_index}]  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"[{accelerator.process_index}]  Total optimization steps = {args.max_train_steps}")
    logger.info(f"[{accelerator.process_index}]  dtype = {weight_dtype}")
    logger.info(f"[{accelerator.process_index}]  dpo loss weight = {args.dpo_loss_weight}")
    logger.info(f"[{accelerator.process_index}]  beta dpo = {args.beta_dpo}")
    logger.info(f"[{accelerator.process_index}]  dpo mask = {args.dpo_mask}")
    logger.info(f"[{accelerator.process_index}]  dpo new = {args.dpo_new}")
    logger.info(f"[{accelerator.process_index}]  mse loss weight = {args.mse_loss_weight}")
    logger.info(f"[{accelerator.process_index}]  learning rate = {args.learning_rate}")
    logger.info(f"[{accelerator.process_index}]  loss = {args.dpo_loss_weight} * dpo_loss + {args.mse_loss_weight} * mse_loss")
    logger.info(f"[{accelerator.process_index}]  Metric = {args.metrics.metric}" if args.metrics.enable else f"[{accelerator.process_index}] metrics are disabled")
    logger.info(f"[{accelerator.process_index}]  Scaling = {args.metrics.scaling}" if args.metrics.enable else f"[{accelerator.process_index}] metrics scaling is not used")
    logger.info(f"[{accelerator.process_index}]  Image annotation file = {args.train_json_dir}")
    logger.info(f"[{accelerator.process_index}]  Score file = {args.metrics.score_file}" if args.metrics.enable else f"[{accelerator.process_index}] metrics score file is not used")
    logger.info(f"[{accelerator.process_index}]  Note: {args.note}")

    global_step = 0
    first_epoch = 0

    # -----------------------------------------------------------------------
    # Resume from checkpoint
    # -----------------------------------------------------------------------

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            if model_type == "flux":
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
            else:
                logger.info(
                    f"[{accelerator.process_index}] Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                raise ValueError(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist."
                )
        else:
            if model_type == "flux":
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.output_dir, path))
                global_step = int(path.split("-")[1])
                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
            else:
                logger.info(
                    f"[{accelerator.process_index}] Resuming from checkpoint {path}"
                )
                accelerator.load_state(
                    os.path.join(args.output_dir, path), map_location="cpu"
                )
                global_step = int(path.split("-")[1])
                initial_global_step = global_step
                first_epoch = global_step // num_update_steps_per_epoch
                first_epoch = 0
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # Flux flow-matching sigma lookup
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    check_first = True
    image_logs = None

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    for epoch in range(first_epoch, args.num_train_epochs):
        if model_type == "flux":
            logger.info(f"[{accelerator.process_index}] epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            if model_type == "flux":
                # ---- Flux step ----
                if args.train_base_model:
                    models_to_accumulate = [transformer]
                with accelerator.accumulate(models_to_accumulate):
                    accelerator.wait_for_everyone()
                    win_pixel_values, loss_pixel_values, masks, prompts = (
                        batch["pixel_values"],
                        batch["inpainting_pixel_values"],
                        batch["masks"],
                        batch["input_ids"],
                    )

                    if not args.metrics.enable:
                        loss_pixel_values = (
                            1 - batch["masks"]
                        ) * loss_pixel_values + batch["masks"] * win_pixel_values

                    feed_in_image = torch.cat(
                        [win_pixel_values, loss_pixel_values], dim=0
                    )

                    feed_in_image = image_processor.preprocess(
                        feed_in_image, height=args.height, width=args.width
                    ).to(dtype=vae.dtype, device=accelerator.device)
                    masks = mask_processor.preprocess(
                        masks, height=args.height, width=args.width
                    ).to(dtype=vae.dtype, device=accelerator.device)

                    if accelerator.is_main_process:
                        if global_step == 0:
                            from torchvision.transforms.functional import to_pil_image

                            if feed_in_image.dtype == torch.bfloat16:
                                image_1 = feed_in_image.to(dtype=torch.float32).to(
                                    "cpu"
                                )
                                image_2 = masks.to(dtype=torch.float32).to("cpu")
                            else:
                                image_1 = feed_in_image.to("cpu")
                                image_2 = masks.to("cpu")
                            to_pil_image((image_1[0] + 1) / 2).save(
                                f"{os.path.join(args.output_dir, args.logging_dir, 'training_image_checks_gt.png')}"
                            )
                            to_pil_image((image_1[1] + 1) / 2).save(
                                f"{os.path.join(args.output_dir, args.logging_dir, 'training_image_checks_flux.png')}"
                            )
                            if image_2.ndim == 4:
                                to_pil_image(image_2[0].squeeze()).save(
                                    f"{os.path.join(args.output_dir, args.logging_dir, 'training_image_checks_mask.png')}"
                                )
                            else:
                                to_pil_image(image_2[0]).save(
                                    f"{os.path.join(args.output_dir, args.logging_dir, 'training_image_checks_mask.png')}"
                                )

                    masks = masks.repeat(2, 1, 1, 1)
                    batch_size = win_pixel_values.shape[0] * 2

                    prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                        prompts, text_encoders, tokenizers
                    )

                    if check_first:
                        logger.info(
                            f"[{accelerator.process_index}] masks.shape: {masks.shape}, pixel_values.shape: {win_pixel_values.shape}, inpainting_pixel_values.shape: {loss_pixel_values.shape}"
                        )
                        logger.info(
                            f"[{accelerator.process_index}] prompt_embeds.shape: {prompt_embeds.shape}, pooled_prompt_embeds.shape: {pooled_prompt_embeds.shape}"
                        )

                    model_input = encode_images_to_latents(
                        vae, feed_in_image, weight_dtype, args.height, args.width
                    )

                    masked_image = feed_in_image * (1 - masks)
                    masked_image = masked_image.to(
                        dtype=weight_dtype, device=accelerator.device
                    )
                    inpaint_cond, _, _ = prepare_fill_with_mask(
                        image_processor=image_processor,
                        mask_processor=mask_processor,
                        vae=vae,
                        vae_scale_factor=vae_scale_factor,
                        masked_image=masked_image,
                        mask=masks,
                        width=args.width,
                        height=args.height,
                        batch_size=batch_size,
                        num_images_per_prompt=1,
                        device=accelerator.device,
                        dtype=weight_dtype,
                    )

                    inpaint_cond = inpaint_cond.to(
                        dtype=weight_dtype, device=accelerator.device
                    )

                    if check_first:
                        logger.info(
                            f"[{accelerator.process_index}] feed_in_image.shape: {feed_in_image.shape}"
                        )
                        logger.info(
                            f"[{accelerator.process_index}] inpaint_cond.shape: {inpaint_cond.shape}"
                        )
                        logger.info(
                            f"[{accelerator.process_index}] model_input.shape: {model_input.shape}"
                        )

                    latent_image_ids = prepare_latents(
                        vae_scale_factor,
                        batch_size,
                        args.height,
                        args.width,
                        weight_dtype,
                        accelerator.device,
                    )

                    noise = torch.randn_like(
                        model_input, device=accelerator.device, dtype=weight_dtype
                    )
                    noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)
                    bsz = model_input.shape[0]

                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                    timesteps = noise_scheduler.timesteps[indices].to(
                        device=model_input.device
                    )
                    timesteps = timesteps.chunk(2)[0].repeat(2)

                    sigmas = get_sigmas(
                        timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                    )
                    noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                    packed_noisy_model_input = FluxFillPipeline._pack_latents(
                        noisy_model_input,
                        batch_size=model_input.shape[0],
                        num_channels_latents=model_input.shape[1],
                        height=model_input.shape[2],
                        width=model_input.shape[3],
                    )

                    if check_first:
                        logger.info(
                            f"[{accelerator.process_index}] packed_noisy_model_input.shape: {packed_noisy_model_input.shape}"
                        )

                    guidance = torch.full(
                        [1], args.guidance_scale, device=accelerator.device
                    )
                    guidance = guidance.expand(model_input.shape[0])

                    if inpaint_cond is not None:
                        packed_noisy_model_input = torch.cat(
                            [packed_noisy_model_input, inpaint_cond], dim=2
                        )

                    if check_first:
                        logger.info(
                            f"[{accelerator.process_index}] After concat packed_noisy_model_input.shape: {packed_noisy_model_input.shape}"
                        )

                    model_pred = transformer(
                        hidden_states=packed_noisy_model_input,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        return_dict=False,
                    )[0]

                    model_pred = FluxFillPipeline._unpack_latents(
                        model_pred,
                        height=args.height,
                        width=args.width,
                        vae_scale_factor=vae_scale_factor,
                    )

                    weighting = compute_loss_weighting_for_sd3(
                        weighting_scheme=args.weighting_scheme, sigmas=sigmas
                    )
                    target = noise - model_input

                    loss_diff = model_pred - target
                    if args.dpo_mask:
                        loss_masks = F.interpolate(
                            masks, size=(model_input.shape[-2], model_input.shape[-1])
                        )
                        loss_diff = loss_diff * loss_masks
                    model_losses = (weighting.float() * loss_diff.pow(2)).mean(
                        dim=[1, 2, 3]
                    )
                    model_losses_w, model_losses_l = model_losses.chunk(2)
                    raw_model_loss = 0.5 * (
                        model_losses_w.mean() + model_losses_l.mean()
                    )
                    model_diff = model_losses_w - model_losses_l

                    with torch.no_grad():
                        ref_model_pred = ref_transformer(
                            hidden_states=packed_noisy_model_input,
                            timestep=timesteps / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]

                        ref_model_pred = FluxFillPipeline._unpack_latents(
                            ref_model_pred,
                            height=args.height,
                            width=args.width,
                            vae_scale_factor=vae_scale_factor,
                        )

                        ref_loss_diff = ref_model_pred - target
                        if args.dpo_mask:
                            ref_loss_diff = ref_loss_diff * loss_masks
                        ref_losses = ref_loss_diff.pow(2).mean(dim=[1, 2, 3])
                        ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                        ref_diff = ref_losses_w - ref_losses_l

                    scale_term = -0.5 * args.beta_dpo
                    inside_term = scale_term * (model_diff - ref_diff)
                    dpo_loss = -1 * F.logsigmoid(inside_term).mean()

                    loss = dpo_loss

                    avg_dpo_loss = accelerator.gather(dpo_loss).mean().item()
                    avg_model_mse = accelerator.gather(raw_model_loss).mean().item()

                    if args.do_test and not torch.isfinite(loss.detach()).all():
                        raise RuntimeError(f"Smoke test loss is not finite: {loss.detach()}")

                    accelerator.backward(loss)
                    if args.do_test and accelerator.sync_gradients:
                        smoke_grad_name = None
                        smoke_grad_norm = None
                        for name, param in transformer.named_parameters():
                            if not param.requires_grad or param.grad is None:
                                continue
                            grad = param.grad.detach()
                            if not torch.isfinite(grad).all():
                                raise RuntimeError(
                                    f"Smoke test found non-finite gradient in {name}"
                                )
                            grad_norm = grad.float().norm().item()
                            if grad_norm > 0:
                                smoke_grad_name = name
                                smoke_grad_norm = grad_norm
                                break
                        if smoke_grad_name is None:
                            raise RuntimeError(
                                "Smoke test did not find a finite nonzero gradient on any trainable Flux parameter."
                            )
                        logger.info(
                            f"Smoke test gradient check passed: {smoke_grad_name} grad_norm={smoke_grad_norm:.6e}"
                        )
                        accelerator.print(
                            f"Smoke test gradient check passed: {smoke_grad_name} grad_norm={smoke_grad_norm:.6e}"
                        )

                    if accelerator.sync_gradients:
                        if args.train_base_model:
                            params_to_clip = transformer.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            else:
                # ---- BrushNet step ----
                accelerator.wait_for_everyone()

                with accelerator.accumulate(brushnet):
                    with accelerator.autocast():
                        pixel_values, inpainting_pixel_values = (
                            batch["pixel_values"],
                            batch["inpainting_pixel_values"],
                        )
                        if not args.metrics.enable:
                            inpainting_pixel_values = (
                                1 - batch["masks"]
                            ) * inpainting_pixel_values + batch["masks"] * pixel_values

                        if accelerator.is_main_process:
                            if step == 0:
                                from torchvision.transforms.functional import (
                                    to_pil_image,
                                )

                                if pixel_values.dtype == torch.bfloat16:
                                    image_1_win = pixel_values.to(
                                        dtype=torch.float32
                                    ).to("cpu")
                                    image_1_loss = inpainting_pixel_values.to(
                                        dtype=torch.float32
                                    ).to("cpu")
                                    image_2 = batch["masks"].to(dtype=torch.float32).to(
                                        "cpu"
                                    )
                                else:
                                    image_1_win = pixel_values.to("cpu")
                                    image_1_loss = inpainting_pixel_values.to("cpu")
                                    image_2 = batch["masks"].to("cpu")
                                to_pil_image((image_1_win[0] + 1) / 2).save(
                                    f"{os.path.join(args.output_dir, 'logs', 'training_image_checks_win.png')}"
                                )
                                to_pil_image((image_1_loss[0] + 1) / 2).save(
                                    f"{os.path.join(args.output_dir, 'logs', 'training_image_checks_loss.png')}"
                                )
                                if image_2.ndim == 4:
                                    to_pil_image(image_2[0].squeeze()).save(
                                        f"{os.path.join(args.output_dir, 'logs', 'training_image_checks_mask.png')}"
                                    )
                                else:
                                    to_pil_image(image_2[0]).save(
                                        f"{os.path.join(args.output_dir, 'training_image_checks_mask.png')}"
                                    )

                        if args.dpo_new:
                            feed_pixel_values = torch.cat(
                                [inpainting_pixel_values, pixel_values]
                            )
                        else:
                            feed_pixel_values = torch.cat(
                                [pixel_values, inpainting_pixel_values]
                            )

                        if check_first:
                            logger.info(
                                f"[{accelerator.process_index}] feed_pixel_values shape: {feed_pixel_values.shape}"
                            )
                            logger.info(
                                f"[{accelerator.process_index}] mask shape: {batch['masks'].shape}"
                            )

                        latents = vae.encode(
                            feed_pixel_values.to(dtype=weight_dtype)
                        ).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                        if check_first:
                            logger.info(
                                f"[{accelerator.process_index}] latents shape: {latents.shape}"
                            )

                        (
                            feed_win_conditioning_pixel_values,
                            feed_loss_conditioning_pixel_values,
                        ) = (
                            batch["conditioning_pixel_values"],
                            batch["conditioning_inpainting_pixel_values"],
                        )
                        conditioning_win_latents = vae.encode(
                            feed_win_conditioning_pixel_values.to(dtype=weight_dtype)
                        ).latent_dist.sample()
                        conditioning_loss_latents = vae.encode(
                            feed_loss_conditioning_pixel_values.to(dtype=weight_dtype)
                        ).latent_dist.sample()
                        conditioning_win_latents = (
                            conditioning_win_latents * vae.config.scaling_factor
                        )
                        conditioning_loss_latents = (
                            conditioning_loss_latents * vae.config.scaling_factor
                        )

                        masks = torch.nn.functional.interpolate(
                            batch["masks"],
                            size=(latents.shape[-2], latents.shape[-1]),
                        )

                        conditioning_win_latents = torch.concat(
                            [conditioning_win_latents, masks], 1
                        )
                        conditioning_loss_latents = torch.concat(
                            [conditioning_loss_latents, masks], 1
                        )

                        if check_first:
                            logger.info(
                                f"[{accelerator.process_index}] conditioning_win_latents shape: {conditioning_win_latents.shape}"
                            )
                            logger.info(
                                f"[{accelerator.process_index}] conditioning_loss_latents shape: {conditioning_loss_latents.shape}"
                            )
                            logger.info(
                                f"[{accelerator.process_index}] masks shape: {masks.shape}"
                            )

                        if args.dpo_new:
                            conditioning_latents = torch.cat(
                                [conditioning_loss_latents, conditioning_win_latents]
                            )
                        else:
                            conditioning_latents = torch.cat(
                                [conditioning_win_latents, conditioning_loss_latents]
                            )

                        if check_first:
                            logger.info(
                                f"[{accelerator.process_index}] conditioning_latents shape: {conditioning_latents.shape}"
                            )

                        noise = torch.randn_like(latents)
                        bsz = latents.shape[0]
                        timesteps = torch.randint(
                            0,
                            noise_scheduler.config.num_train_timesteps,
                            (bsz,),
                            device=latents.device,
                        )
                        timesteps = timesteps.long()
                        timesteps = timesteps.chunk(2)[0].repeat(2)
                        noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                        noisy_latents = noise_scheduler.add_noise(
                            latents, noise, timesteps
                        )

                        encoder_hidden_states = text_encoder(
                            batch["input_ids"], return_dict=False
                        )[0]
                        encoder_hidden_states = encoder_hidden_states.repeat(2, 1, 1)

                        (
                            down_block_res_samples,
                            mid_block_res_sample,
                            up_block_res_samples,
                        ) = brushnet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            brushnet_cond=conditioning_latents,
                            return_dict=False,
                        )

                        model_pred = unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            down_block_add_samples=[
                                sample.to(dtype=weight_dtype)
                                for sample in down_block_res_samples
                            ],
                            mid_block_add_sample=mid_block_res_sample.to(
                                dtype=weight_dtype
                            ),
                            up_block_add_samples=[
                                sample.to(dtype=weight_dtype)
                                for sample in up_block_res_samples
                            ],
                            return_dict=False,
                        )[0]

                        if noise_scheduler.config.prediction_type == "epsilon":
                            target = noise
                        elif noise_scheduler.config.prediction_type == "v_prediction":
                            target = noise_scheduler.get_velocity(
                                latents, noise, timesteps
                            )
                        else:
                            raise ValueError(
                                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                            )

                        loss_diff = model_pred - target
                        if args.dpo_mask:
                            loss_masks = 1 - masks
                            loss_masks = loss_masks.repeat(2, 1, 1, 1)
                            loss_diff = loss_diff * loss_masks
                        model_losses = loss_diff.pow(2).mean(dim=[1, 2, 3])

                        model_losses_w, model_losses_l = model_losses.chunk(2)
                        raw_model_loss = 0.5 * (
                            model_losses_w.mean() + model_losses_l.mean()
                        )
                        model_diff = model_losses_w - model_losses_l

                        with torch.no_grad():
                            (
                                ref_down_block_res_samples,
                                ref_mid_block_res_sample,
                                ref_up_block_res_samples,
                            ) = ref_brushnet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                                brushnet_cond=conditioning_latents,
                                return_dict=False,
                            )

                            ref_pred = unet(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                                down_block_add_samples=[
                                    sample.to(dtype=weight_dtype)
                                    for sample in ref_down_block_res_samples
                                ],
                                mid_block_add_sample=ref_mid_block_res_sample.to(
                                    dtype=weight_dtype
                                ),
                                up_block_add_samples=[
                                    sample.to(dtype=weight_dtype)
                                    for sample in ref_up_block_res_samples
                                ],
                                return_dict=False,
                            )[0]

                            ref_loss_diff = ref_pred - target
                            if args.dpo_mask:
                                ref_loss_diff = ref_loss_diff * loss_masks
                            ref_losses = ref_loss_diff.pow(2).mean(dim=[1, 2, 3])
                            ref_losses_w, ref_losses_l = ref_losses.chunk(2)
                            ref_diff = ref_losses_w - ref_losses_l
                            raw_ref_loss = ref_losses.mean()

                        scale_term = -0.5 * args.beta_dpo
                        inside_term = scale_term * (model_diff - ref_diff)

                        dpo_loss = -1 * F.logsigmoid(inside_term).mean()
                        loss = dpo_loss

                    avg_loss = accelerator.gather(loss).mean().item()
                    avg_dpo_loss = accelerator.gather(dpo_loss).mean().item()
                    avg_raw_model_mse = accelerator.gather(raw_model_loss).mean().item()
                    avg_ref_mse = accelerator.gather(raw_ref_loss).mean().item()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = brushnet.parameters()
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # ---- End-of-batch (shared) ----

            if model_type == "brushnet":
                logger.info(
                    f"[{accelerator.process_index}] global_step: {global_step} / {args.max_train_steps}, step_loss: {avg_loss}, lr: {lr_scheduler.get_last_lr()[0]}"
                )

            if accelerator.sync_gradients:
                check_first = False
                progress_bar.update(1)
                global_step += 1

                if model_type == "flux":
                    logs = {
                        "step_loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    accelerator.log(logs, step=global_step)
                    accelerator.log(
                        {"model_mse_unaccumulated": avg_model_mse}, step=global_step
                    )
                    accelerator.log(
                        {"dpo_loss_unaccumulated": avg_dpo_loss}, step=global_step
                    )
                else:
                    logs = {
                        "step_loss": avg_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    accelerator.log({"train_loss": avg_loss}, step=global_step)
                    accelerator.log(logs, step=global_step)
                    accelerator.log(
                        {"model_mse_unaccumulated": avg_raw_model_mse},
                        step=global_step,
                    )
                    accelerator.log(
                        {"ref_mse_unaccumulated": avg_ref_mse}, step=global_step
                    )
                    accelerator.log(
                        {"dpo_loss_unaccumulated": avg_dpo_loss}, step=global_step
                    )

                # Checkpointing
                if model_type == "flux":
                    save_condition = (
                        accelerator.distributed_type == DistributedType.DEEPSPEED
                        or accelerator.is_main_process
                    )
                else:
                    save_condition = accelerator.is_main_process

                if save_condition and not args.do_test:
                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [
                                    d
                                    for d in checkpoints
                                    if d.startswith("checkpoint")
                                ]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )

                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = (
                                        len(checkpoints)
                                        - args.checkpoints_total_limit
                                        + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"[{accelerator.process_index}] {len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"[{accelerator.process_index}] removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        if model_type == "flux":
                                            removing_checkpoint = os.path.join(
                                                args.output_dir,
                                                removing_checkpoint,
                                                "pytorch_model",
                                            )
                                            if os.path.exists(removing_checkpoint):
                                                shutil.rmtree(removing_checkpoint)
                                        else:
                                            removing_checkpoint = os.path.join(
                                                args.output_dir, removing_checkpoint
                                            )
                                            shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(
                            f"[{accelerator.process_index}] Saved state to {save_path}"
                        )

                # Validation
                if model_type == "flux":
                    if (
                        not args.do_test
                        and (
                            global_step % args.validation_steps == 0
                            or global_step == 1
                        )
                        and accelerator.is_main_process
                    ):
                        logger.info(
                            f"[{accelerator.process_index}] validating...",
                            main_process_only=False,
                        )
                        pipeline = FluxFillPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            transformer=accelerator.unwrap_model(transformer),
                            torch_dtype=weight_dtype,
                            vae=vae,
                            tokenizer=tokenizer_one,
                            tokenizer_2=tokenizer_two,
                            text_encoder=text_encoder_one,
                            text_encoder_2=text_encoder_two,
                        ).to(accelerator.device)
                        image_logs = log_validation_flux(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            epoch=epoch,
                            step=global_step,
                            is_final_validation=False,
                        )
                        logger.info(
                            f"[{accelerator.process_index}] validation done.",
                            main_process_only=False,
                        )
                else:
                    if accelerator.is_main_process:
                        if (
                            not args.do_test
                            and args.validation_prompt is not None
                            and (
                                global_step % args.validation_steps == 0
                                or global_step == 1
                            )
                        ):
                            image_logs = log_validation_brushnet(
                                vae,
                                text_encoder,
                                tokenizer,
                                unet,
                                brushnet,
                                args,
                                accelerator,
                                weight_dtype,
                                global_step,
                            )

            progress_bar.set_postfix(**logs)
            if model_type == "flux":
                logger.info(
                    f"global_step: {global_step} / {args.max_train_steps}, step_loss: {loss.detach().item()}, lr: {lr_scheduler.get_last_lr()[0]}"
                )

            if global_step >= args.train_steps:
                accelerator.wait_for_everyone()
                accelerator.end_training()
                if args.do_test:
                    return

    # -----------------------------------------------------------------------
    # Final save
    # -----------------------------------------------------------------------

    if args.do_test:
        logger.info("Smoke test mode enabled; skipping final model save.")
        accelerator.wait_for_everyone()
        accelerator.end_training()
        return

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if model_type == "flux":
            transformer = unwrap_model(transformer)
            pipeline = FluxFillPipeline.from_pretrained(
                args.pretrained_model_name_or_path, transformer=transformer
            )
            pipeline.save_pretrained(args.output_dir)

            pipeline = FluxFillPipeline.from_pretrained(
                args.output_dir,
                revision=args.revision,
                variant=args.variant,
                torch_dtype=weight_dtype,
            )
        else:
            brushnet = unwrap_model(brushnet)
            brushnet.save_pretrained(args.output_dir)

            image_logs = None
            if args.validation_prompt is not None:
                image_logs = log_validation_brushnet(
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    unet=unet,
                    brushnet=None,
                    args=args,
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    step=global_step,
                    is_final_validation=True,
                )

            if args.push_to_hub:
                save_model_card_brushnet(
                    repo_id,
                    image_logs=image_logs,
                    base_model=args.pretrained_model_name_or_path,
                    repo_folder=args.output_dir,
                )
                upload_folder(
                    repo_id=repo_id,
                    folder_path=args.output_dir,
                    commit_message="End of training",
                    ignore_patterns=["step_*", "epoch_*"],
                )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main(args, model_type)
