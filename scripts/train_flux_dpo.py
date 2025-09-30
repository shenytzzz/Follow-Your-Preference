###############################################################################################
# Modified from https://github.com/nftblackmagic/catvton-flux/blob/main/train_flux_inpaint.py #
###############################################################################################

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

# better to modify /path/to/miniconda3/envs/flux/lib/python3.12/site-packages/diffusers/image_processor.py:746 to close the futurewarning

"""Flux Transformer DPO training (inpainting).

Trains a subset of transformer blocks using a DPO objective comparing win vs
loss generations, with a frozen reference transformer for baseline losses.
"""

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

import numpy as np
import torch
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
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    PretrainedConfig,
    T5EncoderModel,
    T5TokenizerFast,
)
from train_utils import (
    prepare_fill_with_mask,
    prepare_latents,
    encode_images_to_latents,
)
from diffusers import FluxTransformer2DModel, FluxFillPipeline

# from src.flux.pipeline_flux_inpaint import FluxInpaintingPipeline
from diffusers.image_processor import VaeImageProcessor
from deepspeed.runtime.engine import DeepSpeedEngine

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
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
from diffusers.utils.torch_utils import is_compiled_module
from datasets import load_dataset

from omegaconf import OmegaConf

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.30.2")


def check_args(args):
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


full_args = OmegaConf.load("/root/test_env/configs/configs_flux.yaml")
args = full_args.train
check_args(args)
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


def log_validation(
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

    # run inference
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed
        else None
    )
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
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


def import_model_class_from_model_name_or_path(
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

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
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

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
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


class MyDataset:
    """Collate: produce (win, loss) image tensors + mask for DPO."""
    def __init__(self, resolution, random_mask, tokenizer=None, score_file=None):
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.random_mask = random_mask
        # with open(self.file_path, "r", encoding="utf-8") as f:
        #     self.inpainting_annotations = json.load(f)
        if score_file is not None:
            with open(score_file, "r") as f:
                self.scores = json.load(f)
        else:
            raise ValueError(
                "`score_file` must be provided to load the scoring results."
            )

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

    # def get_len(self):
    #     return len(self.inpainting_annotations)

    def __call__(self, examples):
        pixel_values = []
        # conditioning_pixel_values=[]
        masks = []
        input_ids = []
        # conditioning_inpainting_pixel_values=[]
        inpainting_pixel_values = []
        # image_ids=[]

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
            # masks are different from BrushNet
            # mask = 1 - mask

            w, h, _ = mask.shape
            if w > h:
                scale = self.resolution / h
            else:
                scale = self.resolution / w
            w_new = int(np.ceil(w * scale))
            h_new = int(np.ceil(h * scale))

            random_crop = inpainting_crop
            if not args.metrics.enable:
                win_image = cv2.resize(
                    win_image, (w_new, h_new), interpolation=cv2.INTER_CUBIC
                )
                win_image = win_image[
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

            win_image = cv2.cvtColor(win_image, cv2.COLOR_BGR2RGB)
            lose_image = cv2.cvtColor(lose_image, cv2.COLOR_BGR2RGB)
            win_image = (win_image.astype(np.float32) / 127.5) - 1.0
            lose_image = (lose_image.astype(np.float32) / 127.5) - 1.0

            mask = mask.astype(np.float32)

            pixel_values.append(torch.tensor(win_image).permute(2, 0, 1))
            inpainting_pixel_values.append(torch.tensor(lose_image).permute(2, 0, 1))
            masks.append(torch.tensor(mask).permute(2, 0, 1))
            input_ids.append(caption)
            # image_ids.append(image_id)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        inpainting_pixel_values = torch.stack(inpainting_pixel_values)
        inpainting_pixel_values = inpainting_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()
        masks = torch.stack(masks)
        masks = masks.to(memory_format=torch.contiguous_format).float()

        return {
            "pixel_values": pixel_values,
            "inpainting_pixel_values": inpainting_pixel_values,
            "masks": masks,
            "input_ids": input_ids,
        }


def main(args):
    """Full DPO training loop (data, forward, DPO loss, checkpoints)."""
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
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

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
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
        transformer.requires_grad_(
            False
        )  # Set all parameters to not require gradients by default

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

    # #you can train your own layers
    # for n, param in transformer.named_parameters():
    #     print(n)
    #     if 'single_transformer_blocks' in n:
    #         param.requires_grad = False
    #     elif 'transformer_blocks' in n and '1.attn' in n:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    logger.info(
        f"[{accelerator.process_index}] transformer parameters: {sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000}"
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    ref_transformer.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        if args.train_base_model:
            transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(model, DeepSpeedEngine):
                    # For DeepSpeed models, we need to get the underlying model
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
                    unwrap_model(model), (CLIPTextModelWithProjection, T5EncoderModel)
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

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()
                else:
                    logger.info(
                        f"[{accelerator.process_index}] no weights",
                        main_process_only=False,
                    )

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
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

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Optimization parameters
    if args.train_base_model:
        transformer_parameters_with_lr = {
            "params": transformer.parameters(),
            "lr": args.learning_rate,
        }
        # trainable_params = [p for p in transformer.parameters() if p.requires_grad]
        # transformer_parameters_with_lr = {"params": trainable_params, "lr": args.learning_rate}
        # params_to_optimize = [transformer_parameters_with_lr]

    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
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

    # Dataset and DataLoaders creation:
    train_dataset = load_dataset(
        "json", data_files={"train": args.train_json_dir}, split="train"
    )
    train_dataset_len = len(train_dataset)

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=MyDataset(
            resolution=args.resolution,
            random_mask=args.random_mask,
            score_file=args.metrics.score_file if args.metrics.enable else None,
        ),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            text_ids = text_ids.to(accelerator.device)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.

    # Handle class prompt for prior-preservation.

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
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

    if args.train_base_model:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "flux-inpainting-dpo"
        accelerator.init_trackers(
            tracker_name,
            config=vars(args),
            init_kwargs={"wandb": {"settings": wandb.Settings(code_dir=".")}},
        )
    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"[{accelerator.process_index}]  Num examples = {len(train_dataset)}")
    logger.info(f"[{accelerator.process_index}]  Num batches each epoch = {len(train_dataloader)}")
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
    logger.info(f"  Note: {args.note}")
    global_step = 0
    first_epoch = 0
    epoch = first_epoch

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

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

    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"[{accelerator.process_index}] epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            if args.train_base_model:
                models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                # with accelerator.autocast():
                accelerator.wait_for_everyone()
                # vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
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

                feed_in_image = torch.cat([win_pixel_values, loss_pixel_values], dim=0)

                feed_in_image = image_processor.preprocess(
                    feed_in_image, height=args.height, width=args.width
                ).to(dtype=vae.dtype, device=accelerator.device)
                masks = mask_processor.preprocess(
                    masks, height=args.height, width=args.width
                ).to(dtype=vae.dtype, device=accelerator.device)

                if accelerator.is_main_process:
                    if global_step == 0:
                        from torchvision.transforms.functional import to_pil_image

                        # BECAUSE BS == 1 !!!!!!!!! Modify here as we provide in train_brushnet_dpo.py if you get a bs larger than 1
                        if feed_in_image.dtype == torch.bfloat16:
                            image_1 = feed_in_image.to(dtype=torch.float32).to("cpu")
                            image_2 = masks.to(dtype=torch.float32).to("cpu")
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

                # encode batch prompts when custom prompts are provided for each image -
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

                # prompt_embeds = prompt_embeds.repeat(2, 1, 1)  # repeat for both images
                # pooled_prompt_embeds = pooled_prompt_embeds.repeat(2, 1, 1)  #

                # checked
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
                    masked_image=masked_image,  #
                    mask=masks,  #
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

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(
                    model_input, device=accelerator.device, dtype=weight_dtype
                )
                noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
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

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
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

                # handle guidance
                # guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                guidance = torch.full(
                    [1], args.guidance_scale, device=accelerator.device
                )
                guidance = guidance.expand(model_input.shape[0])

                # print("before concat packed_noisy_model_input.shape", packed_noisy_model_input.shape, "inpaint_cond.shape", inpaint_cond.shape)

                if inpaint_cond is not None:
                    packed_noisy_model_input = torch.cat(
                        [packed_noisy_model_input, inpaint_cond], dim=2
                    )

                if check_first:
                    logger.info(
                        f"[{accelerator.process_index}] After concat packed_noisy_model_input.shape: {packed_noisy_model_input.shape}"
                    )
                # print("guidance", guidance, "pooled_prompt_embeds.shape", pooled_prompt_embeds.shape, "prompt_embeds.shape", prompt_embeds.shape)

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                # print("model_pred.shape", model_pred.shape, "prompt_embeds.shape", prompt_embeds.shape, "packed_noisy_model_input.shape", packed_noisy_model_input.shape, "refnet_image.shape", refnet_image.shape)
                # upscaling height & width as discussed in https://github.com/huggingface/diffusers/pull/9257#discussion_r1731108042
                model_pred = FluxFillPipeline._unpack_latents(
                    model_pred,
                    height=args.height,
                    width=args.width,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                # flow matching loss
                target = noise - model_input

                loss_diff = model_pred - target
                if args.dpo_mask:
                    loss_masks = F.interpolate(
                        masks, size=(model_input.shape[-2], model_input.shape[-1])
                    )
                    # loss_masks = loss_masks.repeat(2,1,1,1)
                    loss_diff = loss_diff * loss_masks
                model_losses = (weighting.float() * loss_diff.pow(2)).mean(
                    dim=[1, 2, 3]
                )
                model_losses_w, model_losses_l = model_losses.chunk(2)
                raw_model_loss = 0.5 * (model_losses_w.mean() + model_losses_l.mean())
                model_diff = model_losses_w - model_losses_l

                with torch.no_grad():
                    ref_model_pred = ref_transformer(
                        hidden_states=packed_noisy_model_input,
                        # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
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

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.train_base_model:
                        params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    # accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                check_first = False
                progress_bar.update(1)
                global_step += 1
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

                if (
                    accelerator.distributed_type == DistributedType.DEEPSPEED
                    or accelerator.is_main_process
                ):
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if accelerator.is_main_process:
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith("checkpoint")
                                ]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = (
                                        len(checkpoints)
                                        - args.checkpoints_total_limit
                                        + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"[{accelerator.process_index}] {len(checkpoints)} checkpoints already exist, removing pytorch_model in {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"[{accelerator.process_index}] removing pytorch_models in checkpoints: {', '.join(removing_checkpoints)}"
                                    )

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            args.output_dir,
                                            removing_checkpoint,
                                            "pytorch_model",
                                        )
                                        if os.path.exists(removing_checkpoint):
                                            shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(
                            f"[{accelerator.process_index}] Saved state to {save_path}"
                        )

                if (
                    global_step % args.validation_steps == 0 or global_step == 1
                ) and accelerator.is_main_process:
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
                    image_logs = log_validation(
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

            progress_bar.set_postfix(**logs)
            logger.info(
                f"global_step: {global_step} / {args.max_train_steps}, step_loss: {loss.detach().item()}, lr: {lr_scheduler.get_last_lr()[0]}"
            )

            if global_step >= args.train_steps:
                accelerator.wait_for_everyone()
                accelerator.end_training()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)

        pipeline = FluxFillPipeline.from_pretrained(
            args.pretrained_model_name_or_path, transformer=transformer
        )

        # save the pipeline
        pipeline.save_pretrained(args.output_dir)

        # Final inference
        # Load previous pipeline
        pipeline = FluxFillPipeline.from_pretrained(
            args.output_dir,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

    accelerator.end_training()


if __name__ == "__main__":
    main(args)
