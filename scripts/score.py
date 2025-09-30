"""Score multiple seeds per image across configurable metrics.

Loads annotations listing a canonical seed-0 image then derives seed variants.
Each metric is loaded only when requested to conserve memory.
"""

import argparse
from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import os
from pathlib import Path
import numpy as np
import torch
import cv2
import json
import logging
from contextlib import nullcontext
from typing import Optional

from datasets import load_dataset
from PIL import Image
import math


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a PowerPoint presentation from images."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="clip_score",
        help="Metric to use for scoring images. Should be in [clip_score, hps, pick_score, image_reward, vqa_score , unified_reward].",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to save the scoring results."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16).",
    )
    parser.add_argument(
        "--annotations_path", type=str, help="Path to the annotations file."
    )
    parser.add_argument("--model", type=str, default="flux")

    return parser.parse_args()


args = parse_args()

if args.annotations_path is None:
    raise ValueError(
        "Please provide the path to the annotations file using --annotations_path."
    )

logging_dir = os.path.join(args.output_dir, f"{args.metric}", "logs")
accelerator_project_config = ProjectConfiguration(
    project_dir=args.output_dir, logging_dir=logging_dir
)
accelerator = Accelerator(
    mixed_precision=args.mixed_precision,
    project_config=accelerator_project_config,
)

logger = get_logger(__name__)


class MetricsCalculator:
    """Lazy loader for metric-specific models and computation helpers."""
    def __init__(
        self,
        metric: str,
        device: str = "cuda",
        accelerator: Optional[Accelerator] = None,
    ):
        self.device = device
        self.metric = metric
        self.accelerator = accelerator
        self.state = accelerator.state if accelerator is not None else PartialState()
        self.rank = getattr(self.state, "process_index", 0)

        if self.metric not in [
            "clip_score",
            "hps",
            "hpsv3",
            "pick_score",
            "aesthetic_score",
            "image_reward",
            "vqa_score",
            "unified_reward",
            "pe_score",
            "lpips",
            "psnr",
        ]:
            raise ValueError(
                f"Unsupported metric: {self.metric}, should be one of [clip_score, hps, pick_score, aesthetic_score, image_reward, vqa_score , unified_reward, pe_score]."
            )

        if self.metric == "clip_score":
            from torchmetrics.multimodal import CLIPScore

            logger.info(f"[{self.rank}] Loading CLIP model...")
            self.clip_metric_calculator = CLIPScore(
                model_name_or_path="openai/clip-vit-large-patch14"
            ).to(device)

        elif self.metric == "image_reward":
            import ImageReward as RM

            logger.info(f"[{self.rank}] Loading ImageReward model...")
            self.imagereward_model = RM.load("ImageReward-v1.0", device=self.device)

        elif self.metric == "hps":
            import hpsv2

            logger.info(f"[{self.rank}] Loading hpsv2 model...")

        elif self.metric == "hpsv3":
            from hpsv3 import HPSv3RewardInferencer
            import huggingface_hub

            logger.info(f"[{self.rank}] Loading hpsv3 model...")
            self.inferencer = HPSv3RewardInferencer(
                device=self.device,
                checkpoint_path=huggingface_hub.hf_hub_download(
                    "MizzenAI/HPSv3", "HPSv3.safetensors", repo_type="model"
                ),
            )

        elif self.metric == "aesthetic_score":
            import open_clip

            logger.info(f"[{self.rank}] Loading aesthetic model...")
            self.clip_model, _, self.clip_preprocess = (
                open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
            )
            self.clip_model = self.clip_model.to(self.device)
            self.aesthetic_model = torch.nn.Linear(768, 1)
            aesthetic_model_ckpt_path = "/root/flux/ckpts/sa_0_4_vit_l_14_linear.pth"
            self.aesthetic_model.load_state_dict(
                torch.load(aesthetic_model_ckpt_path, map_location=self.device)
            )
            self.aesthetic_model = self.aesthetic_model.to(self.device)
            self.aesthetic_model.eval()

        elif self.metric == "pick_score":
            from transformers import AutoProcessor, AutoModel

            logger.info(f"[{self.rank}] Loading PickScore model...")
            self.pick_score_processor = AutoProcessor.from_pretrained(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            )
            self.pick_score_model = (
                AutoModel.from_pretrained("yuvalkirstain/PickScore_v1")
                .eval()
                .to(self.device)
            )

        elif self.metric == "vqa_score":
            import t2v_metrics

            logger.info(f"[{self.rank}] Loading VQAScore model...")
            self.clip_flant5_score = t2v_metrics.VQAScore(
                model="clip-flant5-xxl",
                device=self.device,
                cache_dir="/root/flux/hf_cache/",
            )

        elif self.metric == "unified_reward":
            logger.info(f"[{self.rank}] Preparing unified_reward client...")
            self.api_url = "http://127.0.0.1:8080"
            self.max_workers = 8
            self.vllm_output_path = os.path.join(
                args.output_dir, f"{args.metric}", f"vllm_results_{self.rank}.json"
            )
            os.makedirs(os.path.dirname(self.vllm_output_path), exist_ok=True)
            open(self.vllm_output_path, "w").close()

        elif self.metric == "pe_score":
            logger.info(f"[{self.rank}] Loading PE model...")
            import sys

            sys.path.append("/root/flux/perception_models")
            import core.vision_encoder.pe as pe
            import core.vision_encoder.transforms as transforms

            self.pe_model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)
            self.pe_model = self.pe_model.cuda().to(self.device)
            self.pe_preprocess = transforms.get_image_transform(
                self.pe_model.image_size
            )
            self.pe_tokenizer = transforms.get_text_tokenizer(
                self.pe_model.context_length
            )

        elif self.metric == "lpips":
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            logger.info(f"[{self.rank}] Loading LPIPS model...")
            self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(
                net_type="squeeze"
            ).to(device)

        elif self.metric == "psnr":
            pass

    def autocast(self):
        if self.accelerator is not None:
            return self.accelerator.autocast()
        return nullcontext()

    def calculate_psnr(self, img_pred, img_gt, mask=None):
        """Compute PSNR (optionally over masked area)."""
        img_pred = np.array(img_pred).astype(np.float32) / 255.0
        img_gt = np.array(img_gt).astype(np.float32) / 255.0

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        difference = img_pred - img_gt
        difference_square = difference**2
        difference_square_sum = difference_square.sum()
        difference_size = mask.sum() if mask is not None else np.prod(img_gt.shape)

        mse = difference_square_sum / difference_size

        if mse < 1.0e-10:
            return 1000
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def calculate_pe_score(self, image, prompt):
        """Compute PE score (custom perception encoder similarity)."""
        image = self.pe_preprocess(image).unsqueeze(0).cuda().to(self.device)
        text = self.pe_tokenizer([prompt]).cuda().to(self.device)

        with torch.no_grad(), self.autocast():
            image_features, text_features, logit_scale = self.pe_model(image, text)
            text_probs = logit_scale * image_features @ text_features.T

        return text_probs.item()

    def calculate_unified_reward(self, image_paths: list[str], prompts: list[str]):
        from vllm_client import gen_results  # Wait for the VLLM server to start

        gen_results(
            image_paths=image_paths,
            prompts=prompts,
            output_path=self.vllm_output_path,
            api=self.api_url,
            max_workers=self.max_workers,
        )

    def calculate_vqascore(self, images, prompt):
        """
        Calculate VQA score for a list of images and a prompt.
        """
        with torch.no_grad(), self.autocast():
            scores = (
                self.clip_flant5_score(images=images, texts=[prompt])
                .squeeze()
                .cpu()
                .tolist()
            )
        return scores

    def calculate_pick_score(self, images, prompt):
        image_inputs = self.pick_score_processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.pick_score_processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with self.autocast():
            with torch.no_grad():
                image_embs = self.pick_score_model.get_image_features(**image_inputs)
                image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

                text_embs = self.pick_score_model.get_text_features(**text_inputs)
                text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

                scores = (
                    self.pick_score_model.logit_scale.exp()
                    * (text_embs @ image_embs.T)[0]
                )

            return scores.cpu().tolist()

    def calculate_aesthetic_score(self, img):
        image = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad(), self.autocast():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = self.aesthetic_model(image_features)
        return prediction.item()

    def calculate_clip_similarity(self, img, txt):
        # Implement the logic to calculate CLIP score for images in the folder
        img = np.array(img)
        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)
        with self.autocast():
            score = self.clip_metric_calculator(img_tensor, txt)
        return score.item()

    def calculate_image_reward(self, image, prompt):
        reward = self.imagereward_model.score(prompt, [image])
        return reward

    def calculate_hpsv21_score(self, image, prompt):
        import hpsv2

        # we modified function score in img_score.py in hpsv2 to support assigned device
        result = hpsv2.score(image, prompt, hps_version="v2.1", _device=self.device)
        return [i.item() for i in result]  # Convert to list of floats

    def calculate_hpsv3_score(self, images, prompts):
        result = self.inferencer.reward(images, prompts)
        return [i[0].item() for i in result]  # Convert to list of floats


def main():
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=os.path.join(logging_dir, "log.txt"),
    )
    logger.info(accelerator.state, main_process_only=False)
    rank, num_processes, device = (
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, f"{args.metric}"), exist_ok=True)

    annotations_path = args.annotations_path

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    # explicitly pass accelerator to avoid global reference inside the class
    metrics_calculator = MetricsCalculator(
        metric=args.metric, device=device, accelerator=accelerator
    )

    results = {}

    global_step = 0

    def barrier():
        try:
            accelerator.wait_for_everyone()
        except Exception:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    for i, item in enumerate(annotations):
        if i % num_processes != rank:
            continue

        barrier()

        logger.info(
            f"Processing item {global_step}/{len(annotations) // num_processes}"
        )

        images = [item["image_path"]] + [
            item["image_path"].replace("local_seed_0", f"local_seed_{seed}")
            for seed in range(1, 16)
        ]
        captions = [item["caption"]] * len(images)

        with torch.no_grad():
            if args.metric == "image_reward":
                scores = [
                    metrics_calculator.calculate_image_reward(
                        Image.open(f"{os.path.join(img)}")
                        .convert("RGB")
                        .resize((512, 512)),
                        cap,
                    )
                    for img, cap in zip(images, captions)
                ]
            elif args.metric == "clip_score":
                scores = [
                    metrics_calculator.calculate_clip_similarity(
                        Image.open(f"{os.path.join(img)}")
                        .convert("RGB")
                        .resize((512, 512)),
                        cap,
                    )
                    for img, cap in zip(images, captions)
                ]
            elif args.metric == "hps":
                scores = metrics_calculator.calculate_hpsv21_score(
                    [
                        Image.open(f"{os.path.join(img)}")
                        .convert("RGB")
                        .resize((512, 512))
                        for img in images
                    ],
                    captions[0],
                )
            elif args.metric == "hpsv3":
                scores = metrics_calculator.calculate_hpsv3_score(
                    [f"{os.path.join(img)}" for img in images], captions
                )
            elif args.metric == "aesthetic_score":
                scores = [
                    metrics_calculator.calculate_aesthetic_score(
                        Image.open(f"{os.path.join(img)}")
                        .convert("RGB")
                        .resize((512, 512))
                    )
                    for img in images
                ]
            elif args.metric == "pick_score":
                scores = metrics_calculator.calculate_pick_score(
                    [
                        Image.open(f"{os.path.join(img)}")
                        .convert("RGB")
                        .resize((512, 512))
                        for img in images
                    ],
                    captions[0],
                )
            elif args.metric == "vqa_score":
                scores = metrics_calculator.calculate_vqascore(
                    [f"{os.path.join(img)}" for img in images], captions[0]
                )
            elif args.metric == "unified_reward":
                metrics_calculator.calculate_unified_reward(
                    image_paths=[f"{os.path.join(img)}" for img in images],
                    prompts=captions,
                )
            elif args.metric == "pe_score":
                scores = [
                    metrics_calculator.calculate_pe_score(
                        Image.open(f"{os.path.join(img)}")
                        .convert("RGB")
                        .resize((512, 512)),
                        captions[0],
                    )
                    for img in images
                ]
            else:
                raise ValueError(
                    f"Unsupported metric: {args.metric}. Supported metrics are 'clip_score', 'image_reward', 'hps', 'aesthetic_score', 'pick_score', and 'vqa_score'."
                )

        if args.metric != "unified_reward":
            results[f"{item['image_id']}"] = {}
            for seed in range(16):
                results[f"{item['image_id']}"][f"seed_{seed}"] = {
                    f"{args.metric}": scores[seed]
                }

        global_step += 1

    if args.metric != "unified_reward":
        out_dir = os.path.join(args.output_dir, f"{args.metric}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"scores_{rank}.json"), "w") as f:
            json.dump(results, f)

    try:
        barrier()
        accelerator.end_training()
    except Exception:
        pass


if __name__ == "__main__":
    main()
