"""Distributed evaluation script for inpainting/editing metrics.

Loads a mapping file (BrushBench / EditBench) and computes a set of image &
image-text metrics (CLIP, HPS v2.1/v3, Aesthetic, PickScore, VQA Score,
UnifiedReward, PE Score, PSNR, LPIPS, MSE). Each metric is instantiated and
computed independently to limit GPU memory. Use --use_blended to score blended
images (generated region pasted back with soft mask).
"""

from diffusers import FluxFillPipeline, FluxTransformer2DModel
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
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError
from urllib.request import urlretrieve
from PIL import Image
import open_clip
import os
import hpsv2
import t2v_metrics
from contextlib import nullcontext
from typing import Optional
import ImageReward as RM
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


logger = get_logger(__name__)


def rle2mask(mask_rle, shape):  # height, width
    """Decode simple RLE (start, length, ...) list into a binary mask."""
    starts, lengths = [
        np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])
    ]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)


class MetricsCalculator:
    def __init__(
        self,
        metric,
        rank,
        device="cuda",
        output_path=None,
        accelerator: Optional[Accelerator] = None,
    ):
        self.device = device
        self.metric = metric
        self.rank = rank
        self.output_path = output_path
        self.accelerator = accelerator
        self.state = accelerator.state if accelerator is not None else PartialState()
        if self.metric not in [
            "clip_score",
            "hps",
            "pick_score",
            "psnr",
            "lpips",
            "mse",
            "aesthetic_score",
            "image_reward",
            "vqa_score",
            "unified_reward",
            "pe_score",
            "hpsv3",
        ]:
            raise ValueError(
                f"Unsupported metric: {self.metric}, should be one of [clip_score, hps, pick_score, psnr, lpips, mse, aesthetic_score, image_reward, vqa_score , unified_reward, pe_score]."
            )

        if self.metric == "clip_score":
            from torchmetrics.multimodal import CLIPScore

            logger.info(f"{[self.rank]} Loading CLIP model...")
            # Initialize CLIP model or any other necessary setup for CLIP score
            self.clip_metric_calculator = CLIPScore(
                model_name_or_path="openai/clip-vit-large-patch14"
            ).to(device)

        elif self.metric == "image_reward":
            import ImageReward as RM

            logger.info(f"{[self.rank]} Loading ImageReward model...")
            self.imagereward_model = RM.load("ImageReward-v1.0", device=self.device)

        elif self.metric == "hps":
            import hpsv2

            logger.info(f"{[self.rank]} Loading hpsv2 model...")

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

            logger.info(f"{[self.rank]} Loading aesthetic model...")
            self.clip_model, _, self.clip_preprocess = (
                open_clip.create_model_and_transforms(
                    "ViT-L-14", pretrained="openai", device=device
                )
            )
            self.aesthetic_model = torch.nn.Linear(768, 1)
            aesthetic_model_ckpt_path = "/root/flux/ckpts/sa_0_4_vit_l_14_linear.pth"
            self.aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path))
            self.aesthetic_model = self.aesthetic_model.to(self.device)
            self.aesthetic_model.eval()

        elif self.metric == "pick_score":
            from transformers import AutoProcessor, AutoModel

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

            logger.info(f"{[self.rank]} Loading VQAScore model...")
            self.clip_flant5_score = t2v_metrics.VQAScore(
                model="clip-flant5-xxl",
                device=self.device,
                cache_dir="/root/flux/hf_cache/",
            )

        elif self.metric == "unified_reward":
            self.api_url = "http://127.0.0.1:8080"
            self.max_workers = 8
            self.vllm_output_path = os.path.join(
                self.output_path, f"unified_reward", f"vllm_results_{self.rank}.json"
            )
            os.makedirs(os.path.dirname(self.vllm_output_path), exist_ok=True)
            open(self.vllm_output_path, "w").close()

        elif self.metric == "pe_score":
            logger.info(f"{[self.rank]} Loading PE model...")
            import sys

            sys.path.append("/root/test_env/perception_models")
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

            self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(
                net_type="squeeze"
            ).to(device)

    def autocast(self):
        if self.accelerator is not None:
            return self.accelerator.autocast()
        return nullcontext()

    def calculate_psnr(self, img_pred, img_gt, mask=None):
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
        difference_size = mask.sum()

        mse = difference_square_sum / difference_size

        if mse < 1.0e-10:
            return 1000
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def calculate_lpips(self, img_gt, img_pred, mask=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask

        img_pred_tensor = (
            torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )
        img_gt_tensor = (
            torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        score = self.lpips_metric_calculator(
            img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1
        )
        score = score.cpu().item()

        return score

    def calculate_mse(self, img_pred, img_gt, mask=None):
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
        difference_size = mask.sum()

        mse = difference_square_sum / difference_size

        return mse.item()

    def calculate_pe_score(self, image, prompt):
        image = self.pe_preprocess(image).unsqueeze(0).cuda().to(self.device)
        text = self.pe_tokenizer([prompt]).cuda().to(self.device)

        with torch.no_grad(), self.autocast():
            image_features, text_features, logit_scale = self.pe_model(image, text)
            text_probs = logit_scale * image_features @ text_features.T

        return text_probs.item()

    def calculate_unified_reward(self, image_paths: list[str], prompts: list[str]):
        from vllm_client import gen_results  # Wait for the VLLM server to start

        result, success = gen_results(
            image_paths=image_paths,
            prompts=prompts,
            output_path=self.vllm_output_path,
            api=self.api_url,
            max_workers=self.max_workers,
        )
        if success:
            return result["final_score"]
        else:
            logger.error(
                f"[{self.rank}]Failed to calculate unified reward for images: {image_paths} with prompts: {prompts}"
            )
            result = -1.0
        return result

    def calculate_vqascore(self, images, prompt):
        """
        Calculate VQA score for a list of images and a prompt.
        """
        with torch.no_grad():
            with self.autocast():
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

            return scores.cpu().tolist()[0]

    def calculate_aesthetic_score(self, img):
        image = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            with self.autocast():
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
        return result[0].item()

    def calculate_hpsv3_score(self, images, prompts):
        result = self.inferencer.reward(images, prompts)
        return result[0][0].item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flux_transformer_ckpt_path",
        type=str,
        default="data/ckpt/segmentation_mask_brushnet_ckpt",
    )
    parser.add_argument(
        "--base_model_path", type=str, default="runwayml/stable-diffusion-v1-5"
    )
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="/root/flux/runs/seed_scaling/scaling16/ensemble/checkpoint-2000/eval",
    )
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
    parser.add_argument("--benchmark", type=str, default="brushbench")

    args = parser.parse_args()
    args.blended = args.use_blended

    accelerator = Accelerator()
    rank = accelerator.process_index
    num_processes = accelerator.num_processes
    device = accelerator.device

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

    metrics = [
        "vqa_score",
        "image_reward",
        "hps",
        "aesthetic_score",
        "pick_score",
        "unified_reward",
        "clip_score",
        "pe_score",
        "psnr",
        "lpips",
        "mse",
        "hpsv3",
    ]
    names = [
        "VQA Score",
        "Image Reward",
        "HPS V2.1",
        "Aesthetic Score",
        "PickScore",
        "Unified Reward",
        "CLIP Similarity",
        "Perception Similarity",
        "PSNR",
        "LPIPS",
        "MSE",
        "HPS V3",
    ]
    metric_name_dict = dict(zip(metrics, names))
    if args.blended:
        blended_evaluation_df = pd.DataFrame()
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

    for metric in metrics:
        accelerator.wait_for_everyone()
        metrics_calculator = MetricsCalculator(
            device=device,
            metric=metric,
            rank=rank,
            output_path=logging_dir,
            accelerator=accelerator,
        )
        print(f"[{rank}] evluating metric: {metric}")
        logger.info(f"[{rank}] evluating metric: {metric}", main_process_only=False)

        image_keys = []
        metric_results = []

        for i, item in enumerate(mapping_file):

            if i % num_processes != rank:
                continue

            key = (
                item["image_id"]
                if args.benchmark == "brushbench"
                else item["aos"] + "_" + item["type"]
            )
            print(f"[{rank}] evaluating image {key} ...")
            logger.info(f"[{rank}] evaluating image {key} ...", main_process_only=False)
            if args.benchmark == "brushbench":
                image_path = item["image"]
                mask = item[args.mask_key]
                prompt = item["caption"]

                src_image_path = os.path.join(args.base_dir, image_path)
                src_image = Image.open(src_image_path).resize((512, 512))

                tgt_image_path = os.path.join(args.image_save_path, image_path)
                tgt_image = Image.open(tgt_image_path).resize((512, 512))

                mask = rle2mask(mask, (512, 512))
                mask = 1 - mask[:, :, np.newaxis]

                if args.blended:
                    blended_image_path = tgt_image_path.replace(".jpg", "_blended.jpg")
                    blended_image = Image.open(blended_image_path).resize((512, 512))

            elif args.benchmark == "editbench":
                image_path = item["image_path"]
                masked_image_path = item["masked_image_path"]
                masked_image = cv2.imread(masked_image_path, cv2.IMREAD_COLOR)
                masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
                masked_image = cv2.resize(
                    masked_image, (512, 512), interpolation=cv2.INTER_CUBIC
                )

                image_diff = masked_image.sum(axis=2, keepdims=True)
                mask = np.zeros_like(image_diff)
                mask[image_diff < 15] = 1  # hacking, why can't u just provide the mask
                mask = 1 - mask
                mask = mask.astype(np.uint8)
                prompt = item["caption"]

                src_image = Image.open(image_path).resize((512, 512)).convert("RGB")

                tgt_image_path = os.path.join(
                    args.image_save_path, "images", f"{key}.jpg"
                )
                tgt_image = Image.open(tgt_image_path).resize((512, 512)).convert("RGB")

                if args.blended:
                    blended_image_path = tgt_image_path.replace(".jpg", "_blended.jpg")
                    blended_image = (
                        Image.open(blended_image_path).resize((512, 512)).convert("RGB")
                    )

            evaluation_result = [key]
            if args.blended:
                blended_evaluation_result = [key]

            if metric == metrics[0]:
                image_keys.append(key)

            with torch.no_grad():
                if metric == "image_reward":
                    # metric_result = metrics_calculator.calculate_image_reward(tgt_image,prompt)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_image_reward(
                            blended_image, prompt
                        )

                if metric == "hps":
                    # metric_result = metrics_calculator.calculate_hpsv21_score(tgt_image,prompt)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_hpsv21_score(
                            blended_image, prompt
                        )

                if metric == "hpsv3":
                    if args.blended:
                        metric_result = metrics_calculator.calculate_hpsv3_score(
                            [blended_image_path], [prompt]
                        )

                if metric == "aesthetic_score":
                    # metric_result = metrics_calculator.calculate_aesthetic_score(tgt_image)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_aesthetic_score(
                            blended_image
                        )

                if metric == "psnr":
                    # metric_result = metrics_calculator.calculate_psnr(src_image, tgt_image, mask)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_psnr(
                            src_image, blended_image, mask
                        )

                if metric == "lpips":
                    # metric_result = metrics_calculator.calculate_lpips(src_image, tgt_image, mask)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_lpips(
                            src_image, blended_image, mask
                        )

                if metric == "mse":
                    # metric_result = metrics_calculator.calculate_mse(src_image, tgt_image, mask)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_mse(
                            src_image, blended_image, mask
                        )

                if metric == "pick_score":
                    # metric_result = metrics_calculator.calculate_psnr(src_image, tgt_image, mask)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_pick_score(
                            [blended_image], prompt
                        )

                if metric == "vqa_score":
                    # metric_result = metrics_calculator.calculate_lpips(src_image, tgt_image, mask)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_vqascore(
                            [blended_image_path], prompt
                        )

                if metric == "unified_reward":
                    # metric_result = metrics_calculator.calculate_mse(src_image, tgt_image, mask)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_unified_reward(
                            image_paths=[blended_image_path], prompts=[prompt]
                        )

                if metric == "clip_score":
                    # metric_result = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_clip_similarity(
                            blended_image, prompt
                        )

                if metric == "pe_score":
                    # metric_result = metrics_calculator.calculate_pe_score(tgt_image, prompt)
                    if args.blended:
                        metric_result = metrics_calculator.calculate_pe_score(
                            blended_image, prompt
                        )

                # evaluation_result.append(metric_result)
                if args.blended:
                    metric_results.append(metric_result)

            # evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

        if args.blended:
            if metric == metrics[0]:
                blended_evaluation_df["Image ID"] = image_keys
            blended_evaluation_df[metric_name_dict[metric]] = metric_results

        del metrics_calculator
        torch.cuda.empty_cache()

    print("The averaged evaluation result:")
    # averaged_results=evaluation_df.mean(numeric_only=True)
    # averaged_results.to_csv(os.path.join(args.image_save_path,f"evaluation_result_sum_{rank}.csv"))
    # evaluation_df.to_csv(os.path.join(args.image_save_path,f"evaluation_result_{rank}.csv"))
    if args.blended:
        averaged_blended_results = blended_evaluation_df.mean(numeric_only=True)
        # averaged_blended_results.to_csv(os.path.join(args.image_save_path,f"aaa_evaluation_result_blended_sum_{rank}_hpsv3.csv"))
        blended_evaluation_df.to_csv(
            os.path.join(args.image_save_path, f"evaluation_result_blended_{rank}.csv")
        )

    if args.delete_images:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            for dirpath, dirnames, filenames in os.walk(args.image_save_path):
                for filename in filenames:
                    _, ext = os.path.splitext(filename)
                    if ext.lower() in image_extensions:
                        file_path = os.path.join(dirpath, filename)
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            print(f"cannot delete {file_path}: {e}")

    print(
        f"[{rank}] The generated images and evaluation results are saved in {args.image_save_path}"
    )

    accelerator.wait_for_everyone()
    try:
        if rank == 0:
            files = []
            for file in os.listdir(args.image_save_path):
                if args.blended:
                    if (
                        file.startswith("evaluation_result_")
                        and file.endswith(".csv")
                        and "blended" in file
                    ):
                        files.append(file)
                else:
                    if (
                        file.startswith("evaluation_result_")
                        and file.endswith(".csv")
                        and "blended" not in file
                    ):
                        files.append(file)
            df = pd.concat(
                [pd.read_csv(os.path.join(args.image_save_path, f), index_col=0) for f in files], ignore_index=True
            ).sort_values(by=["Image ID"])
            df.mean(numeric_only=True).to_frame().T.to_csv(
                os.path.join(args.image_save_path, f"avg_evaluation_result.csv")
            )
    except Exception as e:
        print(f"Error in aggregating results: {e}")
    destroy_process_group()
