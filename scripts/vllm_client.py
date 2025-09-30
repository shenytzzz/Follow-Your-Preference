"""Parallel client utilities for querying a UnifiedReward vLLM server.

`gen_results` spawns processes to fetch model scores for (image, prompt) pairs
and returns the first completed result (designed for interactive probing).
"""

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
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    filename=f"/root/flux/runs/score/unified_reward/logs_2/log_server.txt",
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)


class VLMessageClient:
    """Minimal wrapper around vLLM chat/completions with image encoding."""
    def __init__(self, api_url):
        self.api_url = api_url
        self.session = requests.Session()

    def _encode_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_prompt(self, prompt):
        # return (
        #             "You are presented with a generated image and its associated text caption. Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n\n"
        #             "1. Evaluate each word in the caption based on how well it is visually represented in the image. Assign a numerical score to each word using the format:\n"
        #             "   Word-wise Scores: [[\"word1\", score1], [\"word2\", score2], ..., [\"wordN\", scoreN], [\"[No_mistakes]\", scoreM]]\n"
        #             "   - A higher score indicates that the word is less well represented in the image.\n"
        #             "   - The special token [No_mistakes] represents whether all elements in the caption were correctly depicted. A high score suggests no mistakes; a low score suggests missing or incorrect elements.\n\n"
        #             "2. Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
        #             "- Alignment Score: How well the image matches the caption in terms of content.\n"
        #             "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
        #             "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
        #             "Output your evaluation using the format below:\n\n"
        #             "---\n\n"
        #             "Word-wise Scores: [[\"word1\", score1], ..., [\"[No_mistakes]\", scoreM]]\n\n"
        #             "Alignment Score (1-5): X\n"
        #             "Coherence Score (1-5): Y\n"
        #             "Style Score (1-5): Z\n\n"
        #             f"Your task is provided as follows:\nText Caption: [{prompt}]"
        #         )
        # return f'You are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nExtract key elements from the provided text caption, evaluate their presence in the generated image using the format: \'element (type): value\' (where value=0 means not generated, and value=1 means generated), and assign a score from 1 to 5 after \'Final Score:\'.\nYour task is provided as follows:\nText Caption: [{prompt}]'
        return f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"  # taken from flowgrpo

    def build_messages(self, item):
        content = []
        for i, img in enumerate(item["images"]):
            img_path = os.path.join(img)
            base64_image = self._encode_image(img_path)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        content.append({"type": "text", "text": self._build_prompt(item["prompt"])})
        return [{"role": "user", "content": content}]

    def infer(self, item):
        messages = self.build_messages(item)
        payload = {
            "model": "UnifiedReward",
            "messages": messages,
            "temperature": 0,
            "max_tokens": 4096,
        }
        response = self.session.post(
            f"{self.api_url}/v1/chat/completions", json=payload, timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def process_item(item, output_file, api_url, max_retries=5):
    client = VLMessageClient(api_url)
    attempt = 0
    while attempt < max_retries:
        try:
            attempt += 1
            output = client.infer(item)
            try:
                parts = re.split(r"(?i)final score[:\s]*", output)
                last_part = parts[-1].strip()

                match = re.search(r"([\d\.]+)", last_part)
                if match:
                    final_score = float(match.group(1))
                else:
                    logging.error(
                        f"No numeric score found in the last part: {last_part}"
                    )
                    final_score = -1.0

            except Exception as e:
                error_message = (
                    f"Error parsing final score from output: {output}. Error: {e}"
                )
                logging.error(error_message)
                final_score = -1.0

            if final_score < -0.5:
                result = {
                    "question": item["problem"],
                    "image_path": item["images"],
                    "image_id": f"{item['images'][0].split('/')[-2]}_{item['images'][0].split('/')[-1].split('.')[0]}",
                    "model_output": output,
                    "final_score": final_score,
                    "success": False,
                }
            else:
                result = {
                    "question": item["problem"],
                    "image_path": item["images"],
                    "image_id": f"{item['images'][0].split('/')[-2]}_{item['images'][0].split('/')[-1].split('.')[0]}",
                    "model_output": output,
                    "final_score": final_score,
                    "success": True,
                }
            return result, True
        except Exception as e:
            if attempt == max_retries:
                result = {
                    "question": item["problem"],
                    "image_path": item["images"],
                    "image_id": f"{item['images'][0].split('/')[-2]}_{item['images'][0].split('/')[-1].split('.')[0]}",
                    "error": str(e),
                    "attempt": attempt,
                    "success": False,
                }
                return result, False
            else:
                sleep_time = min(2**attempt, 10)
                sleep_time = sleep_time / 100
                logging.warning(
                    f"Retrying (attempt {attempt}/{max_retries}) after {sleep_time}s due to error: {e}"
                )
                time.sleep(sleep_time)


def gen_results(
    image_paths: list[str], prompts: list[str], output_path, api, max_workers
):

    task = partial(process_item, output_file=output_path, api_url=api)

    messages = [
        {
            "prompt": f"{prompt}",
            "images": [f"{img_path}"],
            "problem": "",
        }
        for img_path, prompt in zip(image_paths, prompts)
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, item) for item in messages]

        with open(output_path, "a") as fout:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result, success = future.result()
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    return result, success

                except Exception as e:
                    logging.error(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://127.0.0.1:8080")
    parser.add_argument("--prompt_path", default="XXXX.json")
    parser.add_argument("--image_root", default="/root/flux/examples/flux/src")
    parser.add_argument("--output_path", default="/root/vllm/results.json")
    parser.add_argument(
        "--max_workers", type=int, default=8, help="Number of worker processes"
    )
    args = parser.parse_args()

    # with open(args.prompt_path, "r") as f:
    #     test_data = json.load(f)

    open(args.output_path, "w").close()

    test_data = [
        {
            "prompt": "a red pepper on a white background",
            "images": ["test_image.jpg"],
            "problem": "",
        },
    ]
    task = partial(process_item, output_file=args.output_path, api_url=args.api_url)

    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [executor.submit(task, item) for item in test_data]

        with tqdm(total=len(test_data), desc="inferencing...") as pbar, open(
            args.output_path, "a"
        ) as fout:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result, success = future.result()
                    print(result)
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    if success:
                        success_count += 1
                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix(processed=f"{success_count}/{len(test_data)}")

    print(f"\nStatistics:")
    print(f"Total data: {len(test_data)}")
    print(f"Success ratio: {success_count} ({success_count / len(test_data):.2%})")


if __name__ == "__main__":
    main()
