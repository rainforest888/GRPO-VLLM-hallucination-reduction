"""
POPE evaluation script for Qwen3-VL-2B-Instruct
Runs inference on POPE questions (random, popular, adversarial) and saves answers.
Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    python pope_inference.py
"""
import json
import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# ---- Paths ----
MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
OUTPUT_DIR = r"G:\sample\Qwen3vl\router_project\pope_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Load model & processor ----
print("Loading Qwen3-VL model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
print("Model loaded.\n")


def answer_yes_no(output_text: str) -> str:
    """Heuristic to extract yes/no from model output."""
    text = output_text.strip().lower()
    # If the first sentence contains a period, keep only the first sentence
    if "." in text:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split()
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def run_pope_subset(subset_name: str):
    """Run inference on one POPE subset and save answers."""
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset_name}.json")
    output_file = os.path.join(OUTPUT_DIR, f"coco_pope_{subset_name}_answers.json")

    # Read questions (one JSON object per line)
    questions = [json.loads(line) for line in open(pope_file, "r", encoding="utf-8")]

    results = []
    for q in tqdm(questions, desc=f"POPE {subset_name}"):
        image_path = os.path.join(IMAGE_DIR, q["image"])
        text = q["text"]

        # Build chat messages following Qwen3-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text + " Please answer yes or no."},
                ],
            }
        ]

        # Apply chat template → tokenize, add image tokens
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=32)

        # Decode only the new tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Heuristic parse → yes/no
        final_answer = answer_yes_no(raw_output)

        results.append({
            "question": text,
            "answer": final_answer,
            "raw_output": raw_output,  # keep raw for debugging
        })

    # Save results (one JSON per line)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  -> Saved {len(results)} answers to {output_file}")


if __name__ == "__main__":
    for subset in ["random", "popular", "adversarial"]:
        run_pope_subset(subset)
    print("\nAll done!")
