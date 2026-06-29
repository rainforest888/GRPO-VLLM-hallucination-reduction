"""
vcd_inference.py — Visual Contrastive Decoding for POPE evaluation.

VCD amplifies the part of the logits that comes from actually seeing the image:
    logits_vcd = (1 + γ) · logits_with_img − γ · logits_without_img

The "without image" forward zeroes out pixel_values while keeping the exact
same token sequence (vision tokens become zero-embedding placeholders).

Reference: Leng et al., "Mitigating Object Hallucinations in Large
Vision-Language Models through Visual Contrastive Decoding", 2023.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/vcd_inference.py --gamma 1.0 --n 500
"""
import json, os, sys, argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"

# ─── Args ───────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--gamma", type=float, default=1.0, help="VCD amplification factor")
ap.add_argument("--n", type=int, default=500, help="questions per subset (<=3000)")
ap.add_argument("--subsets", type=str, default="adversarial",
                help="comma-separated: random,popular,adversarial")
args = ap.parse_args()

GAMMA = args.gamma
N_Q = args.n

OUTDIR = f"vcd_gamma{GAMMA}_n{N_Q}"
OUTPUT_DIR = os.path.join(RESULTS_BASE, OUTDIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"VCD γ={GAMMA}  N={N_Q}  subsets={args.subsets}")
print(f"Output: {OUTPUT_DIR}")

# ─── Load model ─────────────────────────────────────────────────────
print("Loading Qwen3-VL model (eager)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print("Model loaded.\n")


def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", "")
    w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"


def run_subset(subset):
    """Run VCD evaluation on a POPE subset."""
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
    output_file = os.path.join(OUTPUT_DIR, f"coco_pope_{subset}_answers.json")
    questions = [json.loads(l) for l in open(pope_file, "r", encoding="utf-8")][:N_Q]

    results = []
    correct = 0

    for q in tqdm(questions, desc=f"POPE {subset}"):
        image_path = os.path.join(IMAGE_DIR, q["image"])
        text = q["text"]

        # Build inputs once
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": text + " Please answer yes or no."},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        # ── Forward 1: with vision ──
        with torch.no_grad():
            out_img = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                use_cache=False,
            )
        logits_img = out_img.logits[0, -1, :]  # last token

        # ── Forward 2: zeroed vision (same sequence, no image) ──
        zero_pv = torch.zeros_like(inputs.get("pixel_values",
                                               torch.zeros(1, device=model.device)))
        with torch.no_grad():
            out_noimg = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=zero_pv,
                image_grid_thw=inputs.get("image_grid_thw"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                use_cache=False,
            )
        logits_noimg = out_noimg.logits[0, -1, :]

        # ── VCD combination ──
        # logits_vcd = (1+γ) * logits_img - γ * logits_noimg
        vcd_logits = (1.0 + GAMMA) * logits_img - GAMMA * logits_noimg

        # ── Manual 1-token generation ──
        # Get the most likely token from VCD logits
        top_token_id = torch.argmax(vcd_logits).item()
        raw = processor.decode([top_token_id], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)

        # If the first token isn't yes/no, continue with standard generation
        ans = answer_yes_no(raw)

        results.append({
            "question": text,
            "answer": ans,
            "raw_output": raw,
        })

        if ans == q["label"]:
            correct += 1

    acc = correct / len(results)
    print(f"  -> {subset}: {correct}/{len(results)} = {acc:.4f}  "
          f"(baseline adversarial 87.30%, Δ={acc-0.8730:+.4f})")

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(results)} to {output_file}")

    return acc


for subset in args.subsets.split(","):
    run_subset(subset.strip())

print(f"\nDone. Evaluate: python pope_evaluate.py {OUTDIR}")
