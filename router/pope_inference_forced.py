"""
pope_inference_forced.py — POPE evaluation with a SINGLE forced strategy
across all layers (ablation: None / UAC / AdaIAT individually).

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/pope_inference_forced.py <strategy> [output_dirname]
      strategy: none | uac | adaiat
      output_dirname: defaults to <strategy>_only
"""
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

# ─── Args ───────────────────────────────────────────────────────────
strategy = sys.argv[1] if len(sys.argv) > 1 else "none"
out_name = sys.argv[2] if len(sys.argv) > 2 else f"{strategy}_only"
OUTPUT_DIR = os.path.join(RESULTS_BASE, out_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Forced strategy: {strategy}")
print(f"Output dir: {OUTPUT_DIR}")

# ─── Load model ─────────────────────────────────────────────────────
print("Loading Qwen3-VL model (eager)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
print("Model loaded.\n")

# ─── Load calibration ───────────────────────────────────────────────
calib_path = os.path.join(CHECKPOINT_DIR, "calibration.pt")
if not os.path.exists(calib_path):
    print("ERROR: calibration.pt not found. Run calibration.py first.")
    sys.exit(1)
calib = torch.load(calib_path, map_location="cpu", weights_only=False)

# ─── RouterManager in force mode ────────────────────────────────────
mgr = RouterManager(model, calib)
mgr.mode = "force"
mgr.force_strategy = strategy
mgr.wrap_all()
print(f"Wrapped. Force strategy = {strategy}")
print("Alphas (fixed 0.5):", {k: round(mgr.get_alpha(k).item(), 4)
                              for k in sorted(mgr.raw_alphas.keys())})


def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", "")
    w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"


def run_subset(subset):
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
    output_file = os.path.join(OUTPUT_DIR, f"coco_pope_{subset}_answers.json")
    questions = [json.loads(l) for l in open(pope_file, "r", encoding="utf-8")]

    results = []
    for q in tqdm(questions, desc=f"POPE {subset}"):
        image_path = os.path.join(IMAGE_DIR, q["image"])
        text = q["text"]
        mgr.clear_cache()

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": text + " Please answer yes or no."},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=32)
        gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        results.append({"question": text, "answer": answer_yes_no(raw), "raw_output": raw})

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(results)} to {output_file}")


for subset in ["random", "popular", "adversarial"]:
    run_subset(subset)

mgr.unwrap_all()
print(f"\nDone. Next: python pope_evaluate.py {out_name}")
