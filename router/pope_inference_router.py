"""
pope_inference_router.py — POPE evaluation with trained Router (argmax mode).

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/pope_inference_router.py [checkpoint_path]
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
OUTPUT_DIR = r"G:\sample\Qwen3vl\router_project\pope_results\router_v1"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load model ─────────────────────────────────────────────────────
print("Loading Qwen3-VL model (eager attention)...")
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

# ─── Router weights ─────────────────────────────────────────────────
checkpoint = sys.argv[1] if len(sys.argv) > 1 else os.path.join(CHECKPOINT_DIR, "router_weights_final.pt")
if not os.path.exists(checkpoint):
    print(f"ERROR: Router checkpoint not found: {checkpoint}")
    for f in sorted(os.listdir(CHECKPOINT_DIR)):
        print(f"  {f}")
    sys.exit(1)
print(f"Loading router weights from: {checkpoint}")
router_state = torch.load(checkpoint, map_location="cpu", weights_only=False)

# ─── RouterManager (inference) ──────────────────────────────────────
# Must match training's active_layers (5-18 per paper 2411.16724v3)
ACTIVE_LM_LAYERS = list(range(5, 19))
active_layers = {f"lm.{i}" for i in ACTIVE_LM_LAYERS}
manager = RouterManager(model, calib, active_layers=active_layers)
manager.load_state_dict(router_state)
manager.to("cuda")
manager.eval()
manager.mode = "argmax"
manager.wrap_all()
print(f"Router loaded. {manager.num_routers} routers, 8 alpha blocks.")
print("Alphas:", {k: round(manager.get_alpha(k).item(), 4) for k in sorted(manager.raw_alphas.keys())})


def answer_yes_no(output_text: str) -> str:
    text = output_text.strip().lower()
    if "." in text:
        text = text.split(".")[0]
    text = text.replace(",", "")
    words = text.split()
    if "no" in words or "not" in words:
        return "no"
    return "yes"


def run_pope_subset(subset_name: str):
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset_name}.json")
    output_file = os.path.join(OUTPUT_DIR, f"coco_pope_{subset_name}_answers.json")
    questions = [json.loads(line) for line in open(pope_file, "r", encoding="utf-8")]

    strategy_counts = {"uac": 0, "adaiat": 0, "none": 0}
    results = []

    for q in tqdm(questions, desc=f"POPE {subset_name}"):
        image_path = os.path.join(IMAGE_DIR, q["image"])
        text = q["text"]

        manager.clear_cache()

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": text + " Please answer yes or no."},
        ]}]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Required for UAC (dict W lookup) and AdaIAT-U (question positions)
        manager._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()
        full_text = text + " Please answer yes or no."
        q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
        all_ids = inputs["input_ids"][0]
        q_t = torch.tensor(q_ids, device=all_ids.device)
        q_pos = torch.arange(0, 1)
        for s in range(len(all_ids) - len(q_ids) + 1):
            if (all_ids[s:s + len(q_ids)] == q_t).all():
                q_pos = torch.arange(s, s + len(q_ids))
                break
        manager._current_q_pos = q_pos

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=32)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        final_answer = answer_yes_no(raw_output)

        # Track strategy usage
        for name, idx in manager._decisions.items():
            for d in manager.descs:
                if d["name"] == name:
                    s = d["strategies"]
                    strategy_counts[s[min(idx, len(s) - 1)]] += 1
                    break

        results.append({
            "question": text,
            "answer": final_answer,
            "raw_output": raw_output,
        })

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  -> Saved {len(results)} answers to {output_file}")
    total = sum(strategy_counts.values())
    if total > 0:
        print(f"  Strategy usage: " + ", ".join(
            f"{k}: {v/total:.2%}" for k, v in sorted(strategy_counts.items())
        ))

if __name__ == "__main__":
    for subset in ["random", "popular", "adversarial"]:
        run_pope_subset(subset)

    manager.unwrap_all()
    print("Done!")
