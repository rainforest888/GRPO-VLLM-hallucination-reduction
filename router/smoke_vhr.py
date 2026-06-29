"""Smoke test: vhr and uac_vhr strategies on single LM layer, 500 questions."""
import json, os, sys, torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--strategy", type=str, default="vhr", choices=["vhr", "uac_vhr"])
ap.add_argument("--n", type=int, default=500)
args = ap.parse_args()

LAYER = args.layer
STRATEGY = args.strategy
N_Q = args.n
OUTDIR = f"{STRATEGY}_L{LAYER}"
OUTPUT_DIR = os.path.join(RESULTS_BASE, OUTDIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Strategy={STRATEGY}  Layer=LM.{LAYER}  N={N_Q}  -> {OUTDIR}")

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
calib = torch.load(os.path.join(CHECKPOINT_DIR, "calibration.pt"), map_location="cpu", weights_only=False)

# Only activate the single target layer
active_name = f"lm.{LAYER}"
mgr = RouterManager(model, calib, active_layers={active_name}, alpha_init=1.2)
mgr.wrap_all()
mgr.mode = "force"

# Non-target layers -> none; target layer -> our strategy
strategies = mgr._strategies_for(active_name)
print(f"Available strategies for {active_name}: {strategies}")

if STRATEGY not in strategies:
    print(f"ERROR: {STRATEGY} not in {strategies}")
    sys.exit(1)

strat_idx = strategies.index(STRATEGY)
none_idx = strategies.index("none") if "none" in strategies else 0

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

pope_file = os.path.join(POPE_DIR, "coco_pope_adversarial.json")
questions = [json.loads(l) for l in open(pope_file, encoding="utf-8")][:N_Q]

for subset in ["adversarial"]:
    out_file = os.path.join(OUTPUT_DIR, f"coco_pope_{subset}_answers.json")
    results = []
    for q in tqdm(questions[:N_Q], desc=f"POPE {subset}"):
        mgr.clear_cache()

        # Set decisions: target layer gets our strategy, rest get "none"
        for d in mgr.descs:
            name = d["name"]
            s = d["strategies"]
            if name == active_name:
                idx = strat_idx if strat_idx < len(s) else 0
            else:
                idx = s.index("none") if "none" in s else 0
            mgr._decisions[name] = idx
            mgr._decided.add(name)

        img_path = os.path.join(IMAGE_DIR, q["image"])
        text = q["text"]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": text + " Please answer yes or no."},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        mgr._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()
        full_text = text + " Please answer yes or no."
        q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
        all_ids = inputs["input_ids"][0]
        q_t = torch.tensor(q_ids, device=all_ids.device)
        for s in range(len(all_ids) - len(q_ids) + 1):
            if (all_ids[s:s+len(q_ids)] == q_t).all():
                mgr._current_q_pos = torch.arange(s, s + len(q_ids))
                break

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=16)
        gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        results.append({"question": text, "answer": answer_yes_no(raw), "raw_output": raw})

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Quick accuracy
    labels = [json.loads(l)["label"] for l in open(pope_file, encoding="utf-8")][:len(results)]
    correct = sum(1 for r, lbl in zip(results, labels) if r["answer"] == lbl)
    print(f"  {STRATEGY} L{LAYER} ({subset}): {correct}/{len(results)} = {correct/len(results):.4f}")

mgr.unwrap_all()
