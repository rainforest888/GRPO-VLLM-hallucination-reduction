"""
oracle_test.py — 3×3×3 grid search over shallow/middle/deep layer groups.

Groups: shallow(LM 5-9) × middle(LM 10-14) × deep(LM 15-18)
Each group: UAC / AdaIAT / None → 27 combinations total.
Tests on Adversarial POPE subset (3000 questions per combo).

Saves results to pope_results/oracle/.
"""
import json, os, sys, itertools
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"

GROUPS = {
    'shallow': list(range(5, 10)),
    'middle': list(range(10, 15)),
    'deep': list(range(15, 19)),
}

STRATEGIES = ['uac', 'adaiat', 'none']
SUBSET = 'adversarial'
ALL_LAYERS = sorted(set(sum((list(g) for g in GROUPS.values()), [])))

# CLI: --n 500 (fast screen) or --n 3000 (full)
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000, help="questions per combo (500 fast, 3000 full)")
ap.add_argument("--topk", type=int, default=0, help="if >0: only run top-K combos from a prior fast run (reads oracle_summary.json)")
args = ap.parse_args()
N_QUESTIONS = args.n
TOPK = args.topk

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
calib = torch.load(os.path.join(CHECKPOINT_DIR, "calibration.pt"), map_location="cpu", weights_only=False)

# Build RouterManager with ALL layers as active (we'll use force mode per forward)
active_names = {f"lm.{i}" for i in ALL_LAYERS}
mgr = RouterManager(model, calib, active_layers=active_names, alpha_init=1.2)
mgr.wrap_all()
mgr.mode = "force"

pope_file = os.path.join(POPE_DIR, f"coco_pope_{SUBSET}.json")
questions = [json.loads(l) for l in open(pope_file, "r", encoding="utf-8")]

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

combos = list(itertools.product(STRATEGIES, repeat=3))
fname = lambda s,m,d: f"sh_{s}_mid_{m}_dp_{d}"

# ── Handle --topk: filter to only the top-K combos from a prior fast (500) run ──
if TOPK > 0:
    summary_path = os.path.join(RESULTS_BASE, "oracle", "oracle_summary.json")
    if os.path.exists(summary_path):
        prior = json.load(open(summary_path, encoding="utf-8"))
        # prior keys look like "sh_uac_mid_none_dp_adaiat"
        prior_sorted = sorted(prior.items(), key=lambda x: -x[1])
        top_names = {name for name, _ in prior_sorted[:TOPK]}
        combos = [c for c in combos if fname(*c) in top_names]
        print(f"--topk {TOPK}: filtered to {len(combos)} combos: {[fname(*c) for c in combos]}")
    else:
        print(f"WARNING: --topk {TOPK} but no oracle_summary.json found, running all 27 combos")

questions = [json.loads(l) for l in open(pope_file, "r", encoding="utf-8")]
if N_QUESTIONS < len(questions):
    questions = questions[:N_QUESTIONS]

all_results = {}
for s_shallow, s_middle, s_deep in tqdm(combos, desc="Oracle combos"):
    name = fname(s_shallow, s_middle, s_deep)
    out_dir = os.path.join(RESULTS_BASE, "oracle", name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"coco_pope_{SUBSET}_answers.json")

    # Build decision map
    dmap = {}
    for l in GROUPS['shallow']:
        dmap[f"lm.{l}"] = s_shallow
    for l in GROUPS['middle']:
        dmap[f"lm.{l}"] = s_middle
    for l in GROUPS['deep']:
        dmap[f"lm.{l}"] = s_deep

    # Skip if already done (resume support)
    if os.path.exists(out_file):
        existing = [json.loads(l) for l in open(out_file, encoding='utf-8')]
        if len(existing) >= N_QUESTIONS:
            print(f"  {name}: already exists ({len(existing)} answers), skipping")
            c = sum(1 for e, q in zip(existing, questions[:len(existing)]) if e['answer'] == q['label'])
            all_results[name] = c / max(len(existing), 1)
            continue
        else:
            # Partial: resume from where left off
            results = existing[:]
            correct = sum(1 for r in results if any(
                r['question'] == q['text'] and r['answer'] == q['label']
                for q in questions[:len(results)]))
            total = len(results)
            start_idx = total
    else:
        results = []
        correct = 0
        total = 0
        start_idx = 0

    for i in tqdm(range(start_idx, len(questions)), desc=name, leave=False):
        q = questions[i]
        img_path = os.path.join(IMAGE_DIR, q["image"])
        text = q["text"]
        mgr.clear_cache()

        # Set force strategy per layer (must do BEFORE _decide is called)
        for layer_name in dmap:
            strat = dmap[layer_name]
            strategies_for_layer = mgr._strategies_for(layer_name)
            if strat in strategies_for_layer:
                idx = strategies_for_layer.index(strat)
            elif "none" in strategies_for_layer:
                idx = strategies_for_layer.index("none")
            else:
                idx = 0
            mgr._decisions[layer_name] = idx
            mgr._decided.add(layer_name)

        # Non-group layers → none
        for d in mgr.descs:
            if d["name"] not in dmap and d["name"] not in mgr._decided:
                s = d["strategies"]
                mgr._decisions[d["name"]] = s.index("none") if "none" in s else 0
                mgr._decided.add(d["name"])

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": text + " Please answer yes or no."},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        # Required for UAC (dict W lookup) and AdaIAT-U (question pos)
        mgr._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()
        full_text = text + " Please answer yes or no."
        q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
        all_ids = inputs["input_ids"][0]
        q_t = torch.tensor(q_ids, device=all_ids.device)
        q_pos = torch.arange(0, 1)  # default fallback
        for s in range(len(all_ids) - len(q_ids) + 1):
            if (all_ids[s:s+len(q_ids)] == q_t).all():
                q_pos = torch.arange(s, s + len(q_ids))
                break
        mgr._current_q_pos = q_pos

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=16)
        gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        ans = answer_yes_no(raw)
        results.append({"question": text, "answer": ans, "raw_output": raw})
        if ans == q["label"]: correct += 1
        total += 1

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = correct / max(total, 1)
    all_results[name] = acc
    print(f"  {name}: Acc={acc:.4f} ({correct}/{total})")

# Sorted results
print("\n===== ORACLE RESULTS (Adversarial) =====")
sorted_results = sorted(all_results.items(), key=lambda x: -x[1])
for name, acc in sorted_results:
    sham = name.split('_')[1]
    smid = name.split('_')[3]
    sd = name.split('_')[5]
    print(f"  {acc:.4f}  shallow={sham:8} middle={smid:8} deep={sd:8}")

# Save summary
summary_path = os.path.join(RESULTS_BASE, "oracle", "oracle_summary.json")
with open(summary_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSummary saved to {summary_path}")

mgr.unwrap_all()
