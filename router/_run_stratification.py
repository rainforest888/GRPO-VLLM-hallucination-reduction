"""
_run_stratification.py — Compare single-layer vs multi-layer random strategy
to prove per-layer diversity is necessary.

Test on baseline errors:
  1. Single-layer random: pick ONE layer, random strategy → can it fix the error?
  2. Multi-layer random: pick 3 DIFFERENT layers, random strategies each → can it fix?
  3. Multi-layer SAME: pick 3 layers, all SAME random strategy → can it fix?

If multi-layer DIFFERENT > multi-layer SAME > single-layer:
  → proves BOTH "more layers better" AND "different layers need different strategies"

If multi-layer DIFFERENT == multi-layer SAME > single-layer:
  → only proves "more layers better", NOT "different strategies per layer"

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/_run_stratification.py --n_sample 500 --k_samples 40
"""
import json, os, sys, random, argparse, torch
from collections import defaultdict
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--n_sample", type=int, default=500)
ap.add_argument("--k_samples", type=int, default=40)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed); torch.manual_seed(args.seed)
os.environ['TQDM_DISABLE'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
for p in model.parameters(): p.requires_grad = False
model.eval()
print("Model loaded.")

calib = torch.load(os.path.join(CHECKPOINT_DIR, "calibration.pt"), map_location="cpu", weights_only=False)
active_set = {f"lm.{i}" for i in range(5, 19)}
ALL_LAYERS = list(range(5, 19))
ALL_STRATS = ["uac", "adaiat", "vhr", "uac_vhr", "none"]

# Exclude "none" from random strategies — we want actual interventions
ACTIVE_STRATS = ["uac", "adaiat", "vhr", "uac_vhr"]

mgr = RouterManager(model, calib, active_layers=active_set, alpha_init=0.0)
mgr.wrap_all()
mgr.mode = "force_per_layer"

adv_file = os.path.join(POPE_DIR, "coco_pope_adv_10000.json")
questions = [json.loads(l) for l in open(adv_file, encoding="utf-8")]
random.shuffle(questions)
questions = questions[:args.n_sample]

# ─── Phase 1: Find baseline errors ────────────────────────────────
print("\n=== Phase 1: Baseline ===")
error_indices = []
error_data = []
for i, q in enumerate(tqdm(questions, desc="Baseline")):
    mgr.clear_cache()
    mgr._force_per_layer = {}
    img_path = os.path.join(IMAGE_DIR, q["image"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": q["text"] + " Please answer yes or no."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    mgr._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=4)
    raw = processor.decode(gen[0, inputs.input_ids.shape[1]:],
                           skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ans = answer_yes_no(raw)
    if ans != q["label"]:
        error_indices.append(i)
        error_data.append({"idx": i, "q": q, "inputs": inputs})

baseline_acc = 1 - len(error_indices) / args.n_sample
print(f"Baseline: {args.n_sample - len(error_indices)}/{args.n_sample} = {baseline_acc:.4f}")
print(f"Errors to analyze: {len(error_indices)}")

if len(error_indices) == 0:
    print("No errors found. Need more samples.")
    mgr.unwrap_all()
    sys.exit(0)

# ─── Phase 2: Compare 3 conditions on errors ───────────────────────
conditions = [
    ("single_layer", 1, True),    # 1 layer, 1 strategy (no difference possible)
    ("multi_same", 3, True),      # 3 layers, same strategy
    ("multi_different", 3, False), # 3 layers, different strategies each
]

K = args.k_samples

# For each error, for each condition, run K samples
results = {c[0]: [] for c in conditions}  # condition → [n_correct per error]

for ei, ed in enumerate(tqdm(error_data, desc="Error analysis")):
    inputs = ed["inputs"]
    q = ed["q"]

    for cond_name, n_layers, force_same in conditions:
        n_correct = 0
        for _ in range(K):
            mgr.clear_cache()
            picked = random.sample(ALL_LAYERS, min(n_layers, len(ALL_LAYERS)))
            if force_same:
                strat = random.choice(ACTIVE_STRATS)
                assignment = {f"lm.{li}": ALL_STRATS.index(strat) for li in picked}
            else:
                assignment = {}
                for li in picked:
                    s = random.choice(ACTIVE_STRATS)
                    assignment[f"lm.{li}"] = ALL_STRATS.index(s)

            mgr._force_per_layer = assignment
            mgr._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()

            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=4)
            raw = processor.decode(gen[0, inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ans = answer_yes_no(raw)
            if ans == q["label"]:
                n_correct += 1

        results[cond_name].append(n_correct)

    torch.cuda.empty_cache()

mgr.unwrap_all()

# ─── Analysis ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"STRATIFICATION ANALYSIS")
print(f"{'='*60}")
print(f"Baseline errors: {len(error_indices)}")
print(f"K samples per error per condition: {K}")

for cond_name, n_layers, force_same in conditions:
    arr = results[cond_name]
    n_any = sum(1 for v in arr if v > 0)   # at least 1 in K fixed it
    n_many = sum(1 for v in arr if v >= 3)  # ≥3 of K fixed it
    mean_hits = sum(arr) / len(arr)  # average hits per error
    print(f"\n  {cond_name} ({n_layers} layer{'s' if n_layers>1 else ''}, "
          f"{'same' if force_same else 'different'} strategy):")
    print(f"    Any fix (≥1/K):   {n_any}/{len(error_indices)} ({n_any/len(error_indices):.1%})")
    print(f"    Frequent fix (≥3/K): {n_many}/{len(error_indices)} ({n_many/len(error_indices):.1%})")
    print(f"    Mean hits/error:  {mean_hits:.2f} of {K} ({mean_hits/K:.1%})")

# ─── Statistical test: multi_same vs multi_different ──────────────
import numpy as np
same_arr = np.array(results["multi_same"])
diff_arr = np.array(results["multi_different"])
single_arr = np.array(results["single_layer"])

print(f"\n--- Statistical comparison ---")

# Paired: for each error, is multi_different > multi_same?
diff_vs_same = diff_arr - same_arr
n_diff_better = sum(1 for v in diff_vs_same if v > 0)
n_same_better = sum(1 for v in diff_vs_same if v < 0)
n_tie = sum(1 for v in diff_vs_same if v == 0)
mean_diff = diff_vs_same.mean()
# Paired t-test
se_diff = diff_vs_same.std() / np.sqrt(len(diff_vs_same)) if len(diff_vs_same) > 1 else 1.0
t_diff = mean_diff / se_diff if se_diff > 0 else 0
print(f"\n  multi_different vs multi_same:")
print(f"    Different better: {n_diff_better}, Same better: {n_same_better}, Tie: {n_tie}")
print(f"    Mean diff: {mean_diff:.3f} hits/error")
print(f"    t-stat: {t_diff:.2f}")

# Multi vs single
multi_vs_single = same_arr - single_arr
mean_ms = multi_vs_single.mean()
se_ms = multi_vs_single.std() / np.sqrt(len(multi_vs_single))
t_ms = mean_ms / se_ms if se_ms > 0 else 0
print(f"\n  multi_same vs single_layer:")
print(f"    Mean diff: {mean_ms:.3f} hits/error")
print(f"    t-stat: {t_ms:.2f}")

diff_vs_single = diff_arr - single_arr
mean_ds = diff_vs_single.mean()
se_ds = diff_vs_single.std() / np.sqrt(len(diff_vs_single))
t_ds = mean_ds / se_ds if se_ds > 0 else 0
print(f"\n  multi_different vs single_layer:")
print(f"    Mean diff: {mean_ds:.3f} hits/error")
print(f"    t-stat: {t_ds:.2f}")

# ─── The verdict ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"VERDICT")
print(f"{'='*60}")

max_any_single = sum(1 for v in single_arr if v > 0)
max_any_same = sum(1 for v in same_arr if v > 0)
max_any_diff = sum(1 for v in diff_arr if v > 0)

print(f"Single-layer (random):     {max_any_single}/{len(error_indices)} errors fixable")
print(f"Multi-same (random):       {max_any_same}/{len(error_indices)} errors fixable")
print(f"Multi-different (random):  {max_any_diff}/{len(error_indices)} errors fixable")

if n_diff_better > n_same_better and t_diff > 2:
    print(f"\n>>> per-layer diversity MATTERS: different strats > same strat")
    print(f"    ({n_diff_better} errors benefit from diverse strategies)")
elif t_diff < 1:
    print(f"\n>>> per-layer diversity DOES NOT matter: diff strat == same strat")
    print(f"    Additional layers help (multi > single), but strategy diversity does not add value.")
else:
    print(f"\n>>> Inconclusive: need more samples or K")

if t_ms > 2:
    print(f"    Multiple layers ARE better than single layer (important)")
else:
    print(f"    Multiple layers are NOT better than single layer.")

print(f"\nInterpretation:")
if n_diff_better > n_same_better + 10 and max_any_diff > max_any_same + 5:
    print(f"  Strong evidence for per-layer strategy diversity.")
    print(f"  → RL router with per-layer independent decisions IS justified.")
elif n_diff_better > n_same_better and t_diff > 1.5:
    print(f"  Weak evidence for per-layer strategy diversity.")
    print(f"  → RL router MAY help, but effect is small.")
else:
    print(f"  No evidence for per-layer strategy diversity.")
    print(f"  → Random multi-layer already captures most of the oracle gain.")
    print(f"  → Per-layer independent decisions may not be worth the complexity.")

print(f"\nDONE")
