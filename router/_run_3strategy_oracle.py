"""
_run_3strategy_oracle.py — Oracle: test 3 strategies individually on adversarial 10000,
find baseline errors, count how many each strategy fixes, measure combination upper bound.

Strategies (per-layer, LM 5-18):
  A = adaiat-u all layers (best single strategy)
  B = uac all layers
  C = random layer+strategy combo (per-question oracle)

Then per-baseline-error question: which strategies fix it?
  → A alone fixes X, B alone fixes Y, C alone fixes Z
  → Union(A,B,C) fixes U
  → Upper bound: how many could be fixed if oracle picks the best per question?

This directly answers: "Do different questions need different strategies?"

For C, we sample K random per-layer combos and take the best per question.
This gives a realistic upper bound without exhaustive 3^14 search.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/_run_3strategy_oracle.py --n_sample 200 --k_random 20 --sparse_k 3
"""
import json, os, sys, random, argparse
import torch, torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--n_sample", type=int, default=200, help="Number of questions to test")
ap.add_argument("--k_random", type=int, default=20, help="Random combos per question for oracle C")
ap.add_argument("--sparse_k", type=int, default=3, help="Active layers per random forward")
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
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"

# ─── Load ──────────────────────────────────────────────────────────
_tok = __import__('transformers').AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

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
STRAT_IDX = {s: i for i, s in enumerate(ALL_STRATS)}

# ─── Load questions ────────────────────────────────────────────────
adv_file = os.path.join(POPE_DIR, "coco_pope_adv_10000.json")
questions = [json.loads(l) for l in open(adv_file, encoding="utf-8")]
random.shuffle(questions)
questions = questions[:args.n_sample]
print(f"Testing {args.n_sample} questions")

# ─── Helper: single forward with fixed strategy ────────────────────
def forward_fixed(q, strategy, mgr):
    """Run one question with `strategy` forced on all LM 5-18 layers.
    Returns (answer, is_correct)."""
    mgr.clear_cache()
    # Set force_per_layer for all active layers
    idx = STRAT_IDX.get(strategy, 4)  # default "none"
    mgr._force_per_layer = {f"lm.{li}": idx for li in ALL_LAYERS}
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
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=4)
    raw = processor.decode(gen[0, inputs.input_ids.shape[1]:],
                           skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ans = answer_yes_no(raw)
    return ans, (ans == q["label"])

def forward_random(q, mgr, sparse_k):
    """Run one question with random per-layer strategy assignment.
    Returns (answer, is_correct, assignment_dict)."""
    mgr.clear_cache()
    picked = random.sample(ALL_LAYERS, min(sparse_k, len(ALL_LAYERS)))
    assignment = {}
    for li in picked:
        s = random.choice(ALL_STRATS)
        assignment[f"lm.{li}"] = STRAT_IDX[s]
    mgr._force_per_layer = dict(assignment)
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
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=4)
    raw = processor.decode(gen[0, inputs.input_ids.shape[1]:],
                           skip_special_tokens=True, clean_up_tokenization_spaces=False)
    ans = answer_yes_no(raw)
    return ans, (ans == q["label"]), assignment

# ─── Create RouterManager ──────────────────────────────────────────
# We need separate managers for "force all layers" (A, B) vs "force_per_layer" (C)
mgr_all = RouterManager(model, calib, active_layers=active_set, alpha_init=0.0)
mgr_all.wrap_all()
mgr_all.mode = "force_per_layer"

mgr_random = RouterManager(model, calib, active_layers=active_set, alpha_init=0.0)
# mgr_random will be wrapped on a different model... but we only have one model.
# Need to share the model but switch manager. Let's use the same manager.
# Actually we can just use one manager and change _force_per_layer each time.
# Just use mgr_all for everything — force_per_layer supports both cases.

print("\n=== Phase 1: Baseline (all none) ===")
baseline_results = []  # (answer, correct)
for q in tqdm(questions, desc="Baseline"):
    ans, correct = forward_fixed(q, "none", mgr_all)
    baseline_results.append((ans, correct))

baseline_correct = sum(1 for _, c in baseline_results if c)
baseline_wrong = sum(1 for _, c in baseline_results if not c)
print(f"Baseline: {baseline_correct}/{args.n_sample} = {baseline_correct/args.n_sample:.4f}")
print(f"Baseline wrong: {baseline_wrong}")

# Collect indices of baseline errors
error_indices = [i for i, (_, c) in enumerate(baseline_results) if not c]
print(f"Error indices (first 10): {error_indices[:10]}")
print(f"Total errors: {len(error_indices)}")

# ─── Phase 2: Test A (adaiat-u all layers) on errors ───────────────
print(f"\n=== Phase 2: Strategy A = adaiat-u (L5-18) ===")
A_fixed = {}  # error_idx → correct?
for idx in tqdm(error_indices, desc="A: adaiat-u"):
    ans, correct = forward_fixed(questions[idx], "adaiat", mgr_all)
    A_fixed[idx] = correct

A_fixed_count = sum(1 for v in A_fixed.values() if v)
print(f"A (adaiat-u) fixes: {A_fixed_count}/{len(error_indices)}")

# ─── Phase 3: Test B (uac all layers) on errors ────────────────────
print(f"\n=== Phase 3: Strategy B = uac (L5-18) ===")
B_fixed = {}
for idx in tqdm(error_indices, desc="B: uac"):
    ans, correct = forward_fixed(questions[idx], "uac", mgr_all)
    B_fixed[idx] = correct

B_fixed_count = sum(1 for v in B_fixed.values() if v)
print(f"B (uac) fixes: {B_fixed_count}/{len(error_indices)}")

# ─── Phase 4: Oracle C — random combos per error question ──────────
print(f"\n=== Phase 4: Strategy C = random per-layer combos (K={args.k_random} each, sparse_k={args.sparse_k}) ===")
C_results = {}  # error_idx → {"best_correct": bool, "any_correct": bool, "n_correct": int}
for idx in tqdm(error_indices, desc="C: random combos"):
    n_correct = 0
    for _ in range(args.k_random):
        ans, correct, assignment = forward_random(questions[idx], mgr_all, args.sparse_k)
        if correct:
            n_correct += 1
    C_results[idx] = {
        "any_correct": n_correct > 0,
        "n_correct": n_correct,
        "best_correct": n_correct > 0,
    }

C_any_count = sum(1 for v in C_results.values() if v["any_correct"])
C_total_correct = sum(v["n_correct"] for v in C_results.values())
print(f"C (random K={args.k_random}): any-correct={C_any_count}/{len(error_indices)}")
print(f"C total correct across all random samples: {C_total_correct}")

# ─── Phase 5: Analysis ─────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"COMBINATION ANALYSIS")
print(f"{'='*60}")
print(f"Baseline errors: {len(error_indices)}")
print(f"A (adaiat-u) fixes:  {A_fixed_count}")
print(f"B (uac) fixes:       {B_fixed_count}")
print(f"C (random combo) any: {C_any_count} (best of {args.k_random} random)")

# Union
fixed_by_A = {i for i, v in A_fixed.items() if v}
fixed_by_B = {i for i, v in B_fixed.items() if v}
fixed_by_C = {i for i, v in C_results.items() if v["any_correct"]}

union_all = fixed_by_A | fixed_by_B | fixed_by_C
union_AB = fixed_by_A | fixed_by_B
union_AC = fixed_by_A | fixed_by_C
union_BC = fixed_by_B | fixed_by_C

print(f"\nUnion A∪B:   {len(union_AB)}")
print(f"Union A∪C:   {len(union_AC)}")
print(f"Union B∪C:   {len(union_BC)}")
print(f"Union A∪B∪C: {len(union_all)}")

# Overlap analysis
only_A = fixed_by_A - fixed_by_B - fixed_by_C
only_B = fixed_by_B - fixed_by_A - fixed_by_C
only_C = fixed_by_C - fixed_by_A - fixed_by_B
AB = (fixed_by_A & fixed_by_B) - fixed_by_C
AC = (fixed_by_A & fixed_by_C) - fixed_by_B
BC = (fixed_by_B & fixed_by_C) - fixed_by_A
ABC = fixed_by_A & fixed_by_B & fixed_by_C

print(f"\nVenn breakdown of fixed errors:")
print(f"  Only A:     {len(only_A)}")
print(f"  Only B:     {len(only_B)}")
print(f"  Only C:     {len(only_C)}")
print(f"  A∩B only:   {len(AB)}")
print(f"  A∩C only:   {len(AC)}")
print(f"  B∩C only:   {len(BC)}")
print(f"  A∩B∩C:      {len(ABC)}")
print(f"  Sum:         {len(only_A)+len(only_B)+len(only_C)+len(AB)+len(AC)+len(BC)+len(ABC)}")
print(f"  Total union: {len(union_all)}")

# Key insight
print(f"\n{'='*60}")
print(f"BOTTOM LINE")
print(f"{'='*60}")
print(f"Baseline accuracy: {baseline_correct/args.n_sample:.4f}")
print(f"Best single strategy: A (adaiat-u) → {baseline_correct + A_fixed_count}/{args.n_sample} = {(baseline_correct + A_fixed_count)/args.n_sample:.4f}")
print(f"Oracle union upper bound:  {baseline_correct + len(union_all)}/{args.n_sample} = {(baseline_correct + len(union_all))/args.n_sample:.4f}")

if len(only_A) > 0 or len(only_B) > 0 or len(only_C) > 0:
    print(f"\n>>> DIFFERENT STRATEGIES FIX DIFFERENT QUESTIONS!")
    print(f"    {len(only_A)} questions ONLY fixed by adaiat-u, not others")
    print(f"    {len(only_B)} questions ONLY fixed by uac, not others")
    print(f"    {len(only_C)} questions ONLY fixed by random combo, not others")
    print(f"    This PROVES context-dependent routing is necessary.")
else:
    print("The strategies don't actually fix disjoint questions.")

print(f"\nConclusion: if union >> best single, per-layer routing has value.")
print(f"  Union A∪B∪C = {len(union_all)}, Best single = {max(A_fixed_count, B_fixed_count, C_any_count)}")
if len(union_all) > max(A_fixed_count, B_fixed_count, C_any_count):
    print(f"  → Yes! {len(union_all) - max(A_fixed_count, B_fixed_count, C_any_count)} additional fixes from combining strategies.")

# ─── Save ──────────────────────────────────────────────────────────
out_path = os.path.join(CHECKPOINT_DIR, "3strategy_oracle_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "config": {"n_sample": args.n_sample, "k_random": args.k_random, "sparse_k": args.sparse_k},
        "baseline": {"correct": baseline_correct, "wrong": baseline_wrong, "acc": baseline_correct / args.n_sample},
        "A_adaiat": {"fixed": A_fixed_count},
        "B_uac": {"fixed": B_fixed_count},
        "C_random": {"any_correct": C_any_count, "total_correct": C_total_correct},
        "union": {"AB": len(union_AB), "AC": len(union_AC), "BC": len(union_BC), "ABC": len(union_all)},
        "disjoint": {"only_A": len(only_A), "only_B": len(only_B), "only_C": len(only_C)},
    }, f, indent=2, ensure_ascii=False)

mgr_all.unwrap_all()
print(f"\nSaved: {out_path}")
print("DONE")
