"""
_verify_signal.py — Counterfactual per-layer per-strategy signal verification.

For each question, run K random forwards (random layers + random strategies),
then regress reward against (layer, strategy) indicators to estimate
per-layer per-strategy counterfactual effects β.

If β coefficients are indistinguishable from noise, per-layer routing
cannot be learned — the candidate strategies simply don't have
differential effects at single-question per-layer resolution.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/_verify_signal.py --n_questions 20 --k_samples 50 --sparse_k 3
"""
import json, os, sys, random, argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--n_questions", type=int, default=20)
ap.add_argument("--k_samples", type=int, default=50, help="random forwards per question")
ap.add_argument("--sparse_k", type=int, default=3, help="active layers per forward")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ['TQDM_DISABLE'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

ACTIVE_LAYERS = list(range(5, 19))  # LM 5-18
ALL_STRATEGIES = ["uac", "adaiat", "vhr", "uac_vhr", "none"]

# ─── Token IDs ─────────────────────────────────────────────────────
_tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
YES_IDS = torch.tensor(sorted(set(
    _tok(s, add_special_tokens=False).input_ids[0]
    for s in ["yes", "Yes", " yes", " Yes"]
))).cuda()
NO_IDS = torch.tensor(sorted(set(
    _tok(s, add_special_tokens=False).input_ids[0]
    for s in ["no", "No", " no", " No"]
))).cuda()

# ─── Load model (frozen) ──────────────────────────────────────────
print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
for p in model.parameters():
    p.requires_grad = False
model.eval()

calib = torch.load(os.path.join(CHECKPOINT_DIR, "calibration.pt"),
                   map_location="cpu", weights_only=False)

active_set = {f"lm.{i}" for i in ACTIVE_LAYERS}
mgr = RouterManager(model, calib, active_layers=active_set, alpha_init=0.0)
mgr.wrap_all()
mgr.mode = "force_per_layer"

# ─── Helpers ──────────────────────────────────────────────────────
def make_inputs(image_path, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question + " Please answer yes or no."},
    ]}]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)


def reward_fn(last_logit, label):
    """log P(correct token) — same as GRPO reward"""
    logp = F.log_softmax(last_logit.float(), dim=-1)
    ids = YES_IDS if label == "yes" else NO_IDS
    return torch.logsumexp(logp[ids], dim=0).item()


# ─── Load hard questions ──────────────────────────────────────────
hard_file = os.path.join(POPE_DIR, "coco_pope_hard_10000.json")
all_qs = [json.loads(l) for l in open(hard_file, encoding="utf-8")]
# Take a diverse sample — prefer unique images
seen_imgs = set()
questions = []
for q in all_qs:
    if q["image"] not in seen_imgs:
        seen_imgs.add(q["image"])
        questions.append(q)
    if len(questions) >= args.n_questions:
        break
random.shuffle(questions)
questions = questions[:args.n_questions]
print(f"Using {len(questions)} questions from {len(seen_imgs)} unique images")

# ─── Main loop ────────────────────────────────────────────────────
# For each question: run K random forwards
# Record: for each (layer, strategy), (sum reward, count)

all_data = []  # list of dicts per question

for qi, q in enumerate(tqdm(questions, desc="Questions")):
    img_path = os.path.join(IMAGE_DIR, q["image"])
    label = q["label"]
    inputs = make_inputs(img_path, q["text"])

    # Precompute baseline reward (all none)
    mgr.clear_cache()
    mgr._force_per_layer = {}  # empty → all none
    mgr._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()
    with torch.no_grad():
        emb = model.get_input_embeddings()(inputs.input_ids)
        base_out = model.model(
            inputs_embeds=emb, attention_mask=inputs.attention_mask,
            pixel_values=inputs.get("pixel_values", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
            use_cache=False,
        )
        base_logit = model.lm_head(base_out.last_hidden_state[0:1, -1:, :])[0, -1]
        R_baseline = reward_fn(base_logit, label)
    del base_out, emb

    # Per-layer-strategy accumulator
    ls_rewards = defaultdict(list)  # (layer_name, strategy) → [rewards]

    # Run K random forwards
    for k in range(args.k_samples):
        mgr.clear_cache()
        picked_layers = random.sample(ACTIVE_LAYERS, min(args.sparse_k, len(ACTIVE_LAYERS)))
        assignment = {}
        for li in picked_layers:
            name = f"lm.{li}"
            strategy = random.choice(ALL_STRATEGIES)
            assignment[name] = ALL_STRATEGIES.index(strategy)

        mgr._force_per_layer = dict(assignment)
        mgr._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()

        with torch.no_grad():
            emb = model.get_input_embeddings()(inputs.input_ids)
            base_out = model.model(
                inputs_embeds=emb, attention_mask=inputs.attention_mask,
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                use_cache=False,
            )
            last_logit = model.lm_head(base_out.last_hidden_state[0:1, -1:, :])[0, -1]
            R = reward_fn(last_logit, label)
        del base_out, emb

        for name, strat_idx in assignment.items():
            strategy = ALL_STRATEGIES[strat_idx]
            ls_rewards[(name, strategy)].append(R)

    # Aggregate per (layer, strategy)
    q_data = {
        "question": q["text"],
        "image": q["image"],
        "label": label,
        "R_baseline": R_baseline,
        "k": args.k_samples,
    }
    layer_strat_stats = {}
    for (ln, s), rewards in ls_rewards.items():
        arr = np.array(rewards)
        layer_strat_stats[f"{ln}:{s}"] = {
            "n": len(arr),
            "mean": arr.mean(),
            "std": arr.std(),
            "delta_vs_baseline": arr.mean() - R_baseline,
        }
    q_data["layer_strategy"] = layer_strat_stats
    all_data.append(q_data)
    torch.cuda.empty_cache()

mgr.unwrap_all()

# ─── Analysis ─────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SIGNAL VERIFICATION RESULTS ({args.n_questions} qs × {args.k_samples} samples = "
      f"{args.n_questions * args.k_samples} forwards)")
print(f"{'='*70}")

# 1. Global: per-strategy delta vs baseline (averaged across all layers)
print(f"\n--- 1. Per-strategy average delta vs baseline (all layers, all questions) ---")
strat_deltas = defaultdict(list)
for qd in all_data:
    for key, stats in qd["layer_strategy"].items():
        strat = key.split(":")[1]
        strat_deltas[strat].append(stats["delta_vs_baseline"])

for s in ALL_STRATEGIES:
    if s == "none":
        continue
    arr = np.array(strat_deltas[s])
    t_stat = arr.mean() / (arr.std() / max(np.sqrt(len(arr)), 1))
    p_value = 2 * (1 - max(0, min(1,
        np.exp(-0.5 * t_stat**2)  # rough normal approx
    )))
    print(f"  {s:10s}: mean_delta={arr.mean():.6f}  std={arr.std():.6f}  "
          f"n={len(arr)}  |t|={abs(t_stat):.2f}  significant={'YES' if abs(t_stat) > 2 else 'no'}")

# 2. Per-layer: does ANY layer show significant strategy differentiation?
print(f"\n--- 2. Per-layer significance: does strategy choice matter? ---")
# For each layer, do ANOVA F-test: between-strategy variance vs within-strategy variance
layer_signals = []
for li in ACTIVE_LAYERS:
    ln = f"lm.{li}"
    strat_means = defaultdict(list)
    for qd in all_data:
        for s in ALL_STRATEGIES:
            key = f"{ln}:{s}"
            if key in qd["layer_strategy"]:
                strat_means[s].append(qd["layer_strategy"][key]["delta_vs_baseline"])

    # F-test: is between-strategy variance > within?
    if len(strat_means) < 2:
        continue
    all_values = []
    groups = []
    for s, vals in strat_means.items():
        if len(vals) >= 3:
            all_values.extend(vals)
            groups.append(vals)

    if len(groups) < 2:
        continue

    # One-way ANOVA
    grand_mean = np.mean(all_values)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_within = sum(sum((v - np.mean(g))**2 for v in g) for g in groups)
    df_between = len(groups) - 1
    df_within = len(all_values) - len(groups)
    if df_within < 1 or ss_within < 1e-12:
        continue
    F = (ss_between / df_between) / (ss_within / df_within)
    # p-value from F distribution (rough)
    layer_signals.append((li, F, ss_between, ss_within))

layer_signals.sort(key=lambda x: -x[1])
print(f"  Top-5 layers with strongest strategy differentiation:")
for li, F, ssb, ssw in layer_signals[:5]:
    significant = "YES" if F > 3 else "no "
    print(f"    LM.{li:2d}: F={F:.3f}  SS_between={ssb:.6f}  SS_within={ssw:.6f}  sig={significant}")
print(f"  Bottom-3 layers:")
for li, F, ssb, ssw in layer_signals[-3:]:
    print(f"    LM.{li:2d}: F={F:.3f}")

# 3. Per-question: does strategy choice matter for specific questions?
print(f"\n--- 3. Per-question: max strategy delta range ---")
for qd in all_data[:5]:
    deltas = [stats["delta_vs_baseline"] for key, stats in qd["layer_strategy"].items()
              if not key.endswith(":none")]
    if deltas:
        d_max = max(deltas)
        d_min = min(deltas)
        print(f"  [{qd['label']}] range=[{d_min:.4f}, {d_max:.4f}] "
              f"Q: {qd['question'][:70]}")

# 4. The bottom line
print(f"\n--- 4. Bottom line ---")
all_delta_magnitudes = []
for qd in all_data:
    for key, stats in qd["layer_strategy"].items():
        if not key.endswith(":none"):
            all_delta_magnitudes.append(abs(stats["delta_vs_baseline"]))
arr = np.array(all_delta_magnitudes)
print(f"  Mean |Δ| per (layer, strategy): {arr.mean():.4f}")
print(f"  Std  |Δ| : {arr.std():.4f}")
print(f"  Max  |Δ| : {arr.max():.4f}")
print(f"  95th %ile |Δ|: {np.percentile(arr, 95):.4f}")

# Is the average delta > 0?
non_zero = sum(1 for d in all_delta_magnitudes if d > 0.01)
print(f"  Fraction with |Δ| > 0.01: {non_zero}/{len(all_delta_magnitudes)} ({non_zero/len(all_delta_magnitudes):.1%})")
print(f"\n  Interpretation: if mean |Δ| < ~0.02 (log-prob), the per-layer per-strategy")
print(f"  effect is indistinguishable from sampling noise in a yes/no binary task.")
print(f"  This means NO router can learn meaningful context-dependent decisions")
print(f"  at single-question per-layer resolution — the candidate strategies simply")
print(f"  don't produce differential effects strong enough at this granularity.")

# Save
out_path = os.path.join(CHECKPOINT_DIR, "verify_signal_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "config": {"n_questions": args.n_questions, "k_samples": args.k_samples,
                   "sparse_k": args.sparse_k},
        "per_strategy_delta": {s: {"mean": np.mean(strat_deltas[s]).item(),
                                   "std": np.std(strat_deltas[s]).item()}
                               for s in strat_deltas},
        "global_mean_abs_delta": arr.mean().item(),
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {out_path}")
print("DONE")
