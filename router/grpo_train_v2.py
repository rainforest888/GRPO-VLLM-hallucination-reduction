"""
grpo_train_v2.py — GRPO router training with random layer+strategy exploration.

Key idea (user's design):
    Each training step:
    1. Randomly pick k layers from LM 5-18
    2. Each picked layer gets a randomly assigned strategy (uac/adaiat/vhr/uac_vhr/none)
    3. All non-picked layers → "none"
    4. Run forward pass, compute reward = log P(correct answer token)
    5. GRPO: within a group of K samples, advantage = (R - mean(R)) / std(R)
       Policy loss = -advantage * log_prob(router predicts chosen strategy)
       Clipped with PPO-style ratio to prevent extreme updates.
    6. Router per layer learns: given hidden state → predict strategy logits

Why this is novel vs existing methods:
    - Random exploration gives unbiased coverage of the 3^14 = 4.78M combo space
    - Per-layer routers learn context-dependent strategy selection
    - Not a fixed strategy paper re-run — the router IS the contribution

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/grpo_train_v2.py --sparse_k 3 --grpo_k 4 --n_epochs 10
"""
import json, os, sys, random, time
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--sparse_k", type=int, default=3,
                help="Number of randomly selected layers to activate per forward pass")
ap.add_argument("--grpo_k", type=int, default=4,
                help="GRPO group size (samples per group)")
ap.add_argument("--n_epochs", type=int, default=10,
                help="Number of epochs over the training set")
ap.add_argument("--max_samples", type=int, default=8000,
                help="Max training samples per epoch")
ap.add_argument("--valid_samples", type=int, default=500,
                help="Validation samples per epoch")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--grad_accum", type=int, default=4)
ap.add_argument("--clip_eps", type=float, default=0.2,
                help="PPO clip epsilon")
ap.add_argument("--entropy_coef", type=float, default=0.05,
                help="Entropy bonus coefficient")
ap.add_argument("--save_every", type=int, default=200,
                help="Save checkpoint every N GRPO steps")
ap.add_argument("--eval_every", type=int, default=500,
                help="Evaluate on valid set every N GRPO steps")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--resume", type=str, default=None,
                help="Resume from checkpoint path")
args = ap.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager

MODEL_DIR = r"G:\sample\Qwen3vl"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

HARD_FILE = os.path.join(POPE_DIR, "coco_pope_hard_10000.json")

ACTIVE_LM_LAYERS = list(range(5, 19))  # 14 layers

# Strategies available for LM 5-18
ALL_STRATEGIES = ["uac", "adaiat", "vhr", "uac_vhr", "none"]
STRATEGY_TO_IDX = {s: i for i, s in enumerate(ALL_STRATEGIES)}

# ─── Token IDs for yes/no ───────────────────────────────────────────
_tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
YES_IDS = torch.tensor(sorted(set(
    _tok(s, add_special_tokens=False).input_ids[0]
    for s in ["yes", "Yes", " yes", " Yes"]
)))
NO_IDS = torch.tensor(sorted(set(
    _tok(s, add_special_tokens=False).input_ids[0]
    for s in ["no", "No", " no", " No"]
)))
print(f"Yes ids: {YES_IDS.tolist()}, No ids: {NO_IDS.tolist()}")

# ─── Load model (frozen) ────────────────────────────────────────────
print("Loading Qwen3-VL (eager, frozen)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
for p in model.parameters():
    p.requires_grad = False
model.eval()
print("Model loaded & frozen.\n")

# ─── Load calibration ───────────────────────────────────────────────
calib_path = os.path.join(CHECKPOINT_DIR, "calibration.pt")
if not os.path.exists(calib_path):
    print("ERROR: calibration.pt not found. Run calibrate_vhr.py and recalibrate_u.py first.")
    sys.exit(1)
calib = torch.load(calib_path, map_location="cpu", weights_only=False)

# ─── RouterManager ──────────────────────────────────────────────────
active_layers = {f"lm.{i}" for i in ACTIVE_LM_LAYERS}
mgr = RouterManager(model, calib, active_layers=active_layers, alpha_init=0.0)
mgr.wrap_all()
print(f"RouterManager: {mgr.num_routers} routers, active layers LM {min(ACTIVE_LM_LAYERS)}-{max(ACTIVE_LM_LAYERS)}")
print(f"Trainable params: {sum(p.numel() for p in mgr.parameters()):,}")

# ─── Resume checkpoint ──────────────────────────────────────────────
if args.resume and os.path.exists(args.resume):
    sd = torch.load(args.resume, map_location="cpu", weights_only=False)
    mgr.load_state_dict(sd)
    print(f"Resumed from {args.resume}")
    for k in sorted(mgr.raw_alphas.keys()):
        print(f"  alpha {k}: {mgr.get_alpha(k).item():.4f}")


# ─── Helpers ────────────────────────────────────────────────────────
def prepare_inputs(image_path, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question + " Please answer yes or no."},
    ]}]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)


def find_q_positions(inputs, question):
    """Find token positions of the question + prompt in the input sequence."""
    full_text = question + " Please answer yes or no."
    q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
    all_ids = inputs["input_ids"][0]
    q_t = torch.tensor(q_ids, device=all_ids.device)
    for s in range(len(all_ids) - len(q_ids) + 1):
        if (all_ids[s:s + len(q_ids)] == q_t).all():
            return torch.arange(s, s + len(q_ids))
    return None


def reward_from_last_logit(last_logit, label):
    """Log probability of the correct answer token (any yes/no variant)."""
    logp = F.log_softmax(last_logit.float(), dim=-1)
    ids = YES_IDS if label == "yes" else NO_IDS
    return torch.logsumexp(logp[ids.to(logp.device)], dim=0)


# ─── Sample a random layer+strategy assignment ──────────────────────
def random_layer_strategy_assignment(sparse_k):
    """
    Randomly pick `sparse_k` layers from ACTIVE_LM_LAYERS.
    For each picked layer, randomly assign a strategy.
    Returns dict {layer_name: strategy_idx}.

    Non-picked layers default to "none" in the forward pass.
    """
    picked = random.sample(ACTIVE_LM_LAYERS, min(sparse_k, len(ACTIVE_LM_LAYERS)))
    assignment = {}
    for layer_idx in picked:
        name = f"lm.{layer_idx}"
        strategy = random.choice(ALL_STRATEGIES)
        assignment[name] = STRATEGY_TO_IDX[strategy]
    return assignment


# ─── Single sample forward ──────────────────────────────────────────
def sample_forward(inputs, label, question, sparse_k):
    """
    Run one forward pass with random layer+strategy assignment.

    Returns:
        reward: float (log-prob of correct answer)
        path: dict {layer_name: (decision_int, hidden_states_tensor)}
        assignment: dict {layer_name: strategy_idx} (what was applied)
    """
    mgr.eval()
    mgr.mode = "force_per_layer"
    mgr.clear_cache()

    # Random assignment for active layers
    assignment = random_layer_strategy_assignment(sparse_k)

    # Set per-layer force decisions
    mgr._force_per_layer = dict(assignment)

    # Set up context for strategies
    mgr._current_q_pos = find_q_positions(inputs, question)
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
        R = reward_from_last_logit(last_logit, label).item()
        path = mgr.save_path()

    del base_out, last_logit, emb
    return R, path, assignment


# ─── GRPO loss from group ──────────────────────────────────────────
def grpo_loss_from_group(samples):
    """
    samples: list of (R_i, path_i, assignment_i) for K samples.

    Computes GRPO loss: advantage-weighted negative log-prob
    with PPO-style clip on the importance ratio.

    The ratio = exp(new_log_prob - old_log_prob) clips policy drift.
    Since we use random exploration, old_log_prob is uniform(1/5),
    i.e., log(1/5) = -1.609. The ratio represents how much the router
    now prefers the chosen strategy compared to random chance.
    """
    if len(samples) < 2:
        return None, {}

    Rs = torch.tensor([s[0] for s in samples], dtype=torch.float32)
    mean_R = Rs.mean()
    std_R = Rs.std().clamp_min(1e-4)
    advantages = (Rs - mean_R) / std_R  # (K,)

    mgr.train()

    old_logp_uniform = torch.log(torch.tensor(1.0 / len(ALL_STRATEGIES)))

    layer_losses = []
    stats = {"advantages": advantages.tolist(), "rewards": Rs.tolist()}

    for i, (_, path_i, assignment_i) in enumerate(samples):
        if not path_i or not assignment_i:
            continue

        # Load saved inputs for grad-free recomputation
        mgr._saved_inputs = {
            n: h for n, (_, h) in path_i.items()
            if n in mgr._router_map and n in assignment_i
        }

        # Decisions from this sample
        dec_i = {
            n: d for n, (d, _) in path_i.items()
            if n in mgr._router_map and n in assignment_i
        }

        if not dec_i:
            continue

        # Compute NEW log-prob under current router weights
        lp_new = mgr.compute_log_prob_from_saved(dec_i)
        if lp_new is None:
            continue

        adv = advantages[i].detach()

        # PPO-style clipped objective
        # Ratio between new policy and old (random/uniform) policy
        log_ratio = lp_new - old_logp_uniform.to(lp_new.device)
        ratio = log_ratio.exp()
        clipped_ratio = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps)

        loss_i = -torch.min(adv * ratio, adv * clipped_ratio)
        layer_losses.append(loss_i)

        # Track which layers contributed
        for name in dec_i:
            stats.setdefault("active_layers", set()).add(name)

    if not layer_losses:
        return None, stats

    loss = torch.stack(layer_losses).mean()

    # Entropy bonus: encourage exploration
    if path_i:
        ent = mgr.compute_entropy_from_saved(
            {n: d for n, (d, _) in path_i.items() if n in mgr._router_map}
        )
        loss = loss - args.entropy_coef * ent
        stats["entropy"] = ent.item()

    return loss, stats


# ─── Load hard dataset ──────────────────────────────────────────────
print(f"\nLoading hard dataset: {HARD_FILE}")
hard_questions = [json.loads(l) for l in open(HARD_FILE, "r", encoding="utf-8")]
random.shuffle(hard_questions)
print(f"Hard set: {len(hard_questions)} questions")

# Split by image (from the hard set's unique 464 images)
train_ratio = 0.85
all_images = list(set(q["image"] for q in hard_questions))
random.shuffle(all_images)
n_train_img = int(len(all_images) * train_ratio)
train_images = set(all_images[:n_train_img])
valid_images = set(all_images[n_train_img:])

train_qs = [q for q in hard_questions if q["image"] in train_images]
valid_qs = [q for q in hard_questions if q["image"] in valid_images]
print(f"Train: {len(train_qs)} qs from {len(train_images)} images")
print(f"Valid: {len(valid_qs)} qs from {len(valid_images)} images")

# Trim train if needed
if args.max_samples and len(train_qs) > args.max_samples:
    train_qs = train_qs[:args.max_samples]
    print(f"Trimmed train to {len(train_qs)}")


# ─── Training loop ──────────────────────────────────────────────────
optimizer = AdamW(mgr.parameters(), lr=args.lr)
total_grpo_steps = 0
grad_step = 0
best_valid_acc = 0.0

# Log for analysis
strategy_layer_rewards = []  # (step, layer, strategy, reward)

print(f"\n{'='*60}")
print(f"GRPO v2: sparse_k={args.sparse_k}, grpo_k={args.grpo_k}, "
      f"n_epochs={args.n_epochs}")
print(f"Hard set wrong rate: 26.6%, Baseline acc ~73.4%")
print(f"{'='*60}\n")

for epoch in range(args.n_epochs):
    epoch_loss = 0.0
    epoch_grpo = 0
    epoch_reward = 0.0
    epoch_samples = 0

    random.shuffle(train_qs)
    train_iter = train_qs[:args.max_samples] if args.max_samples else train_qs

    pbar = tqdm(range(0, len(train_iter), args.grpo_k),
                desc=f"Epoch {epoch+1}/{args.n_epochs}")

    for base_idx in pbar:
        # Build a GRPO group: K samples for the same question
        # (same question, different layer+strategy assignments)
        q = train_iter[base_idx % len(train_iter)]
        image_path = os.path.join(IMAGE_DIR, q["image"])
        if not os.path.exists(image_path):
            continue

        inputs = prepare_inputs(image_path, q["text"])
        label = q["label"]

        # Collect K samples
        group = []
        for _ in range(args.grpo_k):
            R, path, assignment = sample_forward(inputs, label, q["text"], args.sparse_k)
            if path and assignment:
                group.append((R, path, assignment))
                epoch_reward += R
                epoch_samples += 1

        if len(group) < 2:
            torch.cuda.empty_cache()
            continue

        loss, stats = grpo_loss_from_group(group)
        if loss is None:
            torch.cuda.empty_cache()
            continue

        loss_val = loss.item()
        loss.backward()
        epoch_loss += loss_val
        epoch_grpo += 1
        total_grpo_steps += 1
        grad_step += 1

        # Log strategy-layer-reward
        for R, _, assignment in group:
            for name, strat_idx in assignment.items():
                strategy_layer_rewards.append({
                    "step": total_grpo_steps,
                    "layer": name,
                    "strategy": ALL_STRATEGIES[strat_idx],
                    "reward": R,
                })

        # Gradient accumulation
        if grad_step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(mgr.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        # Progress bar
        avg_R = stats.get("rewards", [0])[0] if stats.get("rewards") else 0
        advs = stats.get("advantages", [])
        adv_range = f"[{min(advs):.2f},{max(advs):.2f}]" if advs else "-"
        pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "R": f"{avg_R:.2f}",
            "A": adv_range,
            "ent": f"{stats.get('entropy', 0):.3f}",
            "steps": total_grpo_steps,
        })

        # Save checkpoint
        if total_grpo_steps % args.save_every == 0:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"router_weights_grpo_v2_step{total_grpo_steps}.pt")
            # Save combined state: router weights + step info
            sd = mgr.state_dict()
            sd["_meta"] = {
                "step": total_grpo_steps,
                "sparse_k": args.sparse_k,
                "grpo_k": args.grpo_k,
                "epoch": epoch,
                "active_layers": ACTIVE_LM_LAYERS,
                "all_strategies": ALL_STRATEGIES,
            }
            torch.save(sd, ckpt_path)

            # Also save strategy-layer-reward log periodically
            if len(strategy_layer_rewards) > 10000:
                log_path = os.path.join(CHECKPOINT_DIR, "grpo_v2_strategy_layer_rewards.jsonl")
                with open(log_path, "a", encoding="utf-8") as f:
                    for entry in strategy_layer_rewards[-5000:]:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                strategy_layer_rewards = strategy_layer_rewards[-5000:]

        # Validation
        if total_grpo_steps % args.eval_every == 0 and len(valid_qs) > 0:
            mgr.eval()
            mgr.mode = "argmax"
            correct = 0
            total = 0
            strategy_counts = defaultdict(int)

            valid_sample = random.sample(valid_qs, min(args.valid_samples, len(valid_qs)))
            for vq in tqdm(valid_sample, desc="Valid", leave=False):
                img_path = os.path.join(IMAGE_DIR, vq["image"])
                if not os.path.exists(img_path):
                    continue
                vinp = prepare_inputs(img_path, vq["text"])
                mgr.clear_cache()
                mgr._current_q_pos = find_q_positions(vinp, vq["text"])
                mgr._current_n_vis = (vinp["mm_token_type_ids"][0] > 0).sum().item()

                with torch.no_grad():
                    gen = model.generate(**vinp, max_new_tokens=4, use_cache=True)
                raw = processor.decode(
                    gen[0, vinp.input_ids.shape[1]:],
                    skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )
                ans = raw.strip().lower()
                if "." in ans:
                    ans = ans.split(".")[0]
                ans = ans.replace(",", "")
                words = ans.split()
                ans = "no" if ("no" in words or "not" in words) else "yes"

                if ans == vq["label"]:
                    correct += 1
                total += 1

                # Track strategy usage
                for name, idx in mgr._decisions.items():
                    for d in mgr.descs:
                        if d["name"] == name:
                            s = d["strategies"]
                            strat = s[min(idx, len(s) - 1)]
                            strategy_counts[strat] += 1
                            break

            valid_acc = correct / max(total, 1)

            strat_pct = {}
            total_s = sum(strategy_counts.values())
            if total_s > 0:
                strat_pct = {k: f"{v/total_s:.1%}" for k, v in strategy_counts.items()}

            print(f"\n[Step {total_grpo_steps}] Valid acc: {valid_acc:.4f} "
                  f"({correct}/{total})  best: {best_valid_acc:.4f}")
            print(f"  Strategy usage: {strat_pct}")

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_path = os.path.join(CHECKPOINT_DIR, "router_weights_grpo_v2_best.pt")
                torch.save(mgr.state_dict(), best_path)
                print(f"  -> Best checkpoint saved: {best_path}")

            mgr.eval()

    # End of epoch gradient flush
    if grad_step % args.grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(mgr.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss / max(epoch_grpo, 1)
    avg_R = epoch_reward / max(epoch_samples, 1)
    print(f"\nEpoch {epoch+1}: avg_loss={avg_loss:.4f}, avg_R={avg_R:.4f}, "
          f"grpo_steps={epoch_grpo}")

    # Print alpha values
    alphas = {k: f"{mgr.get_alpha(k).item():.4f}" for k in sorted(mgr.raw_alphas.keys())}
    print(f"  Alphas: {alphas}")

    # Epoch checkpoint
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"router_weights_grpo_v2_epoch{epoch+1}.pt")
    sd = mgr.state_dict()
    sd["_meta"] = {
        "step": total_grpo_steps, "epoch": epoch + 1,
        "valid_acc": best_valid_acc,
        "sparse_k": args.sparse_k, "grpo_k": args.grpo_k,
    }
    torch.save(sd, ckpt_path)

# ─── Final save ─────────────────────────────────────────────────────
final_path = os.path.join(CHECKPOINT_DIR, "router_weights_grpo_v2_final.pt")
sd = mgr.state_dict()
sd["_meta"] = {
    "step": total_grpo_steps, "best_valid_acc": best_valid_acc,
    "sparse_k": args.sparse_k, "grpo_k": args.grpo_k,
    "active_layers": ACTIVE_LM_LAYERS, "all_strategies": ALL_STRATEGIES,
}
torch.save(sd, final_path)

# Save strategy-layer-reward log
log_path = os.path.join(CHECKPOINT_DIR, "grpo_v2_strategy_layer_rewards.jsonl")
with open(log_path, "a", encoding="utf-8") as f:
    for entry in strategy_layer_rewards:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\n{'='*60}")
print(f"DONE. Best valid acc: {best_valid_acc:.4f}")
print(f"Final checkpoint: {final_path}")
print(f"Reward log: {log_path}")
print(f"{'='*60}")

mgr.unwrap_all()
