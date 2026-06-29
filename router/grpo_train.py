"""
grpo_train.py — GRPO router training (v6). Replaces DPO.

GRPO (Group Relative Policy Optimization):
  - Each step samples K=4 decisions per layer (groups).
  - Reward R = log P(correct answer token) per sample.
  - Within each group, compute advantage:
      A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
  - Policy loss: -A_i * log_softmax(router_logits)[decision_i]
    (clipped to avoid extreme updates).
  - Base model FROZEN. Grad-free through router.

Why GRPO > DPO for this task:
  - Group ranking amplifies weak signals (+0.17% becomes detectable).
  - K=4 samples = more exploration, less conservative collapse.
  - No need for reference model or explicit chosen/rejected pairing.
  - Sparse_k=2: 2 layers active per sample, 4 samples per group for each.
"""
import json, os, sys
from collections import defaultdict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager
from dpo_data import load_pope_questions, split_by_image, POPEDataset

MODEL_DIR = r"G:\sample\Qwen3vl"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

GRPO_K = 4           # group size (samples per group)
GRPO_EPS = 0.2       # clip epsilon for policy ratio
LR = 1e-4
N_EPOCHS = 5
GRAD_ACCUM = 4
MAX_SAMPLES = 1000
VALID_SAMPLES = 200
SAVE_EVERY = 150
ACTIVE_LM_LAYERS = list(range(5, 19))
SPARSE_K = 2  # active layers per forward

_tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
YES_IDS = torch.tensor(sorted(set(
    _tok(s, add_special_tokens=False).input_ids[0] for s in ["yes","Yes"," yes"," Yes"]
)))
NO_IDS = torch.tensor(sorted(set(
    _tok(s, add_special_tokens=False).input_ids[0] for s in ["no","No"," no"," No"]
)))

print(f"Yes ids: {YES_IDS.tolist()}, No ids: {NO_IDS.tolist()}")

print("Loading Qwen3-VL (eager, frozen)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
for p in model.parameters(): p.requires_grad = False
model.eval()
print("Model loaded & frozen.\n")

calib_path = os.path.join(CHECKPOINT_DIR, "calibration.pt")
if not os.path.exists(calib_path):
    print("ERROR: calibration.pt not found. Run recalibrate_u.py first.")
    sys.exit(1)
calib = torch.load(calib_path, map_location="cpu", weights_only=False)

active_layers = {f"lm.{i}" for i in ACTIVE_LM_LAYERS}
mgr = RouterManager(model, calib, active_layers=active_layers, alpha_init=1.2)
mgr.wrap_all()
mgr.gumbel_tau = 0.5
mgr._sparse_k = SPARSE_K
print(f"Trainable: {sum(p.numel() for p in mgr.parameters()):,}, sparse_k={SPARSE_K}")


def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"


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
    full_text = question + " Please answer yes or no."
    q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
    all_ids = inputs["input_ids"][0]
    q_t = torch.tensor(q_ids, device=all_ids.device)
    for s in range(len(all_ids) - len(q_ids) + 1):
        if (all_ids[s:s + len(q_ids)] == q_t).all():
            return torch.arange(s, s + len(q_ids))
    return None


def reward_from_last_logit(last_logit, label):
    logp = F.log_softmax(last_logit.float(), dim=-1)
    ids = YES_IDS if label == "yes" else NO_IDS
    return torch.logsumexp(logp[ids.to(logp.device)], dim=0)


def sample_forward(inputs, label, question):
    """Sample decisions (no_grad), return (R, path_data, old_log_prob).

    The old_log_prob is the log-probability of the sampled decisions under the
    CURRENT policy weights. It is saved as a detached float and used later by
    grpo_loss_from_group as the reference for the importance-sampling ratio.
    """
    mgr.eval(); mgr.mode = "sample"; mgr.clear_cache()
    mgr._current_q_pos = find_q_positions(inputs, question)
    mgr._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()  # n_vis for UAC dict W
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
        # Compute log_prob under the OLD policy (current weights) for PPO ratio
        dec_i = {n: d for n, (d, _) in path.items() if n in mgr._router_map}
        mgr._saved_inputs = {n: h for n, (_, h) in path.items() if n in mgr._router_map}
        old_lp_tensor = mgr.compute_log_prob_from_saved(dec_i)
        old_lp_val = old_lp_tensor.item() if old_lp_tensor is not None else None
    del base_out, last_logit, emb
    return R, path, old_lp_val


def grpo_loss_from_group(samples):
    """
    samples: list of (R_i, path_i, old_lp_i) for K samples.
    old_lp_i is the log_prob under the sampling policy (detached float).

    Compute GRPO loss: sum over active layers of advantage-weighted
    negative log_prob, with PPO-style clip on importance ratio.

    Ratio = exp(new_log_prob - old_log_prob) measures how much the policy
    has moved from the sampling distribution.  A clip of ε=0.2 prevents
    the update from drifting too far per step.
    """
    Rs = torch.tensor([s[0] for s in samples], dtype=torch.float32)
    mean_R = Rs.mean()
    std_R = Rs.std().clamp_min(1e-4)
    advantages = (Rs - mean_R) / std_R  # (K,)

    mgr.train()
    layer_losses = []
    for i, (_, path_i, old_lp_val) in enumerate(samples):
        if not path_i: continue
        dec_i = {n: d for n, (d, _) in path_i.items() if n in mgr._router_map}
        mgr._saved_inputs = {n: h for n, (_, h) in path_i.items() if n in mgr._router_map}
        lp_new = mgr.compute_log_prob_from_saved(dec_i)
        if lp_new is None: continue
        if old_lp_val is None:
            # Fallback: no old log_prob available, use vanilla REINFORCE
            layer_losses.append(-advantages[i].detach() * lp_new)
            continue

        adv = advantages[i].detach()
        # PPO-style clipped objective: ratio = π_new(a|s) / π_old(a|s)
        ratio = (lp_new - old_lp_val).exp()  # differentiates  new ≠ old
        clipped_ratio = ratio.clamp(1.0 - GRPO_EPS, 1.0 + GRPO_EPS)
        loss_i = -torch.min(adv * ratio, adv * clipped_ratio)
        layer_losses.append(loss_i)

    if not layer_losses:
        return None

    loss = torch.stack(layer_losses).mean()
    # entropy bonus: encourage exploration
    if path_i:
        ent = mgr.compute_entropy_from_saved({n: d for n, (d, _) in path_i.items() if n in mgr._router_map})
        loss = loss - 0.02 * ent  # higher than before — encourage exploration
    return loss


# ═══ Main ════════════════════════════════════════════════════════════
def main():
    all_qs = load_pope_questions()
    train_qs, valid_qs = split_by_image(all_qs, train_ratio=0.8, seed=42)
    if MAX_SAMPLES: train_qs = train_qs[:MAX_SAMPLES]
    train_dataset = POPEDataset(train_qs)
    valid_dataset = POPEDataset(valid_qs)
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

    optimizer = AdamW(mgr.parameters(), lr=LR)
    grpo_steps = 0; grad_step = 0

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0; epoch_grpo = 0
        tau = 0.8 - (epoch / max(N_EPOCHS - 1, 1)) * 0.5
        mgr.gumbel_tau = tau
        print(f"\nEpoch {epoch+1}: tau={tau:.2f}")

        pbar = tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for idx in pbar:
            item = train_dataset[idx]
            if not os.path.exists(item["image_path"]): continue
            inputs = prepare_inputs(item["image_path"], item["question"])
            label = item["label"]

            # Sample K paths for this group
            group = []
            for _ in range(GRPO_K):
                R, path, old_lp = sample_forward(inputs, label, item["question"])
                if path and old_lp is not None:
                    group.append((R, path, old_lp))

            if len(group) < 2: continue

            loss = grpo_loss_from_group(group)
            if loss is None: continue

            loss_val = loss.item()
            loss.backward()
            epoch_loss += loss_val; epoch_grpo += 1; grpo_steps += 1; grad_step += 1

            if grad_step % GRAD_ACCUM == 0:
                optimizer.step(); optimizer.zero_grad()
            torch.cuda.empty_cache()

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "grpo": grpo_steps,
                              "tau": f"{tau:.2f}"})

            if grpo_steps % SAVE_EVERY == 0:
                torch.save(mgr.state_dict(), os.path.join(CHECKPOINT_DIR, f"router_weights_step{grpo_steps}.pt"))

        if grad_step % GRAD_ACCUM != 0:
            optimizer.step(); optimizer.zero_grad()

        print(f"\nEpoch {epoch+1}: avg_loss={epoch_loss/max(epoch_grpo,1):.4f}, grpo={epoch_grpo}")
        alphas = {k: f"{mgr.get_alpha(k).item():.4f}" for k in sorted(mgr.raw_alphas.keys())}
        print(f"  Alphas: {alphas}")

        mgr.eval(); mgr.mode = "argmax"
        correct_cnt = 0; total_cnt = 0
        for i in tqdm(range(min(len(valid_dataset), VALID_SAMPLES)), desc="Valid"):
            it = valid_dataset[i]
            if not os.path.exists(it["image_path"]): continue
            inp = prepare_inputs(it["image_path"], it["question"])
            mgr.clear_cache()
            mgr._current_q_pos = find_q_positions(inp, it["question"])
            mgr._current_n_vis = (inp["mm_token_type_ids"][0] > 0).sum().item()
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=4)
            raw = processor.decode(gen[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
            if answer_yes_no(raw) == it["label"]: correct_cnt += 1
            total_cnt += 1
        print(f"  Valid acc: {correct_cnt/max(total_cnt,1):.4f} ({correct_cnt}/{total_cnt})")
        torch.save(mgr.state_dict(), os.path.join(CHECKPOINT_DIR, f"router_weights_epoch{epoch+1}.pt"))

    final_path = os.path.join(CHECKPOINT_DIR, "router_weights_final.pt")
    torch.save(mgr.state_dict(), final_path)
    print(f"\n[OK] Final: {final_path}")
    mgr.unwrap_all()


if __name__ == "__main__":
    main()
