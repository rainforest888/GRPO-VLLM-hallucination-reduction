"""
dpo_train.py — DPO router training with CONTINUOUS reward (dense signal).

Previous failure: binary correct/wrong reward gave only ~5% valid preference
pairs (POPE yes/no → two samples usually same answer). Fix: use the model's
log-prob of the CORRECT answer token as a continuous quality score. Two
samples almost always differ in this score → a preference pair on EVERY sample
(~20x denser signal).

Per-sample flow (4 no_grad forwards + tiny grad backward, grad-free):
  1. sample d1 → forward → R1 = log P(correct_label_token) ; save path1
  2. sample d2 → forward → R2 ; save path2
  3. chosen = argmax(R1,R2), rejected = argmin
  4. DPO loss from saved hidden states (router MLP only, grad-free):
        L = -log σ(β (logπ(chosen) − logπ(rejected)))
  5. backward (only router params get grad)

Router acts only on MIDDLE LM layers (configurable, default 5-20); all other
layers are forced to "none" (no intervention), matching the papers' scope.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/dpo_train.py
"""
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager
from dpo_data import load_pope_questions, split_by_image, POPEDataset

# ─── Paths ──────────────────────────────────────────────────────────
MODEL_DIR = r"G:\sample\Qwen3vl"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── Hyperparameters ────────────────────────────────────────────────
BETA = 0.1
LR = 1e-4
N_EPOCHS = 5
GRAD_ACCUM = 4
MAX_SAMPLES = 1000
VALID_SAMPLES = 200
SAVE_EVERY = 150
# Middle LM layers that get a router (others forced to "none")
ACTIVE_LM_LAYERS = list(range(5, 19))  # LM layers 5–18: vision-enrichment stage (paper: 2411.16724v3)
SPARSE_K = 2  # train: randomly activate only 2 layers per forward (non-sparse for inference)

# ─── Answer token ids (yes/no variants) for continuous reward ───────
_tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
YES_IDS = [_tok(s, add_special_tokens=False).input_ids[0]
           for s in ["yes", "Yes", " yes", " Yes"]]
NO_IDS = [_tok(s, add_special_tokens=False).input_ids[0]
          for s in ["no", "No", " no", " No"]]
YES_IDS = torch.tensor(sorted(set(YES_IDS)))
NO_IDS = torch.tensor(sorted(set(NO_IDS)))
print(f"Yes token ids: {YES_IDS.tolist()}, No token ids: {NO_IDS.tolist()}")

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
    print("ERROR: calibration.pt not found. Run calibration.py first.")
    sys.exit(1)
calib = torch.load(calib_path, map_location="cpu", weights_only=False)

# ─── RouterManager (active = middle LM layers only) ─────────────────
active_layers = {f"lm.{i}" for i in ACTIVE_LM_LAYERS}
mgr = RouterManager(model, calib, active_layers=active_layers, alpha_init=1.2)
mgr.wrap_all()
mgr.gumbel_tau = 0.5
mgr._sparse_k = SPARSE_K  # train: 2 layers active per step; inference: argmax all layers
print(f"Trainable params: {sum(p.numel() for p in mgr.parameters()):,}")


def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", "")
    w = t.split()
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
    """Locate question tokens in the full tokenized sequence. Returns (N_q,) tensor
    of key positions, or None."""
    full_text = question + " Please answer yes or no."
    q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
    all_ids = inputs["input_ids"][0]
    q_t = torch.tensor(q_ids, device=all_ids.device)
    for s in range(len(all_ids) - len(q_ids) + 1):
        if (all_ids[s:s + len(q_ids)] == q_t).all():
            return torch.arange(s, s + len(q_ids))
    return None


def reward_from_last_logit(last_logit, label):
    """R = log P(correct label token) using logsumexp over yes/no variants.
    last_logit: (vocab,) on cuda. label: 'yes' or 'no'."""
    logp = F.log_softmax(last_logit.float(), dim=-1)
    ids = YES_IDS if label == "yes" else NO_IDS
    return torch.logsumexp(logp[ids.to(logp.device)], dim=0)


def sample_forward(inputs, label, question):
    """Sample decisions (no_grad), forward, return (answer, R, path_data).
    R = continuous reward = log P(correct label token)."""
    mgr.eval()
    mgr.mode = "sample"
    mgr.clear_cache()
    mgr._current_q_pos = find_q_positions(inputs, question)  # for AdaIAT-U
    mgr._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()  # n_vis for UAC dict W
    with torch.no_grad():
        emb = model.get_input_embeddings()(inputs.input_ids)
        base_out = model.model(
            inputs_embeds=emb,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.get("pixel_values", None),
            image_grid_thw=inputs.get("image_grid_thw", None),
            use_cache=False,
        )
        last_logit = model.lm_head(base_out.last_hidden_state[0:1, -1:, :])[0, -1]
        ans = answer_yes_no(processor.decode([last_logit.argmax().item()]))
        R = reward_from_last_logit(last_logit, label).item()
        path = mgr.save_path()
    del base_out, last_logit, emb
    return ans, R, path


def dpo_loss_from_paths(path_chosen, path_rejected):
    """Recompute router logπ for both paths from saved detached inputs (grad-free
    over attention). Returns differentiable loss tensor."""
    mgr.train()
    dec_c = {n: d for n, (d, _) in path_chosen.items() if n in mgr._router_map}
    dec_r = {n: d for n, (d, _) in path_rejected.items() if n in mgr._router_map}
    mgr._saved_inputs = {n: h for n, (_, h) in path_chosen.items() if n in mgr._router_map}
    logp_c = mgr.compute_log_prob_from_saved(dec_c)
    ent_c = mgr.compute_entropy_from_saved(dec_c)
    mgr._saved_inputs = {n: h for n, (_, h) in path_rejected.items() if n in mgr._router_map}
    logp_r = mgr.compute_log_prob_from_saved(dec_r)
    ent_r = mgr.compute_entropy_from_saved(dec_r)
    if logp_c is None or logp_r is None:
        return None
    dpo = -F.logsigmoid(BETA * (logp_c - logp_r))
    ent_reg = -0.01 * (ent_c + ent_r) * 0.5
    return dpo + ent_reg


# ═══ Main ════════════════════════════════════════════════════════════
def main():
    all_qs = load_pope_questions()
    train_qs, valid_qs = split_by_image(all_qs, train_ratio=0.8, seed=42)
    if MAX_SAMPLES:
        train_qs = train_qs[:MAX_SAMPLES]
    train_dataset = POPEDataset(train_qs)
    valid_dataset = POPEDataset(valid_qs)
    print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}")

    optimizer = AdamW(mgr.parameters(), lr=LR)
    dpo_steps = 0
    grad_step = 0
    skip = 0

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0
        epoch_dpo = 0
        tau = 0.8 - (epoch / max(N_EPOCHS - 1, 1)) * 0.5
        mgr.gumbel_tau = tau
        print(f"\nEpoch {epoch+1}: tau={tau:.2f}")

        pbar = tqdm(range(len(train_dataset)), desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for idx in pbar:
            item = train_dataset[idx]
            if not os.path.exists(item["image_path"]):
                continue
            inputs = prepare_inputs(item["image_path"], item["question"])
            label = item["label"]

            # ── Phase 1: two samples with continuous reward ──
            ans1, R1, path1 = sample_forward(inputs, label, item["question"])
            ans2, R2, path2 = sample_forward(inputs, label, item["question"])

            if abs(R1 - R2) < 1e-4:
                skip += 1
                continue
            if R1 >= R2:
                chosen, rejected = path1, path2
            else:
                chosen, rejected = path2, path1

            # ── Phase 2: DPO loss from saved paths (grad-free) ──
            loss = dpo_loss_from_paths(chosen, rejected)
            if loss is None:
                continue

            loss_val = loss.item()
            loss.backward()
            del loss
            epoch_loss += loss_val
            epoch_dpo += 1
            dpo_steps += 1
            grad_step += 1

            if grad_step % GRAD_ACCUM == 0:
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.empty_cache()

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "dpo": dpo_steps,
                              "dR": f"{abs(R1-R2):.2f}", "tau": f"{tau:.2f}"})

            if dpo_steps % SAVE_EVERY == 0:
                torch.save(mgr.state_dict(),
                           os.path.join(CHECKPOINT_DIR, f"router_weights_step{dpo_steps}.pt"))

        if grad_step % GRAD_ACCUM != 0:
            optimizer.step(); optimizer.zero_grad()

        print(f"\nEpoch {epoch+1}: avg_loss={epoch_loss/max(epoch_dpo,1):.4f}, "
              f"dpo={epoch_dpo}, skip={skip}")
        alphas = {k: f"{mgr.get_alpha(k).item():.4f}" for k in sorted(mgr.raw_alphas.keys())}
        print(f"  Alphas: {alphas}")

        # ── Validation (argmax router) ──
        mgr.eval(); mgr.mode = "argmax"
        correct_cnt = 0; total_cnt = 0
        for i in tqdm(range(min(len(valid_dataset), VALID_SAMPLES)), desc="Valid"):
            it = valid_dataset[i]
            if not os.path.exists(it["image_path"]): continue
            inp = prepare_inputs(it["image_path"], it["question"])
            mgr.clear_cache()
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=4)
            raw = processor.decode(gen[0, inp.input_ids.shape[1]:], skip_special_tokens=True)
            if answer_yes_no(raw) == it["label"]: correct_cnt += 1
            total_cnt += 1
        print(f"  Valid acc: {correct_cnt/max(total_cnt,1):.4f} ({correct_cnt}/{total_cnt})")

        torch.save(mgr.state_dict(),
                   os.path.join(CHECKPOINT_DIR, f"router_weights_epoch{epoch+1}.pt"))

    final_path = os.path.join(CHECKPOINT_DIR, "router_weights_final.pt")
    torch.save(mgr.state_dict(), final_path)
    print(f"\n[OK] Final: {final_path}")
    mgr.unwrap_all()


if __name__ == "__main__":
    main()
