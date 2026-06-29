"""
adaiat_u_inference.py — AdaIAT adapted for POPE with QUESTION token target.

Original AdaIAT amplifies attention to GENERATED TEXT (Tp). On POPE yes/no
the answer token is produced at prefill where Tp is empty → original target
inapplicable.

Version 1 (V target): amplified attention to image tokens V.
  Result: M=0.95 (<1) — signal reversed, correct answers attend LESS to V.
  POPE: Random +0.27, Popular/Adversarial flat.

Version 2 (U target — THIS FILE): amplifies attention to QUESTION tokens U.
  Calibration found M_mean=1.023 (>1) for layer 15 — signal direction correct.
  The question carries the queried object ("Is there a cat...") and correct
  answers pay MORE attention to it → amplifying U is the right direction.
  Also restricts to vision-enrichment layers 5-18 per paper 2411.16724v3.

Mechanism (same as AdaIAT): per-head M = A^correct_U / A^wrong_U,
  layer threshold, adaptive trigger, post-softmax amplify + row renormalize.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/adaiat_u_inference.py [--layer L] [--alpha A] [--beta B] [--outdir NAME]
"""
import json
import os
import sys
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    repeat_kv, apply_rotary_pos_emb,
)

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
BASELINE_DIR = r"G:\sample\Qwen3vl\router_project\pope_results\baseline"
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--alpha", type=float, default=1.0, help="amplification strength (higher since M>1 now)")
ap.add_argument("--beta", type=float, default=0.5, help="threshold balance coef")
ap.add_argument("--ncalib", type=int, default=60, help="#correct & #wrong for calibration")
ap.add_argument("--outdir", type=str, default=None)
args = ap.parse_args()
LAYER = args.layer
ALPHA = args.alpha
BETA = args.beta
NCALIB = args.ncalib
OUT_NAME = args.outdir or f"adaiat_layer{LAYER}_a{ALPHA}"
OUTPUT_DIR = os.path.join(RESULTS_BASE, OUT_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"AdaIAT-POPE: layer={LAYER}, alpha={ALPHA}, beta={BETA}, out={OUTPUT_DIR}")

EPS = 1e-8

# ─── Load model ─────────────────────────────────────────────────────
print("Loading Qwen3-VL (eager)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
for p in model.parameters():
    p.requires_grad = False

lm_layers = model.model.language_model.layers
attn_mod = lm_layers[LAYER].self_attn
H_heads = model.model.language_model.config.num_attention_heads


# ─── Step 1: Calibrate M and threshold from baseline correct/wrong ──
print("Loading baseline correct/wrong samples for calibration...")
correct_samples, wrong_samples = [], []
for s in ["random", "popular", "adversarial"]:
    a = [json.loads(l) for l in open(os.path.join(BASELINE_DIR, f"coco_pope_{s}_answers.json"), encoding="utf-8")]
    b = [json.loads(l) for l in open(os.path.join(POPE_DIR, f"coco_pope_{s}.json"), encoding="utf-8")]
    for ai, bi in zip(a, b):
        entry = (bi["image"], bi["text"])
        (correct_samples if ai["answer"] == bi["label"] else wrong_samples).append(entry)
print(f"  baseline: {len(correct_samples)} correct, {len(wrong_samples)} wrong")


def capture_attention_to_U(image_name, question):
    """Run one sample, capture attention from last query token to QUESTION tokens
    at LAYER, per head. Returns (H,) vector = mean over U tokens.
    Locates question tokens in the tokenized sequence via substring match."""
    img = Image.open(os.path.join(IMAGE_DIR, image_name)).convert("RGB")
    full_text = question + " Please answer yes or no."
    q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": full_text},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    all_ids = inputs["input_ids"][0]
    # Locate question token positions (q_ids appears contiguously)
    q_tensor = torch.tensor(q_ids, device=all_ids.device)
    q_pos = []
    for s in range(len(all_ids) - len(q_ids) + 1):
        if (all_ids[s:s + len(q_ids)] == q_tensor).all():
            q_pos = list(range(s, s + len(q_ids)))
            break
    if not q_pos:
        return None

    captured = {}
    def hook(module, inp, out):
        if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
            aw = out[1][0, :, -1, :]  # (H, Lk)
            q_idx = torch.tensor(q_pos, device=aw.device, dtype=torch.long)
            captured["u"] = aw[:, q_idx].mean(dim=-1).detach().cpu()  # (H,)
    h = attn_mod.register_forward_hook(hook)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=2)
    h.remove()
    return captured.get("u", None)


print(f"Calibrating on {NCALIB} correct + {NCALIB} wrong samples...")
correct_attn = []  # list of (H,)
for img_name, q in tqdm(correct_samples[:NCALIB], desc="correct"):
    v = capture_attention_to_U(img_name, q)
    if v is not None:
        correct_attn.append(v)
wrong_attn = []
for img_name, q in tqdm(wrong_samples[:NCALIB], desc="wrong"):
    v = capture_attention_to_U(img_name, q)
    if v is not None:
        wrong_attn.append(v)

C = torch.stack(correct_attn, 0)  # (N, H)
W = torch.stack(wrong_attn, 0)    # (N, H)
mean_c = C.mean(0)  # (H,)
mean_w = W.mean(0)  # (H,)
M = (mean_c + EPS) / (mean_w + EPS)  # (H,) per-head amplification ratio
# threshold: per-layer scalar, head-averaged current attention
thr_c = mean_c.mean().item()  # scalar (avg over heads)
thr_w = mean_w.mean().item()
THRESHOLD = thr_w + BETA * (thr_c - thr_w)
print(f"  M (per head): mean={M.mean():.3f}, range=[{M.min():.3f},{M.max():.3f}]")
print(f"  threshold={THRESHOLD:.5f} (wrong_mean={thr_w:.5f}, correct_mean={thr_c:.5f})")
M_gpu = M.cuda()


# ─── Step 2: Install AdaIAT forward on LAYER ────────────────────────
state = {"q_pos": None, "prefill_done": False}  # q_pos tracks question-token key positions
orig_forward = attn_mod.forward


def adaiat_forward(hidden_states, position_embeddings, attention_mask,
                   past_key_values=None, **kwargs):
    is_prefill = not state["prefill_done"]
    if is_prefill:
        state["prefill_done"] = True

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attn_mod.head_dim)
    q = attn_mod.q_norm(attn_mod.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    k = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    v = attn_mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, attn_mod.layer_idx)
    k_attn = repeat_kv(k, attn_mod.num_key_value_groups)
    v_attn = repeat_kv(v, attn_mod.num_key_value_groups)

    attn_w = torch.matmul(q, k_attn.transpose(2, 3)) * attn_mod.scaling
    if attention_mask is not None:
        attn_w = attn_w + attention_mask[:, :, :, :k_attn.shape[-2]]
    attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)

    # ── AdaIAT-U: adaptively amplify last query row's attention to QUESTION ──
    if is_prefill and state["q_pos"] is not None:
        q_pos = state["q_pos"].to(attn_w.device)
        row = attn_w[:, :, -1:, :]  # (1, H, 1, Lk)
        a_q = row[..., q_pos]  # (1, H, 1, n_vis)
        current = a_q.mean().item()  # scalar, head+token averaged
        if current < THRESHOLD:
            # amplify: A_Q *= (1 + alpha * M[h])
            amp = 1.0 + ALPHA * M_gpu.view(1, H_heads, 1, 1).to(attn_w.dtype)
            a_q = a_q * amp
            row[..., q_pos] = a_q
            # renormalize the last row over all keys
            row_sum = row.sum(dim=-1, keepdim=True).clamp_min(EPS)
            attn_w[:, :, -1:, :] = row / row_sum

    out = torch.matmul(attn_w, v_attn)
    out = out.transpose(1, 2).contiguous().reshape(*input_shape, -1).contiguous()
    out = attn_mod.o_proj(out)
    return out, attn_w


attn_mod.forward = adaiat_forward
print("AdaIAT hook installed on layer", LAYER)


# ─── Step 3: POPE inference ─────────────────────────────────────────
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
    triggered = 0
    for q in tqdm(questions, desc=f"POPE {subset}"):
        img = Image.open(os.path.join(IMAGE_DIR, q["image"])).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": q["text"] + " Please answer yes or no."},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        # set runtime question token positions
        full_text = q["text"] + " Please answer yes or no."
        q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
        all_ids = inputs["input_ids"][0]
        q_t = torch.tensor(q_ids, device=all_ids.device)
        q_pos_r = []
        for s in range(len(all_ids) - len(q_ids) + 1):
            if (all_ids[s:s + len(q_ids)] == q_t).all():
                q_pos_r = list(range(s, s + len(q_ids)))
                break
        state["q_pos"] = torch.tensor(q_pos_r) if q_pos_r else None
        state["prefill_done"] = False

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=32)
        gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        results.append({"question": q["text"], "answer": answer_yes_no(raw), "raw_output": raw})
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(results)} to {output_file}")


# ─── Smoke test ─────────────────────────────────────────────────────
print("\nSmoke test (3 samples):")
qs = [json.loads(l) for l in open(os.path.join(POPE_DIR, "coco_pope_random.json"), encoding="utf-8")][:3]
for item in qs:
    img = Image.open(os.path.join(IMAGE_DIR, item["image"])).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": item["text"] + " Please answer yes or no."},
    ]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    state["q_pos"] = (inputs["mm_token_type_ids"][0] > 0).nonzero(as_tuple=True)[0]
    state["prefill_done"] = False
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=16)
    txt = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"  label={item['label']:3} ans={txt[:30]!r}")

print("\nRunning full POPE...")
for subset in ["random", "popular", "adversarial"]:
    run_subset(subset)

attn_mod.forward = orig_forward
print(f"\nDone. Next: python pope_evaluate.py {OUT_NAME}")
