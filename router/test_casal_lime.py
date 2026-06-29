"""
test_casal_lime.py — Evaluate CASAL (steering vector) and LIME (attention rebalance)
on 100 POPE adversarial questions.

Usage: python test_casal_lime.py --method {casal,lime} --scale {0.1,0.5,1.0,2.0}
"""
import json, os, sys, torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"
LAYER = 15
N_Q = 100

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--method", type=str, required=True, choices=["casal", "lime"])
ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--n", type=int, default=N_Q)
args = ap.parse_args()

METHOD = args.method
SCALE = args.scale
N_Q = args.n

OUTDIR = f"{METHOD}_L{LAYER}_s{SCALE}"
OUTPUT_DIR = os.path.join(RESULTS_BASE, OUTDIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Method={METHOD}  Layer=LM.{LAYER}  Scale={SCALE}  N={N_Q}")

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
for p in model.parameters(): p.requires_grad = False

# ─── Load steering vector ───
steering_path = os.path.join(CHECKPOINT_DIR, "casal_steering.pt")
have_steering = os.path.exists(steering_path)
if have_steering:
    steering_data = torch.load(steering_path, map_location="cpu", weights_only=False)
    steering_vec = steering_data["steering"].to(model.device).to(torch.bfloat16)
    print(f"Loaded steering vector: layer={steering_data['layer']}, "
          f"n_correct={steering_data['n_correct']}, n_wrong={steering_data['n_wrong']}")
else:
    print("WARNING: steering vector not found, CASAL will be neutral")
    steering_vec = None

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

# ─── Install hooks ───
if METHOD == "casal":
    # CASAL: subtract scaled steering vector from MLP output
    target_layer = model.model.language_model.layers[LAYER].mlp
    orig_mlp = target_layer.forward
    def casal_mlp_forward(hidden_states):
        out = orig_mlp(hidden_states)
        if steering_vec is not None:
            # Inject steering: push activations toward "correct" distribution
            out = out - SCALE * steering_vec.to(dtype=out.dtype)
        return out
    target_layer.forward = casal_mlp_forward
    print(f"CASAL hook installed on LM{LAYER} MLP")
elif METHOD == "lime":
    # LIME-inspired: rebalance last query row attention weights
    # Give MORE weight to vision token keys, LESS to text token keys
    attn_mod = model.model.language_model.layers[LAYER].self_attn
    orig_attn = attn_mod.forward
    state = {"prefill_done": False}
    def lime_forward(hidden_states, position_embeddings, attention_mask,
                     past_key_values=None, **kw):
        is_prefill = not state["prefill_done"]
        if is_prefill: state["prefill_done"] = True
        m = attn_mod
        inp_shape = hidden_states.shape[:-1]
        hid_shape = (*inp_shape, -1, m.head_dim)
        q = m.q_norm(m.q_proj(hidden_states).view(hid_shape)).transpose(1, 2)
        k = m.k_norm(m.k_proj(hidden_states).view(hid_shape)).transpose(1, 2)
        v = m.v_proj(hidden_states).view(hid_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, m.layer_idx)
        k_attn = repeat_kv(k, m.num_key_value_groups)
        v_attn = repeat_kv(v, m.num_key_value_groups)
        aw = torch.matmul(q, k_attn.transpose(2, 3)) * m.scaling
        if attention_mask is not None:
            aw = aw + attention_mask[:, :, :, :k_attn.shape[-2]]
        aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        # LIME rebalance: find vision token keys and boost their attention
        # mm_token_type_ids are in inputs, but not available here directly.
        # Instead, use a simple heuristic: the first ~N tokens are vision tokens.
        # We detect n_vis from the hook state set by the test loop.
        if is_prefill:
            n_vis = getattr(m, '_lime_n_vis', None)
            if n_vis and n_vis > 0:
                row = aw[:, :, -1:, :]  # (1, H, 1, Lk)
                vis_avg = row[:, :, :, :n_vis].mean(dim=-1, keepdim=True)  # (1,H,1,1)
                txt_avg = row[:, :, :, n_vis:].mean(dim=-1, keepdim=True)
                # Boost vision, suppress text
                row_vis = row[:, :, :, :n_vis] * (1.0 + SCALE / vis_avg.clamp_min(1e-8))
                row_txt = row[:, :, :, n_vis:] * (1.0 - SCALE * 0.5 / txt_avg.clamp_min(1e-8))
                row_new = torch.cat([row_vis, row_txt], dim=-1)
                aw[:, :, -1:, :] = row_new / row_new.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        out = torch.matmul(aw, v_attn)
        out = out.transpose(1, 2).contiguous().reshape(*inp_shape, -1).contiguous()
        return m.o_proj(out), aw
    attn_mod.forward = lime_forward
    print(f"LIME hook installed on LM{LAYER} attention")

# ─── Run evaluation ───
pope_file = os.path.join(POPE_DIR, "coco_pope_adversarial.json")
questions = [json.loads(l) for l in open(pope_file, encoding="utf-8")][:N_Q]
labels = [q["label"] for q in questions]
results = []
correct = 0

for i, q in enumerate(tqdm(questions, desc=f"{METHOD} L{LAYER} s{SCALE}")):
    img_path = os.path.join(IMAGE_DIR, q["image"])
    text = q["text"]

    # Reset LIME state
    if METHOD == "lime":
        state["prefill_done"] = False

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": text + " Please answer yes or no."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    # For LIME: set n_vis on the attention module
    if METHOD == "lime":
        n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()
        attn_mod._lime_n_vis = n_vis

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=8)
    gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    ans = answer_yes_no(raw)
    results.append({"question": text, "answer": ans, "raw_output": raw})
    if ans == labels[i]: correct += 1

acc = correct / N_Q
print(f"\n{METHOD} L{LAYER} scale={SCALE}: {correct}/{N_Q} = {acc:.4f}")

# Save
out_file = os.path.join(OUTPUT_DIR, "coco_pope_adversarial_answers.json")
with open(out_file, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved to {out_file}")

# Quick compare
print(f"\nBaseline adversarial (3000): 0.8730")
print(f"{METHOD} s={SCALE} ({N_Q}): {acc:.4f}  ({acc-0.8730:+.4f})")
