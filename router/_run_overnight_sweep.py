"""
_run_overnight_sweep.py — CAI-only param sweep for the overnight pipeline.

CAI-only (no BRACS barrier): just add scaled caption offset to last-token
attention output. Simpler, fewer params to sweep.

Usage: python _run_overnight_sweep.py --n 200
"""
import json, os, sys, torch, argparse
import torch.nn.functional as F
os.environ['TQDM_DISABLE'] = '1'
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
args = ap.parse_args()

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
OFFSET_FILE = os.path.join(CHECKPOINT_DIR, "cai_offsets.pt")
LM_LAYERS = list(range(5, 19))

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

print(f"Loading model + offsets ({args.n} questions)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
for p in model.parameters(): p.requires_grad = False
model.eval()

offsets_data = torch.load(OFFSET_FILE, map_location="cpu", weights_only=False)
offsets = offsets_data["offsets"]

questions = [json.loads(l) for l in open(
    f"{POPE_DIR}/coco_pope_adversarial.json", encoding="utf-8")][:args.n]

# Pre-load to avoid disk I/O in the sweep loop
all_data = []
for q in questions:
    img = Image.open(f"{IMAGE_DIR}/{q['image']}").convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": q['text'] + " Please answer yes or no."},
    ]}]
    inp = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    all_data.append((inp, q['label']))
print(f"Pre-loaded {len(all_data)}")

orig_forwards = {}
for li in LM_LAYERS:
    if li in offsets:
        orig_forwards[li] = model.model.language_model.layers[li].self_attn.forward

def run_one_alpha(alpha):
    """Install CAI hooks with given alpha, return accuracy."""
    for li, orig in orig_forwards.items():
        off = offsets[li].cuda().bfloat16()
        a = torch.tensor(alpha, dtype=torch.bfloat16, device=model.device)
        m = model.model.language_model.layers[li].self_attn

        def make(_orig, _m, _off, _a):
            def hook(hidden_states, position_embeddings, attention_mask,
                     past_key_values=None, **kw):
                is_pf = past_key_values is None or past_key_values.get_seq_length() == 0
                if not is_pf:
                    return _orig(hidden_states, position_embeddings, attention_mask,
                                 past_key_values=past_key_values, **kw)
                ishape = hidden_states.shape[:-1]
                hshape = (*ishape, -1, _m.head_dim)
                q = _m.q_norm(_m.q_proj(hidden_states).view(hshape)).transpose(1, 2)
                k = _m.k_norm(_m.k_proj(hidden_states).view(hshape)).transpose(1, 2)
                v = _m.v_proj(hidden_states).view(hshape).transpose(1, 2)
                c, s = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, c, s)
                if past_key_values is not None:
                    k, v = past_key_values.update(k, v, _m.layer_idx)
                ka = repeat_kv(k, _m.num_key_value_groups)
                va = repeat_kv(v, _m.num_key_value_groups)
                aw = torch.matmul(q, ka.transpose(2, 3)) * _m.scaling
                if attention_mask is not None:
                    aw = aw + attention_mask[:, :, :, :ka.shape[-2]]
                aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
                ao = torch.matmul(aw, va)  # (1, H, Lq, D_head)
                # CAI: add caption offset to last query position
                ao[:, :, -1:, :] = ao[:, :, -1:, :] + _a * _off.unsqueeze(0).unsqueeze(2)
                out = ao.transpose(1, 2).contiguous().reshape(*ishape, -1).contiguous()
                return _m.o_proj(out), aw
            return hook

        model.model.language_model.layers[li].self_attn.forward = make(orig, m, off, a)

    correct = 0
    for inp, label in all_data:
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=8)
        raw = processor.decode(gen[0, inp.input_ids.shape[1]:],
                               skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if answer_yes_no(raw) == label:
            correct += 1

    # Restore
    for li, orig in orig_forwards.items():
        model.model.language_model.layers[li].self_attn.forward = orig

    return correct / args.n

# Sweep
alphas = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
# Also test with normalization: divide offset by its norm
results = []
print(f"\nSweeping {len(alphas)} alpha values...")
for alpha in alphas:
    acc = run_one_alpha(alpha)
    delta = acc - 0.8730
    results.append((alpha, acc, delta))
    print(f"  α={alpha:.4f}: acc={acc:.4f} Δ={delta:+.4f}", flush=True)

# Also test with different layer subsets
# Only top-7 layers (largest offsets: 12-18)
for nl in [7, 10]:
    alpha = 0.01
    # Temporarily limit offsets to top-nl layers
    offset_norms = [(li, offsets[li].norm().item()) for li in offsets]
    offset_norms.sort(key=lambda x: -x[1])
    top_layers = set(l for l, _ in offset_norms[:nl])
    saved = dict(orig_forwards)
    orig_forwards = {li: saved[li] for li in saved if li in top_layers}
    acc = run_one_alpha(alpha)
    delta = acc - 0.8730
    results.append((f"α={alpha}, top{nl}layers", acc, delta))
    print(f"  α={alpha} top{nl}layers: acc={acc:.4f} Δ={delta:+.4f}", flush=True)
    orig_forwards = saved

# Best
results.sort(key=lambda x: -x[1])
best_acc, best_alpha, best_delta = results[0][1], results[0][0], results[0][2]
best_params = {"alpha": best_alpha, "acc": best_acc, "delta": best_delta}
print(f"\nBest: {best_params}", flush=True)

out_path = os.path.join(CHECKPOINT_DIR, "cai_sweep_best.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2)
print(f"Saved: {out_path}")
