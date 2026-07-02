"""Strict paper-implementation methods on Qwen3-VL-2B CHAIR, 100 images each."""
import json, os, torch, argparse, random
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb
from PIL import Image
import torch.nn.functional as F
from pathlib import Path

MODEL_DIR = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl"
IMAGE_DIR = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\val2014\val2014"
SEG_FILE = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\POPE-main\POPE-main\segmentation\coco_ground_truth_segmentation.json"
OUT_DIR = Path(r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\router_project\pope_results\chair_strict")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--n_images", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)

# Load data
seg_data = [json.loads(l) for l in open(SEG_FILE, encoding="utf-8")]
seen = set(); unique_seg = []
for e in seg_data:
    if e["image"] not in seen: seen.add(e["image"]); unique_seg.append(e)
available = [e for e in unique_seg if os.path.exists(os.path.join(IMAGE_DIR, e["image"]))]
sample = random.sample(available, min(args.n_images, len(available)))
print(f"Sampled {len(sample)} images (shared across methods)")

# Load model
print("Loading Qwen3-VL-2B...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
H = model.model.language_model.config.num_attention_heads
KV_H = model.model.language_model.config.num_key_value_heads
LM_LAYERS = len(model.model.language_model.layers)
print(f"LM: {LM_LAYERS} layers, Q_heads={H}, KV_heads={KV_H}")

# ──────────────────────────────────────────────────────────────
# METHOD: Paper-DAC
# Paper: 2502.01969 "Attention Calibration"
# ──────────────────────────────────────────────────────────────
print("\n=== Paper-DAC (2502.01969) ===")
# Step 1: Get blank-image attention bias at each layer (L5-L25)
BLANK_BIAS = {}
calib_layers = range(5, min(26, LM_LAYERS))
print(f"Calibrating layers {calib_layers.start}-{calib_layers.stop-1} on blank images...")
for L in tqdm(calib_layers, desc="DAC calib"):
    attn_mod = model.model.language_model.layers[L].self_attn
    bias_list = []
    def blank_hook(m, inp, out):
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            bias_list.append(out[1][0, :, -1, :].detach().cpu())  # (H, Lk)
    h = attn_mod.register_forward_hook(blank_hook)
    for e in random.sample(available, min(10, len(available))):
        img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
        msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Describe this image."}]}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        # BLANK image: zero pixel_values
        inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        with torch.no_grad(): model.generate(**inputs, max_new_tokens=2)
        torch.cuda.empty_cache()
    h.remove()
    target_len = bias_list[0].shape[1]
    same_len = [b for b in bias_list if b.shape[1] == target_len]
    BLANK_BIAS[L] = torch.stack(same_len).mean(0) if same_len else bias_list[0]
    print(f"  L{L}: bias shape={BLANK_BIAS[L].shape}, range=[{BLANK_BIAS[L].min():.4f},{BLANK_BIAS[L].max():.4f}]")

# Step 2: DAC forward — subtract blank bias from attention
def make_dac_fwd(attn_mod, blank_bias, L, alpha=0.5):
    def fwd(hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
        bsz, q_len, _ = hidden_states.size()
        hdim = attn_mod.head_dim
        q = attn_mod.q_norm(attn_mod.q_proj(hidden_states).view(bsz,q_len,H,hdim)).transpose(1,2)
        k = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(bsz,q_len,KV_H,hdim)).transpose(1,2)
        v = attn_mod.v_proj(hidden_states).view(bsz,q_len,KV_H,hdim).transpose(1,2)
        cos, sin = position_embeddings; q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None: k, v = past_key_values.update(k, v, attn_mod.layer_idx)
        k2 = repeat_kv(k, attn_mod.num_key_value_groups); v2 = repeat_kv(v, attn_mod.num_key_value_groups)
        aw = torch.matmul(q, k2.transpose(-2,-1)) * attn_mod.scaling
        if attention_mask is not None: aw = aw + attention_mask[:,:,:,:k2.shape[-2]]
        aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
        # Paper-DAC: subtract blank-image position bias from attention (last query row, all key positions)
        Lk = min(blank_bias.shape[1], aw.shape[-1])
        bias = blank_bias[:, :Lk].to(aw.device, aw.dtype).unsqueeze(0).unsqueeze(2)  # (1,H,1,Lk)
        aw[:,:,-1:,:Lk] = aw[:,:,-1:,:Lk] - alpha * bias
        aw[:,:,-1:,:Lk] = torch.clamp(aw[:,:,-1:,:Lk], min=0)  # keep non-negative
        aw[:,:,-1:,:Lk] = aw[:,:,-1:,:Lk] / aw[:,:,-1:,:Lk].sum(dim=-1, keepdim=True).clamp_min(1e-8)
        out = torch.matmul(aw, v2)
        out = out.transpose(1,2).contiguous().reshape(bsz,q_len,-1)
        return attn_mod.o_proj(out), aw
    return fwd

# Run paper-DAC
results_dac = []
hooks = []
for L in BLANK_BIAS:
    attn_mod = model.model.language_model.layers[L].self_attn
    orig = attn_mod.forward
    attn_mod.forward = make_dac_fwd(attn_mod, BLANK_BIAS[L], L, alpha=0.5)
    hooks.append((attn_mod, orig))

for e in tqdm(sample, desc="Paper-DAC"):
    img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Please describe this image in detail."}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad(): gen = model.generate(**inputs, max_new_tokens=64)
    raw = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    results_dac.append({"image_id": e["image_id"], "image": e["image"], "caption": raw.strip()})
    torch.cuda.empty_cache()

for attn_mod, orig in hooks: attn_mod.forward = orig

out = OUT_DIR / "chair_paper_dac.jsonl"
with open(str(out), "w", encoding="utf-8") as f:
    for r in results_dac: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Paper-DAC saved {len(results_dac)} to {out}")

# ──────────────────────────────────────────────────────────────
# METHOD: Paper-VCD (2311.16922)
# ──────────────────────────────────────────────────────────────
print("\n=== Paper-VCD (2311.16922) ===")
# One extra forward with zero pixel_values per sample, LogitsProcessor
class VCDLP:
    def __init__(s, nl, g): s.nl, s.g = nl, g
    def __call__(s, ids, sc):
        p = ids.shape[1]-1
        if p < s.nl.shape[0]: sc = (1+s.g)*sc - s.g*s.nl[p].to(sc.device,sc.dtype)
        return sc

results_vcd = []
for e in tqdm(sample, desc="Paper-VCD"):
    img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Please describe this image in detail."}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    noimg_inputs = {k: v.clone() for k, v in inputs.items()}
    noimg_inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
    with torch.no_grad(): out_noimg = model(**noimg_inputs)
    nl = out_noimg.logits[0].clone()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, logits_processor=[VCDLP(nl, 1.0)])
    raw = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    results_vcd.append({"image_id": e["image_id"], "image": e["image"], "caption": raw.strip()})
    torch.cuda.empty_cache()

out = OUT_DIR / "chair_paper_vcd.jsonl"
with open(str(out), "w", encoding="utf-8") as f:
    for r in results_vcd: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Paper-VCD saved {len(results_vcd)} to {out}")

print("\nAll done. Evaluate each: python chair_evaluate.py chair_strict/chair_paper_dac.jsonl")
