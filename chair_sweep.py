"""3-region layer sweep: UAC on/off per region, 8 combos, 100 images each."""
import json, os, torch, argparse, random, itertools
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb
from PIL import Image
import torch.nn.functional as F

MODEL_DIR = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl"
IMAGE_DIR = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\val2014\val2014"
SEG_FILE = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\POPE-main\POPE-main\segmentation\coco_ground_truth_segmentation.json"
from pathlib import Path
OUT_DIR = Path(r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\router_project\pope_results\chair_sweep")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--n_images", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--alpha", type=float, default=0.77)
args = ap.parse_args()
random.seed(args.seed)

# 3 regions
REGIONS = {"early": range(2, 10), "mid": range(11, 20), "late": range(21, 31)}
ALPHA = args.alpha

# Load seg data
seg_data = [json.loads(l) for l in open(SEG_FILE, encoding="utf-8")]
seen = set(); unique_seg = []
for e in seg_data:
    if e["image"] not in seen: seen.add(e["image"]); unique_seg.append(e)
available = [e for e in unique_seg if os.path.exists(os.path.join(IMAGE_DIR, e["image"]))]
sample = random.sample(available, min(args.n_images, len(available)))
print(f"Sampled {len(sample)} images for all combos")

# Load model once
print("Loading Qwen3-VL-2B...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
LM_LAYERS = len(model.model.language_model.layers)
H = model.model.language_model.config.num_attention_heads
KV_H = model.model.language_model.config.num_key_value_heads

# Pre-calibrate UAC for each layer
W_norms = {}
for rname, rlayers in REGIONS.items():
    for L in rlayers:
        if L >= LM_LAYERS: continue
        attn_mod = model.model.language_model.layers[L].self_attn
        W_list = []
        def calib_hook(m, inp, out):
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                W_list.append(out[1][0,:,-1,:].detach().cpu())
        h = attn_mod.register_forward_hook(calib_hook)
        cal_imgs = random.sample(available, min(10, len(available)))
        for e in tqdm(cal_imgs, desc=f"Calib {rname} L{L}", leave=False):
            img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
            msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Describe this image."}]}]
            inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            with torch.no_grad(): model.generate(**inputs, max_new_tokens=2)
            torch.cuda.empty_cache()
        h.remove()
        target_len = W_list[0].shape[1]
        same_len = [w for w in W_list if w.shape[1] == target_len]
        W_cal = torch.stack(same_len).mean(0) if same_len else W_list[0]
        W = W_cal.float(); mean_W = W.mean() + 1e-8
        W_norms[(rname, L)] = mean_W / (W + 1e-8)
        print(f"  Calib {rname} L{L}: W shape={W.shape}")

def make_uac_fwd(attn_mod, W_norm):
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
        Lk = min(W_norm.shape[1], aw.shape[-1])
        w = W_norm[:,:Lk].to(aw.device, aw.dtype)
        log_w = torch.log(w.clamp_min(1e-6))
        corr = (1.0 + ALPHA * torch.tanh(log_w)).unsqueeze(0).unsqueeze(2)
        row = aw[:,:,-1:,:Lk] * corr[:,:H,:,:]
        aw[:,:,-1:,:Lk] = row / row.sum(dim=-1,keepdim=True).clamp_min(1e-8)
        out = torch.matmul(aw, v2)
        out = out.transpose(1,2).contiguous().reshape(bsz,q_len,-1)
        return attn_mod.o_proj(out), aw
    return fwd

# Enumerate combos: each region is 0 (off) or 1 (on)
results_all = {}
for mask in itertools.product([0, 1], repeat=3):
    combo_name = f"UAC_{'E' if mask[0] else 'e'}_{'M' if mask[1] else 'm'}_{'L' if mask[2] else 'l'}"
    print(f"\n=== Combo {combo_name} ===")
    
    # Install hooks on active regions
    hooks = []
    for i, (rname, rlayers) in enumerate(REGIONS.items()):
        if mask[i]:
            for L in rlayers:
                if (rname, L) not in W_norms: continue
                attn_mod = model.model.language_model.layers[L].self_attn
                orig = attn_mod.forward
                attn_mod.forward = make_uac_fwd(attn_mod, W_norms[(rname, L)])
                hooks.append((attn_mod, orig))
    
    results = []
    for e in tqdm(sample, desc=combo_name):
        img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
        msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Please describe this image in detail."}]}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64)
        raw = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        results.append({"image_id": e["image_id"], "image": e["image"], "caption": raw.strip()})
        torch.cuda.empty_cache()
    
    # Restore
    for attn_mod, orig in hooks:
        attn_mod.forward = orig
    
    out_file = OUT_DIR / f"{combo_name}.jsonl"
    with open(str(out_file), "w", encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert out_file.exists(), f"MISSING: {out_file}"
    results_all[combo_name] = str(out_file)
    print(f"  Saved {len(results)} to {out_file}")

# Print summary
print("\n=== All combos ===")
for name, path in results_all.items():
    print(f"  {name}: {path}")
print("Run: for f in pope_results/chair_sweep/*.jsonl; do python chair_evaluate.py $f; done")
