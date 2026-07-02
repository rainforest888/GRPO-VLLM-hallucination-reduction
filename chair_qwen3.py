"""Qwen3-VL-2B CHAIR baseline + UAC strategy."""
import json, os, torch, argparse, random
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch.nn.functional as F

MODEL_DIR = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl"
BASE = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction"
IMAGE_DIR = os.path.join(BASE, r"Qwen3vl\val2014\val2014")
SEG_FILE = os.path.join(BASE, r"Qwen3vl\POPE-main\POPE-main\segmentation\coco_ground_truth_segmentation.json")

ap = argparse.ArgumentParser()
ap.add_argument("--n_images", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--strategy", type=str, default="none", choices=["none","uac","dola","cad"])
ap.add_argument("--outdir", type=str, default=None)
ap.add_argument("--layer", type=int, default=15)
ap.add_argument("--alpha", type=float, default=0.77)
args = ap.parse_args()

random.seed(args.seed)
outdir_name = args.outdir or f"chair_{args.strategy}"
OUT_DIR = os.path.join(os.path.dirname(__file__), "pope_results", outdir_name)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Qwen3 CHAIR strategy={args.strategy} layer={args.layer}")

# Load seg data
seg_data = [json.loads(l) for l in open(SEG_FILE, encoding="utf-8")]
seen = set(); unique_seg = []
for e in seg_data:
    if e["image"] not in seen: seen.add(e["image"]); unique_seg.append(e)
available = [e for e in unique_seg if os.path.exists(os.path.join(IMAGE_DIR, e["image"]))]
sample = random.sample(available, min(args.n_images, len(available)))
print(f"Sampled {len(sample)} images")

# Load Qwen3-VL
print("Loading Qwen3-VL-2B (bfloat16)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
print(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# Qwen3 attention path: model.model.language_model.layers[N].self_attn
LM_LAYERS = len(model.model.language_model.layers)
H = model.model.language_model.config.num_attention_heads
KV_H = model.model.language_model.config.num_key_value_heads
print(f"LM layers: {LM_LAYERS}, Q heads: {H}, KV heads: {KV_H}")

# UAC calibration
if args.strategy == "uac":
    attn_mod = model.model.language_model.layers[args.layer].self_attn
    W_list = []
    def uac_hook(m, inp, out):
        if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
            W_list.append(out[1][0,:,-1,:].detach().cpu())
    h = attn_mod.register_forward_hook(uac_hook)
    cal_imgs = random.sample(available, min(20, len(available)))
    print(f"UAC calib on {len(cal_imgs)} images...")
    for e in tqdm(cal_imgs):
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
    W_norm = mean_W / (W + 1e-8)
    print(f"UAC calibrated: shape={W.shape}")

# Generate
results = []
for e in tqdm(sample, desc=f"CHAIR {args.strategy}"):
    img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Please describe this image in detail."}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)

    if args.strategy == "uac":
        attn_mod = model.model.language_model.layers[args.layer].self_attn
        orig_fwd = attn_mod.forward
        def uac_fwd(hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
            from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb
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
            corr = (1.0 + args.alpha * torch.tanh(log_w)).unsqueeze(0).unsqueeze(2)
            row = aw[:,:,-1:,:Lk] * corr[:,:H,:,:]
            aw[:,:,-1:,:Lk] = row / row.sum(dim=-1,keepdim=True).clamp_min(1e-8)
            out = torch.matmul(aw, v2)
            out = out.transpose(1,2).contiguous().reshape(bsz,q_len,-1)
            return attn_mod.o_proj(out), aw
        attn_mod.forward = uac_fwd

    if args.strategy == "dola":
        # DoLa: contrast early (L8) vs late (L27) layer logits
        deep_layer = LM_LAYERS - 1
        shallow_layer = min(8, LM_LAYERS - 2)
        class DoLaLP:
            def __init__(s, m, sl, dl, a): s.m, s.sl, s.dl, s.a = m, sl, dl, a
            def __call__(s, ids, sc):
                # This is per-step but we need hidden states — use pre-captured from prefill
                return sc  # placeholder — actual in forward hook
        # Real DoLa: run forward with output_hidden_states, subtract shallow from deep
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        # Get logits from shallow layer hidden state via lm_head
        shallow_hidden = out.hidden_states[shallow_layer][0, -1:, :]  # last token
        deep_hidden = out.hidden_states[deep_layer][0, -1:, :]
        dola_bias = -args.alpha * (model.lm_head(shallow_hidden) - model.lm_head(deep_hidden)).squeeze(0)
        class DoLaLP:
            def __init__(s, bias): s.bias = bias
            def __call__(s, ids, sc):
                pos = ids.shape[1] - 1
                if pos == 0: return sc + s.bias.to(sc.device, sc.dtype)
                return sc
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=64, logits_processor=[DoLaLP(dola_bias)])
    
    elif args.strategy == "cad":
        # CAD: amplify heads with high visual attention diff (with vs without image)
        # No-image forward to capture attention baseline
        noimg_inputs = {k: v.clone() for k, v in inputs.items()}
        noimg_inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
        with torch.no_grad():
            out_noimg = model(**noimg_inputs, output_attentions=True)
        noimg_attn = out_noimg.attentions[args.layer][0,:,-1,:576].cpu().float()  # (H, 576) visual tokens
        with torch.no_grad():
            out_img = model(**inputs, output_attentions=True)
        img_attn = out_img.attentions[args.layer][0,:,-1,:576].cpu().float()
        diff = (img_attn - noimg_attn).abs().mean(dim=1)  # (H,) per-head visual sensitivity
        cad_weights = 1.0 + args.alpha * (diff / diff.max().clamp_min(1e-8))  # higher weight to visual heads
        cad_weights = cad_weights.to(model.device).to(torch.bfloat16)
        # During generate: apply CAD weight to attention module
        attn_mod = model.model.language_model.layers[args.layer].self_attn
        orig_fwd = attn_mod.forward
        from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb
        def cad_fwd(hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
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
            # CAD: amplify visual heads
            w = cad_weights.view(1, H, 1, 1)
            aw = aw * w
            aw = aw / aw.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            out = torch.matmul(aw, v2)
            out = out.transpose(1,2).contiguous().reshape(bsz,q_len,-1)
            return attn_mod.o_proj(out), aw
        attn_mod.forward = cad_fwd
    
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64)
    raw = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    results.append({"image_id": e["image_id"], "image": e["image"], "caption": raw.strip()})
    
    if args.strategy in ("uac", "cad"): attn_mod.forward = orig_fwd
    torch.cuda.empty_cache()

out_file = os.path.join(OUT_DIR, "captions.jsonl")
with open(out_file, "w", encoding="utf-8") as f:
    for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved {len(results)} to {out_file}")
