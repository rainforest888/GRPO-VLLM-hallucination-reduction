"""
calibrate_vhr.py — Compute VHD (Vision-aware Head Divergence) per LM layer/head.

VHD_{l,h} = || A_{l,h}(with_vision) - A_{l,h}(zero_vision) ||_2

Uses 50 COCO images. The "without vision" condition zeroes out pixel_values
while keeping the identical token sequence (vision tokens become zero-embedding
placeholders). This avoids sequence-length mismatches from different image
resolutions.

Output: saved to calibration.pt under key "VHD" as dict {layer_name: (H,) tensor}
"""
import json, os, sys, torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb

MODEL_DIR = r"G:\sample\Qwen3vl"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
N_CALIB = 50
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Loading model (eager)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
for p in model.parameters(): p.requires_grad = False

image_files = sorted(
    [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
)[:N_CALIB]
print(f"Using {len(image_files)} COCO images for VHD calibration")

LM_LAYERS = list(range(0, 28))
attn_cache = {}   # layer_idx -> tensor

for i in LM_LAYERS:
    attn_mod = model.model.language_model.layers[i].self_attn
    orig = attn_mod.forward

    def make_hook(_i, _orig):
        def hook(hidden_states, position_embeddings, attention_mask,
                 past_key_values=None, **kw):
            is_prefill = past_key_values is None or past_key_values.get_seq_length() == 0
            if not is_prefill:
                return _orig(hidden_states, position_embeddings, attention_mask,
                           past_key_values=past_key_values, **kw)
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
            # Capture last-query-row attention: (H, Lk)
            attn_cache.setdefault(_i, []).append(aw[0, :, -1, :].detach().cpu())
            out = torch.matmul(aw, v_attn)
            out = out.transpose(1, 2).contiguous().reshape(*inp_shape, -1).contiguous()
            return m.o_proj(out), aw
        return hook

    attn_mod.forward = make_hook(i, orig)

print("Hooks installed on LM layers 0-27")

fixed_question = "Describe this image in detail."
vhd_all = {}  # layer_idx -> list of (H,) per-head distances

for img_file in tqdm(image_files, desc="VHD calibration"):
    img = Image.open(os.path.join(IMAGE_DIR, img_file)).convert("RGB")

    # Prepare inputs once
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": fixed_question},
    ]}]
    inp = processor.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    # ── Forward 1: normal vision ──
    attn_cache.clear()
    with torch.no_grad():
        _ = model(
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
            pixel_values=inp.get("pixel_values"),
            image_grid_thw=inp.get("image_grid_thw"),
            mm_token_type_ids=inp.get("mm_token_type_ids"),
            use_cache=False,
        )
    real_attn = {k: list(v) for k, v in attn_cache.items()}
    attn_cache.clear()

    # ── Forward 2: zeroed vision ──
    zero_pv = torch.zeros_like(inp.get("pixel_values", torch.zeros(1)))
    with torch.no_grad():
        _ = model(
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
            pixel_values=zero_pv,
            image_grid_thw=inp.get("image_grid_thw"),
            mm_token_type_ids=inp.get("mm_token_type_ids"),
            use_cache=False,
        )
    zero_attn = {k: list(v) for k, v in attn_cache.items()}
    attn_cache.clear()

    # ── Compute per-head VHD ──
    for li in LM_LAYERS:
        if li not in real_attn or li not in zero_attn:
            continue
        r_a = real_attn[li][-1]   # (H, Lk)
        z_a = zero_attn[li][-1]   # (H, Lk)
        diff = (r_a - z_a).pow(2).sum(dim=-1).sqrt()  # (H,)
        vhd_all.setdefault(li, []).append(diff)

# ── Aggregate ──
VHD_dict = {}
print("\nVHD per layer (mean ± std across 50 images):")
for li in sorted(vhd_all.keys()):
    stacked = torch.stack(vhd_all[li])   # (N, H)
    mean_vhd = stacked.mean(dim=0)       # (H,)
    VHD_dict[f"lm.{li}"] = mean_vhd
    top5_idx = mean_vhd.argsort(descending=True)[:5].tolist()
    top5_val = [f"{mean_vhd[h].item():.6f}" for h in top5_idx]
    print(f"  lm.{li:2d}: mean={mean_vhd.mean().item():.6f} ± {mean_vhd.std().item():.6f}  "
          f"top5 heads={top5_idx} vals={top5_val}")

# ── Save ──
calib_path = os.path.join(CHECKPOINT_DIR, "calibration.pt")
existing = torch.load(calib_path, map_location="cpu", weights_only=False)
existing["VHD"] = VHD_dict
torch.save(existing, calib_path)
print(f"\n[OK] VHD saved to calibration.pt (key: 'VHD', {len(VHD_dict)} layers)")
