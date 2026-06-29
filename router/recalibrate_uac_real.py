"""
recalibrate_uac_real.py — Recalibrate UAC W using real COCO images.

Groups images by processor output resolution (n_vis tokens), computes
per-head W = mean(A_real) / A_real for each resolution group separately.
No interpolation needed at inference — exact match to resolution.

Replaces W in calibration.pt (keeps existing AdaIAT-U M/thresholds).

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    python router/recalibrate_uac_real.py [--nimages N] [--seed S]
"""
import json, os, sys, random, argparse
from collections import defaultdict
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--nimages", type=int, default=100)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

MODEL_DIR = r"G:\sample\Qwen3vl"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
EPS = 1e-8
ACTIVE_LAYERS = list(range(5, 19))  # LM 5-18

print(f"Loading model (eager), calibrating on {args.nimages} real COCO images...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
lm = model.model.language_model.layers

# Pick random COCO images (not POPE; just any real images)
all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
random.seed(args.seed)
random.shuffle(all_images)
images = all_images[:args.nimages]
print(f"Selected {len(images)} images from {IMAGE_DIR}")

# Group images by n_vis
print("Grouping images by resolution...")
res_groups = defaultdict(list)  # n_vis → [(image_name, inputs_dict)]
for img_name in tqdm(images, desc="Grouping"):
    img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "Describe this image."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    tt = inputs["mm_token_type_ids"][0]
    n_vis = (tt > 0).sum().item()
    inputs = inputs.to(model.device)
    res_groups[n_vis].append((img_name, inputs))
print("Resolution groups:", {k: len(v) for k, v in sorted(res_groups.items())})

# Calibrate per layer, per resolution
W_new = {}

for li in tqdm(ACTIVE_LAYERS, desc="Layer"):
    name = f"lm.{li}"
    attn_mod = lm[li].self_attn
    W_res = {}  # n_vis → (H, n_vis) tensor

    for n_vis, group in res_groups.items():
        attn_maps = []  # list of (H, n_vis) tensors
        for img_name, inputs in tqdm(group, desc=f"L{li} nvis={n_vis}", leave=False):
            tt = inputs["mm_token_type_ids"][0]
            vis_idx = (tt > 0).nonzero(as_tuple=True)[0]

            cap = {}
            def hook(m, inp, out):
                if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
                    aw = out[1][0, :, -1, :]  # (H, Lk), last query row
                    cap["v"] = aw[:, vis_idx].detach().cpu()  # (H, n_vis)

            h = attn_mod.register_forward_hook(hook)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=2)
            h.remove()
            if "v" in cap:
                attn_maps.append(cap["v"])

        if not attn_maps:
            continue

        # A_real_mean = average over images for this resolution: (H, n_vis)
        stacked = torch.stack(attn_maps, dim=0)  # (N, H, n_vis)
        A_mean = stacked.mean(dim=0)  # (H, n_vis)
        # W = scalar_mean(A_mean) / A_mean
        scalar_mean = A_mean.mean()
        w = (scalar_mean + EPS) / (A_mean + EPS)
        W_res[n_vis] = w

    if W_res:
        W_new[name] = W_res
        total = sum(w.numel() for w in W_res.values())
        print(f"  L{li}: calibrated for {len(W_res)} resolutions, total params={total}")

# ─── Save ───────────────────────────────────────────────────────────
existing = torch.load(os.path.join(CHECKPOINT_DIR, "calibration.pt"), map_location="cpu", weights_only=False)
existing["W"] = W_new  # replace W, keep M + thresholds

save_path = os.path.join(CHECKPOINT_DIR, "calibration.pt")
torch.save(existing, save_path)
print(f"\n[OK] Updated calibration.pt with real-image UAC W ({len(W_new)} layers, {sum(len(v) for v in W_new.values())} resolution groups)")
print(f"Replaced W field; kept M/thresholds intact.")
