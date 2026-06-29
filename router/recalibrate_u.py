"""
Recalibrate M and thresholds for AdaIAT-U (question token target)
across all active LM layers 5-18.

Usage: source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
       python router/recalibrate_u.py
"""
import json, os, sys, torch
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
BASELINE_DIR = r"G:\sample\Qwen3vl\router_project\pope_results\baseline"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
ACTIVE_LAYERS = list(range(5, 19))  # LM 5-18
NCALIB = 80

print("Loading model (eager)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
lm = model.model.language_model.layers

# Load baseline correct/wrong
correct, wrong = [], []
for s in ["random", "popular", "adversarial"]:
    a = [json.loads(l) for l in open(f"{BASELINE_DIR}/coco_pope_{s}_answers.json", encoding="utf-8")]
    b = [json.loads(l) for l in open(f"{POPE_DIR}/coco_pope_{s}.json", encoding="utf-8")]
    for ai, bi in zip(a, b):
        (correct if ai["answer"] == bi["label"] else wrong).append((bi["image"], bi["text"]))
print(f"Baseline: {len(correct)} correct, {len(wrong)} wrong")

def capture_at_layer(layer_idx, image_name, question):
    """Capture attention from last query to question tokens at a specific LM layer."""
    img = Image.open(os.path.join(IMAGE_DIR, image_name)).convert("RGB")
    full_text = question + " Please answer yes or no."
    q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": full_text},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    # Locate question tokens in sequence
    all_ids = inputs["input_ids"][0]
    q_tensor = torch.tensor(q_ids, device=all_ids.device)
    q_pos = []
    for s in range(len(all_ids) - len(q_ids) + 1):
        if (all_ids[s:s + len(q_ids)] == q_tensor).all():
            q_pos = list(range(s, s + len(q_ids)))
            break
    if not q_pos:
        return None

    cap = {}
    def hook(m, inp, out):
        if isinstance(out, tuple) and len(out) == 2 and out[1] is not None:
            aw = out[1][0, :, -1, :]  # (H, Lk)
            q_idx = torch.tensor(q_pos, device=aw.device, dtype=torch.long)
            cap["u"] = aw[:, q_idx].mean(dim=-1).detach().cpu()  # (H,)
    h = lm[layer_idx].self_attn.register_forward_hook(hook)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=2)
    h.remove()
    return cap.get("u", None)

# Existing calibration (keep W)
existing = torch.load(os.path.join(CHECKPOINT_DIR, "calibration.pt"), map_location="cpu", weights_only=False)
M_U = dict(existing.get("M", {}))
thresholds_U = dict(existing.get("thresholds", {}))

print(f"\nRecalibrating AdaIAT-U M/threshold for layers {ACTIVE_LAYERS[0]}-{ACTIVE_LAYERS[-1]}...")
for li in tqdm(ACTIVE_LAYERS, desc="Layer"):
    name = f"lm.{li}"
    c_attn, w_attn = [], []
    for img_name, q in tqdm(correct[:NCALIB], desc=f"L{li} correct", leave=False):
        v = capture_at_layer(li, img_name, q)
        if v is not None: c_attn.append(v)
    for img_name, q in tqdm(wrong[:NCALIB], desc=f"L{li} wrong", leave=False):
        v = capture_at_layer(li, img_name, q)
        if v is not None: w_attn.append(v)
    if not c_attn or not w_attn:
        print(f"  Skip layer {li}: no data")
        continue
    C = torch.stack(c_attn, 0)  # (N, H)
    W = torch.stack(w_attn, 0)
    mc, mw = C.mean(0), W.mean(0)
    M_U[name] = (mc + 1e-8) / (mw + 1e-8)
    per_sample = W.mean(1)  # (N,)
    thresholds_U[name] = (per_sample.mean() + 0.5 * per_sample.std()).item()
    print(f"  L{li}: M_mean={M_U[name].mean():.4f}, M>1 heads={(M_U[name]>1).sum().item()}/16")

save_path = os.path.join(CHECKPOINT_DIR, "calibration.pt")
torch.save({"W": existing["W"], "M": M_U, "thresholds": thresholds_U}, save_path)
print(f"\n[OK] Updated calibration saved with AdaIAT-U M/thresholds to {save_path}")
