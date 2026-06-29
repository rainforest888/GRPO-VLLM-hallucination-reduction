"""
calibration.py — Phase 0: Offline calibration for UAC and AdaIAT.

Uses RouterManager "collect" mode to capture attention weights from
all 52 layers for blank images (UAC) and baseline samples (AdaIAT).

Saves calibration.pt → {W, M, thresholds}
"""
import json, os, sys
from collections import defaultdict
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from router_module import RouterManager

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
BASELINE_DIR = r"G:\sample\Qwen3vl\router_project\pope_results\baseline"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print("Loading model (eager)...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
print("Model loaded.\n")

mgr = RouterManager(model, {"W": {}, "M": {}, "thresholds": {}})
mgr.wrap_all()
mgr.mode = "collect"


def run_one(img, question="Describe this image.", max_tokens=4):
    mgr.clear_cache()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": question},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=max_tokens)
    result = dict(mgr._collected)
    mgr._collected.clear()
    mgr._decided.clear()
    mgr._logits.clear()
    mgr._decisions.clear()
    torch.cuda.empty_cache()
    return result


# ═══ UAC ═══
print(f"\n{'='*60}\nUAC: blank images\n{'='*60}")
blank = Image.new("RGB", (448, 448), color=(0, 0, 0))
all_uac = defaultdict(list)

for _ in tqdm(range(5), desc="Blank passes"):
    maps = run_one(blank, max_tokens=4)
    for name, wlist in maps.items():
        for w in wlist:
            all_uac[name].append(w[0])  # w=(1,H,L,L) → (H,L,L)
    torch.cuda.empty_cache()

W = {}
for name in all_uac:
    # Keep only prefill attention (first call per pass), ignore decode steps
    maps = all_uac[name]
    # Use the first map from each pass (prefill step; shape (H, L, L))
    prefill_maps = [m for m in maps if m.dim() == 3 and m.shape[1] > 10]
    if not prefill_maps:
        print(f"  WARNING: No prefill maps for {name}")
        continue
    try:
        stacked = torch.stack(prefill_maps, dim=0)  # (N, H, L, L)
    except RuntimeError:
        # Different L across passes — use mean of each then average
        stacked = torch.stack([m.mean(dim=0) for m in prefill_maps], dim=0)
        # This gives (N, H, L, L) if same L, but if different, pad to max
        max_l = max(m.shape[-1] for m in prefill_maps)
        padded = []
        for m in prefill_maps:
            if m.shape[-1] < max_l:
                p = torch.zeros(m.shape[0], m.shape[1], max_l)
                p[:, :, :m.shape[-1]] = m
                padded.append(p)
            else:
                padded.append(m)
        stacked = torch.stack(padded, dim=0)
    mean_h = stacked.mean(dim=0)                  # (H, L, L)
    eps = 1e-8
    head_means = mean_h.mean(dim=[1, 2], keepdim=True)  # (H, 1, 1)
    W[name] = (head_means + eps) / (mean_h + eps)
print(f"W: {len(W)} layers")


# ═══ AdaIAT ═══
print(f"\n{'='*60}\nAdaIAT: baseline correct/wrong samples\n{'='*60}")
subsets = ["random", "popular", "adversarial"]
correct, wrong = [], []
for s in subsets:
    a = [json.loads(l) for l in open(f"{BASELINE_DIR}/coco_pope_{s}_answers.json", encoding="utf-8")]
    b = [json.loads(l) for l in open(f"{POPE_DIR}/coco_pope_{s}.json", encoding="utf-8")]
    for ai, bi in zip(a, b):
        (correct if ai["answer"] == bi["label"] else wrong).append((bi["image"], bi["text"]))
print(f"Baseline: {len(correct)} correct, {len(wrong)} wrong")

def collect_atp(questions, label, max_n=100):
    acc = defaultdict(list)
    for i in tqdm(range(min(len(questions), max_n)), desc=label):
        img_name, q_text = questions[i]
        img = Image.open(os.path.join(IMAGE_DIR, img_name)).convert("RGB")
        maps = run_one(img, q_text + " Please answer yes or no.")
        for name, wlist in maps.items():
            for w in wlist:
                # Atp: mean attention from last row to all columns
                atp = w[0, :, -1, :].mean(dim=-1)  # (H,)
                acc[name].append(atp)
        torch.cuda.empty_cache()
    return acc

atp_c = collect_atp(correct, "Correct", 100)
atp_w = collect_atp(wrong, "Wrong", 100)

M = {}
thresholds = {}
for name in set(atp_c) | set(atp_w):
    mc = atp_c.get(name, [])
    mw = atp_w.get(name, [])
    if not mc or not mw:
        continue
    sc, sw = torch.stack(mc, 0), torch.stack(mw, 0)  # (N, H)
    mean_c, mean_w = sc.mean(0), sw.mean(0)
    M[name] = mean_c / (mean_w + 1e-8)  # (H,)
    per_sample = sw.mean(1)  # (N,) avg over heads
    thresholds[name] = (per_sample.mean() + 0.5 * per_sample.std()).item()

print(f"M: {len(M)} layers, thresholds: {len(thresholds)}")


# ═══ Save ═══
save_path = os.path.join(CHECKPOINT_DIR, "calibration.pt")
torch.save({"W": W, "M": M, "thresholds": thresholds}, save_path)
print(f"\n[OK] calibration.pt saved to {save_path}")

mgr.unwrap_all()
