"""Paper-DoLa (2309.03883) on Qwen3 CHAIR, 500 images.
DoLa: contrast early-exit logits vs final logits. Subtract language prior at first token only."""
import json, os, torch, random
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image
from pathlib import Path

MODEL_DIR = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl"
IMAGE_DIR = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\val2014\val2014"
SEG_FILE = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\POPE-main\POPE-main\segmentation\coco_ground_truth_segmentation.json"
OUT_DIR = Path(r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\router_project\pope_results\chair_strict")
OUT_DIR.mkdir(parents=True, exist_ok=True)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--n_images", type=int, default=500)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)

seg_data = [json.loads(l) for l in open(SEG_FILE, encoding="utf-8")]
seen = set(); unique_seg = []
for e in seg_data:
    if e["image"] not in seen: seen.add(e["image"]); unique_seg.append(e)
available = [e for e in unique_seg if os.path.exists(os.path.join(IMAGE_DIR, e["image"]))]
sample = random.sample(available, min(args.n_images, len(available)))
print(f"DoLa: {len(sample)} images")

print("Loading Qwen3-VL-2B...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
LM_LAYERS = len(model.model.language_model.layers)
SHALLOW_L = min(8, LM_LAYERS - 2)  # early layer
DEEP_L = LM_LAYERS - 1              # final layer
print(f"DoLa: shallow=L{SHALLOW_L}, deep=L{DEEP_L}")

# Paper-DoLa: prefill forward with output_hidden_states,
# then use early-exit logit bias at first token only
results = []
for e in tqdm(sample, desc="Paper-DoLa"):
    img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"Please describe this image in detail."}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)

    # Prefill forward: capture early and late hidden states
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    shallow_h = out.hidden_states[SHALLOW_L][0, -1:, :]
    deep_h = out.hidden_states[DEEP_L][0, -1:, :]
    # DoLa bias: -alpha * (early_logits - late_logits) = +alpha * (late - early)
    dola_bias = 0.5 * (model.lm_head(deep_h) - model.lm_head(shallow_h)).squeeze(0)  # (V,)

    class DoLaLP:
        def __init__(s, bias): s.bias = bias
        def __call__(s, ids, sc):
            if ids.shape[1] - 1 == inputs["input_ids"].shape[1]:  # first new token
                return sc + s.bias.to(sc.device, sc.dtype)
            return sc

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, logits_processor=[DoLaLP(dola_bias)])
    raw = processor.decode(gen[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    results.append({"image_id": e["image_id"], "image": e["image"], "caption": raw.strip()})
    torch.cuda.empty_cache()

out = OUT_DIR / "chair_paper_dola.jsonl"
with open(str(out), "w", encoding="utf-8") as f:
    for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Paper-DoLa saved {len(results)} to {out}")
