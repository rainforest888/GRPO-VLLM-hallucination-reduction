"""Quick verification: alpha=0 hooks = baseline"""
import json, os, sys, torch
os.environ['TQDM_DISABLE'] = '1'

# Direct imports — no sys.path manipulation
MODEL_DIR = r"G:\sample\Qwen3vl"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
OFFSET_FILE = r"G:\sample\Qwen3vl\router_project\router\checkpoints\cai_offsets_v2.pt"

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
for p in model.parameters(): p.requires_grad = False
model.eval()
questions = [json.loads(l) for l in open(f"{POPE_DIR}/coco_pope_adversarial.json", encoding="utf-8")][:50]

# Baseline
correct = 0
for q in questions:
    img = Image.open(f"{IMAGE_DIR}/{q['image']}").convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q['text'] + " Please answer yes or no."}]}]
    inp = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    with torch.no_grad(): gen = model.generate(**inp, max_new_tokens=8)
    raw = processor.decode(gen[0, inp.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if answer_yes_no(raw) == q['label']: correct += 1
print(f"Baseline: {correct}/50 = {correct/50:.4f}")

# alpha=0 hooks on o_proj
if os.path.exists(OFFSET_FILE):
    data = torch.load(OFFSET_FILE, map_location="cpu", weights_only=False)
    offsets = {li: v.cuda().bfloat16() for li, v in data["offsets"].items()}
    handles = {}
    for li, steer_vec in offsets.items():
        s = steer_vec
        m = model.model.language_model.layers[li].self_attn.o_proj
        def make_hook(_s):
            def hook(module, input, output):
                return output  # alpha=0, no modification
            return hook
        handles[li] = m.register_forward_hook(make_hook(s))

    correct2 = 0
    for q in questions[:50]:
        img = Image.open(f"{IMAGE_DIR}/{q['image']}").convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q['text'] + " Please answer yes or no."}]}]
        inp = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad(): gen = model.generate(**inp, max_new_tokens=8)
        raw = processor.decode(gen[0, inp.input_ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if answer_yes_no(raw) == q['label']: correct2 += 1

    for h in handles.values(): h.remove()
    print(f"alpha=0 hooks: {correct2}/50 = {correct2/50:.4f}")
    print(f"Match: {correct == correct2}")
else:
    print("No offsets yet — needs calibration")
print("DONE")
