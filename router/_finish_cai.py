"""Quick completion: run CAI v2 popular+adversarial at alpha=0.02.
Standalone — no argparse imports from cai_bracs_v2."""
import json, os, torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image

MODEL_DIR = r"G:\sample\Qwen3vl"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
OUTDIR = r"G:\sample\Qwen3vl\router_project\pope_results\cai_bracs"
OFFSET_FILE = r"G:\sample\Qwen3vl\router_project\router\checkpoints\cai_offsets_v2.pt"
LM_LAYERS = list(range(5, 19))
ALPHA = 0.02

def answer_yes_no(t):
    t = t.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

print("Loading model...", flush=True)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
for p in model.parameters(): p.requires_grad = False
model.eval()

data = torch.load(OFFSET_FILE, map_location="cpu", weights_only=False)
offsets = {int(k) if isinstance(k, str) else k: v for k, v in data["offsets"].items()}
# Fix: keys might be ints or strings
offsets2 = {}
for k, v in data["offsets"].items():
    if isinstance(k, str):
        offsets2[int(k.split(".")[1])] = v
    else:
        offsets2[k] = v
offsets = offsets2

# Install hooks manually
handles = {}
for li in LM_LAYERS:
    if li not in offsets: continue
    s = offsets[li].cuda().bfloat16()
    a = ALPHA
    m = model.model.language_model.layers[li].self_attn.o_proj
    def make_hook(_s, _a):
        def hook(module, input, output):
            mod = output.clone()
            mod[:, -1:, :] = mod[:, -1:, :] + _a * _s.unsqueeze(0).unsqueeze(0)
            return mod
        return hook
    handles[li] = m.register_forward_hook(make_hook(s, a))
print(f"Installed {len(handles)} hooks", flush=True)

# Run popular + adversarial
for subset in ["popular", "adversarial"]:
    pope_file = f"{POPE_DIR}/coco_pope_{subset}.json"
    out_file = f"{OUTDIR}/coco_pope_{subset}_answers.json"
    questions = [json.loads(l) for l in open(pope_file, encoding="utf-8")]
    results = []
    for q in tqdm(questions, desc=subset):
        img = Image.open(f"{IMAGE_DIR}/{q['image']}").convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": q["text"] + " Please answer yes or no."},
        ]}]
        inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=8)
        raw = processor.decode(gen[0, inp.input_ids.shape[1]:],
                               skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.append({"question": q["text"], "answer": answer_yes_no(raw), "raw_output": raw})

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = [json.loads(l)["label"] for l in open(pope_file, encoding="utf-8")]
    correct = sum(1 for r, l in zip(results, labels) if r["answer"] == l)
    print(f"\n  {subset}: {correct}/{len(results)} = {correct/len(results):.4f}", flush=True)

# Cleanup
for h in handles.values():
    h.remove()
print("DONE", flush=True)
