"""Fast sweep of CAI+BRACS alpha/beta on 100 adversarial questions.
All combos run in a single pass — hooks reinstalled for each set of params."""
import json, os, sys, torch
import torch.nn.functional as F
os.environ['TQDM_DISABLE'] = '1'
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb
from PIL import Image

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
OFFSET_FILE = r"G:\sample\Qwen3vl\router_project\router\checkpoints\cai_offsets.pt"
LM_LAYERS = list(range(5, 19))

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
for p in model.parameters(): p.requires_grad = False
model.eval()

offsets_data = torch.load(OFFSET_FILE, map_location="cpu", weights_only=False)
offsets = offsets_data["offsets"]

questions = [json.loads(l) for l in open(f"{POPE_DIR}/coco_pope_adversarial.json", encoding="utf-8")][:100]

# Pre-load all inputs
all_inputs = []
for q in questions:
    img = Image.open(f"{IMAGE_DIR}/{q['image']}").convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": q['text'] + " Please answer yes or no."},
    ]}]
    inp = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        return_dict=True, return_tensors="pt").to(model.device)
    all_inputs.append((inp, q['label']))
print(f"Pre-loaded {len(all_inputs)} inputs")

orig_forwards = {}
for li in LM_LAYERS:
    if li in offsets:
        orig_forwards[li] = model.model.language_model.layers[li].self_attn.forward

def install_hooks(alpha, beta, barrier):
    for li, orig in orig_forwards.items():
        off = offsets[li].cuda().bfloat16()
        b = torch.tensor(beta, dtype=torch.bfloat16, device=model.device)
        barr = float(barrier)
        a = torch.tensor(alpha, dtype=torch.bfloat16, device=model.device)
        m = model.model.language_model.layers[li].self_attn
        def make(_orig, _m, _off, _b, _barr, _a):
            def hook(hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
                is_pf = past_key_values is None or past_key_values.get_seq_length() == 0
                if not is_pf:
                    if past_key_values is not None:
                        return _orig(hidden_states, position_embeddings, attention_mask,
                                     past_key_values=past_key_values, **kw)
                    else:
                        return _orig(hidden_states, position_embeddings, attention_mask, **kw)
        model.model.language_model.layers[li].self_attn.forward = make(orig, m, off, b, barr, a)

def uninstall_hooks():
    for li, orig in orig_forwards.items():
        model.model.language_model.layers[li].self_attn.forward = orig

# Sweep: focus on moderate alphas (offsets are large, so alpha must be small)
sweep_params = [
    (0.01, 1.0, 0.3), (0.02, 1.0, 0.3), (0.05, 1.0, 0.3),
    (0.01, 0.5, 0.3), (0.02, 2.0, 0.3), (0.005, 1.0, 0.3),
    (0.01, 1.0, 0.2), (0.01, 1.0, 0.4), (0.001, 1.0, 0.3),
    (0.0, 0.0, 0.0),  # baseline verification
]

print("\nα      β      bar   Acc     Δ")
print("-" * 40)
best = (0, 0, 0, 0)
for alpha, beta, barrier in sweep_params:
    install_hooks(alpha, beta, barrier)
    correct = 0
    for inp, label in all_inputs:
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=8)
        raw = processor.decode(gen[0, inp.input_ids.shape[1]:], skip_special_tokens=True,
                               clean_up_tokenization_spaces=False)
        if answer_yes_no(raw) == label: correct += 1
    acc = correct / 100
    delta = acc - 0.8730
    print(f"{alpha:.3f}  {beta:.1f}  {barrier:.2f}  {acc:.4f}  {delta:+.4f}")
    if acc > best[0]: best = (acc, alpha, beta, barrier)
    uninstall_hooks()

print(f"\nBest: α={best[1]} β={best[2]} bar={best[3]} acc={best[0]:.4f} Δ={best[0]-0.8730:+.4f}")
