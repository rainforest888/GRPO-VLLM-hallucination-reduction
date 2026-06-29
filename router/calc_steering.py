"""
calc_steering.py — Collect steering vectors for CASAL-style activation intervention.

CASAL computes a "steering vector": v = mean(incorrect_activations) - mean(correct_activations)
from the residual stream at a target layer's MLP output.  Adding -v * scale at inference
pushes activations toward the "correct" distribution.

We collect LM15 MLP output at the LAST query token position for 200 POPE adversarial
questions, then compute per-position steering vectors.

Output: casal_steering.pt saved to checkpoints/
"""
import json, os, sys, torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
N_COLLECT = 200
TARGET_LAYER = 15

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
for p in model.parameters(): p.requires_grad = False

# Hook to capture MLP output at target layer
captures = []  # list of (activation_tensor, is_correct_bool)

target_layer = model.model.language_model.layers[TARGET_LAYER]
orig_mlp_forward = target_layer.mlp.forward

def mlp_hook(hidden_states):
    # We capture the OUTPUT of MLP at the last query token
    result = orig_mlp_forward(hidden_states)
    # hidden_states is (B, L, D), capture last position
    captures.append(result[0, -1, :].detach().cpu())
    return result

target_layer.mlp.forward = mlp_hook

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

pope_file = os.path.join(POPE_DIR, "coco_pope_adversarial.json")
questions = [json.loads(l) for l in open(pope_file, encoding="utf-8")][:N_COLLECT]

correct_acts = []
wrong_acts = []

for q in tqdm(questions, desc="Collecting activations"):
    img_path = os.path.join(IMAGE_DIR, q["image"])
    text = q["text"]
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": text + " Please answer yes or no."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    captures.clear()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=8)
    gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    is_correct = (answer_yes_no(raw) == q["label"])

    # Last capture = last token at LM15 MLP output
    if captures:
        act = captures[-1]  # last position at this layer
        if is_correct:
            correct_acts.append(act)
        else:
            wrong_acts.append(act)

target_layer.mlp.forward = orig_mlp_forward  # restore

print(f"\nCollected: {len(correct_acts)} correct, {len(wrong_acts)} wrong")

if correct_acts and wrong_acts:
    mean_correct = torch.stack(correct_acts).mean(dim=0)
    mean_wrong = torch.stack(wrong_acts).mean(dim=0)
    steering = mean_wrong - mean_correct  # v = wrong - correct

    # Normalize
    steering = steering / steering.norm()

    torch.save({
        "steering": steering,
        "layer": TARGET_LAYER,
        "n_correct": len(correct_acts),
        "n_wrong": len(wrong_acts),
    }, os.path.join(CHECKPOINT_DIR, "casal_steering.pt"))
    print(f"Steering vector saved: norm={steering.norm().item():.4f}, "
          f"max={steering.abs().max().item():.4f}")
else:
    print("ERROR: Not enough samples to compute steering vector")
