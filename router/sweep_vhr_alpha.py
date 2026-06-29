"""Test VHR with multiple alpha values on LM15, 100 questions each."""
import json, os, sys, torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"

ALPHAS = [0.1, 0.3, 0.5, 0.75, 1.0]
STRATEGY = "vhr"  # vhr only; uac_vhr gave same result
LAYER = 15
N_Q = 100

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
calib = torch.load(os.path.join(CHECKPOINT_DIR, "calibration.pt"), map_location="cpu", weights_only=False)

pope_file = os.path.join(POPE_DIR, "coco_pope_adversarial.json")
all_qs = [json.loads(l) for l in open(pope_file, encoding="utf-8")][:N_Q]
labels = [q["label"] for q in all_qs]

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

active_name = f"lm.{LAYER}"
strategies = ["none"]  # outer LM default
none_idx = 0

def test_alpha(alpha_init, strategy):
    """Run 100 questions with given alpha_init and return accuracy."""
    mgr = RouterManager(model, calib, active_layers={active_name}, alpha_init=alpha_init)
    mgr.wrap_all()
    mgr.mode = "force"

    full_strats = mgr._strategies_for(active_name)
    strat_idx = full_strats.index(strategy) if strategy in full_strats else 0
    none_idx_s = full_strats.index("none") if "none" in full_strats else 0

    correct = 0
    for q in tqdm(all_qs, desc=f"alpha={alpha_init}", leave=False):
        mgr.clear_cache()

        for d in mgr.descs:
            name = d["name"]
            s = d["strategies"]
            if name == active_name:
                idx = strat_idx if strat_idx < len(s) else 0
            else:
                idx = s.index("none") if "none" in s else 0
            mgr._decisions[name] = idx
            mgr._decided.add(name)

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

        mgr._current_n_vis = (inputs["mm_token_type_ids"][0] > 0).sum().item()
        full_text = text + " Please answer yes or no."
        q_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids
        all_ids = inputs["input_ids"][0]
        q_t = torch.tensor(q_ids, device=all_ids.device)
        for s in range(len(all_ids) - len(q_ids) + 1):
            if (all_ids[s:s+len(q_ids)] == q_t).all():
                mgr._current_q_pos = torch.arange(s, s + len(q_ids))
                break

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=16)
        gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        if answer_yes_no(raw) == q["label"]:
            correct += 1

    mgr.unwrap_all()
    return correct / N_Q

results = {}
for a in ALPHAS:
    acc = test_alpha(a, STRATEGY)
    results[a] = acc
    print(f"  alpha={a:.2f}: {acc:.4f} ({int(acc*N_Q)}/{N_Q})")

print(f"\n=== Summary: VHR LM15, {N_Q} questions ===")
print(f"Baseline: 0.8730 (3000q)")
for a, acc in sorted(results.items()):
    delta = acc - 0.8730
    print(f"  alpha={a:.2f}: {acc:.4f}  ({delta:+.4f})")
