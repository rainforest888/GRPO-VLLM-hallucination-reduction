"""Quick evaluation of champion oracle combo on all 3 POPE subsets."""
import json, os, sys, torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from router_module import RouterManager

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
RESULTS_DIR = r"G:\sample\Qwen3vl\router_project\pope_results\oracle_champion"

GROUPS = {
    'shallow': list(range(5, 10)),
    'middle': list(range(10, 15)),
    'deep': list(range(15, 19)),
}
STRATEGY_MAP = {'shallow': 'adaiat', 'middle': 'adaiat', 'deep': 'none'}
ALL_LAYERS = list(range(5, 19))

os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
calib = torch.load(os.path.join(CHECKPOINT_DIR, "calibration.pt"), map_location="cpu", weights_only=False)

active_names = {f"lm.{i}" for i in ALL_LAYERS}
mgr = RouterManager(model, calib, active_layers=active_names, alpha_init=1.2)
mgr.wrap_all()
mgr.mode = "force"

# Build decision map
dmap = {}
for group_name, layers in GROUPS.items():
    for l in layers:
        dmap[f"lm.{l}"] = STRATEGY_MAP[group_name]

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

for subset in ["random", "popular", "adversarial"]:
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
    out_file = os.path.join(RESULTS_DIR, f"coco_pope_{subset}_answers.json")
    questions = [json.loads(l) for l in open(pope_file, encoding="utf-8")]

    results = []
    for q in tqdm(questions, desc=f"POPE {subset}"):
        mgr.clear_cache()

        for layer_name in dmap:
            strat = dmap[layer_name]
            strategies_for_layer = mgr._strategies_for(layer_name)
            if strat in strategies_for_layer:
                idx = strategies_for_layer.index(strat)
            elif "none" in strategies_for_layer:
                idx = strategies_for_layer.index("none")
            else:
                idx = 0
            mgr._decisions[layer_name] = idx
            mgr._decided.add(layer_name)

        # Non-group layers → none
        for d in mgr.descs:
            if d["name"] not in dmap and d["name"] not in mgr._decided:
                s = d["strategies"]
                mgr._decisions[d["name"]] = s.index("none") if "none" in s else 0
                mgr._decided.add(d["name"])

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
        q_pos = torch.arange(0, 1)
        for s in range(len(all_ids) - len(q_ids) + 1):
            if (all_ids[s:s+len(q_ids)] == q_t).all():
                q_pos = torch.arange(s, s + len(q_ids))
                break
        mgr._current_q_pos = q_pos

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=16)
        gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        raw = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)[0]
        results.append({"question": text, "answer": answer_yes_no(raw), "raw_output": raw})

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> Saved {len(results)} to {out_file}")

mgr.unwrap_all()
print("Done!")
