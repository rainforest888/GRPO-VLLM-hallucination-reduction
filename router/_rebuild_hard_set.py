"""
_rebuild_hard_set.py — Rebuild hard training set with log-prob confidence filtering.

Approach:
1. Re-run baseline on ALL 3000 adversarial questions
2. For each question, record:
   - answer (yes/no)
   - log_prob of the ANSWERED token (not just the correct one)
   - whether it matched the label
3. Hard set = questions where:
   - answer != label  (baseline WRONG)  OR
   - answer == label AND log_p(answer_token) > -0.3 (low confidence correct)
     (i.e. p < ~74%, model was uncertain despite getting it right)
4. Replicate to 10000, shuffle, save.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/_rebuild_hard_set.py --confidence_threshold -0.3
"""
import json, os, sys, argparse, random
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"

ap = argparse.ArgumentParser()
ap.add_argument("--confidence_threshold", type=float, default=-0.3,
                help="Log-prob below which (more negative = less confident) to include in hard set")
ap.add_argument("--target", type=int, default=10000, help="Target size")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

THRESHOLD = args.confidence_threshold
TARGET = args.target
random.seed(args.seed)

# ─── Load model ─────────────────────────────────────────────────────
print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Token IDs for "yes" and "no" (directly, no special handling)
YES_TOKEN_ID = processor.tokenizer.encode("yes", add_special_tokens=False)[0]
NO_TOKEN_ID = processor.tokenizer.encode("no", add_special_tokens=False)[0]
print(f"Yes token ID: {YES_TOKEN_ID}, No token ID: {NO_TOKEN_ID}")

# ─── Run adversarial evaluation with log-prob ───────────────────────
pope_file = os.path.join(POPE_DIR, "coco_pope_adversarial.json")
questions = [json.loads(l) for l in open(pope_file, "r", encoding="utf-8")]

wrong_entries = []       # (question, label, answer, log_p)
low_confidence_correct = []  # (question, label, answer, log_p)
high_confidence_correct = []  # for statistics only

for q in tqdm(questions, desc="Adversarial + log-prob"):
    img_path = os.path.join(IMAGE_DIR, q["image"])
    text = q["text"]
    label = q["label"]

    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_path},
        {"type": "text", "text": text + " Please answer yes or no."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    # Get logits for the last position (next token prediction)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[0, -1, :]  # last position

    # Log-softmax
    log_probs = F.log_softmax(logits, dim=-1)

    # Get log-prob for "yes" and "no"
    lp_yes = log_probs[YES_TOKEN_ID].item()
    lp_no = log_probs[NO_TOKEN_ID].item()

    # Argmax answer
    if lp_yes >= lp_no:
        answer = "yes"
        answer_logp = lp_yes
    else:
        answer = "no"
        answer_logp = lp_no

    entry = {"question": q, "answer": answer, "label": label,
             "logp_yes": lp_yes, "logp_no": lp_no,
             "logp_answer": answer_logp}

    if answer != label:
        wrong_entries.append(entry)
    elif answer_logp > THRESHOLD:  # less confident than threshold
        low_confidence_correct.append(entry)
    else:
        high_confidence_correct.append(entry)

# ─── Statistics ─────────────────────────────────────────────────────
n_total = len(questions)
n_wrong = len(wrong_entries)
n_low_conf = len(low_confidence_correct)
n_high_conf = len(high_confidence_correct)

print(f"\n{'='*60}")
print(f"CONFIDENCE THRESHOLD: log_p > {THRESHOLD} (i.e. prob < ~{1 - torch.exp(torch.tensor(THRESHOLD)):.0%} area)")
print(f"{'='*60}")
print(f"Total:                    {n_total}")
print(f"Wrong:                    {n_wrong}  ({n_wrong/n_total:.1%})")
print(f"Correct + low confidence: {n_low_conf}  ({n_low_conf/n_total:.1%})")
print(f"Correct + high confidence:{n_high_conf}  ({n_high_conf/n_total:.1%})")
print(f"Hard set size:            {n_wrong + n_low_conf}  "
      f"({(n_wrong+n_low_conf)/n_total:.1%})")

# Confidence distribution stats
wrong_lps = [e["logp_answer"] for e in wrong_entries]
low_lps = [e["logp_answer"] for e in low_confidence_correct]
high_lps = [e["logp_answer"] for e in high_confidence_correct]

if wrong_lps:
    print(f"\nWrong answers:       mean log_p = {sum(wrong_lps)/len(wrong_lps):.4f}  "
          f"min={min(wrong_lps):.4f}  max={max(wrong_lps):.4f}")
if low_lps:
    print(f"Low-conf correct:    mean log_p = {sum(low_lps)/len(low_lps):.4f}  "
          f"min={min(low_lps):.4f}  max={max(low_lps):.4f}")
if high_lps:
    print(f"High-conf correct:   mean log_p = {sum(high_lps)/len(high_lps):.4f}  "
          f"min={min(high_lps):.4f}  max={max(high_lps):.4f}")

# ─── Build hard set ─────────────────────────────────────────────────
hard_entries = wrong_entries + low_confidence_correct
hard_questions = [e["question"] for e in hard_entries]

# Replicate to TARGET
n_base = len(hard_questions)
repeats = TARGET // n_base
remainder = TARGET % n_base
train_set = hard_questions * repeats + hard_questions[:remainder]
random.shuffle(train_set)

# ─── Save ───────────────────────────────────────────────────────────
out_path = os.path.join(POPE_DIR, "coco_pope_hard_10000.json")
with open(out_path, "w", encoding="utf-8") as f:
    for q in train_set:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")

yes_count = sum(1 for q in train_set if q["label"] == "yes")
print(f"\nHard set saved: {len(train_set)} questions ({n_base} unique × {repeats}+{remainder})")
print(f"Label dist: yes={yes_count} ({yes_count/len(train_set):.1%}) "
      f"no={len(train_set)-yes_count} ({1-yes_count/len(train_set):.1%})")
print(f"Saved to: {out_path}")

# ─── Also save metadata for inspection ──────────────────────────────
meta_path = os.path.join(POPE_DIR, "coco_pope_hard_10000_meta.json")
meta = {
    "threshold_logp": THRESHOLD,
    "n_total_adversarial": n_total,
    "n_wrong": n_wrong,
    "n_low_confidence_correct": n_low_conf,
    "n_high_confidence_correct": n_high_conf,
    "unique_hard_questions": n_base,
    "unique_images": len(set(q["image"] for q in hard_questions)),
    "target_size": TARGET,
    "label_distribution": {"yes": yes_count, "no": len(train_set) - yes_count},
    "wrong_mean_logp": sum(wrong_lps) / len(wrong_lps) if wrong_lps else 0,
    "low_conf_mean_logp": sum(low_lps) / len(low_lps) if low_lps else 0,
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"Metadata saved to: {meta_path}")
