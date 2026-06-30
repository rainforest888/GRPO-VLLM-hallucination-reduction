"""
LLaVA POPE Baseline Inference.
Runs full POPE evaluation (random, popular, adversarial) on LLaVA-1.5-7B.

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/llava_project
    python pope_baseline.py
"""
import json, os, sys, torch, argparse
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
OUT_DIR = r"G:\sample\llava_project\pope_results\baseline"
os.makedirs(OUT_DIR, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--subsets", type=str, default="random,popular,adversarial")
ap.add_argument("--max_n", type=int, default=0, help="max questions per subset (0=all)")
args = ap.parse_args()


def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"


print("Loading LLaVA-1.5-7B...")
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    local_files_only=True,
)
processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
print(f"Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

for subset in args.subsets.split(","):
    subset = subset.strip()
    pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
    out_file = os.path.join(OUT_DIR, f"coco_pope_{subset}_answers.json")

    questions = [json.loads(l) for l in open(pope_file, encoding="utf-8")]
    if args.max_n > 0:
        questions = questions[:args.max_n]

    results = []
    yes_count = 0
    for q in tqdm(questions, desc=f"POPE {subset}"):
        img = Image.open(os.path.join(IMAGE_DIR, q["image"])).convert("RGB")

        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": q["text"] + " Please answer yes or no."},
        ]}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=8)
        raw = processor.decode(gen[0, inputs.input_ids.shape[1]:],
                               skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ans = answer_yes_no(raw)
        if ans == "yes": yes_count += 1
        results.append({"question": q["text"], "answer": ans, "raw_output": raw})

    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    labels = [json.loads(l)["label"] for l in open(pope_file, encoding="utf-8")][:len(results)]
    correct = sum(1 for r, l in zip(results, labels) if r["answer"] == l)
    acc = correct / len(results)
    print(f"  {subset}: {correct}/{len(results)} = {acc:.4f}  yes_ratio={yes_count/len(results):.3f}")
    torch.cuda.empty_cache()

print(f"\nBaseline complete. Results in {OUT_DIR}/")
print("Next: write pope_evaluate.py to compute TP/FP/TN/FN/Precision/Recall/F1")
