"""Quick evaluation of all POPE results."""
import json, os

pope_dir = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
results_dir = r"G:\sample\Qwen3vl\router_project\pope_results"

for d in ["baseline", "uac_layer15", "adaiat_u_layer15_a1", "adaiat_layer15_a0.5", "router_v1"]:
    print(f"=== {d} ===")
    for subset in ["random", "popular", "adversarial"]:
        ans_file = os.path.join(results_dir, d, f"coco_pope_{subset}_answers.json")
        pope_file = os.path.join(pope_dir, f"coco_pope_{subset}.json")
        if not os.path.exists(ans_file):
            print(f"  {subset}: MISSING")
            continue
        answers = [json.loads(l) for l in open(ans_file, encoding="utf-8")]
        labels = [json.loads(l)["label"] for l in open(pope_file, encoding="utf-8")]
        c = sum(1 for a, l in zip(answers, labels[:len(answers)]) if a["answer"] == l)
        print(f"  {subset:12}: {c}/{len(answers)} = {c/len(answers):.4f}")
