"""
POPE evaluation script — computes TP/TN/FP/FN, Accuracy, Precision, Recall, F1, Yes-Ratio.
Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    python pope_evaluate.py
"""

import json
import os

POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"


def load_jsonl(path):
    """Load a JSON-lines file into a list of dicts."""
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")]


def evaluate(answers, labels):
    """
    answers: list of {"question": ..., "answer": "yes"/"no"}
    labels:  list of {"question_id": ..., "label": "yes"/"no"}
    Returns dict of metrics.
    """
    # Ensure same order — answers and POPE labels are aligned by index
    preds = [1 if a["answer"] == "yes" else 0 for a in answers]
    golds = [1 if l["label"] == "yes" else 0 for l in labels]

    assert len(preds) == len(golds), f"Mismatch: {len(preds)} preds vs {len(golds)} golds"

    TP = FP = TN = FN = 0
    for p, g in zip(preds, golds):
        if p == 1 and g == 1:
            TP += 1
        elif p == 1 and g == 0:
            FP += 1
        elif p == 0 and g == 0:
            TN += 1
        elif p == 0 and g == 1:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    yes_ratio = sum(preds) / len(preds)

    return {
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Accuracy": round(acc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Yes_Ratio": round(yes_ratio, 4),
    }


def main():
    import sys
    dir_name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    subsets = ["random", "popular", "adversarial"]
    summary = {}

    for subset in subsets:
        label_file = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
        answer_file = os.path.join(RESULTS_BASE, dir_name, f"coco_pope_{subset}_answers.json")

        if not os.path.exists(answer_file):
            print(f"[WARN] Answer file missing for {subset}: {answer_file}")
            continue

        labels = load_jsonl(label_file)
        answers = load_jsonl(answer_file)

        metrics = evaluate(answers, labels)
        print(f"\n=== POPE {subset.upper()} ({dir_name}) ===")
        print(f"  TP={metrics['TP']}\tFP={metrics['FP']}\tTN={metrics['TN']}\tFN={metrics['FN']}")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1:        {metrics['F1']:.4f}")
        print(f"  Yes Ratio: {metrics['Yes_Ratio']:.4f}")
        summary[subset] = metrics

    # Write summary as JSON
    summary_path = os.path.join(RESULTS_BASE, dir_name, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Summary saved to {summary_path} (dir={dir_name})")


if __name__ == "__main__":
    main()
