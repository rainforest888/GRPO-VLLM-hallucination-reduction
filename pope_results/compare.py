"""
POPE comparison script: compares two result directories side by side.
Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    python compare.py baseline with_hook_v1

Produces a markdown table of all metrics and the delta for each POPE subset.
"""
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_summary(dir_name):
    path = os.path.join(BASE_DIR, dir_name, "evaluation_summary.json")
    if not os.path.exists(path):
        print(f"[ERROR] No evaluation_summary.json found in {dir_name}/")
        sys.exit(1)
    return json.load(open(path, "r", encoding="utf-8"))


def format_diff(a, b):
    """Return a string like '+0.0123' or '-0.0056'."""
    delta = b - a
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare.py <baseline_dir> <hook_dir>")
        print("Example: python compare.py baseline with_hook_v1")
        sys.exit(1)

    dir_a = sys.argv[1]
    dir_b = sys.argv[2]

    s_a = load_summary(dir_a)
    s_b = load_summary(dir_b)

    subsets = ["random", "popular", "adversarial"]
    metrics = ["Accuracy", "Precision", "Recall", "F1", "Yes_Ratio"]

    # Markdown table
    print(f"\n## POPE Comparison: `{dir_a}`  vs  `{dir_b}`\n")
    header = "| Subset      | Metric    | " + f"{dir_a:^12} | " + f"{dir_b:^12} | " + "Delta      |"
    sep    = "|------------:|----------:|-------------:|-------------:|-----------:|"
    print(header)
    print(sep)

    for subset in subsets:
        for i, metric in enumerate(metrics):
            a_val = s_a.get(subset, {}).get(metric, 0)
            b_val = s_b.get(subset, {}).get(metric, 0)
            delta = format_diff(a_val, b_val)

            label = subset if i == 0 else ""
            print(f"| {label:<11} | {metric:<9} | {a_val:>11.4f} | {b_val:>11.4f} | {delta:>10} |")

    # Per-subset summary
    print("\n---\n")
    for subset in subsets:
        a = s_a.get(subset, {})
        b = s_b.get(subset, {})
        print(f"### {subset.upper()}")
        print(f"| Metric    | {dir_a}  | {dir_b}  | Delta    |")
        print(f"|----------:|--------:|--------:|---------:|")
        for m in metrics:
            a_val = a.get(m, 0)
            b_val = b.get(m, 0)
            d = format_diff(a_val, b_val)
            name = m.replace("_", " ")
            print(f"| {name:<9} | {a_val:>.4f} | {b_val:>.4f} | {d:>8} |")
        print()


if __name__ == "__main__":
    main()
