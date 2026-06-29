"""
overnight.py — Fully automatic pipeline. No interactive stops, no permissions.
Sequence: UAC test → Oracle fast (500) → Oracle top-5 (3000) → summary → GRPO → POPE → compare

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    python /g/sample/Qwen3vl/router_project/router/overnight.py
    python overnight.py --skip-to-step 2     # resume from step 2
    python overnight.py --timeout 14400       # 4h timeout per step (seconds)
"""
import json, os, subprocess, sys, shutil, time, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--skip-to-step", type=int, default=1, help="resume from this step")
ap.add_argument("--timeout", type=int, default=14400, help="per-step timeout in seconds (default 4h)")
opts = ap.parse_args()

ROUTER_DIR = r"G:\sample\Qwen3vl\router_project\router"
PROJECT_DIR = r"G:\sample\Qwen3vl\router_project"
POPS_DIR = r"G:\sample\Qwen3vl\router_project\pope_results"
ORACLE_DIR = os.path.join(POPS_DIR, "oracle")
CHECKPOINT_DIR = os.path.join(ROUTER_DIR, "checkpoints")
TIMEOUT = opts.timeout
PY = sys.executable
os.makedirs(ORACLE_DIR, exist_ok=True)

STEP = [0]
def run(script, desc, *extra_args):
    STEP[0] += 1
    if STEP[0] < opts.skip_to_step:
        print(f"\n[SKIP] Step {STEP[0]}: {desc} (--skip-to-step={opts.skip_to_step})", flush=True)
        return True
    args = [PY, "-u", script] + list(extra_args)
    print(f"\n{'#'*70}\n# STEP {STEP[0]}: {desc}\n# {' '.join(args)}\n{'#'*70}", flush=True)
    try:
        r = subprocess.run(args, env={**os.environ, "PYTHONUNBUFFERED": "1"}, timeout=TIMEOUT)
        ok = r.returncode == 0
        print(f"\n[{'OK' if ok else 'FAILED ('+str(r.returncode)+')'}] {desc}", flush=True)
        return ok
    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT after {TIMEOUT}s] {desc}", flush=True)
        return False

def find_best_combo():
    """Parse existing oracle results and return (name, acc, S_strategy, M_strategy, D_strategy)."""
    best = None
    for d in os.listdir(ORACLE_DIR):
        f = os.path.join(ORACLE_DIR, d, "coco_pope_adversarial_answers.json")
        if not os.path.exists(f): continue
        try:
            answers = [json.loads(l) for l in open(f, encoding='utf-8')]
        except: continue
        if len(answers) < 10: continue
        popep = os.path.join(PROJECT_DIR, "..", "POPE-main", "POPE-main", "output", "coco", "coco_pope_adversarial.json")
        labels = [json.loads(l)['label'] for l in open(popep, encoding='utf-8')]
        c = sum(1 for a, lbl in zip(answers, labels[:len(answers)]) if a['answer'] == lbl)
        acc = c / len(answers)
        parts = d.split('_')
        name = d; s = parts[1]; m = parts[3]; d_strat = parts[5]
        if best is None or acc > best[1]: best = (name, acc, s, m, d_strat)
    return best

# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print(f"OVERNIGHT PIPELINE START: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Python: {PY}", flush=True)

    # ── Step 1: UAC standalone test (new real-image W, layer 15) ──
    uac_script = os.path.join(ROUTER_DIR, "uac_inference.py")
    run(uac_script, "UAC test (L15, real-image W)", "--layer", "15", "--outdir", "uac_real_L15_v2")
    # Evaluate UAC
    eval_script = os.path.join(PROJECT_DIR, "pope_evaluate.py")
    ok_uac = subprocess.run([PY, eval_script, "uac_real_L15_v2"], capture_output=True, text=True,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"}, timeout=TIMEOUT).returncode == 0
    print(f"UAC eval: {'OK' if ok_uac else 'FAILED'}", flush=True)

    # ── Step 2: Oracle fast screen (500 questions per combo) ──
    oracle_script = os.path.join(ROUTER_DIR, "oracle_test.py")
    run(oracle_script, "Oracle fast screen (500 q/combo)", "--n", "500")

    # ── Step 3: Oracle top-5 at 3000 ──
    # Compute current top-5
    summary = {}
    for d in os.listdir(ORACLE_DIR):
        f = os.path.join(ORACLE_DIR, d, "coco_pope_adversarial_answers.json")
        if not os.path.exists(f): continue
        try: answers = [json.loads(l) for l in open(f, encoding='utf-8')]
        except: continue
        if len(answers) < 10: continue
        popep = os.path.join(PROJECT_DIR, "..", "POPE-main", "POPE-main", "output", "coco", "coco_pope_adversarial.json")
        labels = [json.loads(l)['label'] for l in open(popep, encoding='utf-8')]
        c = sum(1 for a, lbl in zip(answers, labels[:len(answers)]) if a['answer'] == lbl)
        summary[d] = c / len(answers)
    with open(os.path.join(ORACLE_DIR, "oracle_summary.json"), 'w') as f: json.dump(summary, f, indent=2)
    sorted_combos = sorted(summary.items(), key=lambda x: -x[1])
    print(f"\nOracle top-5 from {len(summary)} combos:", flush=True)
    for i, (name, acc) in enumerate(sorted_combos[:5]):
        parts = name.split('_')
        print(f"  {i+1}. {acc:.4f}  S={parts[1]:8} M={parts[3]:8} D={parts[5]:8}", flush=True)

    best = sorted_combos[0] if sorted_combos else None
    if best: print(f"BEST ORACLE: {best[1]:.4f} ({best[0]})", flush=True)

    run(oracle_script, "Oracle top-5 at 3000", "--n", "3000", "--topk", "5")

    # ── Re-evaluate top-5 after full run ──
    summary2 = {}
    for d in os.listdir(ORACLE_DIR):
        f = os.path.join(ORACLE_DIR, d, "coco_pope_adversarial_answers.json")
        if not os.path.exists(f): continue
        try: answers = [json.loads(l) for l in open(f, encoding='utf-8')]
        except: continue
        if len(answers) < 100: continue  # only 3000-answer files
        popep = os.path.join(PROJECT_DIR, "..", "POPE-main", "POPE-main", "output", "coco", "coco_pope_adversarial.json")
        labels = [json.loads(l)['label'] for l in open(popep, encoding='utf-8')]
        c = sum(1 for a, lbl in zip(answers, labels[:len(answers)]) if a['answer'] == lbl)
        summary2[d] = c / len(answers)
    if summary2:
        with open(os.path.join(ORACLE_DIR, "oracle_summary_full.json"), 'w') as f: json.dump(summary2, f, indent=2)
        best2 = sorted(summary2.items(), key=lambda x: -x[1])[0]
        print(f"\nBEST ORACLE (full 3000): {best2[1]:.4f} ({best2[0]})", flush=True)

    # ── Step 4: GRPO training ──
    grpo_script = os.path.join(ROUTER_DIR, "grpo_train.py")
    run(grpo_script, "GRPO router training (sparse_k=2, K=4)")

    # ── Step 5: POPE inference with trained router ──
    router_weights = os.path.join(CHECKPOINT_DIR, "router_weights_final.pt")
    pope_router_script = os.path.join(ROUTER_DIR, "pope_inference_router.py")
    run(pope_router_script, "POPE inference (router argmax)", router_weights)

    # ── Step 6: Evaluate & compare ──
    run(eval_script, "Router evaluation", "router_v1")
    compare_script = os.path.join(POPS_DIR, "compare.py")
    subprocess.run([PY, compare_script, "baseline", "router_v1"],
                   env={**os.environ, "PYTHONUNBUFFERED": "1"}, timeout=TIMEOUT)
    # Also show UAC vs baseline
    subprocess.run([PY, compare_script, "baseline", "uac_real_L15_v2"],
                   env={**os.environ, "PYTHONUNBUFFERED": "1"}, timeout=TIMEOUT)

    elapsed = (time.time() - t0) / 3600
    print(f"\n{'='*70}")
    print(f"OVERNIGHT PIPELINE COMPLETE at {time.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed:.1f}h)")
    best_final = find_best_combo()
    if best_final:
        print(f"Best oracle combo: {best_final[1]:.4f} (S={best_final[2]}, M={best_final[3]}, D={best_final[4]})")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
