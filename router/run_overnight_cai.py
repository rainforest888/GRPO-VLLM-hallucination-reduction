"""
run_overnight_cai.py — Fully automatic overnight pipeline:
  1. CAI+BRACS calibration (cached if exists)
  2. Param sweep on 200 adversarial → pick best alpha/beta/barrier
  3. Full POPE evaluation (random + popular + adversarial, 9000 questions)
  4. Compute evaluation metrics
  5. Compare vs baseline
  6. Save results, commit to git

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project
    python router/run_overnight_cai.py
"""
import json, os, sys, time, subprocess, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--skip_calib", action="store_true", help="Skip calibration if offsets exist")
ap.add_argument("--sweep_n", type=int, default=200, help="Sweep questions")
args = ap.parse_args()

ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROUTER_DIR)
CHECKPOINT_DIR = os.path.join(ROUTER_DIR, "checkpoints")
OFFSET_FILE = os.path.join(CHECKPOINT_DIR, "cai_offsets.pt")
PY = sys.executable

t0 = time.time()
print(f"OVERNIGHT CAI+BRACS START: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

# ═══ Step 1: Calibration ══════════════════════════════════════════
if os.path.exists(OFFSET_FILE) and args.skip_calib:
    print(f"\n[SKIP] Calibration: {OFFSET_FILE} exists", flush=True)
else:
    print(f"\n{'='*60}\nSTEP 1: CAI Calibration (50 images)\n{'='*60}", flush=True)
    r = subprocess.run(
        [PY, "-u", os.path.join(ROUTER_DIR, "cai_bracs_inference.py"),
         "--phase", "calibrate", "--n_captions", "50"],
        cwd=PROJECT_DIR, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        timeout=3600  # 1h max
    )
    print(f"Calibration: {'OK' if r.returncode == 0 else f'FAILED({r.returncode})'}", flush=True)
    if r.returncode != 0:
        sys.exit(1)

# ═══ Step 2: Param sweep ══════════════════════════════════════════
print(f"\n{'='*60}\nSTEP 2: Alpha/Beta sweep ({args.sweep_n} adversarial)\n{'='*60}", flush=True)
sweep_script = os.path.join(ROUTER_DIR, "_run_overnight_sweep.py")
r = subprocess.run(
    [PY, "-u", sweep_script, "--n", str(args.sweep_n)],
    cwd=PROJECT_DIR, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    timeout=7200  # 2h max
)
print(f"Sweep: {'OK' if r.returncode == 0 else f'FAILED({r.returncode})'}", flush=True)

# Read best params
best_path = os.path.join(CHECKPOINT_DIR, "cai_sweep_best.json")
best = {}
if os.path.exists(best_path):
    best = json.load(open(best_path))
    print(f"Best params: α={best.get('alpha')} β={best.get('beta')} bar={best.get('barrier')} "
          f"acc={best.get('acc')} Δ={best.get('delta')}", flush=True)
else:
    best = {"alpha": 0.01, "beta": 1.0, "barrier": 0.3}
    print(f"[WARN] No sweep results, using defaults: {best}", flush=True)

# ═══ Step 3: Full POPE evaluation ═════════════════════════════════
print(f"\n{'='*60}\nSTEP 3: Full POPE (9000 questions)\n{'='*60}", flush=True)
# Use a dedicated inference script
eval_script = os.path.join(ROUTER_DIR, "cai_bracs_inference.py")
r = subprocess.run(
    [PY, "-u", eval_script, "--phase", "evaluate",
     "--alpha", str(best["alpha"]),
     "--beta", str(best["beta"]),
     "--barrier", str(best["barrier"])],
    cwd=PROJECT_DIR, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    timeout=28800  # 8h for 9000 questions
)
print(f"Evaluation: {'OK' if r.returncode == 0 else f'FAILED({r.returncode})'}", flush=True)

# ═══ Step 4: Compute metrics ═══════════════════════════════════════
print(f"\n{'='*60}\nSTEP 4: Evaluation metrics\n{'='*60}", flush=True)
evaluate_script = os.path.join(PROJECT_DIR, "pope_evaluate.py")
r = subprocess.run(
    [PY, evaluate_script, "cai_bracs"],
    cwd=PROJECT_DIR, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    timeout=600, capture_output=True, text=True
)
print(r.stdout[-1000:] if r.stdout else "no output", flush=True)

# ═══ Step 5: Compare vs baseline ═══════════════════════════════════
print(f"\n{'='*60}\nSTEP 5: Compare cai_bracs vs baseline\n{'='*60}", flush=True)
compare_script = os.path.join(PROJECT_DIR, "pope_results", "compare.py")
r = subprocess.run(
    [PY, compare_script, "baseline", "cai_bracs"],
    cwd=PROJECT_DIR, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    timeout=600, capture_output=True, text=True
)
print(r.stdout if r.stdout else "no output", flush=True)

# ═══ Step 6: Git commit ════════════════════════════════════════════
print(f"\n{'='*60}\nSTEP 6: Commit results\n{'='*60}", flush=True)
subprocess.run(["git", "add", "-A"], cwd=PROJECT_DIR)
subprocess.run(["git", "commit", "-m",
    f"overnight: CAI+BRACS full POPE — α={best.get('alpha')} β={best.get('beta')} bar={best.get('barrier')} acc={best.get('acc')}"],
    cwd=PROJECT_DIR)
subprocess.run(["git", "push", "origin", "master"], cwd=PROJECT_DIR)

elapsed = (time.time() - t0) / 3600
print(f"\n{'='*60}")
print(f"DONE at {time.strftime('%Y-%m-%d %H:%M:%S')} ({elapsed:.1f}h)")
print(f"Results: pope_results/cai_bracs/")
print(f"{'='*60}", flush=True)
