"""
overnight_cai_v2.py — Run calibrate → sweep → full POPE overnight. No manual steps.
"""
import json, os, sys, time, subprocess

ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(ROUTER_DIR)
CHECKPOINT_DIR = os.path.join(ROUTER_DIR, "checkpoints")
PY = sys.executable
SCRIPT = os.path.join(ROUTER_DIR, "cai_bracs_v2.py")

t0 = time.time()
print(f"START: {time.strftime('%Y-%m-%d %H:%M:%S')}")

def runstep(desc, *args, timeout=3600):
    print(f"\n{'='*60}\n{desc}\n{'='*60}", flush=True)
    r = subprocess.run([PY, "-u", SCRIPT] + list(args), cwd=PROJECT_DIR,
                       env={**os.environ, "PYTHONUNBUFFERED": "1"},
                       timeout=timeout)
    ok = r.returncode == 0
    print(f"{'OK' if ok else 'FAILED('+str(r.returncode)+')'}: {desc}", flush=True)
    return ok

# Step 1: Calibrate
if not runstep("Step 1: Calibrate caption offsets", "--phase", "calibrate"):
    sys.exit(1)

# Step 2: Sweep
if not runstep("Step 2: Alpha sweep (200q)", "--phase", "sweep", "--n", "200", timeout=7200):
    sys.exit(1)

# Read best params
best = json.load(open(os.path.join(CHECKPOINT_DIR, "cai_sweep_best_v2.json")))
# best = {"param": "α=0.02", "acc": 0.885, "delta": 0.012}
print(f"Best sweep: {best}")
alpha_str = best["param"]
# Parse alpha from param string like "α=0.02" or "top7 α=0.05"
import re
alphas = re.findall(r"α=([\d.]+)", str(alpha_str))
alpha = float(alphas[0]) if alphas else 0.01
print(f"Using α={alpha}")

# Step 3: Full POPE
if not runstep(f"Step 3: Full POPE (α={alpha})", "--phase", "evaluate", "--alpha", str(alpha), timeout=28800):
    sys.exit(1)

# Step 4: Evaluate
print(f"\n{'='*60}\nStep 4: Evaluate\n{'='*60}", flush=True)
eval_script = os.path.join(PROJECT_DIR, "pope_evaluate.py")
r = subprocess.run([PY, eval_script, "cai_bracs"], cwd=PROJECT_DIR,
                   capture_output=True, text=True, timeout=600)
print(r.stdout[-500:])

# Step 5: Compare
compare_script = os.path.join(PROJECT_DIR, "pope_results", "compare.py")
r = subprocess.run([PY, compare_script, "baseline", "cai_bracs"], cwd=PROJECT_DIR,
                   capture_output=True, text=True, timeout=600)
print(r.stdout)

# Step 6: Commit
subprocess.run(["git", "add", "-A"], cwd=PROJECT_DIR)
subprocess.run(["git", "commit", "-m",
    f"overnight: CAI v2 full POPE — best α={alpha}"], cwd=PROJECT_DIR)
subprocess.run(["git", "push", "origin", "master"], cwd=PROJECT_DIR)

print(f"\nDONE at {time.strftime('%Y-%m-%d %H:%M:%S')} ({(time.time()-t0)/3600:.1f}h)")
