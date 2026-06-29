"""Analyze UAC W values from calibration.pt"""
import torch

calib = torch.load(
    r"G:\sample\Qwen3vl\router_project\router\checkpoints\calibration.pt",
    map_location="cpu", weights_only=False
)
W = calib.get("W", {})

print("=" * 70)
print("UAC W Analysis: per-layer, per-resolution deviation from 1.0")
print("W = mean(A) / A  →  W≈1 means uniform attention across vision tokens")
print("=" * 70)

total_elements = 0
total_deviated = 0

for name in sorted(W.keys(), key=lambda x: int(x.split(".")[1])):
    w = W[name]
    if not isinstance(w, dict):
        continue
    layer_devs = []
    for nv, t in sorted(w.items()):
        dev = (t - 1.0).abs()
        pct_dev = (dev > 0.01).float().mean().item()
        pct_big = (dev > 0.05).float().mean().item()
        layer_devs.append((nv, t, pct_dev, pct_big))
        total_elements += t.numel()
        total_deviated += (dev > 0.01).sum().item()

    # Summarize per layer
    avg_pct = sum(d[2] for d in layer_devs) / max(len(layer_devs), 1)
    avg_pct_big = sum(d[3] for d in layer_devs) / max(len(layer_devs), 1)
    n_res = len(layer_devs)
    sample_nv, sample_t, _, _ = layer_devs[0]
    print(f"\n{name} ({n_res} resolutions):")
    print(f"  avg fraction with |W-1|>0.01: {avg_pct:.4f}  |  |W-1|>0.05: {avg_pct_big:.4f}")
    print(f"  sample n_vis={sample_nv}: shape={sample_t.shape}, ")
    print(f"    min={sample_t.min().item():.4f}  max={sample_t.max().item():.4f}  mean={sample_t.mean().item():.4f}  std={sample_t.std().item():.4f}")

    # Show worst resolution
    worst = max(layer_devs, key=lambda x: x[2])
    print(f"  worst n_vis={worst[0]}: |W-1|>0.01 = {worst[2]:.3f}, |W-1|>0.05 = {worst[3]:.3f}")

print(f"\n{'=' * 70}")
print(f"OVERALL: {total_deviated}/{total_elements} elements deviate >0.01 ({100*total_deviated/max(total_elements,1):.2f}%)")

# Also check M values for AdaIAT
M = calib.get("M", {})
print(f"\n{'=' * 70}")
print("AdaIAT M Analysis (per-head amplification factor)")
print("=" * 70)
for name in sorted(M.keys(), key=lambda x: int(x.split(".")[1])):
    m = M[name]
    print(f"{name}: shape={m.shape}, min={m.min().item():.4f}, max={m.max().item():.4f}, "
          f"mean={m.mean().item():.4f}, std={m.std().item():.4f}")
    dev = (m - 1.0).abs()
    pct = (dev > 0.01).float().mean().item()
    print(f"  fraction with |M-1|>0.01: {pct:.3f}")

# Check thresholds
thresh = calib.get("thresholds", {})
print(f"\n{'=' * 70}")
print("AdaIAT Thresholds")
print("=" * 70)
for name in sorted(thresh.keys(), key=lambda x: int(x.split(".")[1])):
    print(f"{name}: {thresh[name]}")
