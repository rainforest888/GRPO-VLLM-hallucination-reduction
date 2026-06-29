# GRPO-VLLM Hallucination Reduction

**Per-layer router trained via GRPO to dynamically select attention correction strategies for reducing object hallucination in Qwen3-VL-2B-Instruct.**

[English](#english) | [中文](README_zh.md)

---

## Overview

Vision Language Models like Qwen3-VL suffer from **object hallucination** — generating objects not present in the image. This project addresses hallucination by training a lightweight per-layer router that dynamically selects attention correction strategies (UAC, AdaIAT, VHR, UAC+VHR, or None) for each transformer layer, optimized via **GRPO (Group Relative Policy Optimization)**.

### Key Innovation

Instead of applying a fixed intervention strategy across all layers, the router **learns per-layer, context-dependent decisions**:
- Each of 14 LM layers (5–18) has its own small MLP router
- Given hidden states, the router picks among 5 strategies
- GRPO amplifies weak per-sample reward signals through group-relative advantage

### Strategies

| Strategy | Mechanism | Paper |
|----------|-----------|-------|
| **UAC** | Calibrates attention weights with log-space correction | [Uncertainty-aware Attention Calibration](https://arxiv.org/abs/2502.01969) |
| **AdaIAT-U** | Adaptive attention amplification targeting question tokens | [Adaptive Inference-Time Attention Tuning](https://arxiv.org/abs/2603.04908) |
| **VHR** | Vision-aware Head Re-weighting based on vision divergence | Custom |
| **UAC+VHR** | Combined UAC + VHR intervention | Custom |
| **None** | Pass-through (no intervention) | — |

---

## Results

### POPE Benchmark (Qwen3-VL-2B-Instruct)

| Subset | Baseline | Best Router | Δ |
|--------|:--------:|:-----------:|:--:|
| Random | 91.53% | 91.60% | +0.07 |
| Popular | 89.10% | 89.10% | ±0.00 |
| **Adversarial** | 87.30% | **87.50%** | **+0.20** |

### Strategy Distribution (Router v4)
`adaiat-U: 15.4% | none: 84.6% | uac: 0.0%`

The router learns a conservative strategy — intervene only when signals support it.

---

## Architecture

```
Qwen3-VL-2B (frozen) → RouterManager (wraps 52 attention forwards)
                          → LM layers 5-18 each have LayerRouter
                          → LayerRouter: hidden → pool → MLP(256) → 5-class logits
                          → {UAC, AdaIAT-U, VHR, UAC+VHR, None}
                          → grad-free GRPO: detached hidden states, loss through router MLP only
```

- **Grad-free design**: ~4.9 GB GPU RAM (safe on 8 GB laptop GPU), ~0.4s/step
- **Calibration**: W (UAC), M_U + threshold (AdaIAT-U), VHD (VHR)
- **GRPO reward**: R = log P(correct answer token)
- **Sparse sampling**: only k layers active per forward pass to isolate signals

---

## Project Structure

```
router_project/
├── README.md                          # This file
├── README_zh.md                       # Chinese version
├── SUMMARY.md                         # Full technical summary
├── pope_inference.py                  # Baseline inference
├── pope_evaluate.py                   # Metric computation (TP/FP/TN/FN/Acc/Prec/Rec/F1)
├── pope_results/
│   ├── baseline/                      # Baseline results
│   ├── uac_layer15/                   # UAC single-layer
│   ├── adaiat_u_layer15_a1/           # AdaIAT-U single-layer
│   ├── router_v1/                     # Router v4 results
│   ├── vhr_L15/                       # VHR results
│   ├── vcd_gamma1.0_n500/             # VCD results
│   ├── oracle/                        # Oracle search results
│   ├── oracle_champion/               # Best oracle combo
│   └── compare.py                     # Side-by-side comparison
└── router/
    ├── router_module.py               # RouterManager + LayerRouter (core)
    ├── strategies.py                  # UAC / AdaIAT / VHR / UAC+VHR implementations
    ├── grpo_train.py                  # GRPO training v1
    ├── grpo_train_v2.py               # GRPO training v2 (random exploration)
    ├── dpo_train.py                   # DPO training (earlier approach)
    ├── dpo_data.py                    # POPE data loading & train/valid split
    ├── calibration.py                 # W (UAC) + M/threshold (AdaIAT)
    ├── calibrate_vhr.py               # VHD calibration for VHR strategy
    ├── calc_steering.py               # CASAL-style steering vectors
    ├── recalibrate_u.py               # Re-calibrate AdaIAT-U
    ├── recalibrate_uac_real.py        # UAC re-calibration on real images
    ├── pope_inference_router.py       # Router inference (argmax)
    ├── pope_inference_forced.py       # Forced single-strategy inference
    ├── uac_inference.py               # Standalone UAC evaluation
    ├── adaiat_inference.py            # Standalone AdaIAT evaluation
    ├── adaiat_u_inference.py          # Standalone AdaIAT-U evaluation
    ├── vcd_inference.py               # Visual Contrastive Decoding
    ├── smoke_vhr.py                   # VHR/UAC+VHR smoke test
    ├── oracle_test.py                 # Oracle strategy search
    ├── test_casal_lime.py             # CASAL & LIME evaluation
    ├── sweep_vhr_alpha.py             # VHR alpha sweep
    ├── _rebuild_hard_set.py           # Hard set construction
    ├── _run_champion.py               # Champion combo evaluation
    ├── _pope_analysis.py              # Analysis utilities
    ├── _analyze_calib.py              # Calibration analysis
    ├── _eval_all.py                   # Batch evaluation
    ├── overnight.py                   # Fully automatic pipeline
    ├── overnight_log_*.txt            # Overnight run logs
    └── checkpoints/
        ├── calibration.pt             # W + M + thresholds + VHD
        ├── router_weights_final.pt    # Final router weights (DPO)
        └── router_weights_grpo_v2_*.pt # GRPO checkpoint weights
```

---

## Setup

### Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB) |
| CUDA | PyTorch 2.12.0 |
| Model | Qwen3-VL-2B-Instruct |
| Framework | transformers 5.12.1 |
| Python | conda env `qwen3vl` |

### Activation

```bash
source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
cd /g/sample/Qwen3vl/router_project
```

### External Dependencies

| Resource | Path |
|----------|------|
| Model weights | `G:\sample\Qwen3vl\` |
| POPE questions | `G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco\` |
| COCO images | `G:\sample\Qwen3vl\val2014\val2014\` |

---

## Usage

### Quick Start — Evaluate Baseline

```bash
python pope_evaluate.py
```

### Calibration

```bash
# UAC + AdaIAT calibration
python router/calibration.py

# VHR calibration (vision divergence)
python router/calibrate_vhr.py
```

### GRPO Training (Recommended)

```bash
# v2 with random exploration (best results)
python router/grpo_train_v2.py --sparse_k 3 --grpo_k 4 --n_epochs 10

# v1 with sparse sampling
python router/grpo_train.py
```

### DPO Training (Earlier Approach)

```bash
python router/dpo_train.py
```

### Evaluate with Trained Router

```bash
# Router argmax inference
python router/pope_inference_router.py router/checkpoints/router_weights_final.pt

# Compute metrics
python pope_evaluate.py router_v1

# Compare with baseline
python pope_results/compare.py baseline router_v1
```

### Standalone Strategy Evaluation

```bash
# UAC single-layer
python router/uac_inference.py

# AdaIAT-U single-layer
python router/adaiat_u_inference.py

# VCD (Visual Contrastive Decoding)
python router/vcd_inference.py --gamma 1.0 --n 500

# VHR smoke test
python router/smoke_vhr.py --layer 15 --strategy vhr

# Full overnight pipeline
python router/overnight.py
```

---

## Key Engineering Discoveries

1. **Grad-free replay**: Full backprop through attention → 10.7 GB GPU (OOM). Detached hidden states + router MLP replay → 4.9 GB (safe).

2. **AdaIAT target mismatch**: Original targets "generated text" attention. POPE single-token answers have empty Tp at decision point. Solved by targeting question tokens (U).

3. **Signal direction matters**: AdaIAT-U shows M > 1 (correct answers attend MORE to questions — correct signal). AdaIAT-V shows M < 1 (reversed signal).

4. **Layer range is critical**: LM layers 5–16 all have M > 1 (visual reasoning). Layers 17–18 drop below M < 1 (semantic refinement boundary).

5. **Sparse sampling prevents collapse**: Without sparsity, router collapses to 94%+ "none". With sparse_k=2, signals become detectable.

---

## Limitations & Future Work

| Priority | Direction | Expected Gain | Effort |
|:--------:|-----------|:-------------:|:------:|
| High | Multi-layer UAC combo search | >1% | Medium |
| High | Larger AdaIAT-U alpha (2.0/3.0) | Possible threshold gain | Low |
| Medium | REINFORCE baseline for variance reduction | Cleaner signals | Low |
| Medium | Learnable alpha per layer | Adaptive strength | Medium |
| Low | Sparse training (1–3 active layers/step) | Independent signals | Low |

**Core bottleneck**: Individual strategy gains are +0.1–0.2%. Router needs strategies delivering +1% to unlock its full value.

---

## References

- [UAC: Uncertainty-aware Attention Calibration](https://arxiv.org/abs/2502.01969)
- [AdaIAT: Adaptive Inference-Time Attention Tuning](https://arxiv.org/abs/2603.04908)
- [VCD: Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922)
- [POPE: Polling-based Object Probing Evaluation](https://arxiv.org/abs/2305.10355)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

---

## License

This project is for research purposes. See individual papers for their respective licenses.
