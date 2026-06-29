# GRPO-VLLM 幻觉消减

**基于 GRPO 训练的逐层路由器，动态选择注意力修正策略以缓解 Qwen3-VL-2B-Instruct 的物体幻觉问题。**

[English](README.md) | [中文](#)

---

## 概述

视觉语言模型（如 Qwen3-VL）存在**物体幻觉**问题 — 模型会"看到"图像中并不存在的物体。本项目通过训练一个轻量级逐层路由器，为每一层 transformer 动态选择注意力修正策略（UAC / AdaIAT / VHR / UAC+VHR / None），并通过 **GRPO（Group Relative Policy Optimization）** 进行优化。

### 核心创新

不同于在所有层上应用固定干预策略，路由器**学习逐层、上下文相关的决策**：
- 14 个 LM 层（5–18）各有自己的小型 MLP 路由器
- 给定隐藏状态，路由器从 5 种策略中做出选择
- GRPO 通过组内相对优势放大微弱的逐样本奖励信号

### 策略总览

| 策略 | 机制 | 论文 |
|------|------|------|
| **UAC** | 在 log 空间校准注意力权重 | [Uncertainty-aware Attention Calibration](https://arxiv.org/abs/2502.01969) |
| **AdaIAT-U** | 针对问题 token 的自适应注意力放大 | [Adaptive Inference-Time Attention Tuning](https://arxiv.org/abs/2603.04908) |
| **VHR** | 基于视觉散度重新加权注意力头 | 自定义 |
| **UAC+VHR** | UAC 与 VHR 组合干预 | 自定义 |
| **None** | 直通（不干预） | — |

---

## 实验结果

### POPE 基准测试（Qwen3-VL-2B-Instruct）

| 子集 | Baseline | 最佳 Router | Δ |
|------|:--------:|:-----------:|:--:|
| Random | 91.53% | 91.60% | +0.07 |
| Popular | 89.10% | 89.10% | ±0.00 |
| **Adversarial** | 87.30% | **87.50%** | **+0.20** |

### 策略分布（Router v4）
`adaiat-U: 15.4% | none: 84.6% | uac: 0.0%`

路由器学会了保守策略 — 只在信号支持时才干预。

---

## 架构设计

```
Qwen3-VL-2B (冻结) → RouterManager (包装 52 个注意力前向)
                        → LM 层 5-18 各有一个 LayerRouter
                        → LayerRouter: hidden → pool → MLP(256) → 5 类 logits
                        → {UAC, AdaIAT-U, VHR, UAC+VHR, None}
                        → 无梯度 GRPO：分离 hidden states，损失仅过 router MLP
```

- **无梯度设计**：~4.9 GB 显存（8 GB 笔记本 GPU 安全），~0.4s/步
- **校准数据**：W（UAC）、M_U + threshold（AdaIAT-U）、VHD（VHR）
- **GRPO 奖励**：R = log P（正确答案 token）
- **稀疏采样**：每次前向仅 k 层激活，隔离单层信号

---

## 项目结构

```
router_project/
├── README.md                          # 英文说明
├── README_zh.md                       # 中文说明（本文件）
├── SUMMARY.md                         # 完整技术总结文档
├── pope_inference.py                  # Baseline 推理
├── pope_evaluate.py                   # 指标计算（TP/FP/TN/FN/Acc/Prec/Rec/F1）
├── pope_results/
│   ├── baseline/                      # Baseline 结果
│   ├── uac_layer15/                   # UAC 单层评估
│   ├── adaiat_u_layer15_a1/           # AdaIAT-U 单层评估
│   ├── router_v1/                     # Router v4 结果
│   ├── vhr_L15/                       # VHR 结果
│   ├── vcd_gamma1.0_n500/             # VCD 结果
│   ├── oracle/                        # Oracle 搜索
│   ├── oracle_champion/               # 最佳 Oracle 组合
│   └── compare.py                     # 并排对比脚本
└── router/
    ├── router_module.py               # RouterManager + LayerRouter（核心）
    ├── strategies.py                  # 策略实现：UAC / AdaIAT / VHR / UAC+VHR
    ├── grpo_train.py                  # GRPO 训练 v1（稀疏采样）
    ├── grpo_train_v2.py               # GRPO 训练 v2（随机探索，推荐）
    ├── dpo_train.py                   # DPO 训练（早期方案）
    ├── dpo_data.py                    # POPE 数据加载 + 图像级 train/valid 分割
    ├── calibration.py                 # Phase 0: W（UAC）+ M/threshold（AdaIAT）
    ├── calibrate_vhr.py               # VHD 校准（VHR 策略）
    ├── calc_steering.py               # CASAL 风格 steering vector
    ├── recalibrate_u.py               # 重新校准 AdaIAT-U
    ├── recalibrate_uac_real.py        # 在真实图像上重新校准 UAC
    ├── pope_inference_router.py       # Router argmax 推理
    ├── pope_inference_forced.py       # 强制单策略推理（消融实验用）
    ├── uac_inference.py               # 独立 UAC POPE 评估
    ├── adaiat_inference.py            # 独立 AdaIAT-V 评估
    ├── adaiat_u_inference.py          # 独立 AdaIAT-U 评估
    ├── vcd_inference.py               # 视觉对比解码（VCD）
    ├── smoke_vhr.py                   # VHR/UAC+VHR 冒烟测试
    ├── oracle_test.py                 # Oracle 策略搜索
    ├── test_casal_lime.py             # CASAL & LIME 评估
    ├── sweep_vhr_alpha.py             # VHR alpha 参数扫描
    ├── _rebuild_hard_set.py           # 构建困难训练集
    ├── _run_champion.py               # 冠军组合评估
    ├── _pope_analysis.py              # 分析工具
    ├── _analyze_calib.py              # 校准分析
    ├── _eval_all.py                   # 批量评估
    ├── overnight.py                   # 全自动流水线
    ├── overnight_log_*.txt            # 通宵运行日志
    └── checkpoints/
        ├── calibration.pt             # W + M + thresholds + VHD
        ├── router_weights_final.pt    # DPO 训练最终权重
        └── router_weights_grpo_v2_*.pt # GRPO 检查点
```

---

## 环境配置

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB) |
| CUDA | PyTorch 2.12.0 |
| 模型 | Qwen3-VL-2B-Instruct |
| 框架 | transformers 5.12.1 |
| Python | conda env `qwen3vl` |

### 环境激活

```bash
source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
cd /g/sample/Qwen3vl/router_project
```

### 外部依赖

| 资源 | 路径 |
|------|------|
| 模型权重 | `G:\sample\Qwen3vl\` |
| POPE 问题 | `G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco\` |
| COCO 图像 | `G:\sample\Qwen3vl\val2014\val2014\` |

---

## 使用指南

### 快速开始 — 评估 Baseline

```bash
python pope_evaluate.py
```

### 离线校准

```bash
# UAC + AdaIAT 校准
python router/calibration.py

# VHR 校准（视觉散度）
python router/calibrate_vhr.py
```

### GRPO 训练（推荐）

```bash
# v2：随机探索版本（效果最佳）
python router/grpo_train_v2.py --sparse_k 3 --grpo_k 4 --n_epochs 10

# v1：稀疏采样版本
python router/grpo_train.py
```

### DPO 训练（早期方案）

```bash
python router/dpo_train.py
```

### 使用训练好的 Router 评估

```bash
# Router argmax 推理
python router/pope_inference_router.py router/checkpoints/router_weights_final.pt

# 计算指标
python pope_evaluate.py router_v1

# 与 baseline 对比
python pope_results/compare.py baseline router_v1
```

### 独立策略评估

```bash
# UAC 单层评估
python router/uac_inference.py

# AdaIAT-U 单层评估
python router/adaiat_u_inference.py

# VCD（视觉对比解码）
python router/vcd_inference.py --gamma 1.0 --n 500

# VHR 冒烟测试
python router/smoke_vhr.py --layer 15 --strategy vhr

# 全自动通宵流水线
python router/overnight.py
```

---

## 关键工程发现

1. **无梯度重放**：直接通过注意力计算图反向传播 → 10.7 GB 显存（OOM）。分离 hidden states + 重算 router MLP → 4.9 GB（安全）。

2. **AdaIAT 目标不匹配**：原论文放大对「已生成文本」的注意力。POPE 单 token 答案在决策点 Tp 为空。通过将目标切换为问题 token（U）解决。

3. **信号方向至关重要**：AdaIAT-U 的 M > 1（正确答案更多关注问题 — 信号正确）。AdaIAT-V 的 M < 1（信号反向）。

4. **层范围选择关键**：LM 5–16 层 M > 1（视觉推理区）。17–18 层 M 降至 < 1（语义提炼边界）。与论文 2411.16724v3 的层划分一致。

5. **稀疏采样防止坍缩**：不启用稀疏采样时，router 坍缩至 94%+ 选择 none。sparse_k=2 时信号变得可检测。

6. **GRPO 优于 DPO**：GRPO 的组内相对优势可以放大 +0.1~0.2% 的微弱信号，而 DPO 因 chosen/rejected 奖励差异过小导致保守策略。

---

## 局限与未来方向

| 优先级 | 方向 | 预期收益 | 工作量 |
|:------:|------|:-------:|:-----:|
| 高 | UAC 多层组合搜索 | >1% | 中 |
| 高 | 增大 AdaIAT-U alpha（2.0/3.0） | 潜在阈值突破 | 低 |
| 中 | Router 增加 REINFORCE baseline | 信号更清晰 | 低 |
| 中 | 逐层可学习 alpha | 策略强度自适应 | 中 |
| 低 | 稀疏训练（每步 1–3 层活跃） | 独立信号 | 低 |

**核心瓶颈**：单个策略的增益仅 +0.1~0.2%。需要让候选策略本身达到 +1% 级别才能真正释放 router 的价值。

---

## 参考资料

- [UAC: Uncertainty-aware Attention Calibration](https://arxiv.org/abs/2502.01969)
- [AdaIAT: Adaptive Inference-Time Attention Tuning](https://arxiv.org/abs/2603.04908)
- [VCD: Visual Contrastive Decoding](https://arxiv.org/abs/2311.16922)
- [POPE: Polling-based Object Probing Evaluation](https://arxiv.org/abs/2305.10355)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)

---

## 许可

本项目用于学术研究目的。各参考论文的许可见原文。
