# Router Project — 完整总结文档

**日期**: 2026-06-27 ~ 2026-06-28  
**目标**: 为 Qwen3-VL-2B-Instruct 训练 DPO-gated attention router，对中间层动态选择 UAC / AdaIAT / None 策略来缓解 POPE 物体幻觉。  
**作者**: 用户 + Claude Code (Opus 4.8)  
**项目路径**: `G:\sample\Qwen3vl\router_project\`

---

## 1. 环境与基础配置

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 5060 Laptop (8GB) |
| CUDA | Available (PyTorch 2.12.0) |
| 模型 | Qwen3-VL-2B-Instruct (`G:\sample\Qwen3vl\`) |
| 框架 | transformers 5.12.1, eager attention |
| Python | conda env `qwen3vl` at `G:\Conda` |
| 激活命令 | `source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl` |
| 模型层数 | 24 ViT + 28 LM = **52 层** |
| 视觉编码器 | 24 blocks, hidden 1024, 16 heads |
| 语言模型 | 28 layers, hidden 2048, 16 Q heads / 8 KV heads |
| POPE 数据 | 9000 题 (3 subsets × 3000), 500 COCO val2014 图像 |
| 源论文 | UAC: `G:\chrome下载\2502.01969v2.pdf`, AdaIAT: `G:\chrome下载\2603.04908v1.pdf` |

---

## 2. Baseline 结果

| Subset | Accuracy | Precision | Recall | F1 | Yes Ratio |
|--------|:---:|:---:|:---:|:---:|:---:|
| Random | 0.9153 | 0.9807 | 0.8473 | 0.9092 | 0.4320 |
| Popular | 0.8910 | 0.9229 | 0.8533 | 0.8867 | 0.4623 |
| Adversarial | 0.8730 | 0.8899 | 0.8513 | 0.8702 | 0.4783 |

---

## 3. 策略单独评测（layer 15）

### 3.1 UAC（论文: 2502.01969）

**论文原版**: W = avg(A_blank) / A_blank, A' = W·A。单层解码器，视觉 token 注意力。  

**修正过程**: 初版全 52 层应用 → 生成垃圾。修正为单层 + 仅视觉 token + 行内重归一化。  

| Subset | Baseline | UAC L15 | Δ |
|--------|:---:|:---:|:---:|
| Random | 0.9153 | 0.9170 | +0.17 |
| Popular | 0.8910 | 0.8927 | +0.17 |
| Adversarial | 0.8730 | 0.8727 | **−0.03** |

**结论**: Random/Popular 正增益，Adversarial 中性。增益偏小 (+0.17%)。

### 3.2 AdaIAT-V（论文: 2603.04908，适配到图像 token V）

**为何需要适配**: 原版放大对"已生成文本 Tp"的注意力，POPE yes/no 答案在 prefill 时 Tp 为空——根本性不适用。  

**适配**: 目标换成图像 token V。校准发现 M=0.95 (<1)，信号反向（正确答案反而更少关注图像）。  

| Subset | Baseline | AdaIAT-V L15 | Δ |
|--------|:---:|:---:|:---:|
| Random | 0.9153 | 0.9180 | **+0.27** |
| Popular | 0.8910 | 0.8907 | −0.03 |
| Adversarial | 0.8730 | 0.8730 | 0.00 |

**结论**: Random 最佳但 Adversarial 不涨，且 M<1 信号反向。

### 3.3 AdaIAT-U（目标换成问题 token U）⭐

**关键发现**: 问题 token U 的 M=1.023 (>1)——正确答案确实更多关注问题。这是实验中最关键的信号验证。  

| Subset | Baseline | AdaIAT-U L15, α=1.0 | Δ |
|--------|:---:|:---:|:---:|
| Random | 0.9153 | 0.9163 | +0.10 |
| Popular | 0.8910 | 0.8927 | +0.17 |
| **Adversarial** | 0.8730 | **0.8750** | **+0.20** |

**结论**: 唯一一个三 subset 全正增益的策略，且 Adversarial 首次正数。

### 3.4 策略汇总

| 策略 | 信号方向 | 优点 | 弱点 |
|------|:---:|------|------|
| UAC | W 校准 | Random/Popular +0.17 | Adversarial 不动 |
| AdaIAT-V | M<1 反向 | — | 信号反向，不推荐 |
| **AdaIAT-U** | **M>1 正向** | **全正，Adv +0.20** | 增益仍偏小 |

---

## 4. Router 训练历程

### 4.1 架构

```
基座模型 (frozen) → RouterManager (wrap 52 attention forwards)
                    → 中间 LM 层 5-18 各有一个 LayerRouter
                    → LayerRouter: hidden → pool → MLP → 3-class logits
                    → {UAC, AdaIAT-U, None} per layer
                    → grad-free DPO: 保存 detached hidden states, loss 只过 router MLP
```

- **grad-free 设计**: 显存 4.9GB 封顶（8GB 笔记本 GPU 安全），单步 ~0.4s
- **校准**: W（黑图，UAC）、M_U + threshold（baseline 正确/错误，AdaIAT-U）
- **DPO reward**: R = log P(correct label token)（连续 reward，100% 样本形成 pair）
- **alpha**: sigmoid(raw_alpha), 初始 0.77 (sigmoid(1.2))

### 4.2 演进

| 版本 | 层 | 候选策略 | alpha | 连续 reward | 结果 |
|:---:|------|------|:---:|:---:|------|
| v1 | LM 5-20 | UAC/AdaIAT-V/None | 0.5 | ❌ 二值 (5% 有效 pair) | 50% acc（崩） |
| v2 | LM 5-20 | UAC/AdaIAT-V/None | 0.5 | ✅ 连续 (100%) | ~baseline, 94% none |
| v3 | LM 5-20 | UAC/AdaIAT-V/None | 0.77 | ✅ | ~baseline, 94% none |
| **v4** | **LM 5-18** | **UAC/AdaIAT-U/None** | 0.77 | ✅ | **Adv +0.20** |

### 4.3 v4 最终结果

| Subset | Baseline | Router v4 | Δ |
|--------|:---:|:---:|:---:|
| Random | 0.9153 | 0.9160 | +0.07 |
| Popular | 0.8910 | 0.8910 | 0.00 |
| **Adversarial** | 0.8730 | **0.8750** | **+0.20** |

**策略分布**: `adaiat-U: 15.4% | none: 84.6% | uac: 0.0%`（3 subset 一致）

**各层 M_U 信号**（5-18 层，80 样本校准）:

| 层 | M_mean | M>1 heads | 层 | M_mean | M>1 heads |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 5 | 1.033 | 9/16 | 12 | 1.056 | 8/16 |
| 6 | 1.111 | 12/16 | 13 | 1.021 | 9/16 |
| 7 | 1.021 | 9/16 | 14 | 1.088 | 9/16 |
| 8 | 1.014 | 8/16 | 15 | 1.023 | 9/16 |
| 9 | 1.049 | 9/16 | 16 | 1.086 | 13/16 |
| 10 | 1.034 | 9/16 | 17 | 0.965 | 5/16 |
| 11 | 1.022 | 8/16 | 18 | 0.938 | 4/16 |

5-16 层全 M>1，17-18 进入语义提炼区 M<1——与论文 2411.16724v3 的层划分一致。

---

## 5. 关键工程发现

1. **Replay vs grad-free**: 直接 backprop 过 attention 计算图 → GPU 峰值 10.7GB（8GB 爆显存）。保存 detached hidden states + 重算 router MLP → 4.9GB，可行。

2. **POPE yes/no 与 AdaIAT 不匹配**: AdaIAT 原文放大对"已生成文本"的注意力，POPE 单 token 答案在决策点 Tp 为空。必须换目标（V/U）。

3. **Prefill 检测**: `past_key_values.get_seq_length()==0` 不可靠（同一次前向中，前面的层已追加 KV）。用 `_prefill_done` set 标记。

4. **层范围关键**: 视觉富集层 5-18 有效，17-18 靠近语义提炼边界（M 开始 <1）。

---

## 6. 根本瓶颈分析

**Router 学到了"不要多干预"的理性策略**（84.6% none），原因：

1. 单个候选策略 vs baseline 只有 **+0.1~0.2%** 的微弱增益
2. DPO 的连续 reward (log P 正确答案) 在"干预"和"不干预"时差异极小
3. 14 层同时随机采样 → 每层独立信号被淹没在全局噪声中
4. alpha 固定（未从 DPO 收到梯度）

**要突破必须让候选策略本身达到 +1% 级别增益。**

---

## 7. 下一步建议

| 优先级 | 方向 | 预期收益 | 工作量 |
|:---:|------|:---:|:---:|
| 高 | **UAC 多层组合搜索** (5-18 层, 每层 ± 开/关) | 单层 +0.17 → 多层叠加可达 >1% | 中 |
| 高 | **AdaIAT-U alpha 调大** (试 2.0/3.0) | 可能有临界增益 | 低 |
| 中 | **Router 加 REINFORCE baseline** 减方差 | 信号更清晰 | 低 |
| 中 | **Alpha 可学习** (加 LM log-prob 损失) | 策略强度自适应 | 中 |
| 低 | **Router 稀疏训练** (每步随机 1-3 层活跃) | 单层信号独立 | 低 |
| 低 | **Router 扩展到更多层** (试 1-18 或 5-24) | 覆盖更多注意力阶段 | 低 |

---

## 8. 文件地图

```
G:\sample\Qwen3vl\router_project\
├── README.md
├── pope_evaluate.py <dir>          # 计算 TP/FP/TN/FN/Acc/Prec/Rec/F1/Yes
├── pope_inference.py               # baseline 推理（原始）
├── pope_results/
│   ├── baseline/                   # baseline POPE 结果
│   ├── uac_layer15/                # UAC 单层评测
│   ├── adaiat_layer15_a0.5/        # AdaIAT-V 评测
│   ├── adaiat_u_layer15_a1/        # AdaIAT-U 评测
│   ├── router_v1/                  # Router v4 最终结果
│   └── compare.py <a> <b>          # 并排对比
├── router/
│   ├── router_module.py            # RouterManager + LayerRouter (核心)
│   ├── strategies.py               # apply_uac / apply_ada_iat_lm / apply_ada_iat_visual
│   ├── calibration.py              # Phase 0: W(blank) + M/threshold(baseline)
│   ├── recalibrate_u.py            # 重校准 AdaIAT-U (问题 token)
│   ├── dpo_data.py                 # POPE 加载 + 图像级 train/valid 分割
│   ├── dpo_train.py                # DPO 训练主循环 (grad-free, 连续 reward)
│   ├── pope_inference_router.py    # Router argmax 推理
│   ├── pope_inference_forced.py    # 强制单策略推理（消融用）
│   ├── uac_inference.py            # 独立 UAC POPE 评测
│   ├── adaiat_inference.py         # 独立 AdaIAT-V POPE 评测
│   ├── adaiat_u_inference.py       # 独立 AdaIAT-U POPE 评测
│   └── checkpoints/
│       ├── calibration.pt          # W + M + thresholds
│       └── router_weights_*.pt     # Router 权重 (v4 final 可用)
└── .claude/                        # 自动记忆
```

**外部依赖**（不在本目录）:
- 模型: `G:\sample\Qwen3vl\` (model.safetensors + config)
- POPE 数据: `G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco\`
- 图像: `G:\sample\Qwen3vl\val2014\val2014\`

---

## 10. CAI+BRACS (2026-06-30)

**CAI v2 calibration**: 50 images, caption vs non-caption → per-layer o_proj output offsets (norms 2-85, largest at deeper layers)

**CAI v2 alpha sweep** (200 adversarial): Best α=0.05, acc=0.860, Δ=−1.30%. ALL alphas ≤ baseline.

**结论**: CAI via o_proj steering 不可用。Caption attn offset norms (40-85) 太大，即使微小偏移也破坏 yes/no 决策。

## 11. 所有方法的诚实汇总

| 方法 | Adversarial Δ | 状态 |
|------|:---:|------|
| AdaIAT-U L15 | +0.20% | 唯一正向 |
| UAC L15 | −0.03% | 噪声级 |
| VCD (all γ) | ≤ 0 | 全负 |
| CASAL/LIME | −1~4% | 不可用 |
| VHR/UAC+VHR | −2~4% | 不可用 |
| GRPO v2 Router | −1.8% | 退化 |
| Oracle 3×3×3 | +1.8% | 上界 |
| **CAI v2** | **−1.3%** | **不可用** |

**贯穿规律**: 2B 模型在 POPE 上决策边界已高度校准（87.3%），任何 training-free 全局偏移都会引入等量反例错误。需要更强候选策略或放弃 POPE 天花板。

## 9. 恢复指南

下次打开项目时:

```bash
source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
cd /g/sample/Qwen3vl/router_project

# 查看已有结果
python pope_results/compare.py baseline router_v1

# 重新训练 router
python router/dpo_train.py

# 用训练好的 router 跑 POPE
python router/pope_inference_router.py router/checkpoints/router_weights_final.pt
python pope_evaluate.py router_v1
python pope_results/compare.py baseline router_v1
```

对话记忆自动加载（`MEMORY.md` 在项目目录下），说"继续 router 项目"即可接上。

---

> **一句话总结**: Router 机械上完全跑通（grad-free DPO, 连续 reward, 14 层独立路由），AdaIAT-U 信号方向正确（M>1），Adversarial 首次 +0.20。根本瓶颈在候选策略本身增益太小（+0.1~0.2%），需要多层组合或更大 alpha 提至 +1% 级别才能让 router 真正发挥价值。
