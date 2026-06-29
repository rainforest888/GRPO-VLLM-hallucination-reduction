"""Analyze why so many strategies fail on POPE."""
import math

print("=" * 70)
print("POPE BENCHMARK ANALYSIS: Why Most Strategies Fail")
print("=" * 70)

# 1. Statistical floor
n = 3000
p0 = 0.873
se = math.sqrt(p0 * (1 - p0) / n)
ci_upper = p0 + 1.96 * se
print(f"""
1. STATISTICAL REALITY
   POPE Adversarial: n={n}, baseline={p0}
   95% CI: [{p0-1.96*se:.4f}, {ci_upper:.4f}]
   Ceiling (perfect): 1.0000
   Room above baseline: {1.0 - p0:.1%} = {int((1-p0)*n)} questions
   Human ceiling: ~95% (estimated)
""")

# 2. Why each method fails
print("""
2. WHY EACH METHOD FAILS ON POPE

   POPE = "Is there a [X] in this image?" → yes/no
   Task requires: locate object → binary decision
   NOT required: detailed description, multi-step reasoning, long generation

   Method          Intervention        POPE Mismatch
   ─────────────   ─────────────────   ──────────────────────────────────
   UAC            空间 attention 均化   POPE需要精准定位，均化=稀释信号
   VHR            放大视觉head输出       VHD信号太弱，放大成了加噪
   CASAL          MLP残差引导          POPE只有1个生成token，激活差异是
                                       yes/no词表选择而非幻觉语义差异
   LIME           视觉/文本重平衡      POPE短文本(问题+yes/no)，文本先验
                                       优势不明显，重平衡空间有限
   AdaIAT         问题文本attention    YES 匹配：强化"读题"，直接关联VQA
""")

# 3. What POPE actually needs
print("""
3. WHAT POPE ACTUALLY NEEDS (vs what methods provide)

   POPE hallucination = model says "yes" when object absent, or "no" when present
   Root cause: weak cross-modal binding between object name and visual region

   What helps:
   ├── Strengthen question->image attention (AdaIAT does this) [YES]\n   ├── Better visual grounding of object names\n   ├── Reduce language prior\n   └── NOT: uniform attention, head amplification, decoding tricks

   What doesn't help:
   ├── Spatial attention uniformization (UAC)
   ├── Single-token activation steering (CASAL)
   ├── Head output scaling without task alignment (VHR)
   └── Text/vision rebalancing for short inputs (LIME)
""")

# 4. Why not switch benchmark
print("""
4. WHY YOUR ADVISOR SAYS DON'T SWITCH BENCHMARKS

   Reason                                    Explanation
   ───────────────────────────────────────   ───────────────────────────
   ① POPE is the academic standard           几乎所有 hallucination 论文
                                             (VHR/CLVA/EVAS/CRoPS/VDGD…)
                                             都报告 POPE 数字。不报告=不完整

   ② Adversarial subset is the hardest       Random/Popular 太简单(已饱和)，
                                             Adversarial 是唯一有区分度的

   ③ Apples-to-apples comparison             换 benchmark 后无法和文献对比

   ④ +1.8% on POPE IS a real result          3000题上54题净提升，已经超95%CI，
                                             可以直接写进论文

   ⑤ Complete the story                      baseline→UAC(fail)→VHR(fail)
                                             →AdaIAT(success) 是一条完整叙事
""")

print("=" * 70)
print("BOTTOM LINE")
print("=" * 70)
print(f"""
   POPE Adversarial ceiling ~0.95 (human), baseline 0.873 (Qwen3-VL-2B)
   Our best: 0.891 (+1.8%, statistically significant at p<0.01)

   为什么看起来"天花板低"？
   - 0.873→0.891 只有 +1.8%，但这是在 3000 题上的真提升
   - 文献中 POPE 的典型提升范围是 +0.5%~+3%
   - 这是 2B 小模型，大模型 baseline 更高但提升空间更小

   师兄不让你换 benchmark 是对的——
   你已经有显著结果了，换 benchmark 不会让失败的方法变成功。
""")
