# POPE Baseline Results — Qwen3-VL-2B-Instruct

**Date:** 2026-06-27  
**Model:** Qwen3-VL-2B-Instruct (no hooks/modifications)  
**GPU:** NVIDIA GeForce RTX 5060 Laptop GPU  
**Framework:** transformers 5.12.1, torch 2.12.0  
**Images:** COCO val2014 (500 images, 3 questions/image = 1500 per subset)  

## Results

| Subset         | Accuracy | Precision | Recall | F1     | Yes Ratio |
|---------------:|----------|-----------|--------|--------|-----------|
| Random         | 0.9153   | 0.9807    | 0.8473 | 0.9092 | 0.4320    |
| Popular        | 0.8910   | 0.9229    | 0.8533 | 0.8867 | 0.4623    |
| Adversarial    | 0.8730   | 0.8899    | 0.8513 | 0.8702 | 0.4783    |

### Confusion Matrix

| Subset      |  TP  |  FP  |  TN  |  FN  |
|------------:|:----:|:----:|:----:|:----:|
| Random      | 1271 |  25  | 1475 | 229  |
| Popular     | 1280 | 107  | 1393 | 220  |
| Adversarial | 1277 | 158  | 1342 | 223  |

## Key Observations

- **Random negatives** are easiest (Acc=91.53%): model rarely says "yes" to random unrelated objects.
- **Popular negatives** are harder (Acc=89.10%): common objects co-appearing with the positive ones cause more false positives (FP=107).
- **Adversarial negatives** are hardest (Acc=87.30%): the model has the most difficulty distinguishing co-occurring objects, with highest FP (158) and Yes Ratio (0.4783).

## Comparison

To compare with hook results, run:
```bash
source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
python /g/sample/Qwen3vl/router_project/pope_results/compare.py baseline router_v1
```
