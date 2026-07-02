"""Paper-DAC + Paper-DoLa on Qwen3 POPE adversarial, 500 samples."""
import json, os, torch, random
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb
from PIL import Image
import torch.nn.functional as F
from pathlib import Path

MODEL_DIR = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl"
IMAGE_DIR = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\val2014\val2014"
POPE_FILE = r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\POPE-main\POPE-main\output\coco\coco_pope_adversarial.json"
OUT_DIR = Path(r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\router_project\pope_results\pope_strict")
OUT_DIR.mkdir(parents=True, exist_ok=True)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--method", type=str, default="dac", choices=["dac","dola","baseline"])
args = ap.parse_args()
random.seed(42)

# Load POPE questions (JSONL)
questions = [json.loads(l) for l in open(POPE_FILE, encoding="utf-8")][:args.n]
print(f"{args.method}: {len(questions)} POPE questions")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
    local_files_only=True, attn_implementation="eager")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()
H = model.model.language_model.config.num_attention_heads
KV_H = model.model.language_model.config.num_key_value_heads
LM_LAYERS = len(model.model.language_model.layers)

# Answer token IDs
YES_ID = 9454  # yes
NO_ID = 2750   # no

# ─── Paper-DAC calibration ───
if args.method == "dac":
    print("Calibrating DAC (blank images)...")
    # Quick calibration with 10 blank images across L5-L25
    from collections import defaultdict
    seg_data = [json.loads(l) for l in open(r"G:\claude code_workspace\GRPO-VLLM-hallucination-reduction\Qwen3vl\POPE-main\POPE-main\segmentation\coco_ground_truth_segmentation.json", encoding="utf-8")]
    all_imgs = [e for e in seg_data if os.path.exists(os.path.join(IMAGE_DIR, e["image"]))]
    cal_layers = range(5, min(26, LM_LAYERS))
    BLANK_BIAS = {}
    for L in tqdm(cal_layers, desc="DAC calib"):
        attn_mod = model.model.language_model.layers[L].self_attn
        bias_list = []
        def bh(m,i,o):
            if isinstance(o,tuple) and len(o)>=2 and o[1] is not None:
                bias_list.append(o[1][0,:,-1,:].detach().cpu())
        h = attn_mod.register_forward_hook(bh)
        for e in random.sample(all_imgs, 10):
            img = Image.open(os.path.join(IMAGE_DIR, e["image"])).convert("RGB")
            msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":"a"}]}]
            inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
            inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
            with torch.no_grad(): model.generate(**inputs, max_new_tokens=1)
            torch.cuda.empty_cache()
        h.remove()
        tl = bias_list[0].shape[1]
        sl = [b for b in bias_list if b.shape[1]==tl]
        BLANK_BIAS[L] = torch.stack(sl).mean(0) if sl else bias_list[0]
        print(f"  L{L}: {BLANK_BIAS[L].shape}")
    
    # Install DAC hooks
    hooks = []
    for L in BLANK_BIAS:
        attn_mod = model.model.language_model.layers[L].self_attn
        orig = attn_mod.forward
        bb = BLANK_BIAS[L]
        def make_fwd(am, bias):
            def fwd(hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
                bsz, q_len, _ = hidden_states.size()
                hdim = am.head_dim
                q = am.q_norm(am.q_proj(hidden_states).view(bsz,q_len,H,hdim)).transpose(1,2)
                k = am.k_norm(am.k_proj(hidden_states).view(bsz,q_len,KV_H,hdim)).transpose(1,2)
                v = am.v_proj(hidden_states).view(bsz,q_len,KV_H,hdim).transpose(1,2)
                cos, sin = position_embeddings; q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if past_key_values is not None: k, v = past_key_values.update(k, v, am.layer_idx)
                k2 = repeat_kv(k, am.num_key_value_groups); v2 = repeat_kv(v, am.num_key_value_groups)
                aw = torch.matmul(q, k2.transpose(-2,-1)) * am.scaling
                if attention_mask is not None: aw = aw + attention_mask[:,:,:,:k2.shape[-2]]
                aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
                Lk = min(bias.shape[1], aw.shape[-1])
                b = bias[:,:Lk].to(aw.device,aw.dtype).unsqueeze(0).unsqueeze(2)
                aw[:,:,-1:,:Lk] = aw[:,:,-1:,:Lk] - 0.5 * b
                aw[:,:,-1:,:Lk] = torch.clamp(aw[:,:,-1:,:Lk], min=0)
                aw[:,:,-1:,:Lk] = aw[:,:,-1:,:Lk] / aw[:,:,-1:,:Lk].sum(dim=-1,keepdim=True).clamp_min(1e-8)
                out = torch.matmul(aw, v2)
                out = out.transpose(1,2).contiguous().reshape(bsz,q_len,-1)
                return am.o_proj(out), aw
            return fwd
        attn_mod.forward = make_fwd(attn_mod, bb)
        hooks.append((attn_mod, orig))

# ─── Inference ───
results = []
for q in tqdm(questions, desc=f"POPE {args.method}"):
    img_path = os.path.join(IMAGE_DIR, q["image"])
    if not os.path.exists(img_path): continue
    img = Image.open(img_path).convert("RGB")
    msgs = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q["text"]}]}]
    inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)

    if args.method == "dola":
        # Prefill for DoLa bias
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        shallow_h = out.hidden_states[8][0, -1:, :]
        deep_h = out.hidden_states[LM_LAYERS-2][0, -1:, :]
        dola_bias = 0.5 * (model.lm_head(deep_h) - model.lm_head(shallow_h)).squeeze(0)
        class DoLaLP:
            def __init__(s, bias): s.bias = bias
            def __call__(s, ids, sc):
                if ids.shape[1]-1 == inputs["input_ids"].shape[1]:
                    return sc + s.bias.to(sc.device, sc.dtype)
                return sc
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=2, logits_processor=[DoLaLP(dola_bias)])
    else:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=2)
    
    tok_id = gen[0, inputs.input_ids.shape[1]:][0].item()
    ans = "yes" if tok_id == YES_ID else "no"
    results.append({"question_id": q.get("question_id",0), "image": q["image"], "answer": ans, "label": q["label"]})
    torch.cuda.empty_cache()

# Restore hooks
if args.method == "dac":
    for attn_mod, orig in hooks: attn_mod.forward = orig

# Evaluate
correct = sum(1 for r in results if r["answer"] == r["label"])
acc = correct / len(results)
tp = sum(1 for r in results if r["answer"]=="yes" and r["label"]=="yes")
fp = sum(1 for r in results if r["answer"]=="yes" and r["label"]=="no")
tn = sum(1 for r in results if r["answer"]=="no" and r["label"]=="no")
fn = sum(1 for r in results if r["answer"]=="no" and r["label"]=="yes")
prec = tp/(tp+fp) if (tp+fp) else 0
rec = tp/(tp+fn) if (tp+fn) else 0
f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
yes_ratio = (tp+fp)/len(results)

print(f"\n{args.method} POPE adversarial ({len(results)} samples):")
print(f"  Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} Yes={yes_ratio:.3f}")
print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")

# Save
out = OUT_DIR / f"pope_{args.method}.jsonl"
with open(str(out), "w", encoding="utf-8") as f:
    for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"Saved to {out}")
