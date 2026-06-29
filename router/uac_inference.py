"""
uac_inference.py — UAC standalone evaluation (log-tanh bounded correction v2).

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    python router/uac_inference.py --layer 15 --outdir uac_real_L15
"""
import json, os, sys, argparse, torch, torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb

MODEL_DIR = r"G:\sample\Qwen3vl"; POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"; RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=15); ap.add_argument("--outdir", type=str, default=None)
args = ap.parse_args()
LAYER, OUT_NAME = args.layer, args.outdir or f"uac_v2_L{LAYER}"
OUTPUT_DIR = os.path.join(RESULTS_BASE, OUT_NAME); os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"UAC v2 (log-tanh) layer {LAYER} -> {OUTPUT_DIR}")

# ── Load model ─────────────────────────────────────────────────────
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0", local_files_only=True, attn_implementation="eager")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True); model.eval()
for p in model.parameters(): p.requires_grad = False
attn_mod = model.model.language_model.layers[LAYER].self_attn; EPS = 1e-8

# ── Load real-image W from calibration.pt ──────────────────────────
calib = torch.load(os.path.join(CHECKPOINT_DIR, "calibration.pt"), map_location="cpu", weights_only=False)
W_all = calib.get("W", {}); name = f"lm.{LAYER}"
W_dict = W_all.get(name, None)
if W_dict is None:
    print(f"ERROR: No W found for layer {LAYER} in calibration.pt. Run recalibrate_uac_real.py first.")
    sys.exit(1)
if isinstance(W_dict, dict):
    print(f"Loaded real-image W: {len(W_dict)} resolutions: {sorted(W_dict.keys())}")
else:
    print(f"Loaded legacy W: shape {W_dict.shape}")

# ── Install UAC v2 hook (log-tanh bounded correction) ──────────────
state = {"w_tensor": None, "prefill_done": False}; orig = attn_mod.forward

def uac_forward(hidden_states, position_embeddings, attention_mask, past_key_values=None, **kw):
    is_prefill = not state["prefill_done"]
    if is_prefill: state["prefill_done"] = True
    inp_shape = hidden_states.shape[:-1]; hid_shape = (*inp_shape, -1, attn_mod.head_dim)
    q = attn_mod.q_norm(attn_mod.q_proj(hidden_states).view(hid_shape)).transpose(1,2)
    k = attn_mod.k_norm(attn_mod.k_proj(hidden_states).view(hid_shape)).transpose(1,2)
    v = attn_mod.v_proj(hidden_states).view(hid_shape).transpose(1,2)
    cos,sin=position_embeddings; q,k=apply_rotary_pos_emb(q,k,cos,sin)
    if past_key_values is not None: k,v = past_key_values.update(k,v,attn_mod.layer_idx)
    ka=repeat_kv(k,attn_mod.num_key_value_groups); va=repeat_kv(v,attn_mod.num_key_value_groups)
    aw=torch.matmul(q,ka.transpose(2,3))*attn_mod.scaling
    if attention_mask is not None: aw=aw+attention_mask[:,:,:,:ka.shape[-2]]
    aw=F.softmax(aw,dim=-1,dtype=torch.float32).to(q.dtype)

    if is_prefill and state["w_tensor"] is not None:
        w = state["w_tensor"].to(device=aw.device, dtype=aw.dtype)
        if w.dim() == 3: w = w.squeeze(0)
        Hw, Lw = w.shape; _, Ha, _, La = aw.shape
        if Hw != Ha: w = w[:Ha, :] if Hw >= Ha else w.expand(Ha, -1)
        Lk_apply = min(Lw, La)
        row = aw[:, :, -1:, :Lk_apply]
        # log-tanh bounded: corr ∈ [1-α, 1+α], α=0.77 → [0.23, 1.77]
        log_w = torch.log(w[:, :Lk_apply].clamp_min(1e-6))
        corr = 1.0 + 0.77 * torch.tanh(log_w)
        corr = corr.unsqueeze(0).unsqueeze(2)
        row = row * corr
        aw[:, :, -1:, :Lk_apply] = row / row.sum(dim=-1, keepdim=True).clamp_min(EPS)

    out=torch.matmul(aw,va); out=out.transpose(1,2).contiguous().reshape(*inp_shape,-1).contiguous()
    return attn_mod.o_proj(out), aw

attn_mod.forward = uac_forward; print("UAC v2 hook installed (log-tanh bounded)")

# ── POPE inference ──────────────────────────────────────────────────
def answer_yes_no(t):
    t=t.strip().lower()
    if "." in t:t=t.split(".")[0]
    t=t.replace(",","");w=t.split()
    return "no" if ("no" in w or "not" in w) else "yes"

for subset in ["random","popular","adversarial"]:
    qs=[json.loads(l) for l in open(os.path.join(POPE_DIR,f"coco_pope_{subset}.json"),encoding="utf-8")]
    out_file=os.path.join(OUTPUT_DIR,f"coco_pope_{subset}_answers.json"); results=[]
    for q in tqdm(qs,desc=f"POPE {subset}"):
        img=Image.open(os.path.join(IMAGE_DIR,q["image"])).convert("RGB")
        msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":q["text"]+" Please answer yes or no."}]}]
        inp=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
        # Find n_vis and pick matching W
        n_vis=(inp["mm_token_type_ids"][0]>0).sum().item()
        if isinstance(W_dict, dict):
            keys=sorted(W_dict.keys())
            state["w_tensor"]=W_dict[n_vis] if n_vis in W_dict else W_dict[min(keys,key=lambda k:abs(k-n_vis))]
        else:
            state["w_tensor"]=W_dict
        state["prefill_done"]=False
        with torch.no_grad(): gen=model.generate(**inp,max_new_tokens=16)
        raw=processor.decode(gen[0,inp.input_ids.shape[1]:],skip_special_tokens=True)
        results.append({"question":q["text"],"answer":answer_yes_no(raw),"raw_output":raw})
    with open(out_file,"w",encoding="utf-8") as f:
        for r in results: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"  -> Saved {len(results)} to {out_file}")

attn_mod.forward = orig; print("Done")
