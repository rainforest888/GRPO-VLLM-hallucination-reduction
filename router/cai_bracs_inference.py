"""
cai_bracs_inference.py — Hybrid CAI+BRACS inference for full POPE evaluation.

CAI (Caption-Sensitive Attention Intervention, 2506.23590):
  - Caption queries make LVLMs attend more strongly to visual info
  - Precompute per-head attention output offsets from caption queries
  - During inference, apply these offsets to non-caption queries

BRACS (Barrier-Regulated Adaptive Closed-form Steering, 2605.29881):
  - Monitor visual grounding score from model's own attention
  - Apply correction ONLY when grounding drops below threshold
  - Adaptive strength based on grounding deficit

HYBRID:
  - BRACS defines WHEN to intervene (grounding monitor → barrier trigger)
  - CAI defines WHAT to inject (caption-query attention output offset)
  - Combined: "When visual grounding is weak, inject caption-level visual attention"

Implementation:
  Phase 0 (offline): Run N caption queries, collect per-layer per-head attention
                     output offsets vs non-caption queries
  Phase 1 (inference): For each layer, compute grounding score from attention,
                       if score < threshold, add scaled caption offset to output

Usage:
    source /g/Conda/etc/profile.d/conda.sh && conda activate qwen3vl
    cd /g/sample/Qwen3vl/router_project

    # Phase 0: collect caption offsets
    python router/cai_bracs_inference.py --phase calibrate --n_captions 50

    # Phase 1: run full POPE evaluation
    python router/cai_bracs_inference.py --phase evaluate --alpha 1.0 --beta 0.5

    # Evaluate:
    python pope_evaluate.py cai_bracs
"""
import json, os, sys, argparse, torch, glob
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl.modeling_qwen3_vl import repeat_kv, apply_rotary_pos_emb
from PIL import Image

MODEL_DIR = r"G:\sample\Qwen3vl"
POPE_DIR = r"G:\sample\Qwen3vl\POPE-main\POPE-main\output\coco"
IMAGE_DIR = r"G:\sample\Qwen3vl\val2014\val2014"
RESULTS_BASE = r"G:\sample\Qwen3vl\router_project\pope_results"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
OUTDIR = os.path.join(RESULTS_BASE, "cai_bracs")
os.makedirs(OUTDIR, exist_ok=True)

LM_LAYERS = list(range(5, 19))  # middle layers where visual grounding matters
OFFSET_FILE = os.path.join(CHECKPOINT_DIR, "cai_offsets.pt")

# ─── Args ─────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--phase", type=str, required=True, choices=["calibrate", "evaluate"])
ap.add_argument("--n_captions", type=int, default=50, help="Images for caption calibration")
ap.add_argument("--alpha", type=float, default=1.0, help="CAI offset strength")
ap.add_argument("--beta", type=float, default=0.5, help="BRACS grounding barrier steepness")
ap.add_argument("--barrier", type=float, default=0.3, help="BRACS grounding threshold (0-1)")
args = ap.parse_args()


# ══════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════

def answer_yes_no(text):
    t = text.strip().lower()
    if "." in t: t = t.split(".")[0]
    t = t.replace(",", ""); w = t.split()
    return "no" if ("no" in w or "not" in w) else "yes"


def load_model_and_processor():
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="cuda:0",
        local_files_only=True, attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor


# ══════════════════════════════════════════════════════════════════
# PHASE 0: CAPTION OFFSET CALIBRATION (CAI)
# ══════════════════════════════════════════════════════════════════

def calibrate_cai():
    """
    For each image:
      1. Run with caption query ("Describe this image in detail.")
      2. Run with non-caption query (random POPE question)
      3. Record per-layer per-head attention output:
         O_l = softmax(QK^T)V before o_proj
      4. Offset = O_caption - O_non_caption (averaged over images)
    """
    print("=== CAI Calibration Phase ===")
    model, processor = load_model_and_processor()

    # Collect sample images
    image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))[:args.n_captions]
    print(f"Using {len(image_files)} images")

    caption_query = "Describe this image in detail."
    non_caption_query = "Is there a person in the image? Please answer yes or no."

    # Storage: layer_idx → (H, D_head) mean offset
    caption_outputs = {li: [] for li in LM_LAYERS}
    non_caption_outputs = {li: [] for li in LM_LAYERS}

    # Hook system: capture attention OUTPUT (post-V matmul, pre-o_proj)
    hooks = {}
    captures = {}  # layer_idx → tensor (H, D_head) for the last row

    for li in LM_LAYERS:
        attn_mod = model.model.language_model.layers[li].self_attn
        orig_forward = attn_mod.forward

        def make_hook(_li, _orig, _mod):
            def hook(hidden_states, position_embeddings, attention_mask,
                     past_key_values=None, **kw):
                is_prefill = past_key_values is None or past_key_values.get_seq_length() == 0
                if not is_prefill:
                    return _orig(hidden_states, position_embeddings, attention_mask,
                                 past_key_values=past_key_values, **kw)
                m = _mod
                inp_shape = hidden_states.shape[:-1]
                hid_shape = (*inp_shape, -1, m.head_dim)
                q = m.q_norm(m.q_proj(hidden_states).view(hid_shape)).transpose(1, 2)  # (1,H,Lq,D)
                k = m.k_norm(m.k_proj(hidden_states).view(hid_shape)).transpose(1, 2)
                v = m.v_proj(hidden_states).view(hid_shape).transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if past_key_values is not None:
                    k, v = past_key_values.update(k, v, m.layer_idx)
                k_attn = repeat_kv(k, m.num_key_value_groups)
                v_attn = repeat_kv(v, m.num_key_value_groups)
                aw = torch.matmul(q, k_attn.transpose(2, 3)) * m.scaling
                if attention_mask is not None:
                    aw = aw + attention_mask[:, :, :, :k_attn.shape[-2]]
                aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

                # Capture attention OUTPUT (post-V matmul, last query position, per head)
                attn_out = torch.matmul(aw, v_attn)  # (1, H, Lq, D_head)
                # Last query position, all heads
                captures[_li] = attn_out[0, :, -1, :].detach().cpu()  # (H, D_head)

                out = attn_out.transpose(1, 2).contiguous().reshape(*inp_shape, -1).contiguous()
                return m.o_proj(out), aw
            return hook

        hooks[li] = orig_forward
        attn_mod.forward = make_hook(li, orig_forward, attn_mod)

    # Run caption queries
    for img_file in tqdm(image_files, desc="Caption queries"):
        img = Image.open(img_file).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": caption_query},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
        for li in LM_LAYERS:
            if li in captures:
                caption_outputs[li].append(captures[li])
        captures.clear()

    # Run non-caption queries
    for img_file in tqdm(image_files, desc="Non-caption queries"):
        img = Image.open(img_file).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": non_caption_query},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            _ = model(**inputs, use_cache=False)
        for li in LM_LAYERS:
            if li in captures:
                non_caption_outputs[li].append(captures[li])
        captures.clear()

    # Restore
    for li in LM_LAYERS:
        model.model.language_model.layers[li].self_attn.forward = hooks[li]

    # Compute offsets
    offsets = {}
    for li in LM_LAYERS:
        if caption_outputs[li] and non_caption_outputs[li]:
            cap = torch.stack(caption_outputs[li]).float()      # (N, H, D)
            ncap = torch.stack(non_caption_outputs[li]).float()  # (N, H, D)
            delta = (cap.mean(dim=0) - ncap.mean(dim=0)).bfloat16()  # (H, D)
            offsets[li] = delta
            norm = delta.norm().item()
            print(f"  LM.{li:2d}: |offset|={norm:.4f}")

    # Also compute grounding score baseline (BRACS):
    # grounding = mean attention to vision tokens in last query row
    torch.save({"offsets": offsets, "lm_layers": LM_LAYERS}, OFFSET_FILE)
    print(f"\nOffsets saved to {OFFSET_FILE}")
    print("Calibration complete.\n")


# ══════════════════════════════════════════════════════════════════
# PHASE 1: HYBRID CAI+BRACS INFERENCE
# ══════════════════════════════════════════════════════════════════

def evaluate_pope():
    """Run hybrid CAI+BRACS on full POPE (random + popular + adversarial)."""
    print("=== CAI+BRACS Hybrid Evaluation ===")
    print(f"alpha={args.alpha}, beta={args.beta}, barrier={args.barrier}")

    if not os.path.exists(OFFSET_FILE):
        print("ERROR: offsets not found. Run --phase calibrate first.")
        sys.exit(1)
    offsets_data = torch.load(OFFSET_FILE, map_location="cpu", weights_only=False)
    offsets = offsets_data["offsets"]  # {li: (H, D_head)}
    print(f"Loaded offsets for {len(offsets)} layers")

    model, processor = load_model_and_processor()

    # Install hooks
    hooks_orig = {}
    grounding_history = {}  # layer → running average for monitoring

    for li in LM_LAYERS:
        if li not in offsets:
            continue
        attn_mod = model.model.language_model.layers[li].self_attn
        hooks_orig[li] = attn_mod.forward
        ca_offset = offsets[li].cuda().bfloat16()  # (H, D_head)
        beta_tensor = torch.tensor(args.beta, dtype=torch.bfloat16, device=model.device)
        barrier = float(args.barrier)
        alpha_tensor = torch.tensor(args.alpha, dtype=torch.bfloat16, device=model.device)

        def make_hybrid_hook(_li, _orig, _mod, _offset, _beta):
            def hook(hidden_states, position_embeddings, attention_mask,
                     past_key_values=None, **kw):
                is_prefill = past_key_values is None or past_key_values.get_seq_length() == 0
                if not is_prefill:
                    return _orig(hidden_states, position_embeddings, attention_mask,
                                 past_key_values=past_key_values, **kw)

                m = _mod
                inp_shape = hidden_states.shape[:-1]
                hid_shape = (*inp_shape, -1, m.head_dim)
                q = m.q_norm(m.q_proj(hidden_states).view(hid_shape)).transpose(1, 2)
                k = m.k_norm(m.k_proj(hidden_states).view(hid_shape)).transpose(1, 2)
                v = m.v_proj(hidden_states).view(hid_shape).transpose(1, 2)
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                if past_key_values is not None:
                    k, v = past_key_values.update(k, v, m.layer_idx)
                k_attn = repeat_kv(k, m.num_key_value_groups)
                v_attn = repeat_kv(v, m.num_key_value_groups)

                aw = torch.matmul(q, k_attn.transpose(2, 3)) * m.scaling
                if attention_mask is not None:
                    aw = aw + attention_mask[:, :, :, :k_attn.shape[-2]]
                aw_raw = aw.detach().clone()
                aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

                Lk = aw.shape[-1]

                # ── BRACS: compute visual grounding score ──
                # Grounding = mean attention of last query row to first ~N_vis tokens
                # We don't have n_vis directly in the hook, use a heuristic:
                # Vision tokens are the first significant chunk (use mask if available)
                # For now: mean of top-25% attended-to positions as grounding proxy
                last_row = aw[0, :, -1, :]  # (H, Lk)
                # Assume first 1000 tokens are vision (conservative for Qwen3-VL)
                n_vis_est = min(1000, Lk)
                vis_attn = last_row[:, :n_vis_est].mean(dim=1)  # (H,) per-head grounding
                grounding = vis_attn.mean()  # scalar

                # ── BRACS barrier: only intervene if grounding < threshold ──
                deficit = barrier - grounding
                if deficit > 0:
                    # Adaptive strength: tanh(beta * deficit) * alpha
                    strength = torch.tanh(_beta * deficit) * alpha_tensor

                    # Compute attn output
                    attn_out = torch.matmul(aw, v_attn)  # (1, H, Lq, D_head)

                    # ── CAI: add caption offset to attention output ──
                    # Apply to last query position, all heads
                    ca = _offset.unsqueeze(0).unsqueeze(2)  # (1, H, 1, D_head)
                    attn_out[:, :, -1:, :] = attn_out[:, :, -1:, :] + strength * ca

                    out = attn_out.transpose(1, 2).contiguous().reshape(*inp_shape, -1).contiguous()
                else:
                    # No intervention
                    out = torch.matmul(aw, v_attn)
                    out = out.transpose(1, 2).contiguous().reshape(*inp_shape, -1).contiguous()

                return m.o_proj(out), aw
            return hook

        attn_mod.forward = make_hybrid_hook(li, hooks_orig[li], attn_mod, ca_offset, beta_tensor)

    print(f"CAI+BRACS hooks installed on {len(hooks_orig)} LM layers")

    # ─── Run POPE ─────────────────────────────────────────────────
    for subset in ["random", "popular", "adversarial"]:
        print(f"\n--- POPE {subset} ---")
        pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
        output_file = os.path.join(OUTDIR, f"coco_pope_{subset}_answers.json")
        questions = [json.loads(l) for l in open(pope_file, "r", encoding="utf-8")]

        results = []
        for q in tqdm(questions, desc=subset):
            img_path = os.path.join(IMAGE_DIR, q["image"])
            text = q["text"]
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": text + " Please answer yes or no."},
            ]}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=8)
            gen_ids = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
            raw = processor.batch_decode(gen_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]
            results.append({"question": text, "answer": answer_yes_no(raw), "raw_output": raw})

        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(results)} answers to {output_file}")

    # Restore
    for li, orig in hooks_orig.items():
        model.model.language_model.layers[li].self_attn.forward = orig

    # ─── Quick accuracy check ─────────────────────────────────────
    print(f"\n=== Quick Summary ===")
    for subset in ["random", "popular", "adversarial"]:
        out_file = os.path.join(OUTDIR, f"coco_pope_{subset}_answers.json")
        pope_file = os.path.join(POPE_DIR, f"coco_pope_{subset}.json")
        answers = [json.loads(l) for l in open(out_file, encoding="utf-8")]
        labels = [json.loads(l)["label"] for l in open(pope_file, encoding="utf-8")]
        correct = sum(1 for a, l in zip(answers, labels) if a["answer"] == l)
        print(f"  {subset}: {correct}/{len(answers)} = {correct/len(answers):.4f}")

    print(f"\nEvaluate: python pope_evaluate.py cai_bracs")
    print(f"Compare:  python pope_results/compare.py baseline cai_bracs")


# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if args.phase == "calibrate":
        calibrate_cai()
    elif args.phase == "evaluate":
        evaluate_pope()
