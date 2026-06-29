"""
router_module.py — LayerRouter + RouterManager (v2 — fixed gradient flow).

Design:
  - Replaces each attention module's forward with our own that computes
    Q,K,V → softmax → strategy → V matmul directly.
  - Router decisions happen ONCE per prefill step, then reused for decode.
  - Added "replay" mode: forces router to use pre-recorded decisions and
    returns log_softmax of those decisions as tensors with grad.

Key fix: Phase 1 samples decisions (eval mode, generate), Phase 2 replays
  those SAME decisions (train mode, model.forward) to get grad-tracked log_probs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    repeat_kv, apply_rotary_pos_emb, apply_rotary_pos_emb_vision,
)

from strategies import apply_uac, apply_ada_iat_lm, apply_ada_iat_visual, apply_vhr, apply_uac_vhr


# ─── LayerRouter ────────────────────────────────────────────────────

class LayerRouter(nn.Module):
    def __init__(self, hidden_dim, n_strategies):
        super().__init__()
        self.pool_proj = nn.Linear(hidden_dim, 1, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, n_strategies),
        )

    def forward(self, hidden_states):
        hs = hidden_states.float()
        scores = self.pool_proj(hs)
        w = F.softmax(scores, dim=1)
        pooled = (hs * w).sum(dim=1)
        return self.mlp(pooled)


# ─── Alpha / Strategy mappings ──────────────────────────────────────

def alpha_block(lt, idx):
    if lt == "visual":
        if idx < 6: return "v_0_5"
        elif idx < 12: return "v_6_11"
        elif idx < 18: return "v_12_17"
        else: return "v_18_23"
    else:
        if idx < 7: return "l_0_6"
        elif idx < 14: return "l_7_13"
        elif idx < 21: return "l_14_20"
        else: return "l_21_27"

def strategy_opts(lt, idx):
    if lt == "visual":
        return ["uac", "none"]
    if 5 <= idx <= 18:
        return ["uac", "adaiat", "vhr", "uac_vhr", "none"]
    return ["none"]


# ─── RouterManager ──────────────────────────────────────────────────

class RouterManager:
    def __init__(self, model, calib, active_layers=None, alpha_init=0.0):
        self.model = model
        self.W = calib.get("W", {})
        self.M = calib.get("M", {})
        self.thresholds = calib.get("thresholds", {})
        self.VHD = calib.get("VHD", {})  # VHR: per-layer per-head vision divergence
        self._vhd_medians = {}  # cached medians for VHR (computed once per run)

        self.descs = []
        for i, blk in enumerate(model.model.visual.blocks):
            self.descs.append(dict(name=f"visual.{i}", module=blk.attn,
                                   type="visual", idx=i,
                                   ablock=alpha_block("visual", i),
                                   strategies=strategy_opts("visual", i)))
        for i, lyr in enumerate(model.model.language_model.layers):
            self.descs.append(dict(name=f"lm.{i}", module=lyr.self_attn,
                                   type="lm", idx=i,
                                   ablock=alpha_block("lm", i),
                                   strategies=strategy_opts("lm", i)))

        # Routers — only created for active layers (others always use "none")
        if active_layers is None:
            active_layers = {d["name"] for d in self.descs}  # all active by default
        self._active_layers = set(active_layers)
        self._router_list = nn.ModuleList()
        self._router_map = {}
        for d in self.descs:
            if d["name"] not in self._active_layers:
                continue
            dim = 1024 if d["type"] == "visual" else 2048
            r = LayerRouter(dim, len(d["strategies"]))
            self._router_list.append(r)
            self._router_map[d["name"]] = r
        print(f"Active routable layers: {len(self._router_map)} / {len(self.descs)}")

        # Alphas (sigmoid-constrained to [0,1]) — only for blocks containing active layers
        active_blocks = {d["ablock"] for d in self.descs if d["name"] in self._active_layers}
        blocks = sorted(active_blocks)
        self.raw_alphas = nn.ParameterDict()
        for b in blocks:
            self.raw_alphas[b] = nn.Parameter(torch.tensor(alpha_init))

        dev = next(model.parameters()).device
        self._router_list.to(dev)

        # Mode & state
        self.mode = "sample"       # "sample" | "argmax" | "replay" | "collect" | "force" | "force_per_layer"
        self.force_strategy = None # strategy name for "force" mode (e.g. "uac","adaiat","none")
        self._force_per_layer = {} # dict: name -> strategy_idx for "force_per_layer" mode
        self.gumbel_tau = 1.0      # temperature for gumbel-softmax
        self._sparse_k = None      # if set, only ~k active layers are allowed to pick non-"none";
        self._sparse_active = None # the subset chosen this forward pass
        self._logits = {}          # name → logits tensor
        self._decisions = {}       # name → int
        self._decided = set()
        self._prefill_done = set() # names that already did their prefill decision
        self._saved_inputs = {}    # name → detached hidden_states (grad-free replay)
        self._current_q_pos = None  # tensor of question key positions (set per input)
        self._current_n_vis = None  # int: number of vision tokens (for UAC dict W lookup)
        self._force_decisions = {} # name → int (for replay mode)
        self._collected = {}       # name → list[Tensor] (for calibration)
        self._orig_fns = {}
        self._wrapped = False

    def get_alpha(self, b):
        return torch.sigmoid(self.raw_alphas[b])

    # ── Decision logic ─────────────────────────────────────────────

    def _decide(self, name, hidden_states):
        """Make a router decision for one layer. Saves detached input for
        grad-free replay (avoids backprop through the full attention graph)."""
        if name in self._decided:
            return
        strategies = self._strategies_for(name)

        # Inactive layer: always use "none" (no intervention, no router call)
        if name not in self._router_map:
            d = strategies.index("none") if "none" in strategies else 0
            self._decisions[name] = d
            self._decided.add(name)
            return

        # ── Sparse sampling: in "sample" mode, randomly pick only ~k active
        # layers per forward pass. Non-sampled layers forced to "none".
        # This isolates each layer's decision signal for DPO, preventing the
        # "all-none-drowns-the-signal" collapse seen previously.
        # In "argmax" (inference) mode, ALL layers use their trained router.
        if self.mode == "sample" and self._sparse_k is not None and self._sparse_k > 0:
            if self._sparse_sampled is None:
                active_names = list(self._router_map.keys())
                k = min(self._sparse_k, len(active_names))
                chosen = random.sample(active_names, k)
                self._sparse_sampled = set(chosen)
            if name not in self._sparse_sampled:
                d = strategies.index("none") if "none" in strategies else 0
                self._decisions[name] = d
                self._decided.add(name)
                return

        router = self._router_map[name]
        logits = router(hidden_states)  # (1, C) with grad track

        if self.mode == "argmax":
            d = int(logits.argmax(dim=-1).item())
        elif self.mode == "replay":
            # Force the pre-recorded decision, but keep logits for grad
            d = self._force_decisions.get(name, 0)
        elif self.mode == "collect":
            d = 0
        elif self.mode == "force":
            # Force a single strategy across all layers (for ablation).
            if self.force_strategy in strategies:
                d = strategies.index(self.force_strategy)
            elif "none" in strategies:
                d = strategies.index("none")
            else:
                d = 0
        elif self.mode == "force_per_layer":
            # Per-layer forced strategy (for GRPO random exploration).
            # _force_per_layer must be set before calling clear_cache().
            idx = self._force_per_layer.get(name)
            if idx is not None and idx < len(strategies):
                d = idx
            elif "none" in strategies:
                d = strategies.index("none")
            else:
                d = 0
        else:
            # sample mode: Gumbel-Softmax with configurable tau
            gumbel = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
            d = int(gumbel.argmax(dim=-1).item())

        self._logits[name] = logits
        self._decisions[name] = d
        self._decided.add(name)
        # Save detached input so we can recompute logits WITH grad later
        # without re-running the attention forward (huge memory/compute save).
        self._saved_inputs[name] = hidden_states.detach()

    def _strategy_desc(self, name):
        for d in self.descs:
            if d["name"] == name:
                s = d["strategies"]
                idx = min(self._decisions.get(name, len(s) - 1), len(s) - 1)
                return s[idx], d
        return "none", {}

    def _strategies_for(self, name):
        for d in self.descs:
            if d["name"] == name:
                return d["strategies"]
        return ["none"]

    def _apply_strategy(self, name, attn_w, strategy, desc, attn_out=None):
        """Apply strategy. For vhr/uac_vhr, also needs attn_output (post-V matmul).
        Returns (attn_w, attn_out) where attn_out may be modified by VHR."""
        if strategy == "none":
            return attn_w, attn_out
        alpha = self.get_alpha(desc["ablock"])
        if strategy == "uac":
            w_mat = self.W.get(name)
            if w_mat is not None:
                # If W is a dict {n_vis: tensor}, pick the closest resolution
                if isinstance(w_mat, dict) and self._current_n_vis is not None:
                    nv = self._current_n_vis
                    if nv in w_mat:
                        w_to_use = w_mat[nv]
                    else:
                        closest = min(w_mat.keys(), key=lambda k: abs(k - nv))
                        w_to_use = w_mat[closest]
                else:
                    w_to_use = w_mat
                return apply_uac(attn_w, w_to_use, alpha), attn_out
        elif strategy == "adaiat":
            m = self.M.get(name)
            thresh = self.thresholds.get(name, 0.0)
            if m is not None:
                if desc["type"] == "visual":
                    return apply_ada_iat_visual(attn_w, m, thresh, alpha), attn_out
                else:
                    # LM AdaIAT-U: amplify attention to question tokens
                    q_pos = self._current_q_pos if self._current_q_pos is not None else torch.arange(attn_w.shape[-1])
                    return apply_ada_iat_lm(attn_w, m, thresh, alpha, q_pos), attn_out
        elif strategy == "vhr":
            vhd = self.VHD.get(name)
            if vhd is not None and attn_out is not None:
                attn_out = apply_vhr(attn_out, vhd, alpha, self._vhd_medians, name)
            return attn_w, attn_out
        elif strategy == "uac_vhr":
            # Phase 1: UAC on attention weights
            w_mat = self.W.get(name)
            if w_mat is not None and attn_out is not None:
                if isinstance(w_mat, dict) and self._current_n_vis is not None:
                    nv = self._current_n_vis
                    if nv in w_mat:
                        w_to_use = w_mat[nv]
                    else:
                        closest = min(w_mat.keys(), key=lambda k: abs(k - nv))
                        w_to_use = w_mat[closest]
                else:
                    w_to_use = w_mat
                vhd = self.VHD.get(name)
                if vhd is not None:
                    attn_w, attn_out = apply_uac_vhr(
                        attn_w, attn_out, w_to_use, vhd, alpha,
                        self._vhd_medians, name,
                    )
            return attn_w, attn_out
        return attn_w, attn_out

    # ── Wrap / Unwrap ──────────────────────────────────────────────

    def wrap_all(self):
        if self._wrapped:
            return
        for d in self.descs:
            name = d["name"]
            module = d["module"]
            self._orig_fns[name] = module.forward
            if d["type"] == "lm":
                module.forward = self._make_lm_forward(name, module)
            else:
                module.forward = self._make_vit_forward(name, module)
        self._wrapped = True
        print(f"Wrapped {len(self.descs)} attention modules")

    def unwrap_all(self):
        for d in self.descs:
            if d["name"] in self._orig_fns:
                d["module"].forward = self._orig_fns[d["name"]]
        self._wrapped = False

    # ── LM forward (reimplemented) ─────────────────────────────────

    def _make_lm_forward(self, name, m):
        rm = self

        def forward(hidden_states, position_embeddings, attention_mask,
                   past_key_values=None, **kwargs):
            # Detect first forward pass of a sequence (not per-token generation).
            # past_key_values.get_seq_length() returns >0 DURING prefill because
            # earlier layers in the same forward pass have already appended their KVs.
            # So we check: has any token been generated yet?
            # We track this via a flag: _prefill_done={name} — cleared each clear_cache().
            is_prefill = name not in rm._prefill_done
            if is_prefill:
                rm._prefill_done.add(name)
                rm._decide(name, hidden_states)
            strategy, desc = rm._strategy_desc(name)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, m.head_dim)

            q = m.q_norm(m.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            k = m.k_norm(m.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            v = m.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            if past_key_values is not None:
                k, v = past_key_values.update(k, v, m.layer_idx)

            k_attn = repeat_kv(k, m.num_key_value_groups)
            v_attn = repeat_kv(v, m.num_key_value_groups)

            attn_w = torch.matmul(q, k_attn.transpose(2, 3)) * m.scaling
            if attention_mask is not None:
                attn_w = attn_w + attention_mask[:, :, :, :k_attn.shape[-2]]
            attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_w = F.dropout(attn_w, p=m.attention_dropout, training=m.training)

            # VHR and UAC+VHR need access to the attention output (post-V matmul)
            # before o_proj.  Compute that first, then apply strategy.
            attn_out = torch.matmul(attn_w, v_attn)  # (1, H, Lq, D_head)
            attn_out = attn_out.transpose(1, 2)        # (1, Lq, H, D_head)

            attn_w, attn_out = rm._apply_strategy(name, attn_w, strategy, desc, attn_out)

            if rm.mode == "collect":
                rm._collected.setdefault(name, []).append(attn_w.detach().cpu())

            if attn_out is not None:
                out = attn_out.contiguous()
            else:
                out = torch.matmul(attn_w, v_attn)
                out = out.transpose(1, 2).contiguous()
            out = out.reshape(*input_shape, -1).contiguous()
            out = m.o_proj(out)
            return out, attn_w

        return forward

    # ── ViT forward (reimplemented) ────────────────────────────────

    def _make_vit_forward(self, name, m):
        rm = self

        def forward(hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
            seq_len = hidden_states.shape[0]
            rm._decide(name, hidden_states.unsqueeze(0))
            strategy, desc = rm._strategy_desc(name)

            q, k, v = (
                m.qkv(hidden_states)
                .reshape(seq_len, 3, m.num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )

            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)

            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [torch.split(t, lengths.tolist(), dim=2) for t in (q, k, v)]

            outputs = []
            for qc, kc, vc in zip(*splits):
                attn_w = torch.matmul(qc, kc.transpose(2, 3)) * m.scaling
                attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(qc.dtype)

                attn_w, _ = rm._apply_strategy(name, attn_w, strategy, desc, None)

                if rm.mode == "collect":
                    rm._collected.setdefault(name, []).append(attn_w.detach().cpu())

                out_c = torch.matmul(attn_w, vc)
                outputs.append(out_c)

            out = torch.cat(outputs, dim=1)
            out = out.transpose(1, 2).reshape(seq_len, -1).contiguous()
            out = m.proj(out)
            return out

        return forward

    # ── State management ───────────────────────────────────────────

    def save_decisions(self):
        """Snapshot current decisions (call after Phase 1 generate)."""
        return dict(self._decisions)

    def load_decisions(self, saved):
        """Load decisions for replay mode (call before Phase 2 forward)."""
        self._force_decisions = dict(saved)

    def clear_cache(self):
        self._logits.clear()
        self._decisions.clear()
        self._decided.clear()
        self._prefill_done.clear()
        self._saved_inputs.clear()
        self._collected.clear()
        self._current_q_pos = None
        self._current_n_vis = None
        self._sparse_sampled = None  # re-randomize layer subset each forward

    def save_path(self):
        """Snapshot current decisions + saved inputs (call after a sample forward).
        Returns dict name → (decision_int, detached_hidden_tensor)."""
        return {name: (self._decisions[name], self._saved_inputs[name])
                for name in self._decisions if name in self._saved_inputs}

    def get_total_log_prob(self):
        """Sum of log_softmax across all layers (returns Python float)."""
        total = 0.0
        for name in self._logits:
            logits = self._logits[name]
            idx = self._decisions[name]
            total += F.log_softmax(logits, dim=-1)[0, idx].item()
        return total

    def compute_log_prob_tensor(self):
        """Sum of log_softmax across all layers (returns differentiable tensor).
        Uses freshly computed logits from the last forward."""
        total = None
        for name in self._logits:
            if name in self._decisions:
                logits = self._logits[name]
                idx = self._decisions[name]
                lp = F.log_softmax(logits, dim=-1)[0, idx]
                total = lp if total is None else total + lp
        return total  # None if no decisions

    def compute_log_prob_from_saved(self, decisions):
        """Recompute log_prob WITH grad from saved detached inputs + a given
        decision set. Does NOT re-run the attention forward — only the tiny
        router MLPs. This is the fast/low-memory path for DPO training.

        Args:
            decisions: dict name → int (the decision indices to score)
        Returns:
            differentiable scalar tensor (sum of log_softmax over layers),
            or None if no saved inputs.
        """
        total = None
        for name, d in decisions.items():
            hs = self._saved_inputs.get(name)
            if hs is None:
                continue
            router = self._router_map[name]
            # Recompute logits with grad through router params only.
            # hs is detached, so no graph through attention.
            logits = router(hs)
            lp = F.log_softmax(logits, dim=-1)[0, d]
            total = lp if total is None else total + lp
        return total

    def compute_entropy_from_saved(self, decisions):
        """Average entropy of router distributions from saved inputs."""
        ent = None
        count = 0
        for name in decisions:
            hs = self._saved_inputs.get(name)
            if hs is None:
                continue
            logits = self._router_map[name](hs)
            probs = F.softmax(logits, dim=-1)
            e = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            ent = e if ent is None else ent + e
            count += 1
        if ent is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return ent / max(count, 1)

    def compute_entropy(self):
        """Compute average entropy across all router logits (differentiable)."""
        ent = None
        count = 0
        for logits in self._logits.values():
            probs = F.softmax(logits, dim=-1)
            e = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            ent = e if ent is None else ent + e
            count += 1
        return ent / max(count, 1) if ent is not None else torch.tensor(0.0)

    # ── Utilities ──────────────────────────────────────────────────

    def to(self, dev):
        self._router_list.to(dev)
        return self

    def train(self, mode=True):
        self._router_list.train(mode)
        return self

    def eval(self):
        self._router_list.eval()
        return self

    def parameters(self):
        return list(self._router_list.parameters()) + list(self.raw_alphas.parameters())

    def state_dict(self):
        return {"routers": self._router_list.state_dict(),
                "raw_alphas": self.raw_alphas.state_dict()}

    def load_state_dict(self, sd):
        self._router_list.load_state_dict(sd["routers"])
        self.raw_alphas.load_state_dict(sd["raw_alphas"])

    @property
    def num_routers(self):
        return len(self._router_map)
