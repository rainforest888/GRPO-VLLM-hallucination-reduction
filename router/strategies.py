"""
strategies.py — Attention correction strategies: UAC, AdaIAT-U, VHR, UAC+VHR.

All alpha parameters passed as Tensors (not .item() scalars) to preserve
gradient flow from DPO loss back to router parameters.
"""
import torch
import torch.nn.functional as F


def apply_uac(attn_weights: torch.Tensor, W, alpha: torch.Tensor = None) -> torch.Tensor:
    """
    UAC v2: calibrate the last query row with bounded log-space correction.

    Problem with v1: W = mean(A)/A ranges from 0.01 to 68608 because some heads
    barely attend to certain vision tokens.  The raw correction 1+α·(W-1) then
    amplifies a single token by ~6000×, and row renormalisation crushes
    everything else to zero — the correction self-destructs.

    Fix: work in log space.  log(W) measures how many orders of magnitude a
    position deviates from the mean attention.  tanh bounds this to [-1, 1],
    so the final correction ∈ [1-α, 1+α] regardless of how extreme W gets.
    With α=0.77 the range is [0.23, 1.77] — a mild 1.77× amplification at
    most, comparable to AdaIAT's per-head M factors.
    """
    if not isinstance(W, torch.Tensor):
        return attn_weights

    w = W.to(device=attn_weights.device, dtype=attn_weights.dtype)
    if w.dim() == 3:
        w = w.squeeze(0)
    H_w, Lk_w = w.shape
    _, H_a, _, Lk_a = attn_weights.shape

    if H_w != H_a:
        if H_w == 1:
            w = w.expand(H_a, -1)
        else:
            w = F.interpolate(w.unsqueeze(0), size=(H_a, Lk_w), mode='nearest').squeeze(0)

    Lk_apply = min(Lk_w, Lk_a)
    row = attn_weights[:, :, -1:, :Lk_apply]

    # --- log-tanh correction (bounded) ---
    log_w = torch.log(w[:, :Lk_apply].clamp_min(1e-6))   # log(mean(A)/A)
    corr = 1.0 + alpha * torch.tanh(log_w)                # ∈ [1-α, 1+α]
    corr = corr.unsqueeze(0).unsqueeze(2)                 # (1, H, 1, Lk_apply)

    row = row * corr
    attn_weights[:, :, -1:, :Lk_apply] = row / row.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return attn_weights


def apply_ada_iat_lm(
    attn_weights: torch.Tensor,
    M: torch.Tensor,
    threshold: float,
    alpha: torch.Tensor,
    question_positions: torch.Tensor,
) -> torch.Tensor:
    """
    AdaIAT-U for LM layers: adaptively amplify last query row's attention
    to QUESTION token keys (U target).

    Args:
        attn_weights:      (1, H, Lq, Lk)
        M:                 (H,) per-head amplification factor
        threshold:         scalar trigger threshold
        alpha:             scalar Tensor in [0,1] with grad
        question_positions: (N_q,) tensor of key indices for question tokens
    """
    B, H, Lq, Lk = attn_weights.shape
    q_idx = question_positions.to(device=attn_weights.device, dtype=torch.long)

    a_q = attn_weights[:, :, -1, q_idx]  # (1, H, N_q)
    atp_current = a_q.mean()

    if atp_current < threshold:
        m = M.to(device=attn_weights.device, dtype=attn_weights.dtype)
        amp = 1.0 + alpha * m  # (H,) with grad
        attn_weights[:, :, -1:, q_idx] *= amp.view(1, H, 1, 1)
        row = attn_weights[:, :, -1:, :]
        row_sum = row.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn_weights[:, :, -1:, :] = row / row_sum
    return attn_weights


def apply_ada_iat_visual(
    attn_weights: torch.Tensor,
    M: torch.Tensor,
    threshold: float,
    alpha: torch.Tensor,
    top_k_ratio: float = 0.1,
) -> torch.Tensor:
    B, H, Lq, Lk = attn_weights.shape
    K_top = max(1, int(Lk * top_k_ratio))
    mean_attn = attn_weights.mean(dim=2)
    _, topk_indices = torch.topk(mean_attn, K_top, dim=-1)
    gathered = torch.gather(mean_attn, dim=-1, index=topk_indices)
    atp_current = gathered.mean()
    if atp_current < threshold:
        m = M.to(device=attn_weights.device, dtype=attn_weights.dtype)
        amp = 1.0 + alpha * m
        mask = torch.zeros_like(attn_weights)
        mask.scatter_(-1, topk_indices.unsqueeze(2).expand(-1, H, Lq, -1), 1.0)
        attn_weights = attn_weights * (1.0 + mask * (amp.view(1, H, 1, 1) - 1.0))
        attn_weights = F.softmax(
            attn_weights - attn_weights.max(dim=-1, keepdim=True).values, dim=-1
        )
    return attn_weights


def apply_vhr(
    attn_output: torch.Tensor,
    VHD: torch.Tensor,
    alpha: torch.Tensor,
    median_cache: dict = None,
    layer_name: str = None,
) -> torch.Tensor:
    """
    VHR (Vision-aware Head Reinforcement): scale the attention OUTPUT of
    vision-aware heads by a factor > 1.

    VHD_{l,h} = || A_{l,h}(with_img) - A_{l,h}(without_img) ||_2
    (precomputed offline in calibrate_vhr.py).

    The top 50% heads per layer (VHD > median) are amplified by (1 + alpha).
    This reorients the FFN input toward vision-grounded directions.

    Args:
        attn_output: (1, Lq, H, D_head) — attention output BEFORE o_proj
        VHD:         (H,) per-head divergence scores
        alpha:       scalar Tensor ∈ [0,1], scaling factor
        median_cache:dict from layer_name -> median_VHD (computed once per run)
        layer_name:  str, used for median_cache key

    Returns:
        attn_output with vision-aware heads scaled up
    """
    if VHD is None:
        return attn_output

    vhd = VHD.to(device=attn_output.device, dtype=attn_output.dtype)
    H = vhd.shape[0]

    # Use precomputed median or compute fresh
    if median_cache is not None and layer_name is not None:
        if layer_name not in median_cache:
            median_cache[layer_name] = vhd.median()
        median = median_cache[layer_name]
    else:
        median = vhd.median()

    vision_heads = (vhd > median).nonzero(as_tuple=True)[0]  # top 50%
    if vision_heads.numel() == 0:
        return attn_output

    # Scale selected heads: out[:, :, h, :] *= (1 + alpha)
    # attn_output is (1, Lq, H, D_head)
    scale = 1.0 + alpha  # Tensor with grad
    attn_output[:, :, vision_heads, :] *= scale
    return attn_output


def apply_uac_vhr(
    attn_weights: torch.Tensor,
    attn_output: torch.Tensor,
    W,
    VHD: torch.Tensor,
    alpha: torch.Tensor,
    median_cache: dict = None,
    layer_name: str = None,
) -> tuple:
    """
    Fusion: UAC (attention weight calibration) → VHR (head output scaling).

    Phase 1: UAC log-tanh bounded correction on attention weights
    Phase 2: VHR amplify vision-aware heads in the output

    Returns (corrected_attn_w, corrected_attn_output).
    The caller must use corrected_attn_w for the V matmul, then
    use corrected_attn_output = VHR(...) on the result.

    Since apply_uac returns corrected attn_w and apply_vhr modifies attn_output,
    this is a convenience wrapper that calls both.

    Args:
        attn_weights: (1, H, Lq, Lk)
        attn_output:  (1, Lq, H, D_head) = attn_w @ V, before o_proj
        W:            UAC calibration matrix
        VHD:          per-head divergence
        alpha:        scaling factor tensor

    Returns:
        (attn_weights_corrected, attn_output_corrected)
    """
    # Phase 1: UAC
    aw = apply_uac(attn_weights, W, alpha)
    # Phase 2: VHR
    ao = apply_vhr(attn_output, VHD, alpha, median_cache, layer_name)
    return aw, ao
