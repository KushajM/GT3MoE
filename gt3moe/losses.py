"""Fast-timescale auxiliary losses.

**Alignment.** ``L_align = -mean log p_{b, pi(y_b)}``. Three fixes, each flagged:
  * computed as ``F.cross_entropy(router_logits, assigned)`` -- log-sum-exp stable, versus the
    original's ``-torch.log(g.clamp(min=1e-8))`` on an already-softmaxed tensor;
  * on **clean** router logits, not the noise-perturbed ones the original used for both selection
    and supervision;
  * optionally class-balanced. The original is sample-weighted, so head classes dominate the
    alignment gradient, while the potential ``Phi = sum_c L[pi(c),c]`` weights classes uniformly.
    The fast and slow timescales therefore optimise differently-weighted versions of the same
    objective -- the code accidentally implements a weighted alignment against an unweighted
    potential.
  * ``hinge_topk`` stops applying pressure once the assigned expert is already selected, which
    removes the structural conflict between a hard one-hot target and top-2 routing.

**Diversity.** ``L_comp = mean cos(z_i1, z_i2)`` on expert *logits* has a decisive defect: softmax
is invariant to a global additive shift and expert logits enter the prediction linearly, so
``z_s -> z_s + kappa_s * 1`` leaves the prediction, the CE, the routing and every reported metric
exactly unchanged -- and it is realised inside the network by each expert head's output bias. But
cosine is not shift-invariant (``cos(z1 + kappa*1, z2) -> 1`` as ``kappa -> inf``), so the
optimiser is handed a free descent direction along which the loss falls and the model provably
does not change. Decomposing ``z = a e_y + r``, lowering cosine can also be achieved by shrinking
the true-class logit ``a``, which directly opposes CE, and since correct classification forces
``a1 a2 > 0`` the term applies permanent pressure to exactly the pair most likely to be jointly
right.

``js_confusion`` replaces it with a Jensen-Shannon divergence between the two experts' predictive
distributions **renormalised over C\\{y}**. It is gauge-invariant (defined on probabilities),
bounded, and its gradient w.r.t. ``z_s[y]`` is identically zero, so it cannot be reduced by
degrading the true class. It expresses the property actually wanted for a linearly combined
ensemble -- agree on the answer, disagree about the mistakes -- which is negative *error*
correlation, the covariance term of the bias-variance-covariance decomposition.

Note ``cosine_orig`` remains the DEFAULT so that a no-flags run is the paper.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

#: fp32-safe masking constant. Never -inf: 0 * inf produces NaN in both forward and backward.
NEG = -1e4


def switch_balance_loss(gate_probs: torch.Tensor, top_idx: torch.Tensor, K: int) -> torch.Tensor:
    """Shazeer/Fedus auxiliary balance loss, as the original implemented it."""
    oh = F.one_hot(top_idx, K).float()
    f = oh.sum(1).clamp(max=1).mean(0)
    P = gate_probs.mean(0)
    return K * (f * P).sum()


def align_loss(router_logits: torch.Tensor, labels: torch.Tensor, assignment: torch.Tensor,
               cfg, class_weight: torch.Tensor = None) -> torch.Tensor:
    """router_logits [B,K] (clean by default), labels [B], assignment [C] int64 on device."""
    if cfg.align_loss == "none":
        return router_logits.new_zeros(())

    if assignment.dim() == 2:
        # Overlapping assignment: `assignment` is [C, m] and a sample is correctly routed to ANY
        # of its class's m owners. The right objective is the negative log MARGINAL probability of
        # the owner set, -log sum_{k in owners(y)} p_k, which is the unique generalisation that
        # collapses to plain CE at m = 1 and never prefers one owner over another. Supervising a
        # single designated owner instead would re-impose the hard partition through the back door.
        owners = assignment[labels]                               # [B, m]
        sel = router_logits.gather(1, owners)                     # [B, m]
        per = torch.logsumexp(router_logits, dim=1) - torch.logsumexp(sel, dim=1)
        if cfg.align_class_balanced and class_weight is not None:
            w = class_weight[labels]
            return (per * w).sum() / w.sum().clamp(min=1e-8)
        return per.mean()

    target = assignment[labels]                                   # [B]
    if cfg.align_loss == "nll":
        per = F.cross_entropy(router_logits, target, reduction="none")
    elif cfg.align_loss == "smoothed":
        per = F.cross_entropy(router_logits, target, reduction="none",
                              label_smoothing=float(cfg.align_smooth))
    elif cfg.align_loss == "hinge_topk":
        # zero once the assigned expert is inside the top-k by at least the margin
        k = max(1, min(int(cfg.top_k), router_logits.shape[1]))
        tgt = router_logits.gather(1, target.unsqueeze(1)).squeeze(1)
        if k >= router_logits.shape[1]:
            per = torch.zeros_like(tgt)
        else:
            masked = router_logits.scatter(1, target.unsqueeze(1), float("-inf"))
            kth = masked.topk(k, dim=1).values[:, -1]             # k-th best among the others
            per = F.relu(kth + float(cfg.align_hinge_margin) - tgt)
    else:
        raise ValueError(f"unknown align_loss {cfg.align_loss!r}")

    if cfg.align_class_balanced and class_weight is not None:
        w = class_weight[labels]
        return (per * w).sum() / w.sum().clamp(min=1e-8)
    return per.mean()


def comp_loss(all_logits: torch.Tensor, top_idx: torch.Tensor, labels: torch.Tensor,
              top_gates: torch.Tensor, cfg) -> torch.Tensor:
    """Diversity/redundancy penalty between the two selected experts.

    ``all_logits`` [B,K,C] fp32, ``top_idx`` [B,k], ``top_gates`` [B,k].
    """
    if cfg.comp_loss == "none" or cfg.top_k < 2 or all_logits is None:
        return all_logits.new_zeros(()) if all_logits is not None else torch.zeros(())
    B = all_logits.shape[0]
    ar = torch.arange(B, device=all_logits.device)
    z1 = all_logits[ar, top_idx[:, 0]].float()
    z2 = all_logits[ar, top_idx[:, 1]].float()

    if cfg.comp_loss == "cosine_orig":
        per = F.cosine_similarity(z1, z2, dim=1)
    elif cfg.comp_loss == "centred_cosine_hinge":
        c1 = z1 - z1.mean(-1, keepdim=True)
        c2 = z2 - z2.mean(-1, keepdim=True)
        per = F.relu(F.cosine_similarity(c1, c2, dim=1) - float(cfg.comp_tau)).pow(2)
    elif cfg.comp_loss == "js_confusion":
        C = z1.shape[-1]
        m = F.one_hot(labels, C).bool()
        lq1 = F.log_softmax(z1.masked_fill(m, NEG), dim=-1)
        lq2 = F.log_softmax(z2.masked_fill(m, NEG), dim=-1)
        lm = torch.logaddexp(lq1, lq2) - math.log(2.0)
        js = 0.5 * ((lq1.exp() * (lq1 - lm)).sum(-1) + (lq2.exp() * (lq2 - lm)).sum(-1))
        per = 1.0 - js / math.log(2.0)                            # in [0,1], minimised at JS=log2
    else:
        raise ValueError(f"unknown comp_loss {cfg.comp_loss!r}")

    if cfg.comp_gate_weight:
        # 2*w1*w2 in [0, 0.5]; -> 0 as top-2 degenerates to top-1, so the term switches itself off
        # instead of regularising a pair the model has stopped using.
        per = per * (2.0 * top_gates[:, 0] * top_gates[:, 1]).detach()

    if cfg.comp_conf_gate > 0.0:
        with torch.no_grad():
            pooled = (top_gates.unsqueeze(-1) *
                      torch.gather(all_logits, 1,
                                   top_idx.unsqueeze(-1).expand(-1, -1, all_logits.shape[-1]))
                      ).sum(1)
            p_y = F.softmax(pooled.float(), -1).gather(1, labels[:, None]).squeeze(1)
            keep = p_y <= cfg.comp_conf_gate
        if keep.sum() == 0:
            return all_logits.new_zeros(())        # an empty .mean() is NaN
        per = per[keep]

    return per.mean()


@torch.no_grad()
def gauge_stats(all_logits: torch.Tensor) -> dict:
    """Falsification diagnostic for the gauge argument.

    ``mean_j z_i[j]`` is the component of expert i's logits along the all-ones direction. It is
    exactly the direction the model is invariant to, so if it drifts apart across experts over
    training, the reported ``L_comp`` value is measuring a coordinate change and not diversity.
    """
    means = all_logits.float().mean(-1)                # [B,K]
    per_expert = means.mean(0)                         # [K]
    return dict(gauge_mean_abs=float(per_expert.abs().mean().item()),
                gauge_spread=float(per_expert.std().item()),
                logit_rms=float(all_logits.float().pow(2).mean().sqrt().item()))
