"""Long-tailed classifier losses, for the orthogonality claim the paper defers to future work.

The manuscript states that Balanced Softmax, LDAM-DRW, RIDE and BCL are orthogonal to GT3MoE and
that the combination "likely compounds", with no experiment. These make that testable. The
comparison that matters is *paired*: LT-loss + GT3MoE against LT-loss + dense, on the same loss,
same backbone, same seeds -- because the authors' own unreported run of exactly that composition
(Dense LDAM-DRW 0.5826 vs Folk+LDAM-DRW 0.5617) went the wrong way.

All three take the training class counts and modify only the logits or the per-class weighting, so
they compose with any router.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmax(nn.Module):
    """Ren et al., NeurIPS 2020. Add log prior to the logits, then plain CE."""

    def __init__(self, class_counts, device="cuda"):
        super().__init__()
        n = torch.as_tensor(np.asarray(class_counts), dtype=torch.float32, device=device)
        self.register_buffer("log_prior", (n / n.sum()).clamp(min=1e-12).log())

    def forward(self, logits, target, reduction="mean"):
        return F.cross_entropy(logits.float() + self.log_prior.unsqueeze(0), target,
                               reduction=reduction)


class LDAMLoss(nn.Module):
    """Cao et al., NeurIPS 2019. Per-class margin proportional to n_c^{-1/4}.

    ``set_drw_weights`` applies Deferred Re-Weighting: plain training until ``drw_start``, then
    class-balanced weights. DRW is the half of LDAM-DRW that does the long-tail work, so it must
    not be silently omitted.
    """

    def __init__(self, class_counts, max_margin=0.5, scale=30.0, device="cuda"):
        super().__init__()
        n = np.asarray(class_counts, dtype=np.float64)
        m = 1.0 / np.power(np.maximum(n, 1), 0.25)
        m = m * (max_margin / m.max())
        self.register_buffer("m_list", torch.tensor(m, dtype=torch.float32, device=device))
        self.scale = scale
        self.weight = None

    def set_drw_weights(self, class_counts, beta=0.9999, device="cuda"):
        n = np.asarray(class_counts, dtype=np.float64)
        eff = 1.0 - np.power(beta, np.maximum(n, 1))
        w = (1.0 - beta) / eff
        w = w / w.sum() * len(n)
        self.weight = torch.tensor(w, dtype=torch.float32, device=device)

    def forward(self, logits, target, reduction="mean"):
        x = logits.float()
        margins = self.m_list[target]
        x_m = x.clone()
        x_m.scatter_(1, target.unsqueeze(1), x.gather(1, target.unsqueeze(1)) - margins.unsqueeze(1))
        return F.cross_entropy(self.scale * x_m, target, weight=self.weight, reduction=reduction)


class ClassBalancedCE(nn.Module):
    """Cui et al., CVPR 2019. Weight by the inverse effective number (1-beta^n)/(1-beta)."""

    def __init__(self, class_counts, beta=0.9999, device="cuda"):
        super().__init__()
        n = np.asarray(class_counts, dtype=np.float64)
        eff = 1.0 - np.power(beta, np.maximum(n, 1))
        w = (1.0 - beta) / eff
        w = w / w.sum() * len(n)
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32, device=device))

    def forward(self, logits, target, reduction="mean"):
        return F.cross_entropy(logits.float(), target, weight=self.w, reduction=reduction)


class PlainCE(nn.Module):
    def forward(self, logits, target, reduction="mean"):
        return F.cross_entropy(logits.float(), target, reduction=reduction)


def make_loss(name: str, class_counts, device="cuda"):
    name = (name or "ce").lower()
    if name == "ce":
        return PlainCE()
    if name in ("balsoftmax", "balanced_softmax"):
        return BalancedSoftmax(class_counts, device)
    if name in ("ldam", "ldam_drw"):
        return LDAMLoss(class_counts, device=device)
    if name in ("cb_ce", "cbce"):
        return ClassBalancedCE(class_counts, device=device)
    raise ValueError(f"unknown base_loss {name!r}")
