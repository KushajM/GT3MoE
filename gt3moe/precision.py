"""Single source of truth for the dtype policy.

The original code had a precision discontinuity at ``epoch == warmup_epochs``: the warmup path
used ``GradScaler.scale/unscale_/step`` while the post-warmup path did raw ``autograd.grad`` plus
``opt.step()`` with no scaler at all. It also computed the K x C competence matrix natively in
fp16, because ``folk.accumulate`` was called *outside* the autocast block, so autocast's fp32
promotion list never applied to that ``cross_entropy``.

``Precision`` splits the forward into two regions:
  * backbone   -- autocast (bf16 by default; fp16 reproduces v1)
  * head       -- router + experts + gate + CE + cost matrix + aux losses, forced fp32 by default

and it hard-disables autocast in the head region rather than merely leaving the context, because
"outside the context" is what produced the fp16 cost matrix in the first place.
"""
from __future__ import annotations

import contextlib

import torch

AMP_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}


class Precision:
    def __init__(self, amp: str = "bf16", tf32: str = "off",
                 expert_dtype: str = "fp32", eval_amp: bool = False,
                 device: str = "cuda"):
        assert amp in ("bf16", "fp16", "fp32"), amp
        assert tf32 in ("on", "off"), tf32
        assert expert_dtype in ("fp32", "amp"), expert_dtype
        self.amp = amp
        self.tf32 = tf32
        self.expert_dtype = expert_dtype
        self.eval_amp = eval_amp
        self.device = device
        self.cuda = device == "cuda" and torch.cuda.is_available()

        if self.cuda and amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "bf16 is unsupported on this GPU (Turing/T4 has no bf16 hardware). Use --amp fp16.")

        on = tf32 == "on"
        torch.backends.cuda.matmul.allow_tf32 = on
        torch.backends.cudnn.allow_tf32 = on
        torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------ contexts
    def _autocast(self, enabled: bool = True):
        if not self.cuda or self.amp == "fp32" or not enabled:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=AMP_DTYPE[self.amp])

    def backbone_autocast(self):
        return self._autocast(True)

    def head_autocast(self):
        """Region covering router, experts, gate, CE, cost matrix and aux losses."""
        if self.expert_dtype == "fp32":
            if not self.cuda:
                return contextlib.nullcontext()
            return torch.autocast("cuda", enabled=False)   # hard-disable, nests safely
        return self._autocast(True)

    def eval_autocast(self):
        return self._autocast(self.eval_amp)

    def fp32(self):
        """Force fp32 anywhere (PCGrad dots, EMA, JS loss)."""
        if not self.cuda:
            return contextlib.nullcontext()
        return torch.autocast("cuda", enabled=False)

    # ------------------------------------------------------------------ misc
    def needs_scaler(self) -> bool:
        return self.cuda and self.amp == "fp16"

    def env(self) -> dict:
        d = dict(amp=self.amp, tf32=self.tf32, expert_dtype=self.expert_dtype,
                 eval_amp=self.eval_amp,
                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_tf32=torch.backends.cudnn.allow_tf32,
                 torch=torch.__version__, cuda=torch.version.cuda)
        if self.cuda:
            d["gpu"] = torch.cuda.get_device_name(0)
            d["capability"] = list(torch.cuda.get_device_capability(0))
        else:
            d["gpu"] = "cpu"
            d["capability"] = None
        return d


def assert_finite_grads(params, where: str):
    """The original lost its non-finite guard when it dropped the scaler post-warmup.

    ``scaler.step`` silently skips a step whose gradients contain inf/NaN; the raw
    ``opt.step()`` path does not, so a single overflow poisoned the weights permanently.
    This restores the guard on every path. Returns True if the step should be skipped.
    """
    bad = False
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            bad = True
            break
    return bad
