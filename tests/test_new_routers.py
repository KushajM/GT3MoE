"""Contract and reduction tests for the two 2024-25 routers.

The contract matters more than it looks. ``route`` derives K from ``all_logits``, but AdaMoE
widens its gate to K+N internally and DynMoE activates a variable number of experts per sample.
If either leaked that width into the returned dict, ``switch_balance_loss`` would broadcast-error,
``_sparse_expert`` would index expert weights out of bounds, and the ``index_add_`` accumulators in
evaluate.py would corrupt the utilisation counts -- some of that silently. These tests pin the
width-K invariant so a future change to either router cannot quietly violate it.
"""
from __future__ import annotations

import torch

from gt3moe.config import make_config
from gt3moe.routers import route

B, K, C, D = 32, 8, 100, 16


def _cfg(**kw):
    return make_config("v2", device="cpu", num_experts=K, num_classes=C, **kw)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, K, generator=g),
            torch.randn(B, K, C, generator=g),
            torch.randn(B, D, generator=g))


def _check_contract(r, k):
    """Every router must return width-K gates and in-range expert indices."""
    assert r["gate_probs"].shape == (B, K), r["gate_probs"].shape
    assert torch.allclose(r["gate_probs"].sum(1), torch.ones(B), atol=1e-5)
    ti = r["top_idx"]
    assert ti.dtype == torch.int64
    assert ti.shape[0] == B and ti.shape[1] <= k
    assert int(ti.min()) >= 0 and int(ti.max()) < K
    assert r["out"].shape == (B, C)
    assert r["aux"] is None or torch.is_tensor(r["aux"])


def test_adamoe_and_dynmoe_respect_the_width_K_contract():
    ls, al, ft = _inputs()
    for name, kw in (("adamoe", dict(null_logits=torch.randn(B, 4))),
                     ("dynmoe", dict(tau=torch.zeros(K)))):
        cfg = _cfg(router=name, top_k=2)
        _check_contract(route(name, ls, al, cfg, True, feats=ft, **kw), 2)


def test_adamoe_with_zero_null_experts_selects_what_switch_selects():
    """With no null experts the gate is a plain softmax over K, so the SELECTION must match
    Switch exactly. The gate VALUES still differ: AdaMoE deliberately does not renormalise over
    the selected slots -- dropping the unselected mass is the adaptive-compute mechanism."""
    ls, al, ft = _inputs(1)
    cfg = _cfg(router="adamoe", top_k=3)
    a = route("adamoe", ls, al, cfg, True, feats=ft, null_logits=torch.zeros(B, 0))
    s = route("switch", ls, al, _cfg(router="switch", top_k=3), True, feats=ft)
    assert torch.equal(a["top_idx"].sort(1).values, s["top_idx"].sort(1).values)


def test_adamoe_null_experts_reduce_the_realised_expert_count():
    """Null experts that dominate the gate must actually displace real ones."""
    ls, al, ft = _inputs(2)
    cfg = _cfg(router="adamoe", top_k=4)
    lo = route("adamoe", ls, al, cfg, True, feats=ft, null_logits=torch.full((B, 4), -20.0))
    hi = route("adamoe", ls, al, cfg, True, feats=ft, null_logits=torch.full((B, 4), 20.0))
    assert lo["experts_per_sample"] == 4.0          # nulls never win
    assert hi["experts_per_sample"] < 1.01          # nulls take every slot -> orphan fallback
    assert int(hi["top_idx"].min()) >= 0 and int(hi["top_idx"].max()) < K


def test_dynmoe_threshold_extremes_span_dense_and_top1():
    ls, al, ft = _inputs(3)
    cfg = _cfg(router="dynmoe", top_k=2)
    dense = route("dynmoe", ls, al, cfg, True, feats=ft, tau=torch.full((K,), -1e9))
    top1 = route("dynmoe", ls, al, cfg, True, feats=ft, tau=torch.full((K,), 1e9))
    assert dense["experts_per_sample"] == float(K)
    assert top1["experts_per_sample"] == 1.0
    # tau -> -inf must reproduce the plain softmax-weighted dense combine over all K
    ref = (torch.softmax(ls, dim=1).unsqueeze(-1) * al).sum(1)
    assert torch.allclose(dense["out"], ref, atol=1e-5)
    # tau -> +inf leaves exactly the argmax expert carrying all the mass
    assert torch.equal(top1["gate_probs"].argmax(1), ls.argmax(1))
    assert torch.allclose(top1["gate_probs"].max(1).values, torch.ones(B), atol=1e-5)
