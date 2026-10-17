"""Correctness invariants. Run with: python tests/test_invariants.py

These are the properties whose violation would silently corrupt a result, so each one is asserted
rather than eyeballed.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gt3moe import assign as A
from gt3moe import losses as L
from gt3moe.config import PAPER_V1, make_config
from gt3moe.pcgrad import (build_groups, make_perms, pcgrad_coefficients, pcgrad_naive,
                           set_flat_grad)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------- 1. the solver

def test_exact_solver():
    rng = np.random.default_rng(0)
    bad = 0
    n = 0
    for _ in range(120):
        K = int(rng.integers(2, 4)); C = int(rng.integers(K, 7))
        lo, hi = C // K, -(-C // K)
        if not (K * lo <= C <= K * hi):
            continue
        Lm = rng.random((K, C)) * 3
        inc = rng.integers(0, K, C)
        for lam, tau in ((0.0, 0.0), (0.0, 0.3), (0.7, 0.0)):
            n += 1
            pi = A.solve_exact(Lm, lo, hi, incumbent=inc, hysteresis=tau,
                               congestion_lambda=lam)
            cnt = np.bincount(pi, minlength=K)
            if cnt.max() > hi or cnt.min() < lo:
                bad += 1
                continue
            best = np.inf
            for cand in itertools.product(range(K), repeat=C):
                cand = np.array(cand); cc = np.bincount(cand, minlength=K)
                if cc.max() > hi or cc.min() < lo:
                    continue
                v = Lm[cand, np.arange(C)].sum() + 0.5 * lam * (cc.astype(float) ** 2).sum()
                if tau:
                    v -= tau * (cand == inc).sum()
                best = min(best, v)
            got = (Lm[pi, np.arange(C)].sum()
                   + 0.5 * lam * (cnt.astype(float) ** 2).sum()
                   - (tau * (pi == inc).sum() if tau else 0.0))
            if abs(got - best) > 1e-9:
                bad += 1
    check("exact assignment == brute force (incl. floor, hysteresis, convex congestion)",
          bad == 0, f"{n} instances, {bad} bad")


def test_capacity_and_floor():
    rng = np.random.default_rng(1)
    C = 100
    ok = True
    for K in (8, 16, 32):
        Lm = rng.random((K, C))
        lo, hi = C // K, -(-C // K)
        pi = A.solve_exact(Lm, lo, hi)
        cnt = np.bincount(pi, minlength=K)
        ok &= bool(cnt.min() >= lo and cnt.max() <= hi and (cnt == 0).sum() == 0)
    check("capacity floor removes empty experts at every K", ok)


def test_greedy_matches_original_semantics():
    # the greedy must never exceed the ceiling and must assign every class
    rng = np.random.default_rng(2)
    Lm = rng.random((8, 100))
    pi = A.solve_greedy(Lm, 13)
    cnt = np.bincount(pi, minlength=8)
    check("greedy: all classes assigned, ceiling respected",
          (pi >= 0).all() and cnt.sum() == 100 and cnt.max() <= 13)


def test_greedy_is_suboptimal():
    rng = np.random.default_rng(3)
    C, K = 100, 32
    ratios = []
    for _ in range(30):
        Lm = rng.random((K, C)) * 2 + 0.2
        hi = -(-C // K)
        ratios.append(A.phi(Lm, A.solve_greedy(Lm, hi)) / A.phi(Lm, A.solve_exact(Lm, 0, hi)))
    check("greedy is measurably suboptimal (so the optimality gap is a real number to report)",
          np.mean(ratios) > 1.0, f"mean Phi_greedy/Phi_exact = {np.mean(ratios):.4f}")


# --------------------------------------------------------------------------- 2. the estimator

def test_imputation_lockout():
    """The core algebraic claim: mean(seen) + mu > min(seen) ALWAYS.

    So under Eq (3) an unseen expert can never be the column argmin, for any class, any epoch,
    any |S_c| and any mu > 0. This is what makes the incumbent unfalsifiable.
    """
    rng = np.random.default_rng(4)
    viol = 0
    for _ in range(20000):
        m = int(rng.integers(1, 9))
        seen = rng.random(m) * 10
        mu = float(rng.random() * 2 + 1e-6)
        if seen.mean() + mu <= seen.min():
            viol += 1
    check("imputed cost is ALWAYS strictly worse than the column minimum (lock-out lemma)",
          viol == 0, "20000 random draws, 0 counterexamples")


def test_probe_leaves_no_unmeasured_cells():
    from gt3moe.data import probe_indices
    y = torch.cat([torch.full((int(max(1, 400 * (0.01 ** (c / 99)))),), c) for c in range(100)])
    gen = torch.Generator().manual_seed(0)
    idx = probe_indices(y, 100, 64, gen)
    cls = torch.bincount(y[idx], minlength=100)
    check("probe covers every class (so the K x C matrix has no unmeasured cells)",
          bool((cls > 0).all()), f"min per-class probe samples = {int(cls.min())}")


# --------------------------------------------------------------------------- 3. PCGrad

def test_pcgrad_coefficient_space_is_exact():
    torch.manual_seed(0)
    worst = 0.0
    for t in range(60):
        G = int(torch.randint(2, 9, (1,)).item())
        P = int(torch.randint(5, 60, (1,)).item())
        M = torch.randn(G, P, dtype=torch.float64)
        perms = make_perms(G, torch.Generator().manual_seed(t))
        ref = pcgrad_naive(list(M), perms)
        got = pcgrad_coefficients(M @ M.T, perms) @ M
        worst = max(worst, max(float((got[i] - ref[i]).abs().max()) for i in range(G)))
    check("coefficient-space PCGrad == the naive loop", worst < 1e-10,
          f"worst abs err {worst:.2e}")


def test_feature_space_pcgrad_is_vacuous():
    """Per-sample feature gradients have disjoint support, so no conflict can ever be detected."""
    torch.manual_seed(0)
    B, D, K = 64, 32, 8
    gid = torch.randint(0, K, (B,))
    g = torch.randn(B, D)
    per = [torch.where((gid == k).view(-1, 1), g, torch.zeros_like(g)) for k in range(K)]
    m = max(abs(float((per[i] * per[j]).sum())) for i in range(K) for j in range(K) if i < j)
    check("per-sample feature-space PCGrad detects zero conflicts (provably vacuous)",
          m == 0.0, f"max |<G_i,G_j>| = {m:.1e}")
    pooled = torch.stack([per[k].sum(0) / max(1, int((gid == k).sum())) for k in range(K)])
    gram = pooled @ pooled.T
    off = ~torch.eye(K, dtype=torch.bool)
    check("POOLED feature-space gradients do conflict (so the 1-backward surrogate is viable)",
          bool((gram[off] < 0).any()))


def test_size_weighted_equals_full_batch_ce():
    """size_weighted grouping must reproduce the full-batch CE gradient before projection."""
    torch.manual_seed(0)
    B, C, K = 40, 10, 4
    lin = torch.nn.Linear(6, C)
    x = torch.randn(B, 6)
    y = torch.randint(0, C, (B,))
    asg = torch.randint(0, K, (C,))
    cfg = make_config("v2", num_classes=C, num_experts=K, pcgrad_min_group=1,
                      pcgrad_singleton="merge_to_residual", backbone_ce="size_weighted")

    def grad_of(w):
        lin.zero_grad()
        per = F.cross_entropy(lin(x), y, reduction="none")
        (per * w).sum().backward()
        return torch.cat([p.grad.reshape(-1).clone() for p in lin.parameters()])

    gid, w, _ = build_groups(y, asg, K, cfg, torch.Generator().manual_seed(0))
    g_grouped = grad_of(w)
    g_full = grad_of(torch.full((B,), 1.0 / B))
    rel = float((g_grouped - g_full).abs().max() / g_full.abs().max())
    check("size_weighted group gradient == full-batch CE gradient", rel < 1e-6,
          f"max rel err {rel:.2e}")

    cfg2 = make_config("v2", num_classes=C, num_experts=K, backbone_ce="group_mean",
                       pcgrad_min_group=1, pcgrad_singleton="merge_to_residual")
    _gid2, w2, _ = build_groups(y, asg, K, cfg2, torch.Generator().manual_seed(0))
    ratio = float(w2.max() / w2.min())
    check("group_mean is a real reweighting, not a relabelling", ratio > 1.5,
          f"max/min per-sample backbone weight = {ratio:.2f}x")


# --------------------------------------------------------------------------- 4. losses

def test_js_confusion_properties():
    torch.manual_seed(0)
    B, C = 64, 20
    z1 = torch.randn(B, C, requires_grad=True)
    z2 = torch.randn(B, C, requires_grad=True)
    y = torch.randint(0, C, (B,))
    al = torch.stack([z1, z2], 1)
    ti = torch.stack([torch.zeros(B, dtype=torch.long), torch.ones(B, dtype=torch.long)], 1)
    tg = torch.full((B, 2), 0.5)
    cfg = make_config("v2", comp_loss="js_confusion", comp_beta=1.0, num_classes=C)
    v = L.comp_loss(al, ti, y, tg, cfg)
    check("js_confusion is in [0,1]", bool(0.0 <= float(v) <= 1.0), f"value {float(v):.4f}")
    v.backward()
    gy = z1.grad[torch.arange(B), y].abs().max()
    check("dJS/dz[y] == 0 exactly (cannot be reduced by hurting the true class)",
          float(gy) == 0.0, f"max |grad| at the true class = {float(gy):.2e}")

    # gauge invariance: shift every logit by a per-expert constant
    for k in (1.0, 50.0):
        a = torch.randn(B, C); b = torch.randn(B, C)
        al0 = torch.stack([a, b], 1)
        al1 = torch.stack([a + k, b - k], 1)
        v0 = float(L.comp_loss(al0, ti, y, tg, cfg))
        v1 = float(L.comp_loss(al1, ti, y, tg, cfg))
        check(f"js_confusion is gauge-invariant (shift {k:g})", abs(v0 - v1) < 1e-5,
              f"{v0:.6f} vs {v1:.6f}")
        cfgc = make_config("v2", comp_loss="cosine_orig", num_classes=C)
        c0 = float(L.comp_loss(al0, ti, y, tg, cfgc))
        c1 = float(L.comp_loss(al1, ti, y, tg, cfgc))
        if k == 50.0:
            check("cosine_orig is NOT gauge-invariant (the free descent direction)",
                  abs(c0 - c1) > 0.1, f"{c0:.4f} vs {c1:.4f}")


def test_align_hinge_is_inactive_when_satisfied():
    torch.manual_seed(0)
    B, K, C = 32, 8, 20
    cfg = make_config("v2", align_loss="hinge_topk", top_k=2, num_classes=C, num_experts=K)
    asg = torch.randint(0, K, (C,))
    y = torch.randint(0, C, (B,))
    rl = torch.full((B, K), -5.0)
    rl[torch.arange(B), asg[y]] = 5.0                 # assigned expert is already the top choice
    rl.requires_grad_(True)
    v = L.align_loss(rl, y, asg, cfg)
    check("hinge_topk is exactly zero once the assigned expert is selected", float(v) == 0.0)
    cfg2 = make_config("v2", align_loss="nll", top_k=2, num_classes=C, num_experts=K)
    v2 = L.align_loss(rl, y, asg, cfg2)
    check("plain NLL keeps applying pressure in the same situation", float(v2) > 0.0,
          f"nll = {float(v2):.4f}")


# --------------------------------------------------------------------------- 5. config

def test_paper_preset_differs_only_by_flags():
    a = make_config("v2")
    b = make_config("paper_v1")
    ha, hb = a.hashable(), b.hashable()
    diff = {k for k in ha if ha[k] != hb[k]}
    undocumented = diff - set(PAPER_V1)
    check("paper_v1 has NO undocumented difference from v2", not undocumented,
          f"{len(diff)} fields differ, all listed in PAPER_V1"
          if not undocumented else f"undocumented: {sorted(undocumented)}")
    check("every capacity is feasible at K in {8,16,32}",
          all(make_config("v2", num_experts=K).resolved_cap() for K in (8, 16, 32)))
    r = [make_config("v2").ramp(t) for t in (0, 4, 5, 10, 14, 15, 40)]
    check("ramp w(t) is 0 through warmup then reaches 1", r[0] == 0 and r[1] == 0
          and 0 < r[2] <= 1 and r[-1] == 1.0, str(r))


def main():
    print("\nGT3MoE v2 -- invariant tests\n" + "=" * 70)
    for fn in (test_exact_solver, test_capacity_and_floor, test_greedy_matches_original_semantics,
               test_greedy_is_suboptimal, test_imputation_lockout,
               test_probe_leaves_no_unmeasured_cells,
               test_pcgrad_coefficient_space_is_exact, test_feature_space_pcgrad_is_vacuous,
               test_size_weighted_equals_full_batch_ce, test_js_confusion_properties,
               test_align_hinge_is_inactive_when_satisfied,
               test_paper_preset_differs_only_by_flags):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            FAIL.append(fn.__name__)
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
    print("=" * 70)
    print(f"  {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("  failures: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())


def test_congestion_potential_is_exact_only_under_total_loss_payoff():
    """Theorem A.1 / Remark A.1b.

    Phi(pi) = sum_c w_c L[pi(c),c] + (lam/2) sum_i ell_i^2 is a *weighted* potential when a class's
    payoff is its MEAN per-sample loss, and an *exact* potential when it is the class's TOTAL loss
    contribution. The paper's Definition 4.1 has to state the latter; this pins that down, because
    "exact potential game" is a strictly stronger claim than "weighted potential game" and the
    difference is a single w_c factor that is easy to lose in a rewrite.
    """
    rng = np.random.default_rng(11)
    K, C, lam = 7, 50, 0.35
    worst_mean = worst_total = 0.0
    n = 0
    for _ in range(600):
        L = rng.random((K, C)) * 3.0
        w = rng.random(C) * 2.0 + 0.05
        pi = rng.integers(0, K, C)
        ell = np.zeros(K)
        np.add.at(ell, pi, w)

        def phi(p):
            l = np.zeros(K)
            np.add.at(l, p, w)
            return (w * L[p, np.arange(C)]).sum() + 0.5 * lam * (l ** 2).sum()

        c = int(rng.integers(0, C))
        i, j = int(pi[c]), int(rng.integers(0, K))
        if j == i:
            continue
        n += 1
        d_mean = (L[j, c] + lam * (ell[j] + w[c])) - (L[i, c] + lam * ell[i])
        d_total = w[c] * d_mean
        pi2 = pi.copy()
        pi2[c] = j
        d_phi = phi(pi2) - phi(pi)
        worst_mean = max(worst_mean, abs(d_phi - d_mean))
        worst_total = max(worst_total, abs(d_phi - d_total))

    assert n > 400
    assert worst_total < 1e-9, f"total-loss payoff must be EXACT, off by {worst_total}"
    assert worst_mean > 1.0, "mean-loss payoff is only a weighted potential; if this ever " \
                             "passes exactly, the payoff definition silently changed"


def test_exact_solver_globally_minimises_the_congestion_potential():
    """With congestion_lambda > 0 the LAP replication still returns the GLOBAL optimum.

    The (lam/2)(2s+1) marginal is the Rosenthal expansion of (lam/2) ell^2, and replication keeps
    the constraint matrix totally unimodular -- so this is an optimality claim, not a heuristic.
    Checked against exhaustive enumeration.
    """
    rng = np.random.default_rng(7)
    n = bad = 0
    for _ in range(200):
        K = int(rng.integers(2, 4))
        C = int(rng.integers(K, 7))
        lam = float(rng.choice([0.0, 0.05, 0.3, 1.0]))
        cap_lo, cap_hi = int(rng.integers(0, 2)), C
        if not (K * cap_lo <= C <= K * cap_hi):
            continue
        L = np.round(rng.random((K, C)) * 2, 3)

        def phi(p):
            cnt = np.bincount(p, minlength=K).astype(float)
            return L[p, np.arange(C)].sum() + 0.5 * lam * (cnt ** 2).sum()

        best = min((phi(np.array(p)) for p in itertools.product(range(K), repeat=C)
                    if np.bincount(p, minlength=K).min() >= cap_lo), default=None)
        if best is None:
            continue
        n += 1
        if phi(A.solve_exact(L, cap_lo, cap_hi, congestion_lambda=lam)) > best + 1e-7:
            bad += 1
    assert n > 100 and bad == 0, f"{bad}/{n} instances suboptimal"


def test_solve_multi_is_exact_and_owners_are_distinct():
    """Overlapping assignment must stay GLOBALLY optimal, or the paper's tractability claim dies.

    The constraint matrix (m owners per class, capacity per expert, unit upper bound per pair) is
    a transportation polytope and therefore totally unimodular, so the LP optimum is integral.
    The unit upper bound is load-bearing: a plain replicated LAP would happily assign two of a
    class's m slots to the same expert, which silently reduces m.
    """
    rng = np.random.default_rng(3)
    n = bad = 0
    for _ in range(120):
        K = int(rng.integers(3, 5))
        C = int(rng.integers(3, 6))
        m = int(rng.integers(1, K))
        lo, hi = (m * C) // K, -(-(m * C) // K)
        if not (K * lo <= m * C <= K * hi):
            continue
        L = np.round(rng.random((K, C)) * 2, 3)
        best = None
        for combo in itertools.product(*[list(itertools.combinations(range(K), m))] * C):
            cnt = np.zeros(K, int)
            for c in range(C):
                for k in combo[c]:
                    cnt[k] += 1
            if cnt.min() < lo or cnt.max() > hi:
                continue
            v = sum(L[list(combo[c]), c].sum() for c in range(C))
            best = v if best is None else min(best, v)
        if best is None:
            continue
        o = A.solve_multi(L, m, lo, hi)
        n += 1
        assert o.shape == (C, m)
        assert all(len(set(r.tolist())) == m for r in o), "a class was given the same owner twice"
        cnt = np.bincount(o.reshape(-1), minlength=K)
        assert cnt.min() >= lo and cnt.max() <= hi, f"capacity violated: {cnt}"
        if A.phi_multi(L, o) > best + 1e-6:
            bad += 1
    assert n > 60 and bad == 0, f"{bad}/{n} suboptimal"


def test_multi_owner_align_loss_collapses_to_ce_at_m1():
    """The [C,m] alignment path must be a strict generalisation, not a second code path.

    At m=1 it has to equal plain CE bit-for-bit, or every m=1 arm silently stops being the control.
    At m=K every expert is an owner, so the marginal log-likelihood of the owner set is log 1 = 0.
    """
    cfg = make_config("v2", device="cpu", align_loss="nll", num_experts=8, num_classes=100)
    g = torch.Generator().manual_seed(0)
    rl = torch.randn(64, 8, generator=g)
    y = torch.randint(0, 100, (64,), generator=g)
    pi = torch.randint(0, 8, (100,), generator=g)
    assert torch.allclose(L.align_loss(rl, y, pi, cfg),
                          L.align_loss(rl, y, pi[:, None], cfg), atol=1e-6)
    owners_all = torch.arange(8).repeat(100, 1)
    assert float(L.align_loss(rl, y, owners_all, cfg)) < 1e-6
    # two acceptable owners can only be easier to hit than one
    two = torch.stack([pi, (pi + 1) % 8], 1)
    assert float(L.align_loss(rl, y, two, cfg)) < float(L.align_loss(rl, y, pi, cfg))
