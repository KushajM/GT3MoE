"""Offline replay of the slow timescale from a saved cost-matrix trace.

Every run with ``--save_cost_trace`` writes ``L^(t)`` for every epoch. Because the assignment step
is a pure function of that trace, **every solver ablation can be re-run offline for $0** -- greedy
vs exact, capacity floor on/off, hysteresis sweeps, convex congestion sweeps, the column-shuffled
null, and the lock-out counterfactual.

What replay CANNOT tell you is how a different assignment would have changed *training*, since the
trace was generated under one particular assignment path. So use it to answer questions about the
solver (how suboptimal is the greedy? does hysteresis remove the churn? would the floor have left
experts empty?) and pay for a GPU run only for questions about the learning dynamics.

    python -m gt3moe.replay outputs/gt3moe_v2_K8_s0_abc123_costtrace.npz
"""
from __future__ import annotations

import json
import sys
from typing import Dict, List

import numpy as np

from . import assign as A


def replay(L_trace: np.ndarray, K: int, cap_lo: int, cap_hi: int,
           variants: Dict[str, dict] = None, seed: int = 0) -> Dict:
    """L_trace [T,K,C]. Returns per-variant per-epoch diagnostics."""
    T, Kt, C = L_trace.shape
    assert Kt == K, (Kt, K)
    rng = np.random.default_rng(seed)
    variants = variants or {
        "greedy_v1": dict(solver="greedy", cap_lo=0),
        "exact_ceiling_only": dict(solver="exact", cap_lo=0),
        "exact_floor": dict(solver="exact", cap_lo=cap_lo),
        "exact_floor_hyst0.1": dict(solver="exact", cap_lo=cap_lo, hyst=0.1),
        "exact_floor_hyst0.3": dict(solver="exact", cap_lo=cap_lo, hyst=0.3),
        "exact_congestion0.02": dict(solver="exact", cap_lo=cap_lo, lam=0.02),
        "column_shuffled_null": dict(solver="exact", cap_lo=cap_lo, shuffle=True),
    }
    out: Dict[str, Dict[str, List[float]]] = {}
    for name, v in variants.items():
        pi = (np.arange(C) % K).astype(np.int64)
        rec = {k: [] for k in ("phi", "phi_exact", "optimality_gap", "regret_norm", "churn",
                               "gamma_star", "n_empty", "cmin", "cmax")}
        for t in range(T):
            Lm = L_trace[t].astype(np.float64)
            Lu = A.column_shuffled(Lm, rng) if v.get("shuffle") else Lm
            lo = v.get("cap_lo", cap_lo)
            if v["solver"] == "greedy":
                new = A.solve_greedy(Lu, cap_hi)
            else:
                new = A.solve_exact(Lu, lo, cap_hi, incumbent=pi,
                                    hysteresis=v.get("hyst", 0.0),
                                    congestion_lambda=v.get("lam", 0.0))
            d = A.normalised_regret(Lm, new, lo, cap_hi)
            rec["phi"].append(d["phi"])
            rec["phi_exact"].append(d["phi_exact"])
            rec["optimality_gap"].append(d["optimality_gap"])
            rec["regret_norm"].append(d["regret_norm"])
            rec["churn"].append(int((new != pi).sum()))
            rec["gamma_star"].append(A.gamma_star(Lm, new, lo, cap_hi))
            cnt = np.bincount(new, minlength=K)
            rec["n_empty"].append(int((cnt == 0).sum()))
            rec["cmin"].append(int(cnt.min()))
            rec["cmax"].append(int(cnt.max()))
            pi = new
        out[name] = rec
    return out


def summarise(res: Dict) -> Dict:
    s = {}
    for name, rec in res.items():
        ch = np.asarray(rec["churn"], float)
        nz = np.nonzero(ch)[0]
        s[name] = dict(
            mean_optimality_gap=float(np.mean(rec["optimality_gap"])),
            max_optimality_gap=float(np.max(rec["optimality_gap"])),
            mean_regret_norm=float(np.mean(rec["regret_norm"])),
            total_churn=int(ch.sum()),
            mean_churn=float(ch.mean()),
            last_nonzero_churn_epoch=int(nz[-1]) if nz.size else -1,
            frac_epochs_zero_churn=float((ch == 0).mean()),
            epochs_with_empty_expert=int(np.sum(np.asarray(rec["n_empty"]) > 0)),
            min_classes_per_expert=int(np.min(rec["cmin"])),
            median_gamma_star=float(np.median([g for g in rec["gamma_star"] if np.isfinite(g)])
                                    if any(np.isfinite(rec["gamma_star"])) else float("nan")),
        )
    return s


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__)
        return 2
    path = argv[0]
    z = np.load(path)
    Lt = z["L"]
    T, K, C = Lt.shape
    cap_hi = -(-C // K)
    cap_lo = C // K
    print(f"  trace: T={T} epochs, K={K}, C={C}, cap=[{cap_lo},{cap_hi}]")
    res = replay(Lt, K, cap_lo, cap_hi)
    summ = summarise(res)
    print(json.dumps(summ, indent=2))
    outp = path.replace(".npz", "_replay.json")
    with open(outp, "w") as f:
        json.dump(dict(summary=summ, per_epoch=res), f, indent=2)
    print(f"\n  wrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
