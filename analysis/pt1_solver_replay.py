"""How suboptimal is the balanced greedy, really? Replayed offline from saved cost traces.

Every GT3MoE run persists L^(t) per epoch, and the assignment step is a pure function of that
trace, so exact and greedy solvers can be compared on the same costs without any GPU time. The
traces used for the paper's exact-versus-greedy table are in data/_costtrace.

Caveat that must travel with the numbers: the trace was produced under ONE assignment path, so
this answers "how suboptimal was the partition the greedy chose, on the costs actually observed",
not "what would training have done under a different solver". The latter needs a GPU run.

    python pt1_solver_replay.py   ->  paper_tables/solver_replay.{csv,md}
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gt3moe import assign as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "paper_tables")
CACHE = os.path.join(HERE, "data", "_costtrace")
#: directories whose GT3MoE runs we replay
TARGETS = ["ksweep", "F_c100lt"]


def main():
    os.makedirs(OUT, exist_ok=True)

    jobs = []
    for d in TARGETS:
        for f in glob.glob(os.path.join(HERE, "data", d, "*.json")):
            r = json.load(open(f))
            c = r["config"]
            if c.get("router", "gt3moe") != "gt3moe" or c.get("dense_head"):
                continue
            if c.get("assign_mode") not in (None, "dynamic"):
                continue
            local = os.path.join(CACHE, os.path.basename(f).replace(".json", "_costtrace.npz"))
            if not os.path.exists(local):
                continue
            jobs.append((d, int(c["num_experts"]), c["seed"], local,
                         np.array(r["class_counts"], dtype=np.float64)))
    print(f"  {len(jobs)} GT3MoE traces matched and cached")

    agg = defaultdict(lambda: defaultdict(list))
    for d, K, seed, path, counts in jobs:
        L = np.load(path)["L"]                     # [T,K,C]
        T, _, C = L.shape
        cap_hi = -(-C // K)
        cap_lo = C // K
        # Two capacity regimes, and they are NOT interchangeable. solve_greedy enforces only the
        # ceiling, so it must be scored against the ceiling-only optimum; scoring it against the
        # floor-constrained optimum compares it to a strictly smaller feasible set and can yield a
        # ratio BELOW 1.0, which is self-contradictory. The deployed solver uses the floor, so the
        # cost of the floor is reported as its own column rather than folded into solver quality.
        prev = {v: (np.arange(C) % K).astype(np.int64)
                for v in ("greedy", "exact_ceiling", "exact_floor")}
        cls = np.arange(C)
        for t in range(T):
            Lm = L[t].astype(np.float64)
            new = {
                "greedy": A.solve_greedy(Lm, cap_hi),
                "exact_ceiling": A.solve_exact(Lm, 0, cap_hi, incumbent=prev["exact_ceiling"]),
                "exact_floor": A.solve_exact(Lm, cap_lo, cap_hi, incumbent=prev["exact_floor"]),
            }
            opt_ceiling = float(Lm[A.solve_exact(Lm, 0, cap_hi)[cls], cls].sum())
            for v, pi in new.items():
                agg[(d, K, v)]["phi"].append(float(Lm[pi[cls], cls].sum()))
                agg[(d, K, v)]["phi_exact"].append(opt_ceiling)
                agg[(d, K, v)]["churn"].append(float((pi != prev[v]).sum()))
                prev[v] = pi

    hdr = ["dir", "K", "solver", "epochs", "phi/phi_opt mean", "phi/phi_opt max",
           "churn mean (classes/epoch)", "churn max"]
    rows = []
    for (d, K, v), rec in sorted(agg.items()):
        ratio = np.array(rec["phi"]) / np.maximum(np.array(rec["phi_exact"]), 1e-12)
        ch = np.array(rec["churn"][1:])            # epoch 0 is against the round-robin init
        rows.append([d, K, v, len(rec["phi"]), round(float(ratio.mean()), 4),
                     round(float(ratio.max()), 4),
                     round(float(ch.mean()), 1) if len(ch) else 0.0,
                     round(float(ch.max()), 1) if len(ch) else 0.0])

    with open(os.path.join(OUT, "solver_replay.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    with open(os.path.join(OUT, "solver_replay.md"), "w", encoding="utf-8") as fh:
        fh.write("# Solver replay: balanced greedy vs the exact LAP\n\n"
                 "All three solvers are scored against the **ceiling-only** optimum, which is the "
                 "feasible set `solve_greedy` works in. `exact_floor` additionally enforces the "
                 "capacity floor, so its ratio above 1.0 is the price of the floor, not solver "
                 "suboptimality.\n\n"
                 "Averaged over seeds and epochs. The trace was generated under one assignment "
                 "path, so this measures solver suboptimality on the costs actually observed, not "
                 "the training effect of switching solver.\n\n"
                 "**`solve_br_weighted` is deliberately excluded.** It minimises the sample-mass "
                 "weighted potential plus a quadratic congestion term, not the objective above, "
                 "so a single phi ratio does not compare it to anything. It is also degenerate at "
                 "lambda = 0 when C/K is an integer: cap_lo = cap_hi leaves no feasible "
                 "single-class move, so it returns its incumbent unchanged (measured churn 0.0 at "
                 "K = 4, C = 100). It is evaluated in the E7 arms at its deployed "
                 "congestion_lambda instead.\n\n")
        fh.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(x) for x in r) + " |\n")
    for r in rows:
        print("  " + "  ".join(str(x) for x in r))
    return rows


if __name__ == "__main__":
    main()
