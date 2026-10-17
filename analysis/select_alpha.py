"""Stage-1 balance-weight selection on CIFAR-100-LT validation accuracy (data/U_tune).

The rule, fixed before the runs:

    For each (method, K), take the alpha with the highest mean VALIDATION accuracy over seeds
    3-5. Ties go to the smaller alpha.

Test accuracy is printed for completeness but plays NO part in the selection. The second round,
which extended Switch's grid to 0.001 and 1.0 (data/U_tune1b), is recorded under "stage1b" in
selected_alphas.json, and the values the paper uses are under "final"; both are preserved here.

    python select_alpha.py   ->  analysis/selected_alphas.json
                                 analysis/paper_tables/stage1_tuning.md
"""
from __future__ import annotations

import collections
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METHOD = {("gt3moe", "dynamic"): "gt3", ("gt3moe", "random_balanced"): "rnd",
          ("switch", "dynamic"): "swi"}


def method_of(c):
    return METHOD.get((c.get("router", "gt3moe"), c.get("assign_mode", "dynamic") or "dynamic"))


def main():
    val = collections.defaultdict(dict)   # (method, K, alpha) -> seed -> val acc
    tst = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(HERE, "data", "U_tune", "*.json")):
        r = json.load(open(f))
        if not r.get("test"):
            continue
        c = r["config"]
        m = method_of(c)
        if m is None:
            continue
        key = (m, int(c["num_experts"]), float(c["balance_alpha"]))
        v = (r.get("val_final") or {}).get("acc")
        if v is None:
            continue
        val[key][c["seed"]] = v
        tst[key][c["seed"]] = r["test"]["acc"]

    alphas = sorted({k[2] for k in val})
    Ks = sorted({k[1] for k in val})
    methods = ["gt3", "rnd", "swi"]
    sel, lines = {}, []
    hdr = (f"  {'method':8s}{'K':>4s}" + "".join(f"{('a=' + str(a)):>18s}" for a in alphas)
           + f"{'sel':>9s}{'SE pp':>8s}{'~ties':>6s}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for m in methods:
        for K in Ks:
            cells, best = [], None
            for a in alphas:
                s = val.get((m, K, a))
                if not s:
                    cells.append(f"{'-':>18s}")
                    continue
                mv = float(np.mean(list(s.values())))
                mt = float(np.mean(list(tst[(m, K, a)].values())))
                cells.append(f"{100 * mv:>9.2f}/{100 * mt:<8.2f}")
                # strictly greater -> ties keep the earlier (smaller) alpha
                if best is None or mv > best[1] + 1e-12:
                    best = (a, mv, mt, len(s))
            if best is None:
                continue
            # How separated is the winner? With 3 seeds the SE of a cell mean is ~0.2-0.3pp, so a
            # "best" alpha that is within 1 SE of a rival is a coin flip, not a tuned choice.
            means = {a: float(np.mean(list(val[(m, K, a)].values())))
                     for a in alphas if val.get((m, K, a))}
            sds = [float(np.std(list(val[(m, K, a)].values()), ddof=1)) / np.sqrt(len(val[(m, K, a)]))
                   for a in alphas if val.get((m, K, a)) and len(val[(m, K, a)]) > 1]
            se = float(np.mean(sds)) if sds else float("nan")
            rivals = sum(1 for a, v in means.items()
                         if a != best[0] and (best[1] - v) < se)
            sel[f"{m}_K{K}"] = dict(method=m, K=K, balance_alpha=best[0],
                                    val_acc=round(best[1], 6), test_acc=round(best[2], 6),
                                    n_seeds=best[3], mean_se_pp=round(100 * se, 3),
                                    rivals_within_1se=rivals,
                                    at_grid_edge=bool(best[0] in (min(alphas), max(alphas))))
            lines.append(f"  {m:8s}{K:>4d}" + "".join(cells)
                         + f"{best[0]:>9g}{100 * se:>8.2f}{rivals:>6d}")

    print("  mean VALIDATION acc / mean test acc, over seeds 3-5 (selection uses validation only)")
    print("\n".join(lines))

    out = os.path.join(HERE, "selected_alphas.json")
    prev = json.load(open(out)) if os.path.exists(out) else {}
    rec = dict(rule="max mean validation accuracy over seeds 3-5; ties to smaller alpha",
               grid=alphas, seeds=sorted({s for d in val.values() for s in d}), selected=sel)
    rec.update({k: prev[k] for k in ("stage1b", "final") if k in prev})
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)

    os.makedirs(os.path.join(HERE, "paper_tables"), exist_ok=True)
    with open(os.path.join(HERE, "paper_tables", "stage1_tuning.md"), "w",
              encoding="utf-8") as fh:
        fh.write("# Stage 1 - tuning only\n\nCells are `mean validation acc / mean test acc` over "
                 "seeds 3-5. Selection is on validation only, ties to the smaller alpha. Test "
                 "numbers here are tuning results and are not evidence for any hypothesis.\n\n```\n"
                 + "\n".join(lines) + "\n```\n")

    print(f"\n  -> {out}")
    boundary = [k for k, v in sel.items() if v["balance_alpha"] == min(alphas)]
    if boundary:
        print(f"  NOTE: selected the SMALLEST alpha in the grid for: {', '.join(boundary)}")
        print("        the grid edge may not contain the optimum for these.")
    return sel


if __name__ == "__main__":
    main()
