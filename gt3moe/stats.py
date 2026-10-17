"""Aggregation and the statistics the paper needs.

Three things the plan got wrong or left out, fixed here.

* **Rank and sign tests cannot reach p<0.05 at small S.** Wilcoxon signed-rank has a minimum
  attainable two-sided p of 0.0625 at n=5. The exact sign-flipping permutation test has the same
  structural problem with a floor of ``2 / 2^n``: 0.25 at n=3, 0.0625 at n=5, 0.031 at n=6. So
  **below 6 seeds the permutation p is not evidence of anything** and the paired t-test is the
  usable statistic; ``paired_test`` reports ``p_perm_floor`` and ``p_perm_usable`` so a floored
  value is never misread as a null result. (Observed live: the augmentation batch returned
  p_perm=0.2500 on every metric at n=3 while the t-test gave p=0.0008-0.023 with dz=3.7-20.2.)
* **The Few metric's error bar.** ``Few`` is a mean over ~34 per-class accuracies, so a
  sample-level bootstrap understates its variance. Bootstrap over *classes*, and pair arms on the
  same test set so the comparison removes class difficulty.
* **Multiplicity.** With ~100 reported comparisons, an uncorrected 0.05 threshold yields ~5 false
  positives by construction. Declare ONE primary endpoint (Overall accuracy at K=8) and report
  everything else as secondary with Holm-corrected p-values.

Common random numbers: every arm should be run at the same ``(data_seed, seed)`` pairs, so all
comparisons are paired and the seed variance cancels. That is free power and it is the difference
between a $150 budget buying a publishable result and buying noise.
"""
from __future__ import annotations

import glob
import itertools
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def load_runs(pattern: str) -> List[Dict]:
    out = []
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            d["_path"] = p
            out.append(d)
        except Exception:
            continue
    return out


def table(runs: Sequence[Dict], metrics=("acc", "many", "medium", "few", "f1")) -> Dict:
    """Group by (arm, K) and report mean, sample SD (ddof=1) and n over seeds."""
    from collections import defaultdict
    g = defaultdict(list)
    for r in runs:
        c = r["config"]
        g[(c["arm"], c["num_experts"])].append(r)
    rows = {}
    for key, rs in sorted(g.items()):
        row = {"n_seeds": len(rs), "seeds": sorted(x["config"]["seed"] for x in rs)}
        for m in metrics:
            v = np.array([x["test"][m] for x in rs if "test" in x and m in x["test"]],
                         dtype=float)
            if v.size:
                row[m] = float(v.mean())
                row[m + "_sd"] = float(v.std(ddof=1)) if v.size > 1 else float("nan")
        secs = np.array([x.get("billed_seconds", np.nan) for x in rs], dtype=float)
        row["billed_seconds"] = float(np.nanmean(secs))
        rows[f"{key[0]}|K={key[1]}"] = row
    return rows


def pooled_sd(runs: Sequence[Dict], metric: str = "acc") -> float:
    """Pooled within-arm SD across all arms.

    A 3-seed SD is a useless interval on its own (its own relative error is ~50%). Pooling the
    within-arm deviations across every arm in the grid gives a usable single dispersion estimate at
    zero extra compute, which is the right thing to put in the tables.
    """
    from collections import defaultdict
    g = defaultdict(list)
    for r in runs:
        if "test" not in r or metric not in r["test"]:
            continue
        c = r["config"]
        g[(c["arm"], c["num_experts"])].append(r["test"][metric])
    num, den = 0.0, 0
    for _k, v in g.items():
        v = np.asarray(v, dtype=float)
        if v.size > 1:
            num += ((v - v.mean()) ** 2).sum()
            den += v.size - 1
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def paired_test(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Paired comparison over matched seeds: t-test plus the exact sign-flipping permutation test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.size >= 2, "need matched pairs"
    d = a - b
    n = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 0.0
    from scipy import stats as st
    t = mean / se if se > 0 else np.inf * np.sign(mean) if mean != 0 else 0.0
    p_t = float(2 * st.t.sf(abs(t), df=n - 1)) if se > 0 else (0.0 if mean != 0 else 1.0)
    # exact permutation over the 2^n sign assignments (n<=10 here)
    obs = abs(mean)
    cnt, tot = 0, 0
    for signs in itertools.product((1, -1), repeat=n):
        tot += 1
        if abs(float((np.asarray(signs) * d).mean())) >= obs - 1e-15:
            cnt += 1
    ci = (mean - 1.96 * se, mean + 1.96 * se) if se > 0 else (mean, mean)
    # The sign-flipping test has a hard p-floor of 2/2^n: with n=3 the smallest attainable
    # two-sided p is 0.25, with n=5 it is 0.0625. That is the SAME defect that makes Wilcoxon
    # unusable at these seed counts, and it applies here too -- so at n<6 the permutation p is
    # not evidence of anything and the t-test is the usable statistic. Reported explicitly rather
    # than left to be misread as "not significant".
    p_perm = cnt / tot
    p_floor = 2.0 / (2 ** n)
    return dict(mean_diff=mean, sd_diff=sd, se=se, t=float(t), p_ttest=p_t,
                p_perm=p_perm, p_perm_floor=p_floor,
                p_perm_at_floor=bool(abs(p_perm - p_floor) < 1e-12),
                p_perm_usable=bool(p_floor < 0.05),
                n=n, ci95=list(ci),
                cohens_dz=float(mean / sd) if sd > 0 else float("inf"))


def tost_equivalence(a: Sequence[float], b: Sequence[float], margin: float) -> Dict:
    """Two one-sided tests. Use for the null-expected controls (E1/E2/E3).

    A non-significant difference is NOT evidence of equivalence; TOST is. ``margin`` must be
    pre-registered -- 0.005 (0.5pp) for Overall accuracy is the plan's figure.
    """
    from scipy import stats as st
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = d.size
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    if se == 0:
        return dict(equivalent=bool(abs(d.mean()) < margin), p_lower=0.0, p_upper=0.0,
                    mean_diff=float(d.mean()), margin=margin, n=n)
    t_lo = (d.mean() + margin) / se
    t_hi = (d.mean() - margin) / se
    p_lo = float(st.t.sf(t_lo, df=n - 1))          # H0: diff <= -margin
    p_hi = float(st.t.cdf(t_hi, df=n - 1))         # H0: diff >= +margin
    return dict(equivalent=bool(max(p_lo, p_hi) < 0.05), p_lower=p_lo, p_upper=p_hi,
                mean_diff=float(d.mean()), margin=margin, n=n,
                note="equivalent=True means the arms are statistically indistinguishable "
                     "within the pre-registered margin")


def bootstrap_few(runs_a: Sequence[Dict], runs_b: Sequence[Dict],
                  cfg_few: Sequence[int], n_boot: int = 10000,
                  seed: int = 0) -> Dict:
    """Paired class-level bootstrap of the Few difference.

    ``Few`` is a mean over |few| per-class accuracies. Resample the CLASSES (with replacement),
    keeping the pairing between arms, so class difficulty cancels. A sample-level bootstrap would
    understate the variance of a mean-of-means.
    """
    fa = np.array([r["test"]["per_class_acc"] for r in runs_a], dtype=float).mean(0)
    fb = np.array([r["test"]["per_class_acc"] for r in runs_b], dtype=float).mean(0)
    idx = np.asarray(list(cfg_few), dtype=int)
    da, db = fa[idx], fb[idx]
    rng = np.random.default_rng(seed)
    n = idx.size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n, n)
        diffs[b] = da[s].mean() - db[s].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(mean_diff=float(da.mean() - db.mean()), ci95=[float(lo), float(hi)],
                p_two_sided=float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
                n_classes=int(n))


def holm(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict]:
    """Holm-Bonferroni step-down correction over a family of comparisons."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        prev = adj
        out[k] = dict(p_raw=p, p_holm=adj, reject=adj < alpha)
    return out


def min_detectable_difference(sd: float, n_seeds: int, power: float = 0.8,
                              alpha: float = 0.05) -> float:
    """MDD for a paired design. State this in the paper; it rules out most small claims."""
    from scipy import stats as st
    if n_seeds < 2 or not np.isfinite(sd):
        return float("nan")
    t_a = st.t.ppf(1 - alpha / 2, df=n_seeds - 1)
    t_b = st.t.ppf(power, df=n_seeds - 1)
    return float((t_a + t_b) * sd / np.sqrt(n_seeds))


def report(pattern: str, primary_metric: str = "acc", primary_K: int = 8,
           reference_arm: str = "v2") -> Dict:
    """One-shot summary of a results directory."""
    runs = load_runs(pattern)
    if not runs:
        return {"error": f"no runs matched {pattern!r}"}
    from collections import defaultdict
    by = defaultdict(dict)
    for r in runs:
        c = r["config"]
        by[(c["arm"], c["num_experts"])][c["seed"]] = r
    sd = pooled_sd(runs, primary_metric)
    out = dict(table=table(runs), pooled_sd=sd,
               n_runs=len(runs),
               mdd_3seed=min_detectable_difference(sd, 3),
               mdd_5seed=min_detectable_difference(sd, 5),
               comparisons={})
    ref = by.get((reference_arm, primary_K))
    if ref:
        praw = {}
        for (arm, K), seeds in sorted(by.items()):
            if arm == reference_arm and K == primary_K:
                continue
            common = sorted(set(seeds) & set(ref))
            if len(common) < 2:
                continue
            a = [seeds[s]["test"][primary_metric] for s in common]
            b = [ref[s]["test"][primary_metric] for s in common]
            key = f"{arm}|K={K}"
            res = paired_test(a, b)
            res["tost_0.5pp"] = tost_equivalence(a, b, 0.005)
            out["comparisons"][key] = res
            praw[key] = res["p_ttest"]
        out["holm"] = holm(praw)
    return out


if __name__ == "__main__":
    import sys
    pat = sys.argv[1] if len(sys.argv) > 1 else "outputs/*.json"
    print(json.dumps(report(pat), indent=2, default=str))
