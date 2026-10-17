"""Tables and figures for the paper, from the 3-seed grid (seeds 20-22).

Reads   analysis/data/{P_c100lt,P_mini,P_tiny,P_ltloss,B_break}
        analysis/paper_tables/solver_replay.csv, analysis/selected_alphas.json
Writes  $GT3MOE_PAPER_DIR/tables/*.tex     (tabular bodies, \\input inside the table environments)
        $GT3MOE_PAPER_DIR/figures/*.pdf, $GT3MOE_PAPER_DIR/previews/*.png
        analysis/paper_numbers.json         (every number the text quotes)

GT3MOE_PAPER_DIR defaults to ./paper_artifacts at the repository root.
A cell with fewer than three seeds carries a dagger; a cell with none is a dash.
"""
from __future__ import annotations

import collections
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.environ.get("GT3MOE_PAPER_DIR", os.path.join(os.path.dirname(HERE), "paper_artifacts"))
TAB, FIG = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")
NUMS = os.path.join(HERE, "paper_numbers.json")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

DATASETS = [("P_c100lt", "CIFAR-100-LT"), ("P_mini", "Mini-ImageNet-LT"), ("P_tiny", "Tiny-ImageNet-LT")]
KS = (8, 16, 32)
ROUTERS = [("gt3", "GT3MoE"), ("swi", "Switch"), ("vmoe", "V-MoE"), ("ec", "Expert Choice"),
           ("stable", "StableMoE"), ("soft", "Soft-attention MoE"), ("ada", "AdaMoE"), ("dyn", "DynMoE")]
NAME = dict(ROUTERS)
# Soft-attention MoE runs every expert on every input, so it is not a sparse router.
SPARSE = [t for t, _ in ROUTERS if t != "soft"]
SEEDS = (20, 21, 22)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
COLOR = {"gt3": "#c0392b", "swi": "#2471a3", "vmoe": "#17a589", "ec": "#7d3c98", "stable": "#b9770e",
         "soft": "#5d6d7e", "ada": "#28b463", "dyn": "#d35400", "km": "#922b21", "kmw": "#922b21",
         "rnd": "#e59866"}


def load(d, seeds=None):
    by = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(DATA, d, "*.json")):
        r = json.load(open(f))
        s = r["config"]["seed"]
        if r.get("test") and (seeds is None or s in seeds):
            by[r["config"]["arm"]][s] = r
    return by


def arm(tag, K):
    return "P_dense" if tag == "dense" else f"P_{tag}_K{K}"


def vals(by, a, m, split="test"):
    return [100 * by[a][s][split][m] for s in sorted(by.get(a, {}))]


def stat(v):
    if not v:
        return None
    return dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, n=len(v))


def cell(v, bold=False, sd=True, d=1):
    if not v:
        return "--"
    m = f"{np.mean(v):.{d}f}"
    m = rf"\textbf{{{m}}}" if bold else m
    if sd and len(v) > 1:
        m += rf"{{\scriptsize$\pm${np.std(v, ddof=1):.{d}f}}}"
    return m + (r"$^\dagger$" if len(v) < 3 else "")


def paired(by, a, b, m):
    s = sorted(set(by.get(a, {})) & set(by.get(b, {})))
    d = [100 * (by[a][x]["test"][m] - by[b][x]["test"][m]) for x in s]
    return dict(mean=float(np.mean(d)), per_seed=[round(x, 2) for x in d], n=len(d),
                wins=int(sum(x > 0 for x in d))) if d else None


def best_of(by, tags, K, m):
    means = {t: np.mean(vals(by, arm(t, K), m)) for t in tags if vals(by, arm(t, K), m)}
    return max(means, key=means.get) if means else None


D = {d: load(d) for d, _ in DATASETS}
NUM = {"main": {}, "paired": {}, "routing": {}, "partitions": {}, "convergence": {}, "ltloss": {},
       "balanced": {}}


PREVIEW = os.path.join(OUT, "previews")
os.makedirs(PREVIEW, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(PREVIEW, name + ".png"), bbox_inches="tight", dpi=170)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def write(name, body):
    open(os.path.join(TAB, name), "w", encoding="utf-8").write(body)
    print("  wrote tables/" + name)


# --------------------------------------------------------------------------- main table
def k_header(cols, sub):
    h = " & " + " & ".join(rf"\multicolumn{{{len(sub)}}}{{c}}{{$K{{=}}{K}$}}" for K in cols) + r" \\" + "\n"
    h += "".join(rf"\cmidrule(lr){{{2 + i * len(sub)}-{1 + (i + 1) * len(sub)}}}" for i in range(len(cols))) + "\n"
    return h + " & " + " & ".join(sub * len(cols)) + r" \\" + "\n"


rows = [r"\begin{tabular}{l" + " cc" * len(KS) + "}", r"\toprule", k_header(KS, ["All", "Few"]).rstrip()]
for d, dname in DATASETS:
    by = D[d]
    proto = "fine-tuned" if d == "P_c100lt" else "from scratch"
    rows += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\emph{{{dname} ({proto})}}}} \\"]
    best = {(K, m): best_of(by, SPARSE, K, m) for K in KS for m in ("acc", "few")}
    for t, n in ROUTERS:
        cells = []
        for K in KS:
            for m in ("acc", "few"):
                v = vals(by, arm(t, K), m)
                cells.append(cell(v, bold=(best[(K, m)] == t)))
                NUM["main"].setdefault(d, {}).setdefault(t, {})[f"K{K}_{m}"] = stat(v)
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    va, vf = vals(by, "P_dense", "acc"), vals(by, "P_dense", "few")
    NUM["main"].setdefault(d, {})["dense"] = dict(acc=stat(va), few=stat(vf))
    txt = f"all {cell(va)}, few-shot {cell(vf)}" if va else "--"
    rows.append(rf"\textit{{Dense head}} & \multicolumn{{6}}{{c}}{{\textit{{{txt}}}}} \\")
    for K in KS:
        for b in ("swi", "ec", "soft", "vmoe"):
            for m in ("acc", "few"):
                NUM["paired"].setdefault(d, {}).setdefault(f"K{K}", {})[f"gt3-{b}_{m}"] = \
                    paired(by, arm("gt3", K), arm(b, K), m)
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_main.tex", "\n".join(rows) + "\n")

# --------------------------------------------------------------------------- per-shot appendix
EXTRA = [("km", "k-means, fixed"), ("kmw", "k-means at warm-up, fixed"), ("rnd", "Random, fixed")]
for d, dname in DATASETS:
    by = D[d]
    rows = [r"\begin{tabular}{l" + " ccc" * len(KS) + "}", r"\toprule",
            k_header(KS, ["Many", "Med.", "Few"]).rstrip(), r"\midrule"]
    for t, n in ROUTERS + [x for x in EXTRA if any(vals(by, arm(x[0], K), "acc") for K in KS)]:
        cells = [cell(vals(by, arm(t, K), m), sd=False) for K in KS for m in ("many", "medium", "few")]
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
    va = [cell(vals(by, "P_dense", m), sd=False) for m in ("many", "medium", "few")]
    rows += [r"\midrule", r"\textit{Dense head} & " + " & ".join(va * len(KS)) + r" \\",
             r"\bottomrule", r"\end{tabular}"]
    write(f"tab_pershot_{d[2:]}.tex", "\n".join(rows) + "\n")


# --------------------------------------------------------------------------- routing diagnostics
def own_other(t):
    """Own classes of an expert: those for which it receives the largest share of routed slots."""
    if "competence_acc" not in t or "affinity" not in t:
        return None
    A = np.asarray(t["affinity"], float)
    Q = np.asarray(t["competence_acc"], float)
    owner = A.argmax(1)
    # one expert taking the largest share of nearly every class leaves "own" undefined for the rest
    if np.bincount(owner).max() >= 0.9 * len(owner):
        return None
    own, oth = [], []
    for e in range(Q.shape[0]):
        m = owner == e
        if m.any() and (~m).any():
            own.append(np.nanmean(Q[e, m]))
            oth.append(np.nanmean(Q[e, ~m]))
    return 100 * float(np.mean(own)), 100 * float(np.mean(oth)), len(own)


def diag(r, K):
    t = r["test"]
    oo = own_other(t)
    return dict(ent=t["mean_routing_entropy"] / np.log(K), ece=t["expert_class_entropy"],
                own=oo[0] if oo else np.nan, other=oo[1] if oo else np.nan,
                owners=len(set(np.asarray(t["affinity"]).argmax(1).tolist())),
                cv=t["util_cv"], dead=t["dead_experts_rel"])


DIAG_COLS = [("ent", r"routing$/\ln K$", 2), ("ece", "expert--class", 2),
             ("own", "own", 1), ("other", "other", 1),
             ("owners", "owning", 0), ("cv", "CV", 2), ("dead", "idle", 1)]
Kd = 32
rows = [r"\begin{tabular}{l" + "c" * len(DIAG_COLS) + "}", r"\toprule",
        r" & \multicolumn{2}{c}{Entropy} & \multicolumn{2}{c}{Expert acc.\ (\%)} & Experts & "
        r"\multicolumn{2}{c}{Load} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){7-8}",
        "Router & " + " & ".join(c[1] for c in DIAG_COLS) + r" \\"]
for d, dname in DATASETS:
    by = D[d]
    rows += [r"\midrule", rf"\multicolumn{{{1 + len(DIAG_COLS)}}}{{l}}{{\emph{{{dname}}}}} \\"]
    for t, n in ROUTERS:
        rs = [by[arm(t, Kd)][s] for s in sorted(by.get(arm(t, Kd), {}))]
        if not rs:
            rows.append(f"{n} & " + " & ".join(["--"] * len(DIAG_COLS)) + r" \\")
            continue
        g = [diag(r, Kd) for r in rs]
        agg = {k: float(np.mean([x[k] for x in g])) for k in g[0]}
        NUM["routing"].setdefault(d, {})[t] = dict(agg, n=len(g))
        dag = r"$^\dagger$" if len(g) < 3 else ""
        rows.append(f"{n}{dag} & " + " & ".join(
            (r"--$^\ddagger$" if k in ("own", "other") else "--") if np.isnan(agg[k])
            else f"{agg[k]:.{p}f}" for k, _, p in DIAG_COLS) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_routing.tex", "\n".join(rows) + "\n")

# --------------------------------------------------------------------------- partitions
PART = [("gt3", "GT3MoE (re-solved)"), ("sem", "k-means, fixed"),
        ("rnd", "Random, fixed"), ("swi", "Switch")]
rows = [r"\begin{tabular}{l" + " cc" * len(KS) + "}", r"\toprule", k_header(KS, ["All", "Few"]).rstrip()]
for d, dname in DATASETS:
    by = D[d]
    rows += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\emph{{{dname}}}}} \\"]
    for t, n in PART:
        cells = []
        for K in KS:
            a = arm(("km" if d == "P_c100lt" else "kmw") if t == "sem" else t, K)
            for m in ("acc", "few"):
                v = vals(by, a, m)
                cells.append(cell(v))
                NUM["partitions"].setdefault(d, {}).setdefault(t, {})[f"K{K}_{m}"] = stat(v)
            if t != "gt3":
                for m in ("acc", "few"):
                    NUM["partitions"][d][t][f"K{K}_{m}_vs_gt3"] = paired(by, a, arm("gt3", K), m)
            NUM["partitions"][d][t][f"K{K}_val_acc"] = stat(vals(by, a, "acc", "val_final"))
        rows.append(f"{n} & " + " & ".join(cells) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_partitions.tex", "\n".join(rows) + "\n")

# --------------------------------------------------------------------------- long-tailed losses
LT = load("P_ltloss")
C100 = D["P_c100lt"]
LOSSES = [("ce", "CE"), ("bal", "Balanced Softmax"), ("cb", "CB-CE"), ("ldam", "LDAM-DRW")]
LT_ROUTERS = [("gt3", "GT3MoE"), ("swi", "Switch"), ("ec", "Expert Choice"), ("dense", r"\textit{Dense head}")]


def lt_vals(t, loss, m):
    if loss == "ce":
        return vals(C100, arm(t, 32), m)
    return vals(LT, f"PL_{t}_{loss}" if t == "dense" else f"PL_{t}_{loss}_K32", m)


rows = [r"\begin{tabular}{l" + " cc" * len(LOSSES) + "}", r"\toprule",
        " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{n}}}" for _, n in LOSSES) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(LOSSES))),
        " & " + " & ".join(["All", "Few"] * len(LOSSES)) + r" \\", r"\midrule"]
for t, n in LT_ROUTERS:
    if t == "dense":
        rows.append(r"\midrule")
    cells = []
    for lo, _ in LOSSES:
        for m in ("acc", "few"):
            v = lt_vals(t, lo, m)
            means = {u: np.mean(lt_vals(u, lo, m)) for u, _ in LT_ROUTERS[:3] if lt_vals(u, lo, m)}
            cells.append(cell(v, bold=(t != "dense" and means and max(means, key=means.get) == t)))
            NUM["ltloss"].setdefault(t, {})[f"{lo}_{m}"] = stat(v)
    rows.append(f"{n} & " + " & ".join(cells) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_ltloss.tex", "\n".join(rows) + "\n")

# --------------------------------------------------------------------------- balanced control
B = load("B_break", seeds={6, 7, 8})
BAL = [("gt3", "GT3MoE"), ("rnd", "Random partition, fixed"), ("swi", "Switch"), ("dense", r"\textit{Dense head}")]
rows = [r"\begin{tabular}{lccc}", r"\toprule",
        r" & CIFAR-100 (balanced) & \multicolumn{2}{c}{CIFAR-100-LT} \\", r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}",
        r"Router ($K{=}32$) & All & All & Few \\", r"\midrule"]
for t, n in BAL:
    bv = vals(B, "B_dense" if t == "dense" else f"B_{t}_K32", "acc")
    rows.append(f"{n} & {cell(bv)} & {cell(vals(C100, arm(t, 32), 'acc'))} & {cell(vals(C100, arm(t, 32), 'few'))}" + r" \\")
    NUM["balanced"][t] = stat(bv)
for b in ("swi", "rnd", "dense"):
    NUM["balanced"][f"gt3-{b}"] = paired(B, "B_gt3_K32", "B_dense" if b == "dense" else f"B_{b}_K32", "acc")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_balanced.tex", "\n".join(rows) + "\n")

# --------------------------------------------------------------------------- paired GT3MoE - Switch
# two-sided 95% Student-t quantiles; scipy is not a dependency of the analysis scripts
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def paired_ci(by, a, b, m):
    s = sorted(set(by.get(a, {})) & set(by.get(b, {})))
    d = np.array([100 * (by[a][x]["test"][m] - by[b][x]["test"][m]) for x in s])
    if len(d) < 2:
        return None
    h = T975[len(d) - 1] * d.std(ddof=1) / np.sqrt(len(d))
    return dict(mean=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h),
                wins=int((d > 0).sum()), n=len(d))


def ci_cell(r):
    if r is None:
        return "--"
    return rf"${r['mean']:+.1f}$ {{\scriptsize[${r['lo']:+.1f}$, ${r['hi']:+.1f}$]}}"


SHOTS = [("acc", "All"), ("many", "Many"), ("medium", "Medium"), ("few", "Few")]
rows = [r"\begin{tabular}{ll" + "c" * len(SHOTS) + "}", r"\toprule",
        "Benchmark & $K$ & " + " & ".join(n for _, n in SHOTS) + r" \\"]
NUM["paired_ci"] = {}
for d, dname in DATASETS:
    rows.append(r"\midrule")
    for j, K in enumerate(KS):
        cells = []
        for m, _ in SHOTS:
            r = paired_ci(D[d], arm("gt3", K), arm("swi", K), m)
            NUM["paired_ci"].setdefault(d, {})[f"K{K}_{m}"] = r
            cells.append(ci_cell(r))
        rows.append((dname if j == 0 else "") + f" & {K} & " + " & ".join(cells) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_paired.tex", "\n".join(rows) + "\n")

# --------------------------------------------------------------------------- balance-weight selection
SEL = json.load(open(os.path.join(os.path.dirname(NUMS), "selected_alphas.json")))
grid = ", ".join(f"{a:g}" for a in SEL["grid"])
TUNE = [("gt3", "GT3MoE$^\\ast$"), ("rnd", "Random, fixed"), ("swi", "Switch")]
rows = [r"\begin{tabular}{llcc}", r"\toprule",
        r"Router & Grid searched & $K$ & Selected $\alpha_{\mathrm{bal}}$ (val.\ acc.) \\", r"\midrule"]
NUM["tuning"] = {}
for t, n in TUNE:
    for j, K in enumerate(KS):
        key = f"{t}_K{K}"
        a = SEL["final"][key]
        if t == "swi":
            mv = SEL["stage1b"]["selections"][key]["mean_val_acc"]
            v = next(x for k_, x in mv.items() if float(k_) == a)
            g = ", ".join(f"{x:g}" for x in sorted(SEL["grid"] + [0.001, 1.0]))
        else:
            v = SEL["selected"][key]["val_acc"]
            g = grid
        NUM["tuning"][key] = dict(alpha=a, val_acc=v)
        rows.append((f"{n} & \\{{{g}\\}}" if j == 0 else " & ") + f" & {K} & {a} ({100 * v:.1f})" + r" \\")
    rows.append(r"\midrule" if t != TUNE[-1][0] else "")
rows = [r for r in rows if r] + [r"\bottomrule", r"\end{tabular}"]
NUM["tuning"]["swi_alpha0_val"] = {k: v["diagnostic_alpha0_mean_val_acc"]
                                   for k, v in SEL["stage1b"]["selections"].items()}
write("tab_tuning.tex", "\n".join(rows) + "\n")

# --------------------------------------------------------------------------- exact vs greedy replay
import csv

rep = list(csv.DictReader(open(os.path.join(os.path.dirname(NUMS), "paper_tables", "solver_replay.csv"))))
rows = [r"\begin{tabular}{lcccc}", r"\toprule",
        r" & \multicolumn{2}{c}{Greedy, $\Phi/\Phi^\star$} & \multicolumn{2}{c}{Lower bound enforced, $\Phi/\Phi^\star$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"$K$ & mean & worst epoch & mean & worst epoch \\", r"\midrule"]
NUM["greedy"] = {}
for K in sorted({int(r["K"]) for r in rep}):
    g = next(r for r in rep if int(r["K"]) == K and r["solver"] == "greedy")
    f = next(r for r in rep if int(r["K"]) == K and r["solver"] == "exact_floor")
    NUM["greedy"][f"K{K}"] = dict(mean=float(g["phi/phi_opt mean"]), max=float(g["phi/phi_opt max"]),
                                  floor_mean=float(f["phi/phi_opt mean"]), floor_max=float(f["phi/phi_opt max"]),
                                  epochs=int(g["epochs"]))
    rows.append(f"{K} & {float(g['phi/phi_opt mean']):.3f} & {float(g['phi/phi_opt max']):.2f} & "
                f"{float(f['phi/phi_opt mean']):.3f} & {float(f['phi/phi_opt max']):.2f}" + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
write("tab_greedy.tex", "\n".join(rows) + "\n")

# --------------------------------------------------------------------------- figure: K scaling
fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.9), sharex=True)
for j, (d, dname) in enumerate(DATASETS):
    by = D[d]
    for i, (m, lab) in enumerate((("acc", "overall acc. (%)"), ("few", "few-shot acc. (%)"))):
        ax = axes[i, j]
        for t, n in ROUTERS:
            pts = [(K, vals(by, arm(t, K), m)) for K in KS if vals(by, arm(t, K), m)]
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [np.mean(p[1]) for p in pts]
            e = [np.std(p[1], ddof=1) if len(p[1]) > 1 else 0 for p in pts]
            ax.errorbar(x, y, yerr=e, color=COLOR[t], lw=2.2 if t == "gt3" else 1.0,
                        ls="-" if t != "soft" else ":", marker="o", ms=3 if t == "gt3" else 2,
                        capsize=1.5, label=n, zorder=5 if t == "gt3" else 2)
        dv = vals(by, "P_dense", m)
        if dv:
            ax.axhline(np.mean(dv), color="0.4", ls="--", lw=0.8, label="dense head")
        ax.set_xscale("log", base=2)
        ax.set_xlim(6.5, 39)
        ax.set_xticks(KS)
        ax.set_xticklabels([str(k) for k in KS])
        ax.grid(alpha=0.25, lw=0.5)
        if i == 0:
            ax.set_title(dname)
        if i == 1:
            ax.set_xlabel("number of experts $K$")
        if j == 0:
            ax.set_ylabel(lab)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, "fig_kscaling")


# --------------------------------------------------------------------------- figure: affinity
def ordered(A):
    """Group classes by the expert that receives most of their slots; experts owning none go last."""
    owner = A.argmax(1)
    order_e = sorted(set(owner.tolist()))
    order_e += [e for e in range(A.shape[1]) if e not in order_e]
    rank = {e: i for i, e in enumerate(order_e)}
    rows_ = sorted(range(A.shape[0]), key=lambda c: (rank[owner[c]], -A[c, owner[c]]))
    return A[np.ix_(rows_, order_e)]


AFF = [("gt3", "GT3MoE"), ("km", "k-means, fixed"), ("swi", "Switch"), ("ec", "Expert Choice")]
fig, axes = plt.subplots(1, len(AFF), figsize=(6.75, 2.3))
for ax, (t, n) in zip(axes, AFF):
    r = C100.get(arm(t, 32), {}).get(20)
    if r is None:
        ax.set_axis_off()
        continue
    # affinity rows sum to 1 over the k slots of each sample; times k it is the share of the class's
    # samples that reach the expert
    A = ordered(np.clip(np.asarray(r["test"]["affinity"]) * r["config"]["top_k"], 0, 1))
    im = ax.imshow(A, aspect="auto", cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(n)
    ax.set_xlabel("expert")
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].set_ylabel("class")
cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
cb.set_label("share of the class's samples\nrouted to the expert", fontsize=7)
save(fig, "fig_affinity")

# --------------------------------------------------------------------------- figure: specialisation
SPEC = [("gt3", "GT3MoE")] + [x for x in ROUTERS if x[0] != "gt3"]
fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.2), sharey=False)
for ax, (d, dname) in zip(axes, DATASETS):
    by = D[d]
    xs, own, oth, own_e, oth_e, labs = [], [], [], [], [], []
    for i, (t, n) in enumerate(SPEC):
        rs = [by[arm(t, 32)][s] for s in sorted(by.get(arm(t, 32), {}))]
        g = [own_other(r["test"]) for r in rs]
        if not rs:
            continue
        xs.append(i)
        if any(x is None for x in g):
            own.append(np.nan); oth.append(np.nan); own_e.append(0); oth_e.append(0)
            labs.append(n + "$^\\ddagger$")
            continue
        own.append(np.mean([x[0] for x in g]))
        oth.append(np.mean([x[1] for x in g]))
        own_e.append(np.std([x[0] for x in g], ddof=1) if len(g) > 1 else 0)
        oth_e.append(np.std([x[1] for x in g], ddof=1) if len(g) > 1 else 0)
        labs.append(n + ("$^\\dagger$" if len(g) < 3 else ""))
    ax.set_title(dname + ", $K{=}32$")
    if not xs:
        ax.text(0.5, 0.5, "runs pending", ha="center", va="center", transform=ax.transAxes, color="0.5")
        ax.set_xticks([])
        ax.set_yticks([])
        continue
    xs = np.arange(len(xs))
    ax.bar(xs - 0.2, own, 0.4, yerr=own_e, color="#c0392b", label="own classes", capsize=1.5)
    ax.bar(xs + 0.2, oth, 0.4, yerr=oth_e, color="#aab7b8", label="other classes", capsize=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=6)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
axes[0].set_ylabel("expert's standalone acc. (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
save(fig, "fig_specialisation")

# --------------------------------------------------------------------------- figure: convergence
fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.2))
DS_COL = {"P_c100lt": "#c0392b", "P_mini": "#2471a3", "P_tiny": "#17a589"}
for d, dname in DATASETS:
    by = D[d]
    for K in KS:
        stats_ = []
        for s, r in sorted(by.get(arm("gt3", K), {}).items()):
            h = r["history"]
            C = r["config"]["num_classes"]
            first = r["config"]["warmup_epochs"]
            churn = np.array([e.get("assign/churn", 0) for e in h], float)
            moves = [e for e in range(first + 1, len(h)) if churn[e] > 0]
            stats_.append(dict(seed=s, first_solve=int(churn[first]), C=C,
                               last_move_epoch=int(moves[-1]) if moves else first,
                               max_after_first=int(churn[first + 1:].max()) if len(h) > first + 1 else 0,
                               moves_after_first=int(churn[first + 1:].sum()), epochs=len(h)))
            if K == 32:
                ep = np.arange(len(h))
                phi = np.array([e.get("assign/phi", np.nan) for e in h], float) / C
                lab = dname if s == min(by[arm("gt3", K)]) else None
                axes[0].plot(ep[first:], phi[first:], color=DS_COL[d], lw=0.9, alpha=0.8, label=lab)
                axes[1].plot(ep, churn, color=DS_COL[d], lw=0.9, alpha=0.8, label=lab)
        if stats_:
            NUM["convergence"].setdefault(d, {})[f"K{K}"] = stats_
axes[0].set_yscale("log")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel(r"$\Phi(\pi^{(t)};L^{(t)})\,/\,C$ (nats)")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("classes reassigned")
axes[1].set_yscale("symlog", linthresh=1)
for ax in axes:
    ax.grid(alpha=0.25, lw=0.5)
axes[1].legend(frameon=False)
fig.tight_layout()
save(fig, "fig_convergence")

json.dump(NUM, open(NUMS, "w"), indent=1)
print("  wrote " + NUMS)
