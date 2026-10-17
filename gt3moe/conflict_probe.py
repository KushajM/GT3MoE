"""Is the group-gradient conflict signal or sampling noise?

Runs from a checkpoint, so it costs minutes rather than a training arm.

The problem with measuring conflict from single-step gradients: at K=8 / batch 128 each group
gradient is ~16 samples, and the attenuation of the measured cosine toward the true one is
``SNR/(1+SNR)``. At the SNR implied by this project's own numbers that is ~0.01, so a single-step
cosine reports about 1% of the systematic effect and 99% noise. Window-averaging W steps multiplies
the SNR by W, lifting attenuation to ~0.84 -- you measure ``cos(mu_i, mu_j)`` almost directly.

Two things this corrects in the earlier ``splithalf_conflict``:

* The cross-group split-half term was redundant. The groups already partition the batch, so the
  noise terms are independent within a single batch and there is no shared-sample coupling to
  remove. Computing it across halves just halves n, making it ~2x more attenuated and ~2x noisier
  -- a strictly worse estimator of the same quantity.
* BatchNorm. Group gradients are mechanically anti-correlated at order ``n_i/B`` through shared
  batch statistics, which manufactures conflict unrelated to any task structure. Running with BN in
  eval mode (running statistics) removes it; running both ways quantifies it.

Also reports the quantities that distinguish "PCGrad removes conflict" from "PCGrad is an
accidental gradient-norm equalizer": the ``||g||`` dispersion across groups, the normalised-weight
churn, and the median/p99 of ``proj_shift`` (whose mean is dominated by rare blow-up steps when a
well-fit group's ``||g||^2`` approaches the 1e-12 guard).

    python -m gt3moe.conflict_probe --ckpt outputs/ckpt/gt3moe_s0.pt --steps 800 --window 50
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .config import make_config
from .engine import Trainer
from .pcgrad import build_groups, make_perms, pcgrad_coefficients


def _set_bn_eval(model, enabled: bool):
    """Freeze BN to running statistics so the per-group gradients are genuinely independent."""
    if not enabled:
        return
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


@torch.enable_grad()
def probe(tr: Trainer, steps: int = 800, window: int = 50, bn_eval: bool = True,
          max_groups: int = 8) -> Dict:
    cfg = tr.cfg
    model = tr.model
    scope = tr.updater.scope
    K = model.K
    P = sum(p.numel() for p in scope)

    acc = torch.zeros(K, P, device=tr.dev, dtype=torch.float32)
    cnt = torch.zeros(K, device=tr.dev)
    windows: List[np.ndarray] = []          # per-window [K,K] cosine matrices
    norms: List[np.ndarray] = []            # per-step per-group ||g||
    shifts, mass, churn, stepratio = [], [], [], []
    gsizes = []

    model.train()
    _set_bn_eval(model, bn_eval)
    done = 0
    while done < steps:
        for xb, yb, _ in tr.tr.train_batches(cfg.batch_size, tr.gen, cfg.augment, True):
            if done >= steps:
                break
            o = model(xb, training=True, prec=tr.prec, gen=tr.gen)
            per = F.cross_entropy(o["out"].float(), yb, reduction="none")
            gid, w, gs = build_groups(yb, tr.assign_gpu, K, cfg, tr.cpu_gen)
            gsizes.append(gs.get("min_group_size", float("nan")))
            groups = sorted({int(g) for g in gid.unique().tolist() if g >= 0})[:max_groups]
            if len(groups) < 2:
                done += 1
                continue
            G = []
            for g in groups:
                m = gid == g
                loss = (per[m] * w[m]).sum()
                gr = torch.autograd.grad(loss, scope, retain_graph=True, allow_unused=True)
                G.append(torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                                    for x, p in zip(gr, scope)]).float())
            M = torch.stack(G)
            # Pad to K: the number of surviving groups varies step to step, so a ragged list
            # cannot be np.stack'ed later.
            row = np.full(K, np.nan)
            row[np.asarray(groups)] = M.norm(dim=1).detach().cpu().numpy()
            norms.append(row)
            # what PCGrad would do on this step, without applying it
            gram = (M @ M.T).double()
            A = pcgrad_coefficients(gram, make_perms(len(groups), tr.cpu_gen))
            wc = A.mean(0)
            shifts.append(float((wc - 1.0 / len(groups)).abs().sum()))
            mass.append(float(wc.sum()))
            wn = wc / wc.sum()
            churn.append(float((wn - 1.0 / len(groups)).abs().sum()))
            upd = (wc.to(M.dtype) @ M)
            stepratio.append(float(upd.norm() / M.mean(0).norm().clamp(min=1e-20)))
            for gi, g in enumerate(groups):
                acc[g] += M[gi].detach()
                cnt[g] += 1
            del M, G
            model.zero_grad(set_to_none=True)
            done += 1
            if done % window == 0:
                keep = (cnt > 0).nonzero(as_tuple=True)[0]
                if keep.numel() >= 2:
                    W = acc[keep] / cnt[keep, None]
                    Wn = W / W.norm(dim=1, keepdim=True).clamp(min=1e-20)
                    C = (Wn @ Wn.T).detach().cpu().numpy()
                    full = np.full((K, K), np.nan)
                    ki = keep.cpu().numpy()
                    full[np.ix_(ki, ki)] = C
                    windows.append(full)
                acc.zero_(); cnt.zero_()

    if len(windows) < 2:
        return {"error": "not enough windows"}
    Wst = np.stack(windows)                                   # [T,K,K]
    iu = np.triu_indices(K, 1)
    pairs = Wst[:, iu[0], iu[1]]                              # [T, n_pairs]
    ok = ~np.isnan(pairs).any(0)
    pairs = pairs[:, ok]
    T, npair = pairs.shape
    pbar = pairs.mean(0)
    within = pairs.var(0, ddof=1).mean()
    between = pbar.var(ddof=1)
    se = np.sqrt(within / T)
    # ICC-style variance decomposition: does between-pair variance exceed sampling variance?
    icc = max(0.0, (between - within / T) / max(1e-12, between + within * (1 - 1 / T)))
    sign_persist = float((np.sign(pairs[1:]) == np.sign(pairs[:-1])).mean())
    nrm = np.stack(norms)
    out = dict(
        bn_eval=bn_eval, steps=done, window=window, n_windows=T, n_pairs=int(npair),
        # the headline: window-averaged per-pair cosines
        pair_mean=float(pbar.mean()), pair_min=float(pbar.min()), pair_max=float(pbar.max()),
        pair_se=float(se), pair_neg_frac=float((pbar < 0).mean()),
        n_pairs_below_minus_0p04_at_3se=int(((pbar < -0.04) & (pbar < -3 * se)).sum()),
        icc=float(icc), sign_persistence=sign_persist,
        pair_q10=float(np.quantile(pbar, 0.10)), pair_q90=float(np.quantile(pbar, 0.90)),
        pair_kurtosis=float(((pbar - pbar.mean()) ** 4).mean() / max(1e-20, pbar.var() ** 2) - 3),
        # mechanism: is PCGrad a norm equalizer?
        norm_log_sd=float(np.nanstd(np.log(np.clip(nrm, 1e-20, None)))),
        norm_ratio_max_min=float(np.nanmean(np.nanmax(nrm, 1)
                                            / np.clip(np.nanmin(nrm, 1), 1e-20, None))),
        proj_shift_mean=float(np.mean(shifts)), proj_shift_median=float(np.median(shifts)),
        proj_shift_p99=float(np.quantile(shifts, 0.99)),
        coef_mass_mean=float(np.mean(mass)),
        normalised_weight_churn=float(np.mean(churn)),
        step_norm_ratio_mean=float(np.mean(stepratio)),
        step_norm_ratio_cv=float(np.std(stepratio) / max(1e-12, np.mean(stepratio))),
        min_group_size_mean=float(np.nanmean(gsizes)),
    )
    out["verdict"] = (
        "CONFLICT IS NOISE" if (out["n_pairs_below_minus_0p04_at_3se"] == 0 and icc < 0.1
                                and sign_persist < 0.55)
        else "CONFLICT MAY BE REAL -- inspect per-pair values")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--out", default="conflict_probe.json")
    ap.add_argument("--both_bn", action="store_true", default=True)
    a = ap.parse_args(argv)

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = make_config("v2", **{k: v for k, v in sd.get("cfg", {}).items()}) \
        if "cfg" in sd else make_config("v2")
    cfg.resume = False
    res = {}
    for bn_eval in ([True, False] if a.both_bn else [True]):
        tr = Trainer(cfg)
        tr.model.load_state_dict(sd["model"])
        tr.assignment = np.asarray(sd["assignment"])
        tr.assign_gpu = torch.as_tensor(tr.assignment, device=tr.dev, dtype=torch.long)
        res["bn_eval" if bn_eval else "bn_train"] = probe(tr, a.steps, a.window, bn_eval)
        del tr
        torch.cuda.empty_cache()
    if "bn_eval" in res and "bn_train" in res:
        res["bn_artifact_delta_pair_mean"] = (res["bn_train"]["pair_mean"]
                                              - res["bn_eval"]["pair_mean"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
