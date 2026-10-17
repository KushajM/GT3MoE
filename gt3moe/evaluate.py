"""Metrics and the diagnostics the original never computed.

The four that decide whether the paper's story is true, all eval-only and essentially free:

* **router group accuracy** ``P(pi(y) in top-k(x))`` -- can the router find the assigned expert
  without seeing the label?
* **oracle-routing accuracy** -- force every test sample through ``pi(y)``'s expert. This is the
  ceiling the router is failing to reach.
* **top-1 vs top-2 inference** -- the whole method is top-2; the top-1 number is never reported.
* **the routing/within-group error decomposition** -- with a *blended* output the decomposition
  needs three variables, not two: was the assigned expert selected, would the assigned expert have
  been right, and was the blend right. The 2x2 version silently conflates "router sent it
  elsewhere and that was fine" with "router sent it elsewhere and that lost the sample".

Plus the honest replacement for Figure 5: a fully-measured, held-out competence matrix. The
original computed ``A = np.where(cc>0, cs/cc, 0.0)`` and averaged the zero-filled unmeasured cells
into the "non-primary accuracy" curve, so that curve largely measures *missing data* rather than
incompetence -- and the "primary" curve is training accuracy on augmented data, which is why it
reads 99% against a 56% test accuracy.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, images, cfg, prec, assignment: Optional[np.ndarray] = None,
             batch_size: int = 512, full: bool = True) -> Dict:
    """Single pass over ``images`` producing metrics plus (optionally) all diagnostics."""
    K, C = model.K, model.C
    dev = images.x.device
    was_training = model.training
    model.eval()

    n = images.n
    yt = torch.empty(n, dtype=torch.long, device=dev)
    yp = torch.empty(n, dtype=torch.long, device=dev)
    yp1 = torch.empty(n, dtype=torch.long, device=dev)
    ent = torch.zeros(n, device=dev)
    g2 = torch.zeros(n, device=dev)
    top_hit = torch.zeros(n, dtype=torch.bool, device=dev)
    ora_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    best_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    uni_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    affinity = torch.zeros(C, K, device=dev)
    expert_cls = torch.zeros(K, C, device=dev)
    util = torch.zeros(K, device=dev)
    comp_loss_sum = torch.zeros(K, C, device=dev, dtype=torch.float64)
    comp_ok_sum = torch.zeros(K, C, device=dev, dtype=torch.float64)
    comp_cnt = torch.zeros(K, C, device=dev, dtype=torch.float64)

    asg = None if assignment is None else torch.as_tensor(assignment, device=dev, dtype=torch.long)

    for xb, yb, sl in images.eval_batches(batch_size):
        with prec.eval_autocast():
            o = model(xb, training=False, prec=prec)
        out = o["out"].float()
        B = out.shape[0]
        yt[sl] = yb
        yp[sl] = out.argmax(1)
        ti, tg = o["top_idx"], o["top_gates"]
        # top-1-only inference
        al = o["all_logits"]
        if al is not None:
            al = al.float()
            yp1[sl] = al[torch.arange(B, device=dev), ti[:, 0]].argmax(1)
            uni_ok[sl] = al.mean(1).argmax(1) == yb           # uniform-average-of-all-K ensemble
            if asg is not None:
                oa = asg[yb]
                ora_ok[sl] = al[torch.arange(B, device=dev), oa].argmax(1) == yb
            best_ok[sl] = (al.argmax(-1) == yb.unsqueeze(1)).any(1)
            # fully-measured competence matrix on held-out data
            ce = F.cross_entropy(al.reshape(-1, C), yb.unsqueeze(1).expand(B, K).reshape(-1),
                                 reduction="none").reshape(B, K)
            ok = (al.argmax(-1) == yb.unsqueeze(1)).double()
            lin = (torch.arange(K, device=dev).unsqueeze(0).expand(B, K).reshape(-1) * C
                   + yb.unsqueeze(1).expand(B, K).reshape(-1))
            comp_loss_sum.view(-1).index_add_(0, lin, ce.reshape(-1).double())
            comp_ok_sum.view(-1).index_add_(0, lin, ok.reshape(-1))
            comp_cnt.view(-1).index_add_(0, lin, torch.ones_like(lin, dtype=torch.float64))
        else:
            yp1[sl] = out.argmax(1)

        gp = o["gate_probs"].float().clamp_min(1e-12)
        gp = gp / gp.sum(1, keepdim=True)
        ent[sl] = -(gp * gp.log()).sum(1)
        if tg.shape[1] > 1:
            g2[sl] = tg[:, 1]
        if asg is not None:
            top_hit[sl] = (ti == asg[yb].unsqueeze(1)).any(1)
        # affinity / utilisation (vectorised; the original looped over 10k samples in Python)
        for s in range(ti.shape[1]):
            affinity.view(-1).index_add_(
                0, yb * K + ti[:, s], torch.ones(B, device=dev))
            expert_cls.view(-1).index_add_(
                0, ti[:, s] * C + yb, torch.ones(B, device=dev))
        util.index_add_(0, ti.reshape(-1), torch.ones(ti.numel(), device=dev))

    if was_training:
        model.train()

    yt_np, yp_np, yp1_np = yt.cpu().numpy(), yp.cpu().numpy(), yp1.cpu().numpy()
    res = _metrics(yt_np, yp_np, cfg)
    res["acc_top1_only"] = float((yp1_np == yt_np).mean())
    res["acc_uniform_ensemble"] = float(uni_ok.float().mean().item())
    res["acc_best_expert_oracle"] = float(best_ok.float().mean().item())
    res["mean_routing_entropy"] = float(ent.mean().item())
    res["mean_second_gate"] = float(g2.mean().item())

    cc = affinity.sum(1, keepdim=True).clamp(min=1)
    aff = (affinity / cc).cpu().numpy()
    ecd = expert_cls.cpu().numpy()
    tot = ecd.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(tot > 0, ecd / np.maximum(tot, 1), 0.0)
    ent_e = -(dist * np.log(np.clip(dist, 1e-12, None))).sum(1)
    res["expert_class_entropy"] = float(ent_e[tot[:, 0] > 0].mean()) if (tot > 0).any() else 0.0

    u = (util / max(1, images.n)).cpu().numpy()
    res["util"] = u.tolist()
    res["util_cv"] = float(u.std() / max(1e-12, u.mean()))
    res["util_std"] = float(u.std())
    # a threshold of 1% is not comparable across K; report a K-relative definition too
    res["dead_experts_abs1pct"] = int((u < 0.01).sum())
    res["dead_experts_rel"] = int((u < 0.1 * u.mean()).sum())

    if assignment is not None:
        res["router_group_acc"] = float(top_hit.float().mean().item())
        res["acc_oracle_routing"] = float(ora_ok.float().mean().item())
        res.update(_error_decomposition(yt_np, yp_np, ora_ok.cpu().numpy(),
                                        top_hit.cpu().numpy()))
        res.update(_bucketed(yt_np, top_hit.cpu().numpy(), cfg, "router_group_acc"))

    if full:
        cnt = comp_cnt.cpu().numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            res["competence_loss"] = np.where(cnt > 0, comp_loss_sum.cpu().numpy() / np.maximum(cnt, 1),
                                              np.nan).tolist()
            res["competence_acc"] = np.where(cnt > 0, comp_ok_sum.cpu().numpy() / np.maximum(cnt, 1),
                                             np.nan).tolist()
        res["competence_measured_cells"] = int((cnt > 0).sum())
        res["affinity"] = aff.tolist()
        if assignment is not None:
            res.update(_primary_nonprimary(np.asarray(res["competence_acc"]), assignment))
    return res


def _metrics(yt: np.ndarray, yp: np.ndarray, cfg) -> Dict:
    from sklearn.metrics import f1_score
    C = cfg.num_classes
    pca = np.zeros(C)
    for c in range(C):
        m = yt == c
        if m.any():
            pca[c] = (yp[m] == c).mean()
    out = dict(acc=float((yp == yt).mean()),
               f1=float(f1_score(yt, yp, average="macro", zero_division=0)),
               balanced_acc=float(pca.mean()),
               per_class_acc=pca.tolist())
    for name in ("many", "medium", "few"):
        idx = getattr(cfg, name) or []
        out[name] = float(pca[list(idx)].mean()) if len(idx) else 0.0
    return out


def _bucketed(yt: np.ndarray, ok: np.ndarray, cfg, prefix: str) -> Dict:
    out = {}
    for name in ("many", "medium", "few"):
        idx = set(getattr(cfg, name) or [])
        if not idx:
            continue
        m = np.isin(yt, list(idx))
        if m.any():
            out[f"{prefix}_{name}"] = float(ok[m].mean())
    return out


def _error_decomposition(yt, yp, oracle_ok, routed_to_assigned) -> Dict:
    """Three-variable decomposition of where accuracy is lost.

    A two-cell version conflates the case where the router did not pick the assigned expert but
    the blended output was right anyway with the case where that choice cost the sample.
    """
    blend_ok = yp == yt
    r = routed_to_assigned
    o = oracle_ok
    n = max(1, len(yt))
    return dict(
        frac_routed_ok=float((r & blend_ok).sum() / n),
        frac_routed_wrong=float((r & ~blend_ok).sum() / n),
        # routing error that actually cost a sample: assigned expert would have been right,
        # the router went elsewhere, and the blend was wrong
        frac_routing_error_costly=float((~r & o & ~blend_ok).sum() / n),
        # router went elsewhere and it did not matter
        frac_routing_error_benign=float((~r & blend_ok).sum() / n),
        # nobody could have saved it
        frac_within_group_error=float((~o & ~blend_ok).sum() / n),
        frac_blend_beats_oracle=float((~o & blend_ok).sum() / n),
    )


def _primary_nonprimary(acc_matrix: np.ndarray, assignment: np.ndarray) -> Dict:
    """Held-out primary/non-primary competence with unmeasured cells EXCLUDED, not zero-filled."""
    K, C = acc_matrix.shape
    pa, npa = [], []
    for i in range(K):
        pm = assignment == i
        if pm.any():
            v = acc_matrix[i, pm]
            v = v[~np.isnan(v)]
            if v.size:
                pa.append(float(v.mean()))
        nm = ~(assignment == i)
        v = acc_matrix[i, nm]
        v = v[~np.isnan(v)]
        if v.size:
            npa.append(float(v.mean()))
    return dict(primary_acc_heldout=float(np.mean(pa)) if pa else float("nan"),
                nonprimary_acc_heldout=float(np.mean(npa)) if npa else float("nan"),
                primary_acc_min=float(np.min(pa)) if pa else float("nan"),
                primary_acc_max=float(np.max(pa)) if pa else float("nan"))


def assignment_semantics(assignment: np.ndarray, coarse_labels: Optional[np.ndarray]) -> Dict:
    """NMI/ARI of the learnt partition against CIFAR-100's 20 superclasses. Free interpretability."""
    if coarse_labels is None:
        return {}
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    return dict(assign_nmi_superclass=float(normalized_mutual_info_score(coarse_labels, assignment)),
                assign_ari_superclass=float(adjusted_rand_score(coarse_labels, assignment)))


@torch.no_grad()
def measure_flops(model, cfg, prec, device: str = "cuda") -> Dict:
    """Forward FLOPs, dense vs sparse (top-k only), using torch's native counter."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return {}
    x = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)
    if cfg.channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    out = {}
    for name, sparse in (("dense", False), ("sparse", True)):
        try:
            with FlopCounterMode(display=False) as fc:
                model(x, training=False, prec=prec, sparse=sparse)
            out[f"fwd_flops_{name}"] = int(fc.get_total_flops())
        except Exception:
            pass
    return out
