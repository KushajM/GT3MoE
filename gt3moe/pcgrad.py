"""Assignment-grouped PCGrad on the shared backbone.

Two structural facts drive this module.

**Fact 1 -- the projection is exactly a linear recombination.** Every inner product in PCGrad is
between a running projected vector ``p_i = sum_j a_ij g_j`` and an *original* ``g_j``. So the whole
algorithm can be run in the G-dimensional coefficient basis given only the G x G Gram matrix, and
the final update is ``w @ M`` for a single coefficient vector ``w``. This is exact, not an
approximation: it reproduces the naive loop bit-for-bit up to fp accumulation order, while
replacing ~G^2 full-length axpy passes and ~G^2 host syncs with one GEMM and one host sync.

**Fact 2 -- per-sample feature-space PCGrad is vacuous here.** Group i's loss depends only on
group i's samples' features, and the groups partition the batch, so the per-sample feature-gradient
matrices ``G_i in R^{BxD}`` have *disjoint support* and ``<G_i, G_j> = 0`` identically. No conflict
can ever be detected. The conflict arises entirely from the shared Jacobian, which is why it must
be measured in parameter space.

What that argument does *not* kill is **pooling**: the per-group mean feature gradients
``ghat_i in R^D`` are D-vectors, each averaging a different set of samples, and they conflict
freely. ``pcgrad_impl="feature"`` projects those and needs one backward instead of G -- a
surrogate, flagged as such, with an agreement-rate diagnostic against the exact path.

``pcgrad_impl="captured"`` is exact for a single-layer scope and also needs only two backwards:
for a layer with input ``a`` and output-gradient ``delta``, ``dL_i/dW = sum_{b in B_i} delta_b a_b^T``,
and one backward of the summed per-sample loss already yields per-sample-separated ``delta``. The
Gram matrix is formed from those, the coefficients solved, and one backward of
``sum_j w_j L_j`` applies the recombination exactly to the full backbone.

Also note the backbone objective itself: ``backbone_ce="group_mean"`` (the original) gives each
sample an effective weight of ``1/(G |B_i|)`` rather than ``1/B``, which under a long tail is an
undisclosed class-balanced reweighting -- plausibly a large part of the paper's tail result.
``"size_weighted"`` provably reproduces the full-batch CE gradient before projection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- core projection

def pcgrad_coefficients(gram: torch.Tensor, perm: Sequence[Sequence[int]]) -> torch.Tensor:
    """Exact PCGrad in coefficient space.

    ``gram`` is [G,G] with ``gram[i,j] = <g_i, g_j>``. ``perm[i]`` is the visit order for row i
    (the original drew a fresh ``torch.randperm(K)`` per i). Returns ``A`` [G,G] float64 such that
    the projected gradient is ``p_i = sum_j A[i,j] g_j``.
    """
    G = gram.shape[0]
    Gr = gram.double()
    A = torch.eye(G, dtype=torch.float64, device=gram.device)
    diag = Gr.diagonal().clone()
    for i in range(G):
        for j in perm[i]:
            if j == i:
                continue
            dot = torch.dot(A[i], Gr[:, j])            # == <p_i, g_j>, exactly
            if dot < 0:
                # p_i <- p_i - (dot / <g_j,g_j>) g_j  is just a scalar update to A[i,j]
                A[i, j] = A[i, j] - dot / (diag[j] + 1e-12)
    return A


def pcgrad_naive(grads: List[torch.Tensor], perm: Sequence[Sequence[int]]) -> List[torch.Tensor]:
    """Reference implementation, used only to test ``pcgrad_coefficients``."""
    proj = [g.clone() for g in grads]
    for i in range(len(grads)):
        for j in perm[i]:
            if j == i:
                continue
            dot = torch.dot(proj[i], grads[j])
            if dot < 0:
                proj[i] = proj[i] - (dot / (torch.dot(grads[j], grads[j]) + 1e-12)) * grads[j]
    return proj


def make_perms(G: int, gen: torch.Generator) -> List[List[int]]:
    """One fresh permutation per row, matching the original's ``torch.randperm(K)`` per i."""
    return [torch.randperm(G, generator=gen).tolist() for _ in range(G)]


def splithalf_conflict(per_sample_ce, gid, K, scope_params, gen,
                       max_groups: int = 8) -> Dict[str, float]:
    """Split-half gradient SNR: is the measured conflict signal or sampling noise?

    A pairwise cosine computed within one batch cannot distinguish the two, because each group
    gradient is built from ~16 samples and therefore lives in a <=16-dimensional subspace where
    chance cosines are O(1/sqrt(16)). Splitting the batch into disjoint halves decouples them:

      same-group  cos(g_i^(1), g_i^(2))  -> gradient SNR, the only quantity a single batch can
                                            supply that is not already available in-batch.

    CORRECTION: an earlier version also returned a cross-group cross-half cosine, described as
    "free of shared-sample coupling". That was wrong -- the groups already partition the batch, so
    the noise terms are independent within one batch and there is no coupling to remove. Computed
    across halves it just halves n, making it ~2x more attenuated and ~2x noisier than the in-batch
    pairwise cosine: a strictly worse estimator of the same quantity. It is retained only as a
    cross-check and must not be read as the conflict measurement.

    NOTE ON BATCHNORM: both halves come from ONE forward pass, so they share batch statistics and
    are NOT independent. BN also anti-correlates group gradients mechanically at order n_i/B. Use
    ``gt3moe.conflict_probe`` (BN in eval mode, window-averaged) for the real measurement.
    """
    dev = per_sample_ce.device
    B = per_sample_ce.shape[0]
    perm = torch.randperm(B, generator=gen).to(dev)
    half = B // 2
    halves = (perm[:half], perm[half:])
    groups = sorted({int(g) for g in gid.unique().tolist() if g >= 0})[:max_groups]
    if len(groups) < 2:
        return {}

    G = [[], []]
    for h, idx in enumerate(halves):
        mask_h = torch.zeros(B, dtype=torch.bool, device=dev)
        mask_h[idx] = True
        for g in groups:
            m = (gid == g) & mask_h
            if m.sum() < 2:
                G[h].append(None)
                continue
            loss = per_sample_ce[m].mean()
            gs = torch.autograd.grad(loss, scope_params, retain_graph=True, allow_unused=True)
            G[h].append(torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1)
                                   for x, p in zip(gs, scope_params)]))
    same, cross = [], []
    for i, gi in enumerate(groups):
        a, b = G[0][i], G[1][i]
        if a is not None and b is not None:
            same.append(float(F.cosine_similarity(a, b, dim=0)))
        for j, _gj in enumerate(groups):
            if i == j:
                continue
            a, b = G[0][i], G[1][j]
            if a is not None and b is not None:
                cross.append(float(F.cosine_similarity(a, b, dim=0)))
    if not same or not cross:
        return {}
    same_t = torch.tensor(same)
    cross_t = torch.tensor(cross)
    return dict(
        sh_same_mean=float(same_t.mean()),            # gradient SNR proxy
        sh_cross_mean=float(cross_t.mean()),          # cross-check ONLY, see docstring
        sh_cross_neg_frac=float((cross_t < 0).float().mean()),
        sh_cross_std=float(cross_t.std()) if cross_t.numel() > 1 else 0.0,
        # bimodality guard: mean~0 could hide strong-conflict and strong-alignment cancelling
        sh_cross_q10=float(cross_t.quantile(0.10)), sh_cross_q90=float(cross_t.quantile(0.90)),
        sh_n_groups=float(len(groups)),
    )


def conflict_stats(gram: torch.Tensor) -> Dict[str, float]:
    G = gram.shape[0]
    if G < 2:
        return dict(neg_pair_frac=0.0, mean_cos=0.0, n_groups=float(G))
    d = gram.diagonal().clamp(min=1e-20).sqrt()
    cos = gram / (d[:, None] * d[None, :])
    off = ~torch.eye(G, dtype=torch.bool, device=gram.device)
    return dict(neg_pair_frac=float((cos[off] < 0).float().mean().item()),
                mean_cos=float(cos[off].mean().item()),
                n_groups=float(G))


# --------------------------------------------------------------------------- grouping

def build_groups(labels: torch.Tensor, assignment: torch.Tensor, K: int, cfg,
                 gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Return (gid [B], weights [B], stats).

    ``gid`` is the group index per sample (``-1`` for samples excluded from every group).
    ``weights`` is the per-sample coefficient of the backbone CE, so the caller never has to
    reason about the reweighting again.
    """
    B = labels.shape[0]
    if cfg.pcgrad_groups == "random":
        # matched group-size distribution, but semantically meaningless partition (the E4(d) control)
        gid = assignment[labels]
        # Draw on the CPU generator and move, rather than passing a CPU generator to a cuda
        # randperm (which raises "Expected a 'cuda' device type for generator but found 'cpu'").
        # This also keeps ALL pcgrad-related randomness on one stream, matching make_perms().
        perm = torch.randperm(B, generator=gen).to(labels.device)
        gid = gid[perm]
    else:
        gid = assignment[labels]

    counts = torch.bincount(gid, minlength=K).float()
    small = counts < cfg.pcgrad_min_group
    if cfg.pcgrad_singleton == "merge_to_residual":
        # pool all sub-threshold groups into one pseudo-group, so no sample is dropped and the
        # backbone objective remains the full-batch CE under size_weighted
        resid = small[gid]
        gid = torch.where(resid, torch.full_like(gid, K), gid)
        counts = torch.bincount(gid, minlength=K + 1).float()
        keep_mask = counts >= 1
    else:  # "drop" -- the original behaviour
        gid = torch.where(small[gid], torch.full_like(gid, -1), gid)
        counts = torch.bincount(gid.clamp(min=0), minlength=K).float()
        counts = torch.where(small, torch.zeros_like(counts), counts)
        keep_mask = counts >= cfg.pcgrad_min_group

    valid = gid >= 0
    G_used = int(keep_mask.sum().item())

    # per-sample backbone weight
    w = torch.zeros(B, device=labels.device, dtype=torch.float32)
    if cfg.backbone_ce == "full_batch":
        w[:] = 1.0 / max(1, B)
    elif cfg.backbone_ce == "size_weighted":
        w[valid] = 1.0 / max(1, B)                    # sums to |valid|/B == full-batch CE grad
    else:  # group_mean, the original: 1/(G |B_i|)
        cn = counts.clamp(min=1)
        w[valid] = 1.0 / (max(1, G_used) * cn[gid[valid]])

    dropped = (~valid).float()
    stats = dict(
        n_groups_used=float(G_used),
        mean_group_size=float(counts[keep_mask].mean().item()) if G_used else 0.0,
        min_group_size=float(counts[keep_mask].min().item()) if G_used else 0.0,
        dropped_frac=float(dropped.mean().item()),
    )
    return gid, w, stats


def dropped_by_bucket(labels: torch.Tensor, gid: torch.Tensor,
                      bucket_masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Fraction of samples excluded from the backbone gradient, split many/medium/few.

    At K=32 with batch 64 the mean group size is 2 and the ``< min_group`` rule silently removes a
    large share of few-shot samples from the backbone update entirely. Nothing in the original
    logged this, and it is a confound in the ``+PCGrad`` ablation row.
    """
    out = {}
    bad = gid < 0
    for name, cls_mask in bucket_masks.items():
        m = cls_mask[labels]
        if m.any():
            out[f"dropped_frac_{name}"] = float(bad[m].float().mean().item())
    return out


# --------------------------------------------------------------------------- the backbone step

class BackboneUpdater:
    """Computes the backbone gradient under the selected grouping and projection."""

    def __init__(self, cfg, model, scope_params: List[torch.nn.Parameter]):
        self.cfg = cfg
        self.model = model
        self.scope = scope_params
        self.all_bb = list(model.backbone.parameters())
        self._group_ema: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ impl: loop (exact)
    def _loop(self, per_sample_ce, gid, w, K, gen, retain=True):
        cfg = self.cfg
        groups = sorted({int(g) for g in gid.unique().tolist() if g >= 0})
        if cfg.pcgrad_max_groups and len(groups) > cfg.pcgrad_max_groups:
            sel = torch.randperm(len(groups), generator=gen)[:cfg.pcgrad_max_groups].tolist()
            groups = [groups[i] for i in sel]
            scale = 1.0
        else:
            scale = 1.0
        losses = []
        for g in groups:
            m = gid == g
            losses.append((per_sample_ce[m] * w[m]).sum())
        if len(losses) < 2:
            tot = (per_sample_ce * w).sum()
            flat = _flat_grad(tot, self.all_bb, retain_graph=retain)
            return flat, dict(n_groups_projected=float(len(losses)), fallback=1.0), losses

        M = torch.stack([_flat_grad(l, self.scope, retain_graph=True) for l in losses])  # [G,P]
        with torch.autocast("cuda", enabled=False) if M.is_cuda else _null():
            gram = (M @ M.T).double()
        st = conflict_stats(gram)
        if cfg.pcgrad_project:
            A = pcgrad_coefficients(gram, make_perms(len(losses), gen))
            wcoef = A.mean(0)
        else:
            wcoef = torch.full((len(losses),), 1.0 / len(losses),
                               dtype=torch.float64, device=gram.device)
        st["proj_shift"] = float(
            (wcoef - torch.full_like(wcoef, 1.0 / len(losses))).abs().sum().item())

        if self.cfg.pcgrad_scope == "full":
            flat = (wcoef.to(M.dtype) @ M) * scale
        else:
            # coefficients measured on the scope, applied exactly to the full backbone via ONE
            # backward of sum_j w_j L_j  (the update is a linear recombination, Fact 1)
            del M
            combo = sum(float(c) * l for c, l in zip(wcoef.tolist(), losses))
            flat = _flat_grad(combo, self.all_bb, retain_graph=retain)
        return flat, {**st, "fallback": 0.0}, losses

    # ------------------------------------------------------------------ impl: feature (surrogate)
    def _feature(self, per_sample_ce, gid, w, K, feats, gen, retain=True):
        cfg = self.cfg
        tot = (per_sample_ce * w).sum()
        g = torch.autograd.grad(tot, feats, retain_graph=True)[0].float()      # [B,D], ONE backward
        valid = gid >= 0
        D = g.shape[1]
        ghat = torch.zeros(K + 1, D, device=g.device, dtype=g.dtype)
        cnt = torch.zeros(K + 1, device=g.device, dtype=g.dtype)
        idx = gid.clamp(min=0)
        ghat.index_add_(0, idx[valid], g[valid])
        cnt.index_add_(0, idx[valid], torch.ones(int(valid.sum()), device=g.device, dtype=g.dtype))
        keep = (cnt >= 1).nonzero(as_tuple=True)[0]
        if keep.numel() < 2:
            feats.backward(gradient=g, retain_graph=retain)
            return None, dict(n_groups_projected=float(keep.numel()), fallback=1.0), None
        gk = ghat[keep] / cnt[keep, None]
        if cfg.pcgrad_group_ema > 0:
            if self._group_ema is None or self._group_ema.shape != ghat.shape:
                self._group_ema = torch.zeros_like(ghat)
            r = cfg.pcgrad_group_ema
            self._group_ema[keep] = r * self._group_ema[keep] + (1 - r) * gk
            gk = self._group_ema[keep]
        gram = (gk @ gk.T).double()
        st = conflict_stats(gram)
        if cfg.pcgrad_project:
            A = pcgrad_coefficients(gram, make_perms(gk.shape[0], gen))
            proj = A.to(gk.dtype) @ gk                             # [Gk,D]
            delta = torch.zeros_like(ghat)
            delta[keep] = proj - gk
            gmod = g + delta[idx] * valid.unsqueeze(1).to(g.dtype)
        else:
            gmod = g
        feats.backward(gradient=gmod, retain_graph=retain)
        return None, {**st, "fallback": 0.0}, None

    # ------------------------------------------------------------------ dispatch
    def compute(self, per_sample_ce, gid, w, K, feats, gen, retain=True):
        """Returns (flat_grad_or_None, stats). If None, gradients are already accumulated."""
        if self.cfg.pcgrad_impl == "feature":
            return self._feature(per_sample_ce, gid, w, K, feats, gen, retain)[:2]
        flat, st, _ = self._loop(per_sample_ce, gid, w, K, gen, retain)
        return flat, st


# --------------------------------------------------------------------------- helpers

class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _flat_grad(loss, params, retain_graph=True) -> torch.Tensor:
    gs = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=False)
    return torch.cat([g.reshape(-1) for g in gs])


def set_flat_grad(params, flat: torch.Tensor):
    off = 0
    for p in params:
        n = p.numel()
        chunk = flat[off:off + n].view_as(p)
        if p.grad is None:
            p.grad = chunk.detach().clone()
        else:
            p.grad.copy_(chunk)
        off += n
    assert off == flat.numel(), (off, flat.numel())


def add_flat_grad(params, flat: torch.Tensor):
    off = 0
    for p in params:
        n = p.numel()
        chunk = flat[off:off + n].view_as(p)
        if p.grad is None:
            p.grad = chunk.detach().clone()
        else:
            p.grad.add_(chunk)
        off += n
