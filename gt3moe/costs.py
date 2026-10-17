"""The per-expert per-class competence matrix.

This is the crux of the method, and the original's estimator has three defects:

1. **It is on-policy.** ``accumulate`` computed ``F.cross_entropy`` for all K experts as a [B,K]
   tensor and then indexed only the two routed columns, discarding K-2. So the unbiased matrix was
   already being computed, for free, and thrown away. Because the router is trained to select
   low-loss experts, ``E[Z | routed] = m + Cov(Z, 1_routed)/p`` with ``Cov < 0``: every measured
   entry is optimistic, and the optimism grows as the selection probability falls.

2. **The imputation rule makes the incumbent unfalsifiable.** With unseen cells filled by
   ``mean(S_c) + mu`` and ``mean(S_c) >= min(S_c)`` for any set, the fill is *strictly worse than
   the column minimum* for every class, every epoch, every ``|S_c|`` and any ``mu > 0``. So
   ``argmin`` always lies in the already-seen set and an unseen expert can never win. The paper's
   stated justification for mu -- "small enough that an unseen expert can still win if every
   measured expert performs poorly on c" -- is arithmetically impossible under its own Eq (3),
   because the fill is defined *relative* to the seen set.

3. **It is a running average over a changing model**, on augmented training data, with dropout
   active, accumulated across the whole epoch.

``cost_source="probe"`` fixes all three: one end-of-epoch pass in eval mode over a per-class-capped
subset, evaluating all K experts, so ``n_{i,c} = n_c > 0`` for every cell and Eq (3) is dead code.
``cost_source="train_topk"`` reproduces the original exactly.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


class CostTracker:
    def __init__(self, cfg, device: str = "cuda"):
        self.cfg = cfg
        K, C = int(cfg.num_experts), int(cfg.num_classes)
        self.K, self.C = K, C
        self.device = device
        # EMA state, plus the epoch counter the bias correction needs
        self.ema = torch.full((K, C), float(cfg.ema_init), device=device, dtype=torch.float64)
        self.t = 0
        # per-epoch accumulators (GPU, no host syncs in the hot loop)
        self.zero_()
        self.last: Dict[str, float] = {}
        self.prev_L: Optional[torch.Tensor] = None

    def zero_(self):
        K, C, d = self.K, self.C, self.device
        z = lambda: torch.zeros((K, C), device=d, dtype=torch.float64)
        self.loss_sum, self.loss_cnt = z(), z()
        self.acc_sum, self.acc_cnt = z(), z()
        self.route_cnt = z()
        # router log-likelihood per (expert, class); see `probe`
        self.lp_sum, self.lp_cnt = z(), z()
        self.clamp_hits = torch.zeros((), device=d, dtype=torch.float64)
        self.clamp_tot = torch.zeros((), device=d, dtype=torch.float64)

    # ------------------------------------------------------------------ train-time accumulation
    @torch.no_grad()
    def accumulate(self, all_logits: torch.Tensor, top_idx: torch.Tensor,
                   labels: torch.Tensor):
        """all_logits [B,K,C] fp32, top_idx [B,k], labels [B].

        ``route_cnt`` always counts only genuinely routed traffic, so utilisation diagnostics stay
        honest regardless of ``cost_source``.
        """
        cfg = self.cfg
        B, K, C = all_logits.shape
        ce = F.cross_entropy(all_logits.reshape(-1, C).float(),
                             labels.unsqueeze(1).expand(B, K).reshape(-1),
                             reduction="none").reshape(B, K)
        self.clamp_tot += ce.numel()
        self.clamp_hits += (ce > cfg.cost_clamp).sum().double()
        ce = ce.clamp(max=cfg.cost_clamp)
        correct = (all_logits.argmax(-1) == labels.unsqueeze(1)).double()

        flat_cls = labels.unsqueeze(1).expand(B, K).reshape(-1)
        flat_exp = torch.arange(K, device=all_logits.device).unsqueeze(0).expand(B, K).reshape(-1)
        flat_lin = flat_exp * C + flat_cls

        # routed traffic (always)
        rc = top_idx.reshape(-1) * C + labels.unsqueeze(1).expand_as(top_idx).reshape(-1)
        self.route_cnt.view(-1).index_add_(0, rc, torch.ones_like(rc, dtype=torch.float64))

        if cfg.cost_source == "train_dense":
            self.loss_sum.view(-1).index_add_(0, flat_lin, ce.reshape(-1).double())
            self.loss_cnt.view(-1).index_add_(0, flat_lin,
                                              torch.ones_like(flat_lin, dtype=torch.float64))
            self.acc_sum.view(-1).index_add_(0, flat_lin, correct.reshape(-1))
            self.acc_cnt.view(-1).index_add_(0, flat_lin,
                                             torch.ones_like(flat_lin, dtype=torch.float64))
        elif cfg.cost_source == "train_topk":
            sel_ce = ce.gather(1, top_idx).reshape(-1).double()
            sel_ok = correct.gather(1, top_idx).reshape(-1)
            ones = torch.ones_like(rc, dtype=torch.float64)
            self.loss_sum.view(-1).index_add_(0, rc, sel_ce)
            self.loss_cnt.view(-1).index_add_(0, rc, ones)
            self.acc_sum.view(-1).index_add_(0, rc, sel_ok)
            self.acc_cnt.view(-1).index_add_(0, rc, ones)
        # cost_source == "probe": nothing accumulated here

    # ------------------------------------------------------------------ probe pass
    @torch.no_grad()
    def probe(self, model, images, idx: torch.Tensor, prec, batch_size: int = 512):
        """Dense eval-mode pass: fills loss_sum/cnt and acc_sum/cnt for ALL K x C cells.

        Called at the epoch boundary with the model in ``eval()`` mode, so there is no dropout,
        no augmentation and no within-epoch drift. Cost at C=100, probe_per_class=64 is one
        forward over <=6,400 images, roughly 10-20% of a training epoch's forward cost and well
        under 10% of its total cost.
        """
        cfg = self.cfg
        K, C = self.K, self.C
        was_training = model.training
        model.eval()
        for i in range(0, idx.numel(), batch_size):
            sl = idx[i:i + batch_size]
            xb = images.normalise(images.x[sl])
            yb = images.y[sl]
            with prec.eval_autocast():
                o = model(xb, training=False, prec=prec)
            al = o["all_logits"].float()                          # [B,K,C]
            B = al.shape[0]
            # PLAIN cross-entropy, deliberately, even when cfg.base_loss is an LT loss.
            #
            # The alternative -- measure the cost matrix in whatever loss the network is training
            # on -- is tempting because it would make Phi literally the training objective. It is
            # rejected for a concrete reason: LDAM multiplies its logits by s=30, so its CE lands
            # far above cost_clamp=20 and the matrix would be dominated by the clamp rather than by
            # expert competence, making the assignment sensitive to a loss-scaling hyperparameter
            # with no routing meaning. Balanced Softmax and CB-CE are reweightings of the same
            # underlying competence, so measuring that competence directly is the stable choice.
            #
            # Consequence to state in the paper: Remark A.1b's identity "Phi's first term IS the
            # training objective" holds for base_loss='ce'. Under an LT loss, Phi is the potential
            # of a competence game whose relation to the training objective is monotone but not
            # equality.
            ce = F.cross_entropy(al.reshape(-1, C),
                                 yb.unsqueeze(1).expand(B, K).reshape(-1),
                                 reduction="none").reshape(B, K)
            self.clamp_tot += ce.numel()
            self.clamp_hits += (ce > cfg.cost_clamp).sum().double()
            ce = ce.clamp(max=cfg.cost_clamp)
            ok = (al.argmax(-1) == yb.unsqueeze(1)).double()
            flat_cls = yb.unsqueeze(1).expand(B, K).reshape(-1)
            flat_exp = torch.arange(K, device=al.device).unsqueeze(0).expand(B, K).reshape(-1)
            lin = flat_exp * C + flat_cls
            ones = torch.ones_like(lin, dtype=torch.float64)
            self.loss_sum.view(-1).index_add_(0, lin, ce.reshape(-1).double())
            self.loss_cnt.view(-1).index_add_(0, lin, ones)
            self.acc_sum.view(-1).index_add_(0, lin, ok.reshape(-1))
            self.acc_cnt.view(-1).index_add_(0, lin, ones)
            # Router log-likelihood, free on this pass: log p(expert i | x) averaged over the
            # samples of class c. This is the only place the ROUTER's opinion enters the slow
            # timescale. Without it the two-timescale loop is one-directional -- L picks pi, pi
            # supervises the router, and nothing ever tells the assignment which partitions the
            # router is actually able to express. Measured consequence: P(top-1 == pi(y)) is only
            # ~0.83 on an 8-way problem a ResNet-50 feature should find easy, and the resulting
            # oracle-to-deployed gap is 15.0pp, larger than any learned baseline's.
            lp = torch.log_softmax(o["router_logits"].float(), dim=1)      # [B,K]
            self.lp_sum.view(-1).index_add_(0, lin, lp.reshape(-1).double())
            self.lp_cnt.view(-1).index_add_(0, lin, ones)
        if was_training:
            model.train()

    # ------------------------------------------------------------------ epoch boundary
    @torch.no_grad()
    def finalize(self, class_counts: np.ndarray) -> Dict:
        """Build l^(t), impute if asked, EMA-smooth, and return (L, accuracy matrix, diagnostics).

        ``L`` is returned on the CPU as float64 numpy, because every consumer (the LAP solver, the
        potential, the churn diagnostics) is a CPU/scipy operation.
        """
        cfg = self.cfg
        K, C = self.K, self.C
        cnt = self.loss_cnt
        seen = cnt > 0
        ell = torch.where(seen, self.loss_sum / cnt.clamp(min=1), torch.full_like(cnt, float("nan")))

        n_seen = int(seen.sum().item())
        imputed_frac = 1.0 - n_seen / (K * C)

        if cfg.impute:
            # Eq (3): mean over the seen experts of the column, plus a positive margin.
            col_seen = seen.sum(0)                                # [C]
            col_mean = torch.where(
                col_seen > 0,
                torch.nansum(torch.where(seen, ell, torch.zeros_like(ell)), dim=0)
                / col_seen.clamp(min=1),
                torch.full_like(col_seen, float(cfg.ema_init), dtype=ell.dtype))
            fill = (col_mean + cfg.impute_margin).unsqueeze(0).expand(K, C)
            ell = torch.where(seen, ell, fill)
        else:
            # With cost_source="probe" every cell is measured, so this branch is a no-op guard.
            if n_seen < K * C:
                col_seen = seen.sum(0)
                col_mean = torch.where(
                    col_seen > 0,
                    torch.nansum(torch.where(seen, ell, torch.zeros_like(ell)), dim=0)
                    / col_seen.clamp(min=1),
                    torch.full_like(col_seen, float(cfg.ema_init), dtype=ell.dtype))
                ell = torch.where(seen, ell, col_mean.unsqueeze(0).expand(K, C))

        # optional SE-weighted shrinkage toward the column mean, for tail cells whose estimate
        # rests on a handful of samples (a 4-image class has SE of order the imputation margin)
        if cfg.cost_shrink > 0:
            n_eff = cnt.clamp(min=1)
            w = n_eff / (n_eff + cfg.cost_shrink)
            colm = ell.mean(0, keepdim=True)
            ell = w * ell + (1 - w) * colm

        # EMA with optional bias correction. The original seeded the EMA at a constant 5.0 and
        # never corrected it, which is what produces the apparent "80% drop in a single step" in
        # the paper's Figure 2: the curve is the 5.0 seed washing out, not best-response descent.
        d = float(cfg.ema_delta)
        self.t += 1
        self.ema = d * self.ema + (1 - d) * ell
        if cfg.ema_bias_correct:
            seed_w = d ** self.t
            L = (self.ema - seed_w * float(cfg.ema_init)) / max(1e-12, 1.0 - seed_w)
        else:
            L = self.ema.clone()

        acc = torch.where(self.acc_cnt > 0, self.acc_sum / self.acc_cnt.clamp(min=1),
                          torch.full_like(self.acc_cnt, float("nan")))

        Ln = L.detach().cpu().numpy().astype(np.float64)
        accn = acc.detach().cpu().numpy()
        measured = seen.detach().cpu().numpy()

        # Global L-inf drift, plus the quantity that actually governs assignment stability.
        # An argmin over a column is invariant to any change common to that column, and most
        # epoch-to-epoch drift IS common-mode (the whole matrix falls together as training
        # proceeds). Measured: global drift exceeds gamma*/(2C) by 218-310x while the assignment
        # is nonetheless stable, so the global bound cannot be the explanation. The centred,
        # per-class drift below is the differential quantity to compare against gamma_c.
        if self.prev_L is not None:
            D = Ln - self.prev_L
            drift = float(np.abs(D).max())
            Dc = D - D.mean(axis=0, keepdims=True)          # remove the per-column common mode
            per_class = np.abs(Dc).max(axis=0)              # [C]
            drift_c_p50 = float(np.median(per_class))
            drift_c_max = float(per_class.max())
        else:
            drift = drift_c_p50 = drift_c_max = float("nan")
        self.prev_L = Ln.copy()

        # per-bucket imputed fraction: the tail is where the estimator is weakest and where the
        # paper's headline gains live
        buckets = {}
        cc = np.asarray(class_counts)
        for name, m in (("many", cc > 100), ("medium", (cc >= 20) & (cc <= 100)), ("few", cc < 20)):
            if m.any():
                buckets[f"imputed_frac_{name}"] = float(1.0 - measured[:, m].mean())

        srt = np.sort(Ln, axis=0)
        gamma = srt[1] - srt[0] if K > 1 else np.zeros(C)
        diag = dict(
            imputed_frac=imputed_frac,
            measured_cells=n_seen,
            drift_Linf=drift,
            drift_centred_p50=drift_c_p50, drift_centred_max=drift_c_max,
            clamp_rate=float((self.clamp_hits / self.clamp_tot.clamp(min=1)).item()),
            gamma_mean=float(gamma.mean()), gamma_min=float(gamma.min()),
            gamma_median=float(np.median(gamma)),
            L_mean=float(Ln.mean()), L_std=float(Ln.std()),
            n_exact_ties=int((gamma == 0).sum()),
            **buckets)
        # Router NLL per cell: -log p(expert i | class c). Cells the probe never reached (only
        # possible when cost_source != "probe") fall back to the column mean, which is neutral --
        # it can neither attract nor repel the assignment.
        lp_n = (self.lp_sum / self.lp_cnt.clamp(min=1)).detach().cpu().numpy()
        seen = (self.lp_cnt > 0).detach().cpu().numpy()
        route_nll = -lp_n
        if not seen.all():
            col = np.where(seen.any(0), (route_nll * seen).sum(0) / np.maximum(seen.sum(0), 1),
                           np.log(max(K, 2)))
            route_nll = np.where(seen, route_nll, col[None, :])

        self.last = diag
        return dict(L=Ln, acc=accn, measured=measured, diag=diag,
                    route_nll=route_nll,
                    route_cnt=self.route_cnt.detach().cpu().numpy())

    # ------------------------------------------------------------------ checkpointing
    def state_dict(self):
        return dict(ema=self.ema.cpu(), t=self.t,
                    prev_L=None if self.prev_L is None else torch.from_numpy(self.prev_L))

    def load_state_dict(self, sd):
        self.ema = sd["ema"].to(self.device)
        self.t = int(sd["t"])
        self.prev_L = None if sd["prev_L"] is None else sd["prev_L"].numpy()
