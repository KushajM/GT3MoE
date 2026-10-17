"""Training loop, epoch boundary, checkpointing and the budget guard.

Order of operations at the epoch boundary matters and the original got it wrong: it called
``folk.update(ep)`` (which reassigns) *before* ``eval_model``, and it appended to
``history['reassign']`` only after warmup while appending to the other histories every epoch, so
every churn/T* plot is shifted by exactly ``warmup_epochs``. Here every per-epoch record is a
single dict in one list, indexed by the true epoch, so misalignment is impossible.

Spot-safety: a checkpoint carries model, optimizer, scheduler, scaler, cost-matrix EMA state, the
assignment, the RNG states, the epoch index and the *cumulative billed seconds* across restarts.
Results are written to ``ckpt_dir`` (S3-synced on SageMaker) as well as ``out_dir``, because
``/opt/ml/model`` is only uploaded on clean exit and is lost on every spot interruption.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from . import assign as A
from . import evaluate as E
from . import losses as L
from .costs import CostTracker
from .data import (GPUImages, build_cifar100_lt, build_generic_lt,
                   group_stratified_order, probe_indices)
from .model import backbone_scope_params, build_model, count_params
from .pcgrad import (BackboneUpdater, add_flat_grad, build_groups, dropped_by_bucket,
                     set_flat_grad, splithalf_conflict)
from .precision import Precision, assert_finite_grads


# --------------------------------------------------------------------------- utilities

def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or n.endswith(".bias")) else decay).append((n, p))

    def split(names_params, pred):
        return [p for n, p in names_params if pred(n)]

    is_bb = lambda n: n.startswith("backbone")
    is_rt = lambda n: n.startswith("router")
    is_ex = lambda n: not (is_bb(n) or is_rt(n))
    groups = []
    for pred, lr, tag in ((is_bb, cfg.lr_backbone, "backbone"),
                          (is_rt, cfg.lr_router, "router"),
                          (is_ex, cfg.lr_experts, "experts")):
        d = split(decay, pred)
        nd = split(no_decay, pred)
        if d:
            groups.append(dict(params=d, lr=lr, weight_decay=cfg.weight_decay, name=tag))
        if nd:
            groups.append(dict(params=nd, lr=lr, weight_decay=0.0, name=tag + "_nodecay"))
    cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    kw = {} if cfg.optimizer == "adamw" else dict(weight_decay=0.0)
    opt = cls(groups, **kw)
    # cfg.lr_warmup_epochs and cfg.lr_schedule were parsed, hashed into config_hash() and then
    # never read -- the scheduler was unconditionally a bare cosine. That is survivable when the
    # trunk is pretrained and lr_backbone is 5e-5, and it is not survivable from scratch: a random
    # ResNet-50 at head-scale LR with no warmup is the classic bf16 divergence case.
    #
    # The warmup==0 branch is kept byte-identical to the previous behaviour on purpose, so every
    # already-published fine-tuned run still reproduces exactly.
    wu = max(0, int(getattr(cfg, "lr_warmup_epochs", 0)))
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cfg.epochs - wu), eta_min=1e-6)
    if wu == 0:
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=wu)
    sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[wu])
    return opt, sch


@torch.no_grad()
def class_mean_features(model, images, cfg, prec, batch_size=512) -> np.ndarray:
    """L2-normalised class-mean features from the (pretrained) backbone, for the k-means init."""
    C, D = cfg.num_classes, cfg.feat_dim
    dev = images.x.device
    s = torch.zeros(C, D, device=dev, dtype=torch.float64)
    n = torch.zeros(C, device=dev, dtype=torch.float64)
    was = model.training
    model.eval()
    for xb, yb, _ in images.eval_batches(batch_size):
        with prec.eval_autocast():
            f = model.pooled_features(xb)
        s.index_add_(0, yb, f.double())
        n.index_add_(0, yb, torch.ones_like(yb, dtype=torch.float64))
    if was:
        model.train()
    mu = (s / n.clamp(min=1).unsqueeze(1)).cpu().numpy()
    nrm = np.linalg.norm(mu, axis=1, keepdims=True)
    return mu / np.clip(nrm, 1e-8, None)


# --------------------------------------------------------------------------- the trainer

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
        cfg.device = self.dev
        seed_everything(cfg.seed, cfg.deterministic)
        self.prec = Precision(cfg.amp, cfg.tf32, cfg.expert_dtype, cfg.eval_amp, self.dev)

        if cfg.dataset == "cifar100lt":
            d = build_cifar100_lt(cfg.data_root, cfg.num_classes, cfg.imbalance_ratio,
                                  cfg.val_frac, cfg.img_size, cfg.data_seed, smoke=cfg.smoke)
        else:
            d = build_generic_lt(cfg.data_root, cfg.dataset, cfg.num_classes,
                                 cfg.imbalance_ratio, cfg.val_frac, cfg.img_size,
                                 cfg.data_seed, smoke=cfg.smoke,
                                 cap_per_class=cfg.train_cap_per_class)
        cfg.class_counts = d["class_counts"].tolist()
        cfg.many, cfg.medium, cfg.few = d["many"].tolist(), d["medium"].tolist(), d["few"].tolist()
        self.class_counts = d["class_counts"]
        cl = cfg.channels_last
        aug = dict(rrc_min_scale=cfg.rrc_min_scale, jitter_strength=cfg.jitter_strength)
        self.tr = GPUImages(d["xtr"], d["ytr"], self.dev, channels_last=cl, **aug)
        self.va = GPUImages(d["xva"], d["yva"], self.dev, channels_last=cl, **aug)
        self.te = GPUImages(d["xte"], d["yte"], self.dev, channels_last=cl, **aug)

        self.gen = torch.Generator(device=self.dev).manual_seed(cfg.seed)
        self.cpu_gen = torch.Generator().manual_seed(cfg.seed)
        if cfg.augment == "frozen_once":
            self.tr.freeze_augmentation(self.gen)

        self.model = build_model(cfg).to(self.dev)
        self.opt, self.sch = make_optimizer(self.model, cfg)
        if (self.prec.needs_scaler() and cfg.pcgrad and not cfg.dense_head
                and not cfg.legacy_scalerless_postwarmup):
            raise ValueError(
                "amp=fp16 with pcgrad requires a choice: the manual autograd.grad path produces "
                "UNSCALED gradients, so mixing it with a GradScaler is unsound. Use --amp bf16 "
                "(recommended on Ampere) or set --legacy_scalerless_postwarmup to reproduce the "
                "original's behaviour deliberately.")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.prec.needs_scaler())
        self.costs = CostTracker(cfg, self.dev)
        from .lt_losses import make_loss
        self.base_loss = make_loss(cfg.base_loss, self.class_counts, self.dev)
        self._drw_done = False

        K, C = self.model.K, self.model.C
        cap_lo, cap_hi = cfg.resolved_cap() if not cfg.dense_head else (0, C)
        self.cap_lo, self.cap_hi = cap_lo, cap_hi
        rng = np.random.default_rng(cfg.seed)
        self.np_rng = rng
        if cfg.dense_head:
            pi0 = np.zeros(C, dtype=np.int64)
        elif cfg.assign_init == "kmeans":
            feats = class_mean_features(self.model, self.tr, cfg, self.prec)
            pi0 = A.init_assignment("kmeans", None, K, C, rng, class_features=feats)
            pi0 = A.project_to_capacity(pi0, K, cap_lo, cap_hi)
        else:
            pi0 = A.init_assignment(cfg.assign_init, None, K, C, rng)
            pi0 = A.project_to_capacity(pi0, K, cap_lo, cap_hi)
        if cfg.assign_mode == "random_balanced":
            pi0 = A.project_to_capacity(rng.permutation(C) % K, K, cap_lo, cap_hi)
        elif cfg.assign_mode == "round_robin":
            pi0 = (np.arange(C) % K).astype(np.int64)
        self.m = max(1, int(getattr(cfg, "class_multiplicity", 1)))
        if self.m > 1:
            # Round-robin seed: owners(c) = {(c*m + j) mod K}. Balanced in class count by
            # construction and distinct by construction whenever m <= K, so it is feasible before
            # the first cost matrix exists. The epoch-boundary solve replaces it immediately.
            pi0 = np.stack([((np.arange(C) * self.m + j) % K) for j in range(self.m)], 1)
            pi0 = np.sort(pi0, axis=1).astype(np.int64)
        self.assignment = pi0
        self.assign_gpu = torch.as_tensor(pi0, device=self.dev, dtype=torch.long)
        #: [C] primary owner, kept 1-D so every existing diagnostic and checkpoint field is
        #: unchanged at m=1 and still well-defined at m>1.
        self.primary = pi0[:, 0] if pi0.ndim == 2 else pi0
        self.primary_gpu = torch.as_tensor(self.primary, device=self.dev, dtype=torch.long)

        scope = backbone_scope_params(self.model.backbone, cfg.backbone, cfg.pcgrad_scope)
        self.updater = BackboneUpdater(cfg, self.model, scope)
        self.bb_params = list(self.model.backbone.parameters())
        self.rt_params = [p for n, p in self.model.named_parameters() if n.startswith("router")]
        self.ex_params = [p for n, p in self.model.named_parameters()
                          if not n.startswith("backbone") and not n.startswith("router")]

        self.bucket_masks = {}
        for name in ("many", "medium", "few"):
            m = torch.zeros(C, dtype=torch.bool, device=self.dev)
            idx = getattr(cfg, name) or []
            if idx:
                m[torch.as_tensor(idx, device=self.dev)] = True
            self.bucket_masks[name] = m
        cw = 1.0 / np.maximum(1, self.class_counts)
        self.class_weight = torch.as_tensor(cw / cw.mean(), device=self.dev, dtype=torch.float32)

        self.epoch = 0
        self.history: List[Dict] = []
        self.cost_trace: List[np.ndarray] = []
        self.billed_seconds = 0.0
        self._t0 = time.time()
        self.backward_count = 0
        self.params = count_params(self.model)

        self.ckpt_dir = cfg.ckpt_dir or os.path.join(cfg.out_dir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.ckpt_dir, f"{cfg.run_name}_s{cfg.seed}.pt")
        if cfg.resume:
            self.maybe_resume()

    # ------------------------------------------------------------------ one step
    def train_step(self, xb, yb) -> Dict[str, float]:
        cfg = self.cfg
        self.opt.zero_grad(set_to_none=True)
        o = self.model(xb, training=True, prec=self.prec, gen=self.gen, labels=yb)
        out = o["out"].float()
        if cfg.expert_loss == "per_expert" and o.get("all_logits") is not None:
            # Each expert carries the full objective on every sample, at equal weight. This is the
            # difference between an ensemble of complete classifiers and a gated mixture whose
            # members only ever learn a residual -- the distinction that makes m=8 not RIDE.
            al = o["all_logits"].float()                                   # [B,K,C]
            per_sample_ce = torch.stack(
                [self.base_loss(al[:, k, :], yb, reduction="none") for k in range(al.shape[1])],
                dim=1).mean(1)
        elif cfg.expert_loss == "per_expert_routed" and o.get("all_logits") is not None:
            al = o["all_logits"].float()                                   # [B,K,C]
            ar = torch.arange(al.shape[0], device=al.device)
            gates = o["top_gates"].float()                                 # [B,k], renormalised
            per_sample_ce = sum(
                gates[:, j] * self.base_loss(al[ar, o["top_idx"][:, j], :], yb, reduction="none")
                for j in range(gates.shape[1]))
        else:
            per_sample_ce = self.base_loss(out, yb, reduction="none")
        ce = per_sample_ce.mean()

        w = cfg.ramp(self.epoch)
        la = out.new_zeros(())
        lc = out.new_zeros(())
        lb = out.new_zeros(())
        if o.get("router_aux") is not None:
            lb = o["router_aux"]
        if not cfg.dense_head and cfg.router == "gt3moe":
            rl = o["router_logits"] if cfg.align_on_clean_logits else o["router_logits_sel"]
            if w > 0:
                la = L.align_loss(rl, yb, self.assign_gpu, cfg, self.class_weight)
                lc = L.comp_loss(o["all_logits"], o["top_idx"], yb, o["top_gates"], cfg)
            # balance loss: full strength during warmup, then ramped down as alignment ramps up
            if cfg.balance_after_warmup == "keep" or self.epoch < cfg.warmup_epochs:
                bw = 1.0
            elif cfg.balance_after_warmup == "ramp_down":
                bw = max(0.0, 1.0 - w)
            else:
                bw = 0.0
            if bw > 0:
                lb = bw * L.switch_balance_loss(o["gate_probs"], o["top_idx"], self.model.K)
        aux = w * cfg.align_alpha * la + w * cfg.comp_beta * lc + cfg.balance_alpha * lb

        use_pcgrad = (cfg.pcgrad and not cfg.dense_head and cfg.router == "gt3moe"
                      and self.epoch >= cfg.warmup_epochs
                      and (self.step_idx % max(1, cfg.pcgrad_every) == 0))
        stats: Dict[str, float] = {}

        # A GradScaler cannot be combined with the manual autograd.grad path: the manual gradients
        # are unscaled, so calling scaler.unscale_ on them would divide real gradients by the
        # scale factor. That is exactly the inconsistency the original shipped (warmup scaled,
        # post-warmup not). Construction refuses the combination unless the legacy flag asks for it.
        scaled = self.prec.needs_scaler() and not use_pcgrad

        if not use_pcgrad:
            total = ce + aux
            if scaled:
                self.scaler.scale(total).backward()
            else:
                total.backward()
            self.backward_count += 1
        else:
            gid, wts, gstats = build_groups(yb, self.primary_gpu, self.model.K, cfg, self.cpu_gen)
            stats.update(gstats)
            stats.update(dropped_by_bucket(yb, gid, self.bucket_masks))
            # 1. grouped (optionally projected) backbone gradient from the per-sample CE
            flat, pst = self.updater.compute(per_sample_ce, gid, wts, self.model.K,
                                             o["feats"], self.cpu_gen, retain=True)
            stats.update(pst)
            if cfg.splithalf_every and self.step_idx % cfg.splithalf_every == 0:
                stats.update(splithalf_conflict(per_sample_ce, gid, self.model.K,
                                                self.updater.scope, self.cpu_gen))
            self.backward_count += int(pst.get("n_groups", 1.0)) + 1
            if flat is not None:
                set_flat_grad(self.bb_params, flat)
            # 2. router and expert heads see the plain full-batch CE plus the aux losses, as in
            #    the original (the backbone and the heads therefore optimise differently-weighted
            #    objectives -- undocumented in the paper, so it is logged here)
            heads = self.rt_params + self.ex_params
            gs = torch.autograd.grad(ce + aux, heads, retain_graph=True, allow_unused=True)
            for p, g in zip(heads, gs):
                if g is not None:
                    p.grad = g.detach().clone() if p.grad is None else p.grad.add_(g)
            # 3. the aux losses also reach the backbone (paper Eq (10))
            if aux.requires_grad:
                gb = torch.autograd.grad(aux, self.bb_params, retain_graph=False,
                                         allow_unused=True)
                add_flat_grad_safe(self.bb_params, gb)

        # ---- clipping and the step
        if scaled:
            self.scaler.unscale_(self.opt)
        if assert_finite_grads(self.model.parameters(), "step"):
            stats["skipped_step"] = 1.0
            self.opt.zero_grad(set_to_none=True)
            if scaled:
                self.scaler.update()
            return dict(ce=float(ce.detach()), align=float(la.detach()),
                        comp=float(lc.detach()), balance=float(lb.detach()), **stats)
        if cfg.grad_clip > 0:
            if cfg.grad_clip_mode == "global":
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            else:
                for grp in (self.bb_params, self.rt_params, self.ex_params):
                    if grp:
                        torch.nn.utils.clip_grad_norm_(grp, cfg.grad_clip)
        if scaled:
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            self.opt.step()

        with torch.no_grad():
            if (not cfg.dense_head and cfg.router == "gt3moe" and o["all_logits"] is not None
                    and o["all_logits"].shape[1] == self.costs.K):
                self.costs.accumulate(o["all_logits"].detach().float(),
                                      o["top_idx"].detach(), yb)
        return dict(ce=float(ce.detach()), align=float(la.detach()),
                    comp=float(lc.detach()), balance=float(lb.detach()), **stats)

    # ------------------------------------------------------------------ one epoch
    def train_epoch(self) -> Dict:
        cfg = self.cfg
        self.model.train()
        agg: Dict[str, float] = {}
        # Per-key counts: some stats (the split-half probe) are logged only every Nth step, so
        # dividing every key by the total step count understates them by exactly that factor.
        agg_n: Dict[str, int] = {}
        nb = 0
        order = None
        if cfg.sampler == "group_stratified" and not cfg.dense_head:
            order = group_stratified_order(self.tr.y, self.primary_gpu, cfg.batch_size,
                                           self.model.K, self.gen)
        self.step_idx = 0
        t0 = time.time()
        for xb, yb, _ in self.tr.train_batches(cfg.batch_size, self.gen, cfg.augment,
                                               cfg.drop_last, order=order):
            s = self.train_step(xb, yb)
            for k, v in s.items():
                agg[k] = agg.get(k, 0.0) + float(v)
                agg_n[k] = agg_n.get(k, 0) + 1
            nb += 1
            self.step_idx += 1
        self.sch.step()
        rec = {f"train/{k}": v / max(1, agg_n.get(k, nb)) for k, v in agg.items()}
        rec["train/seconds"] = time.time() - t0
        rec["train/steps"] = nb
        return rec

    # ------------------------------------------------------------------ epoch boundary
    def slow_step(self) -> Dict:
        cfg = self.cfg
        rec: Dict = {}
        if cfg.dense_head or cfg.router != "gt3moe":
            return rec
        if cfg.cost_source == "probe":
            src = self.va if cfg.probe_split == "val" else self.tr
            gen = torch.Generator(device=self.dev).manual_seed(cfg.data_seed)
            idx = probe_indices(src.y, cfg.num_classes, cfg.probe_per_class, gen)
            self.costs.probe(self.model, src, idx, self.prec, cfg.eval_batch_size)
        got = self.costs.finalize(self.class_counts)
        rec.update({f"cost/{k}": v for k, v in got["diag"].items()})
        Lm = got["L"]
        if cfg.routability_mu > 0 and got.get("route_nll") is not None:
            rn = got["route_nll"]
            # Centre per class before mixing. Only differences WITHIN a column steer the
            # assignment, and an uncentred NLL would otherwise let a class whose routing is
            # uniformly uncertain inflate its whole column and drag the capacity allocation
            # around for no reason.
            rn = rn - rn.mean(axis=0, keepdims=True)
            Lm = Lm + cfg.routability_mu * rn
            rec["cost/routability_spread"] = float(rn.max(0).mean() - rn.min(0).mean())
            rec["cost/routability_agree"] = float(
                (rn.argmin(0) == self.primary).mean())
        if cfg.save_cost_trace:
            self.cost_trace.append(Lm.astype(np.float32))

        # From scratch there are no pretrained features to cluster at initialisation, so the
        # k-means partition is built from the model's own features once warm-up ends and then held
        # fixed. Alignment is off during warm-up (w = 0), so the placeholder partition used until
        # then never reaches the loss.
        if cfg.assign_mode == "kmeans_at_warmup" and self.epoch + 1 == cfg.warmup_epochs:
            feats = class_mean_features(self.model, self.tr, cfg, self.prec)
            K, C = self.model.K, self.model.C
            pi = A.init_assignment("kmeans", None, K, C, self.np_rng, class_features=feats)
            pi = A.project_to_capacity(pi, K, self.cap_lo, self.cap_hi)
            rec["assign/kmeans_at_warmup_churn"] = int((pi != self.primary).sum())
            self.assignment = pi
            self.assign_gpu = torch.as_tensor(pi, device=self.dev, dtype=torch.long)
            self.primary = pi
            self.primary_gpu = self.assign_gpu
        frozen = (cfg.assign_mode in ("kmeans_only", "random_balanced", "round_robin",
                                      "kmeans_at_warmup")
                  or (cfg.assign_mode == "frozen_at" and cfg.assign_freeze_epoch >= 0
                      and self.epoch >= cfg.assign_freeze_epoch))
        due = ((self.epoch + 1) % max(1, cfg.assign_every) == 0)
        if (self.epoch >= cfg.warmup_epochs) and not frozen and due:
            pi, adiag = A.solve(cfg, Lm, self.assignment, self.epoch, self.class_counts,
                                self.np_rng)
            self.assignment = pi
            self.assign_gpu = torch.as_tensor(pi, device=self.dev, dtype=torch.long)
            self.primary = pi[:, 0] if pi.ndim == 2 else pi
            self.primary_gpu = torch.as_tensor(self.primary, device=self.dev, dtype=torch.long)
        else:
            adiag = A.normalised_regret(Lm, self.primary, self.cap_lo, self.cap_hi)
            adiag["churn"] = 0
            adiag["gamma_star"] = A.gamma_star(Lm, self.primary, self.cap_lo, self.cap_hi)
        rec.update({f"assign/{k}": v for k, v in adiag.items()})
        self.costs.zero_()
        return rec

    # ------------------------------------------------------------------ the run
    def fit(self) -> Dict:
        cfg = self.cfg
        aff_epochs = cfg.affinity_epoch_set()
        while self.epoch < cfg.epochs:
            # ``>=``, not ``==``: a spot interruption that resumes past the milestone used to skip
            # the freeze entirely and silently produce an unfrozen StableMoE. The body is
            # idempotent, so re-running it each epoch costs nothing.
            if (cfg.router == "stablemoe" and self.epoch >= cfg.stable_phase1_epochs
                    and next(self.model.router.parameters()).requires_grad):
                for p_ in self.model.router.parameters():      # freeze routing, keep training
                    p_.requires_grad = False
                print(f"  [stablemoe] router frozen at epoch {self.epoch}", flush=True)
            if (cfg.base_loss in ("ldam", "ldam_drw") and not self._drw_done
                    and self.epoch >= cfg.drw_start_epoch):
                self.base_loss.set_drw_weights(self.class_counts, device=self.dev)
                self._drw_done = True
                print(f"  [ldam] DRW weights on at epoch {self.epoch}", flush=True)
            rec = {"epoch": self.epoch}
            rec.update(self.train_epoch())
            rec.update(self.slow_step())
            do_eval = ((self.epoch + 1) % max(1, cfg.eval_every) == 0
                       or self.epoch == cfg.epochs - 1)
            if do_eval:
                m = E.evaluate(self.model, self.va, cfg, self.prec,
                               None if cfg.dense_head else self.primary,
                               cfg.eval_batch_size, full=False)
                rec.update({f"val/{k}": v for k, v in m.items()
                            if not isinstance(v, (list, dict))})
            self.history.append(rec)
            self.epoch += 1
            self._log(rec)
            self.save_checkpoint()
            if self._budget_exceeded():
                print(f"[budget] stopping cleanly after epoch {self.epoch}", flush=True)
                return self.finish(partial=True)
        return self.finish(partial=False)

    def finish(self, partial: bool) -> Dict:
        cfg = self.cfg
        res = dict(config=cfg.to_dict(), config_hash=cfg.config_hash(),
                   precision=self.prec.env(), params=self.params,
                   history=self.history, partial=partial,
                   billed_seconds=self.billed_seconds + (time.time() - self._t0),
                   backward_count=self.backward_count,
                   assignment=np.asarray(self.primary).tolist(),
                   owners=np.asarray(self.assignment).tolist(),
                   class_multiplicity=self.m,
                   class_counts=np.asarray(self.class_counts).tolist())
        if cfg.eval_at_end:
            res["test"] = E.evaluate(self.model, self.te, cfg, self.prec,
                                     None if cfg.dense_head else self.primary,
                                     cfg.eval_batch_size, full=cfg.diagnostics)
            res["val_final"] = E.evaluate(self.model, self.va, cfg, self.prec,
                                          None if cfg.dense_head else self.primary,
                                          cfg.eval_batch_size, full=False)
        if cfg.conflict_probe_steps > 0:
            from .conflict_probe import probe as _cprobe
            res["conflict_probe"] = {}
            for bn_eval in (True, False):
                try:
                    res["conflict_probe"]["bn_eval" if bn_eval else "bn_train"] = _cprobe(
                        self, cfg.conflict_probe_steps, cfg.conflict_probe_window, bn_eval)
                except Exception as e:  # never let a diagnostic kill a completed run
                    res["conflict_probe"]["error_%s" % bn_eval] = f"{type(e).__name__}: {e}"
        if cfg.measure_flops and self.dev == "cuda":
            res["flops"] = E.measure_flops(self.model, cfg, self.prec, self.dev)
        if torch.cuda.is_available():
            res["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
        self._write_results(res)
        return res

    # ------------------------------------------------------------------ io
    def _log(self, rec):
        ks = ["epoch", "train/ce", "train/align", "train/comp", "val/acc", "val/few",
              "assign/churn", "assign/optimality_gap", "cost/imputed_frac", "train/seconds"]
        parts = [f"{k.split('/')[-1]}={rec[k]:.4g}" if isinstance(rec.get(k), float)
                 else f"{k.split('/')[-1]}={rec.get(k)}" for k in ks if k in rec]
        print("  " + " ".join(parts), flush=True)

    def _write_results(self, res):
        cfg = self.cfg
        name = f"{cfg.run_name}_{cfg.arm}_K{cfg.num_experts}_s{cfg.seed}_{cfg.config_hash()}.json"
        for d in {cfg.out_dir, self.ckpt_dir}:      # ckpt_dir is S3-synced; out_dir may be lost
            with open(os.path.join(d, name), "w") as f:
                json.dump(res, f, default=_jsonable)
        if cfg.save_cost_trace and self.cost_trace:
            np.savez_compressed(
                os.path.join(self.ckpt_dir, name.replace(".json", "_costtrace.npz")),
                L=np.stack(self.cost_trace))

    def save_checkpoint(self):
        cfg = self.cfg
        sd = dict(
            epoch=self.epoch, model=self.model.state_dict(), opt=self.opt.state_dict(),
            sch=self.sch.state_dict(), scaler=self.scaler.state_dict(),
            costs=self.costs.state_dict(), assignment=self.assignment,
            history=self.history, backward_count=self.backward_count,
            billed_seconds=self.billed_seconds + (time.time() - self._t0),
            cost_trace=np.stack(self.cost_trace) if self.cost_trace else None,
            rng=dict(torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                     numpy=np.random.get_state(), python=random.getstate(),
                     gen=self.gen.get_state(), cpu_gen=self.cpu_gen.get_state(),
                     np_rng=self.np_rng.bit_generator.state),
        )
        tmp = self.ckpt_path + ".tmp"
        torch.save(sd, tmp)
        os.replace(tmp, self.ckpt_path)             # atomic; a killed job never leaves a torn file

    def maybe_resume(self):
        if not os.path.exists(self.ckpt_path):
            return
        sd = torch.load(self.ckpt_path, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(sd["model"])
        self.opt.load_state_dict(sd["opt"])
        self.sch.load_state_dict(sd["sch"])
        self.scaler.load_state_dict(sd["scaler"])
        self.costs.load_state_dict(sd["costs"])
        self.assignment = np.asarray(sd["assignment"])
        self.assign_gpu = torch.as_tensor(self.assignment, device=self.dev, dtype=torch.long)
        self.primary = self.assignment[:, 0] if self.assignment.ndim == 2 else self.assignment
        self.primary_gpu = torch.as_tensor(self.primary, device=self.dev, dtype=torch.long)
        self.history = sd["history"]
        self.epoch = int(sd["epoch"])
        self.backward_count = int(sd.get("backward_count", 0))
        self.billed_seconds = float(sd.get("billed_seconds", 0.0))
        ct = sd.get("cost_trace")
        self.cost_trace = list(ct) if ct is not None else []
        r = sd["rng"]
        torch.set_rng_state(r["torch"].cpu() if hasattr(r["torch"], "cpu") else r["torch"])
        if r["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in r["cuda"]])
        np.random.set_state(r["numpy"])
        random.setstate(r["python"])
        self.gen.set_state(r["gen"])
        self.cpu_gen.set_state(r["cpu_gen"])
        self.np_rng.bit_generator.state = r["np_rng"]
        self._t0 = time.time()
        print(f"[resume] epoch {self.epoch}, billed {self.billed_seconds:.0f}s", flush=True)

    def _budget_exceeded(self) -> bool:
        b = self.cfg.budget_seconds
        if b <= 0:
            return False
        spent = self.billed_seconds + (time.time() - self._t0)
        per_epoch = spent / max(1, self.epoch)
        return spent + 1.3 * per_epoch > b          # leave room for one more epoch + write-out


def add_flat_grad_safe(params, grads):
    """Accumulate a list of (possibly None) gradients into ``p.grad``."""
    for p, g in zip(params, grads):
        if g is None:
            continue
        p.grad = g.detach().clone() if p.grad is None else p.grad.add_(g)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set,)):
        return sorted(o)
    return str(o)
