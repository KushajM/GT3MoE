"""GT3MoE v2 configuration.

Two presets matter:
  * ``paper_v1``  -- reproduces the original notebook code path, including its defects.
  * ``v2``        -- the corrected defaults.

Every difference between them is a single flag, so any accuracy delta is attributable.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional


@dataclass
class Config:
    # ---------------------------------------------------------------- run identity
    run_name: str = "gt3moe"
    arm: str = "v2"                      # free-form label written into the results JSON
    out_dir: str = "./outputs"
    ckpt_dir: str = ""                   # "" -> out_dir/ckpt ; SageMaker: /opt/ml/checkpoints
    seed: int = 0                        # TRAIN seed: init, aug, dropout, router noise, pcgrad order
    data_seed: int = 1234                # DATA seed: val split + long-tail subsample. Fix across seeds.

    # ---------------------------------------------------------------- data
    dataset: str = "cifar100lt"
    # Cap train images per class AFTER long-tailing. Only used for the balanced control:
    # full CIFAR-100 has 4.6x the images of CIFAR-100-LT, so a bare rho=1 run confounds
    # "balanced" with "more data"; cap=78 matches N and separates the two.
    train_cap_per_class: int = 0
    data_root: str = "./data"
    num_classes: int = 100
    imbalance_ratio: float = 100.0
    val_frac: float = 0.15
    img_size: int = 96
    batch_size: int = 128                # v1: 64
    eval_batch_size: int = 512
    augment: str = "per_batch"           # per_batch | frozen_once (= the v1 bug) | none | strong
    # "strong" = RandomResizedCrop + flip + colour jitter. Every run in this benchmark reaches
    # train CE ~0.00 under "per_batch", so the whole comparison sits in a memorising regime;
    # this is the arm that tests whether the dense-head advantage survives outside it.
    rrc_min_scale: float = 0.35          # min area fraction for RandomResizedCrop
    jitter_strength: float = 0.4         # brightness/contrast/saturation jitter
    drop_last: bool = True               # v1: False
    sampler: str = "random"              # random | group_stratified
    channels_last: bool = True

    # ---------------------------------------------------------------- model
    backbone: str = "resnet50"           # resnet50 | densenet201 | mobilenetv2
    backbone_weights: str = "IMAGENET1K_V2"   # the paper claims V2; the code used V1
    num_experts: int = 8
    top_k: int = 2
    expert_hidden: int = 512
    dropout_rate: float = 0.3
    router_noise: float = 0.1
    expert_impl: str = "bmm"             # bmm | loop (v1)
    # Experts are replicated TRUNK BLOCKS (layer4) rather than MLP heads on one shared vector, so
    # each expert computes its own representation. At K=8 that is 128M params vs a dense ResNet-50's
    # 23.5M, at identical top-1 forward FLOPs (layer4 is 19.8% of the forward; see
    # model.make_backbone_split). The router then reads pooled layer3 features, because routing must
    # precede the expert.
    trunk_experts: bool = False
    dense_head: bool = False             # True -> single-head control, no MoE at all
    router: str = "gt3moe"               # gt3moe | switch | vmoe | expert_choice | stablemoe
                                         #   | softmoe | hash | ride | adamoe | dynmoe
    capacity_factor: float = 1.25        # vmoe token capacity
    vmoe_priority: bool = True           # True = real BPR (drop lowest gate); False = v1 order
    stable_phase1_epochs: int = 20       # stablemoe: freeze the router after this many epochs
    adamoe_null_experts: int = 4         # adamoe: N null experts joining the gate
    dynmoe_tau_init: float = 0.0         # dynmoe: initial per-expert activation threshold
    dynmoe_sparsity_beta: float = 0.01   # dynmoe: pull mean active-expert count toward top_k
    base_loss: str = "ce"                # ce | balsoftmax | ldam_drw | cb_ce
    # How the classification loss reaches the experts.
    #   "blend"      -- one loss on the gated/blended output. Every expert therefore learns a
    #                   RESIDUAL contribution to a mixture, never a standalone classifier. Measured:
    #                   members score 0.24-0.53 alone on the 100-way test set vs a dense head's
    #                   0.713, at every m and every top_k.
    #   "per_expert" -- RIDE's scheme: each expert gets its OWN full loss on EVERY sample at equal
    #                   weight, and inference is a uniform logit average. This is the only setting
    #                   under which "complete experts" is a true description, and it is what makes
    #                   `router="ride"` a faithful RIDE-core baseline rather than a relabelled
    #                   gated mixture.
    expert_loss: str = "blend"           # blend | per_expert | per_expert_routed
    #   "per_expert_routed" -- each SELECTED expert gets the base loss on its own logits, weighted by
    #                   its renormalised gate, instead of one loss on the gated mixture. With
    #                   base_loss="balsoftmax" the logit adjustment acts on each expert's prediction.
    # Cao et al. defer re-weighting at 160/200 epochs, i.e. the last 20% of the schedule.
    # 48/60 keeps that fraction; the previous 40 gave DRW a third of the run, which is a
    # different method, not a shorter one.
    drw_start_epoch: int = 48            # LDAM-DRW: epoch at which class-balanced weights switch on

    # ---------------------------------------------------------------- precision
    amp: str = "bf16"                    # bf16 | fp16 (v1) | fp32
    tf32: str = "off"                    # v1 set the flags True but ran on a T4 where they no-op
    expert_dtype: str = "fp32"           # fp32 | amp (v1 computed the cost matrix in fp16)
    eval_amp: bool = False
    legacy_scalerless_postwarmup: bool = False   # True reproduces the v1 epoch-5 discontinuity

    # ---------------------------------------------------------------- optimisation
    epochs: int = 60
    optimizer: str = "adamw"             # adamw (what the paper claims) | adam (what it ran)
    weight_decay: float = 0.05
    lr_backbone: float = 5e-5
    lr_router: float = 3e-4
    lr_experts: float = 2e-4
    lr_schedule: str = "cosine"
    lr_warmup_epochs: int = 0
    grad_clip: float = 5.0
    grad_clip_mode: str = "per_group"    # per_group | global (v1)

    # ---------------------------------------------------------------- slow timescale
    assign_mode: str = "dynamic"         # dynamic | frozen_at | kmeans_only | random_balanced
                                         #   | round_robin | column_shuffled | kmeans_at_warmup
    assign_freeze_epoch: int = -1
    assign_solver: str = "exact"         # exact | greedy (v1) | br_weighted
    assign_init: str = "kmeans"          # kmeans | round_robin | random
    cap_hi: int = -1                     # -1 -> ceil(C/K)
    cap_lo: int = -1                     # -1 -> floor(C/K); 0 reproduces v1 (dead experts possible)
    assign_hysteresis: float = 0.0       # incumbent bonus inside the LAP, in nats
    # Overlapping assignment: each class is owned by this many experts. 1 is the original hard
    # partition; K gives every expert every class, i.e. a plain deep ensemble. This is the axis
    # that sets C/K -- the effective width of each expert's classification problem -- and the
    # measurements say that, not the router, is the binding constraint (acc_uniform_ensemble is
    # 0.5255 against a dense head's 0.7112, while routing errors cost only 5.98pp). Set top_k >= m
    # or the router cannot reach all of a class's owners.
    class_multiplicity: int = 1
    assign_every: int = 1
    congestion_lambda: float = 0.0       # >0 -> exact convex count-congestion in the LAP (Path A+)
    congestion_weighted: bool = False    # sample-mass weights; requires br_weighted
    warmup_epochs: int = 5               # tau_w
    ramp_epochs: int = 10                # tau_r
    # Ramp the alignment loss in from epoch 0 instead of after warm-up, so the experts learn the
    # initial partition before the first competence solve. The slow step still waits for warm-up.
    align_from_start: bool = False

    # ---------------------------------------------------------------- cost matrix
    cost_source: str = "probe"           # probe | train_dense | train_topk (v1)
    probe_split: str = "train_eval"      # train_eval | val  (val leaks balanced labels: ablation only)
    probe_per_class: int = 64
    # Routability-aware assignment. The assignment cost becomes
    #     L'[i,c] = L[i,c] + routability_mu * ( -log p_router(i | class c) ),
    # so pi prefers groupings the router can actually express. 0.0 reproduces the current method
    # exactly. See CostTracker.probe for why: the two timescales are otherwise coupled in one
    # direction only (L -> pi -> align loss -> router), and nothing feeds the router's ability
    # back into the assignment. Both terms are in nats, so mu is a pure trade-off weight.
    routability_mu: float = 0.0
    ema_delta: float = 0.7
    ema_bias_correct: bool = True
    ema_init: float = 5.0
    impute: bool = False                 # v1: True (mean-of-seen + margin)
    impute_margin: float = 0.5
    cost_clamp: float = 20.0             # v1: 10.0, which truncates where decisions are hardest
    cost_shrink: float = 0.0             # >0 -> SE-weighted shrink of tail cells toward column mean

    # ---------------------------------------------------------------- fast timescale
    align_loss: str = "nll"              # nll | smoothed | hinge_topk | none
    align_alpha: float = 0.7             # the paper's own swept optimum; the code shipped 1.0
    align_smooth: float = 0.05
    align_hinge_margin: float = 0.0
    align_on_clean_logits: bool = True   # v1 used the noise-perturbed softmax
    align_class_balanced: bool = False   # match the potential's uniform-over-classes weighting
    comp_loss: str = "cosine_orig"       # cosine_orig (v1) | js_confusion | centred_cosine_hinge | none
    comp_beta: float = 0.3               # js_confusion wants ~0.005 (gradient-norm matched)
    comp_tau: float = 0.5
    comp_gate_weight: bool = False       # weight by 2*w1*w2 so it fades as top-2 -> top-1
    comp_conf_gate: float = 0.0
    balance_alpha: float = 0.1
    balance_after_warmup: str = "ramp_down"   # ramp_down | off (v1) | keep

    # ---------------------------------------------------------------- PCGrad
    pcgrad: bool = True
    pcgrad_impl: str = "loop"            # loop (exact, G backwards) | captured (2 backwards) | feature
    pcgrad_scope: str = "full"           # full | layer4 | layer4.2
    pcgrad_project: bool = True          # False -> grouping/reweighting only, the E4(b) control
    pcgrad_groups: str = "assignment"    # assignment | random, the E4(d) control
    backbone_ce: str = "group_mean"      # group_mean (v1) | size_weighted | full_batch
    pcgrad_min_group: int = 2            # v1: 2, and it silently drops smaller groups
    pcgrad_singleton: str = "drop"       # drop (v1) | merge_to_residual
    pcgrad_every: int = 1
    pcgrad_max_groups: int = 0           # 0 = all; else a random rescaled subset per step
    pcgrad_group_ema: float = 0.0        # feature impl only
    pcgrad_batched_backward: bool = False
    splithalf_every: int = 0            # >0: run the split-half conflict probe every N steps
    conflict_probe_steps: int = 0        # >0: window-averaged conflict probe after training
    conflict_probe_window: int = 50

    # ---------------------------------------------------------------- evaluation
    eval_every: int = 5                  # v1 evaluated every epoch (~27% of runtime, never used)
    eval_at_end: bool = True
    diagnostics: bool = True
    save_cost_trace: bool = True         # persist L^(t) so solver ablations replay offline for $0
    affinity_epochs: str = "1,5,10,20,30,40,50,60"
    measure_flops: bool = True

    # ---------------------------------------------------------------- infra
    device: str = "cuda"
    num_workers: int = 2
    budget_seconds: float = 0.0          # >0 -> checkpoint and exit cleanly before this
    resume: bool = True
    deterministic: bool = False
    smoke: bool = False                  # tiny run for CI / local CPU test

    # ---------------------------------------------------------------- derived at runtime
    class_counts: Optional[List[int]] = field(default=None, repr=False)
    many: Optional[List[int]] = field(default=None, repr=False)
    medium: Optional[List[int]] = field(default=None, repr=False)
    few: Optional[List[int]] = field(default=None, repr=False)
    feat_dim: int = 0
    git_sha: str = ""
    pkg_versions: dict = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- helpers
    def resolved_cap(self):
        """Return (cap_lo, cap_hi) with defaults resolved and feasibility asserted."""
        C, K = self.num_classes, self.num_experts
        hi = self.cap_hi if self.cap_hi > 0 else -(-C // K)      # ceil(C/K)
        lo = self.cap_lo if self.cap_lo >= 0 else C // K         # floor(C/K)
        lo = min(lo, hi)
        if not (K * lo <= C <= K * hi):
            raise ValueError(f"infeasible capacity: K={K} lo={lo} hi={hi} C={C}")
        return lo, hi

    def ramp(self, epoch: int) -> float:
        """w(t) from Eq (5). Zero during warmup, linear over ramp_epochs, then one."""
        if self.align_from_start:
            return min(1.0, (epoch + 1) / max(1, self.ramp_epochs))
        if epoch < self.warmup_epochs:
            return 0.0
        return min(1.0, (epoch - self.warmup_epochs + 1) / max(1, self.ramp_epochs))

    def affinity_epoch_set(self):
        return {int(t) for t in str(self.affinity_epochs).split(",") if t.strip()}

    def hashable(self) -> dict:
        skip = {"out_dir", "ckpt_dir", "data_root", "git_sha", "pkg_versions",
                "class_counts", "many", "medium", "few", "feat_dim", "run_name",
                "device", "num_workers", "budget_seconds", "resume"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in skip}

    def config_hash(self) -> str:
        blob = json.dumps(self.hashable(), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def stamp_provenance(self) -> "Config":
        try:
            self.git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                text=True, timeout=5).strip()
        except Exception:
            self.git_sha = "unknown"
        vers = {}
        for mod in ("torch", "torchvision", "numpy", "scipy", "sklearn"):
            try:
                vers[mod] = __import__(mod).__version__
            except Exception:
                vers[mod] = "absent"
        self.pkg_versions = vers
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- presets

#: Exactly the original notebook code path, including every defect the audit identified.
PAPER_V1 = dict(
    arm="paper_v1",
    batch_size=64, augment="frozen_once", drop_last=False, sampler="random",
    channels_last=False, backbone_weights="IMAGENET1K_V1", expert_impl="loop",
    amp="fp16", expert_dtype="amp", tf32="off", legacy_scalerless_postwarmup=True,
    optimizer="adam", weight_decay=0.0, grad_clip_mode="global",
    assign_solver="greedy", cap_lo=0, assign_hysteresis=0.0, congestion_lambda=0.0,
    cost_source="train_topk", impute=True, cost_clamp=10.0, ema_bias_correct=False,
    align_loss="nll", align_alpha=1.0, align_on_clean_logits=False,
    comp_loss="cosine_orig", comp_beta=0.3, comp_gate_weight=False,
    balance_after_warmup="off",
    pcgrad_impl="loop", pcgrad_scope="full", backbone_ce="group_mean",
    pcgrad_min_group=2, pcgrad_singleton="drop", eval_every=1,
)

#: The corrected defaults (identical to the dataclass defaults).
V2 = dict(arm="v2")

PRESETS = {"paper_v1": PAPER_V1, "v2": V2}


def make_config(preset: str = "v2", **overrides) -> Config:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    cfg = Config(**PRESETS[preset])
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"unknown config key {k!r}")
        setattr(cfg, k, v)
    return cfg.stamp_provenance()
