"""Launch GT3MoE batches on SageMaker Training via raw boto3.

Each job is a plain training job on the public PyTorch deep-learning container, with a ``data``
channel carrying the code payload and ``run.sh`` as the container entrypoint. A batch of arms runs
sequentially inside one job, each arm with its own checkpoint prefix, so that every arm of a paired
comparison shares one job on one GPU: cross-GPU differences (cuDNN algorithm choice, bf16 tensor-core
behaviour) would otherwise leak into the comparison. Parallelise across batches (datasets, seeds),
never across the arms of one comparison.

Configuration comes from the environment:

    AWS_PROFILE                  credentials profile (optional; the default chain is used otherwise)
    AWS_REGION                   region (default us-west-2)
    GT3MOE_S3_BUCKET             S3 bucket for payloads, datasets and checkpoints
    GT3MOE_SAGEMAKER_ROLE_ARN    IAM role that SageMaker Training assumes

    python sagemaker/launch_boto3.py --batch P_c100lt_K32 --dry_run
    python sagemaker/launch_boto3.py --batch P_c100lt_K32 --instance ml.g5.2xlarge
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms as ARMS  # noqa: E402

REGION = os.environ.get("AWS_REGION", "us-west-2")
PROFILE = os.environ.get("AWS_PROFILE")
BUCKET = os.environ.get("GT3MOE_S3_BUCKET", "")
ROLE = os.environ.get("GT3MOE_SAGEMAKER_ROLE_ARN", "")
# the AWS-published PyTorch training container
IMAGE = (f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
         "pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker")
PREFIX = "gt3moe-v2"

PRICE = {"ml.g5.xlarge": 1.515, "ml.g4dn.xlarge": 0.736, "ml.g5.2xlarge": 1.69,
         "ml.g5.4xlarge": 2.03, "ml.g6.xlarge": 1.10, "ml.g6e.xlarge": 2.24,
         "ml.g5.8xlarge": 3.06, "ml.g5.16xlarge": 5.29,
         # added for the cross-dataset fan-out: six batches need six distinct types, because
         # the training-job quota is 1 PER TYPE.
         "ml.g6.2xlarge": 1.27, "ml.g6.4xlarge": 1.73, "ml.g6e.2xlarge": 2.80,
         "ml.g6.8xlarge": 2.518, "ml.g6.16xlarge": 4.246}

#: Hours are MEASURED, not extrapolated (job gt3moe-v2-calib-20260910-121700 on ml.g5.xlarge):
#:   K=8   4.08 s/epoch pre-warmup, 18.22 s/epoch with PCGrad  -> 0.284 h for 60 epochs
#:   K=32  4.60 s/epoch pre-warmup, 46.24 s/epoch with PCGrad  -> 0.713 h for 60 epochs
#: The PCGrad path costs 4.5x (K=8) / 10.1x (K=32) the plain path, which is the whole cost story.
#: My pre-calibration guesses (1.25 h / 2.90 h) were 4.4x too high.
# Every model in the paper's comparison table, one entry each. Used unchanged by all four
# cross-dataset batches so the corpora are compared arm-for-arm.
DSET_ARMS = ["R_dense", "R_gt3moe", "R_switch", "R_softmoe", "R_vmoe",
             "R_expert_choice", "R_hash", "R_stablemoe", "M_ride_ce"]

#: The extended router table: the nine above plus two recent (2024-25) routers.
TMLR_ARMS = DSET_ARMS + ["R_adamoe", "R_dynmoe"]

#: The augmentation diagnostic needs the dense reference, the method, the nearest baseline, the
#: best sparse router and the only arm that beats dense -- not the full table.
AUG_ARMS = ["R_dense", "R_gt3moe", "R_switch", "R_softmoe", "M_ride_ce"]

#: From-scratch protocol. Needed because Mini/Tiny-ImageNet are BUILT FROM ImageNet-1k train and
#: ImageNet-LT's train split IS a subset of it, so an IMAGENET1K_V2 trunk has already seen the
#: images we evaluate on. Everything here is a deliberate departure from the fine-tuned preset:
#:   * lr_backbone 5e-5 is a fine-tuning LR; a random trunk needs head-scale.
#:   * lr_warmup_epochs was dead code until now; without it a random ResNet-50 at 1e-3 in bf16
#:     diverges in the first epochs.
#:   * assign_init="kmeans" clusters PRETRAINED class-mean features before training starts. On a
#:     random trunk those are noise, so the partition would be arbitrary.
_SCRATCH = dict(backbone_weights="none", lr_backbone=1e-3, lr_warmup_epochs=5,
                assign_init="round_robin")

#: 200-epoch variant for the small corpora. Every epoch-ABSOLUTE schedule field has to move with
#: the horizon or it fires at the wrong fraction of training: at 200 epochs the stock
#: drw_start_epoch=48 would run DRW for 76% of the schedule instead of the last 20%.
SCRATCH200 = dict(_SCRATCH, epochs=200, warmup_epochs=17, ramp_epochs=33,
                  drw_start_epoch=160, stable_phase1_epochs=67, assign_freeze_epoch=20,
                  eval_every=10)

#: 60-epoch variant for ImageNet-LT. probe_per_class drops 64 -> 16 because at C=1000 the
#: competence probe would otherwise touch sum_c min(n_c,64) ~ 47k images every epoch, a ~41%
#: surcharge on a 116k-image corpus.
SCRATCH60 = dict(_SCRATCH, epochs=60, probe_per_class=16)

TIN_NPZ = "tinyimagenet_lt_r100_v0.15_s96_d1234.npz"
MIN_NPZ = "miniimagenet_lt_r100_v0.15_s96_d1234.npz"
INL_NPZ = "imagenet_lt_r1_v0.15_s96_d1234.npz"

BATCHES = {
    # ~10 min. Validates the whole path AND replaces the extrapolated timing model.
    "calib": dict(arms=["E0_calib"], seeds=[0], ks=[8, 32], hours=0.5),
    # The augmentation bug. Must precede everything: if it moves Overall by points, every
    # pre-registered equivalence margin downstream has to be re-derived.
    "aug": dict(arms=["A_frozen_aug", "A_per_batch"], seeds=[0, 1, 2], ks=[8], hours=1.7),
    # E1 decides whether the two-timescale story is true at all.
    "decisive": dict(arms=["E1_frozen", "v2"], seeds=[0, 1, 2, 3, 4], ks=[8], hours=3.2),
    "controls": dict(arms=["E2_kmeans_only", "E3_random_balanced"], seeds=[0, 1, 2], ks=[8],
                     hours=1.7),
    "pcgrad": dict(arms=["E4b_reweight_only", "E4c_size_weighted", "E4d_random_groups",
                         "E4e_no_pcgrad"], seeds=[0, 1, 2], ks=[8], hours=3.5),
    "estimator": dict(arms=["E5_train_topk"], seeds=[0, 1, 2], ks=[8], hours=0.9),
    "dense": dict(arms=["E8_dense_head"], seeds=[0, 1, 2], ks=[8], hours=0.9),
    # Motivated by measurement, not theory: the cosine term drives the top-2 logit cosine to
    # -0.9475 while non-primary experts are 0.43% accurate, so the gated blend injects an
    # actively-wrong anti-correlated vector and costs 6.6pp against routing to the assigned
    # expert alone. js_confusion cannot fight CE on the true class (dJS/dz[y] == 0 exactly).
    # "v2" is included as a WITHIN-JOB anchor. Cross-job comparison is invalid here: the same
    # config and seed in a different job differs by 0.42-0.84pp (mean 0.60pp) from cudnn.benchmark
    # and nondeterministic kernels, which is larger than most effects being measured.
    "losses": dict(arms=["L_js", "L_no_comp", "L_hinge_cos", "v2"], seeds=[0, 1, 2], ks=[8],
                   hours=3.5),
    # The augmentation x PCGrad 2x2 (all four cells within ONE job), plus the loss arms motivated
    # by comp driving the top-2 logit cosine to -0.9475, plus the E4d re-run now that the
    # cuda/cpu generator bug is fixed. v2 and E4e_no_pcgrad double as the within-job anchors.
    # The clean PCGrad decomposition, ALL within one job (cross-job noise is 0.60pp):
    #   v2   = projection + group_mean          E4b  = no projection + group_mean
    #   E4a  = no projection + full_batch CE    E4e  = the whole path off
    # v2-E4b isolates the PROJECTION; E4b-E4a isolates the REWEIGHTING; E4a~E4e is a consistency
    # check, since no-projection + full-batch CE should reproduce the plain gradient.
    "clean": dict(arms=["E4a_full_batch", "E4b_reweight_only", "v2", "E4e_no_pcgrad", "D_probe"],
                  seeds=[0, 1, 2], ks=[8], hours=3.3),
    # The comparative table the paper does not currently have any evidence for.
    "baselines": dict(arms=["R_switch", "R_vmoe", "R_expert_choice", "R_stablemoe", "R_softmoe",
                            "R_hash", "R_gt3moe", "R_dense"],
                      seeds=[0, 1, 2], ks=[8], hours=2.4),
    # LT-loss orthogonality, GT3MoE vs dense on the same loss.
    # Anchors (plain CE) run FIRST so the "does the LT loss help at all?" baseline exists even if
    # the job is cut short, and each LT loss then runs GT3MoE and dense back to back.
    "ltloss": dict(arms=["R_gt3moe", "R_dense", "T_gt3_bal", "T_dense_bal",
                         "T_gt3_ldam", "T_dense_ldam", "T_gt3_cb", "T_dense_cb"],
                   seeds=[0, 1, 2], ks=[8], hours=2.4),
    # Two questions in one job, sharing R_gt3moe as the anchor so both are paired against the same
    # runs (a second job would reintroduce the 0.60pp cross-job floor for no reason).
    #   (a) the congestion term, which is what licenses the "weighted congestion game" framing:
    #       unweighted at three lambdas, plus the mass-weighted w_c = n_c variant;
    #   (b) the baselines AS PUBLISHED -- Switch and StableMoE at their canonical top-1, and
    #       Switch/V-MoE at their canonical aux weight 0.01 rather than the paper's 0.1.
    # R_gt3moe is listed FIRST so that the anchor exists even if the job is cut short.
    #   (c) routability-aware assignment, which is the only arm here aimed at the bottleneck the
    #       baselines job actually identified (GT3MoE has the highest oracle and the largest gap
    #       to it). mu=0 is R_gt3moe, so the sweep carries its own null.
    "e7": dict(arms=["R_gt3moe", "B3_route_lo", "B3_route_mid", "B3_route_hi",
                     "R_switch_canon", "R_switch_top1", "R_vmoe_canon", "R_stablemoe_top1",
                     "E7_congestion", "E7_mass_weighted", "E7_cong_lo", "E7_cong_hi"],
               seeds=[0, 1, 2], ks=[8], hours=3.6),
    # Seeds 3-5 for the four arms the headline table actually rests on. With n=3 the exact
    # sign-flip permutation test bottoms out at 2/2^3 = 0.25, so NOTHING in the baselines job can
    # reach significance no matter how large the effect -- Soft-MoE is +0.97pp acc / +2.26pp few
    # over Switch and still reports p=0.50 / 0.25. At n=6 the floor is 2/2^6 = 0.031.
    # Pairing survives the split: each seed's delta is measured between arms run in the SAME job,
    # so job is a randomised block and seeds 3-5 pool with seeds 0-2 from the baselines job.
    # Also carries the parameter-matched 2x2 (R_dense_wide / R_gt3moe_narrow), because that
    # question only exists because of R_dense's win and needs the same anchors to be paired.
    "seeds345": dict(arms=["R_gt3moe", "R_softmoe", "R_switch", "R_dense",
                           "R_dense_wide", "R_gt3moe_narrow",
                           "B3_route_lo", "B3_route_vlo", "B3_route_vvlo"],
                     seeds=[3, 4, 5], ks=[8], hours=3.1),
    # The overlapping-assignment sweep. T_dense_bal is the bar (0.7112) and T_gt3_bal (0.6793) is
    # the m=1, top_k=2 anchor already measured, so both endpoints are known before this runs.
    # m=8 doubles as the RIDE-style complete-expert baseline.
    "msweep": dict(arms=["T_dense_bal", "M_m1_bal", "M_m2_bal", "M_m4_bal", "M_m8_bal"],
                   seeds=[0, 1, 2], ks=[8], hours=1.9),
    # Trunk-level experts vs the dense control, all four arms inside ONE job so the comparison is
    # paired. Launch this batch three times with --seeds 0 / 1 / 2 on three instance types: that
    # parallelises across SEEDS while keeping every subtracted pair on one GPU architecture.
    "trunk": dict(arms=["T_dense_bal", "X_trunk_gt3_bal", "X_trunk_switch_bal",
                        "X_trunk_ride_bal"],
                  seeds=[0, 1, 2], ks=[8], hours=1.1),
    # Concurrency probe. GPU quotas are PER INSTANCE TYPE (43 separate quota codes, each 1), not
    # a global cap -- the docstring above was wrong about that. This batch is launched once per
    # type to verify that three jobs actually run at the same time.
    "conc": dict(arms=["Z_tiny"], seeds=[0], ks=[8], hours=0.3),

    # ----------------------------------------------------------------- cross-dataset generality
    # The CIFAR-100-LT result (dense head beats every MoE; GT3MoE only overtakes Switch at K>=16)
    # rests on ONE corpus with C/K = 12.5. These three batches ask whether it travels. The arm
    # list is byte-identical across all three so every cross-dataset difference is the corpus.
    #
    #   D_tiny  200 classes, C/K = 25 at K=8 -- the axis the m-sweep implicated, doubled.
    #   D_mini  100 classes at ImageNet resolution/difficulty -- the corpus the paper CLAIMS
    #           but has never actually been run on.
    #   D_c100  rho=1, the balanced control: is the dense head's win about long-tailedness at
    #           all, or does it hold on balanced data too? NOTE this also multiplies N by 4.6,
    #           so it confounds balance with data volume; D_c100n is the matched-N companion
    #           (78/class = 7,800 train, same size as CIFAR-100-LT) that separates them.
    "D_tiny": dict(arms=DSET_ARMS, seeds=[0, 1, 2], ks=[8], hours=5.0, tag="tin",
                   assets=["tinyimagenet_lt_r100_v0.15_s96_d1234.npz"],
                   over=dict(dataset="tinyimagenet_lt", num_classes=200, imbalance_ratio=100.0)),
    "D_mini": dict(arms=DSET_ARMS, seeds=[0, 1, 2], ks=[8], hours=2.8, tag="min",
                   assets=["miniimagenet_lt_r100_v0.15_s96_d1234.npz"],
                   over=dict(dataset="miniimagenet_lt", num_classes=100, imbalance_ratio=100.0)),
    # ~25 min/run at 42.5k images, so this one is launched three times with --seeds 0 / 1 / 2
    # rather than as a single 11-hour job; job stays the randomised block either way.
    "D_c100": dict(arms=DSET_ARMS, seeds=[0, 1, 2], ks=[8], hours=4.5, tag="c1b",
                   over=dict(dataset="cifar100", num_classes=100, imbalance_ratio=1.0)),
    # The K crossover, re-run on a SECOND corpus. On CIFAR-100-LT GT3MoE loses to Switch at K=4/8
    # and wins at K=16/32, and that crossover is now the only positive claim left standing -- the
    # K=8 edge does not replicate (pooled -0.07pp, p=0.53, 8/15). If the crossover is real it must
    # appear on MiniImageNet-LT too; if it does not, it was a CIFAR artefact. K=8 for both arms is
    # already in D_mini, so this completes the sweep at three Ks on a second corpus.
    "D_mini_k": dict(arms=["R_gt3moe_K16", "R_switch_K16", "R_gt3moe_K32", "R_switch_K32"],
                     seeds=[0, 1, 2], ks=[8], hours=1.6, tag="mnk",
                     assets=["miniimagenet_lt_r100_v0.15_s96_d1234.npz"],
                     over=dict(dataset="miniimagenet_lt", num_classes=100,
                               imbalance_ratio=100.0)),
    # K sweep on a THIRD corpus. The crossover threshold moved between CIFAR-100-LT (between K=8
    # and K=16) and MiniImageNet-LT (between K=16 and K=32), so it is governed by neither K nor
    # C/K alone. Tiny has C=200, so K=16 gives C/K=12.5 (= CIFAR at K=8) and K=32 gives C/K=6.25
    # (= CIFAR at K=16). Running both discriminates: if C/K governs, tiny loses at K=16 and wins
    # at K=32; if raw K governs, it should track CIFAR at both.
    "D_tiny_k": dict(arms=["R_gt3moe_K16", "R_switch_K16", "R_gt3moe_K32", "R_switch_K32"],
                     seeds=[0, 1, 2], ks=[8], hours=3.0, tag="tnk",
                     assets=["tinyimagenet_lt_r100_v0.15_s96_d1234.npz"],
                     over=dict(dataset="tinyimagenet_lt", num_classes=200,
                               imbalance_ratio=100.0)),
    "D_c100n": dict(arms=DSET_ARMS, seeds=[0, 1, 2], ks=[8], hours=2.8, tag="c1n",
                    over=dict(dataset="cifar100", num_classes=100, imbalance_ratio=1.0,
                              train_cap_per_class=78)),

    # ================================================================== FINAL TMLR PROGRAM ======
    # 11 arms x 6 corpora x 2 protocols x 3 seeds. Launch ONE JOB PER SEED (--seeds 0 / 1 / 2) on
    # distinct instance types: every arm-pair then stays inside one job on one GPU, and seeds pool
    # across jobs as a randomised block. ``hours`` is quoted for the batch as defined (3 seeds) and
    # the launcher scales it down when --seeds selects a subset.
    #
    # ---- fine-tuned protocol (IMAGENET1K_V2 trunk, 60 epochs) --------------------------------
    # The three CIFAR corpora are clean under this protocol -- CIFAR-100 comes from 80 Million
    # Tiny Images, not ImageNet -- so the penalty decomposition stays here at full 11 arms.
    # Mini/Tiny are contaminated under it and are reported as such, against their scratch twins.
    "F_c100lt": dict(arms=TMLR_ARMS, seeds=[0, 1, 2], ks=[8], hours=3.4, tag="fcl",
                     over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0)),
    "F_c100bal": dict(arms=TMLR_ARMS, seeds=[0, 1, 2], ks=[8], hours=14.5, tag="fcb",
                      over=dict(dataset="cifar100", num_classes=100, imbalance_ratio=1.0)),
    "F_c100n": dict(arms=TMLR_ARMS, seeds=[0, 1, 2], ks=[8], hours=2.9, tag="fcn",
                    over=dict(dataset="cifar100", num_classes=100, imbalance_ratio=1.0,
                              train_cap_per_class=78)),
    "F_mini": dict(arms=TMLR_ARMS, seeds=[0, 1, 2], ks=[8], hours=3.4, tag="fmi",
                   assets=[MIN_NPZ],
                   over=dict(dataset="miniimagenet_lt", num_classes=100, imbalance_ratio=100.0)),
    "F_tiny": dict(arms=TMLR_ARMS, seeds=[0, 1, 2], ks=[8], hours=6.4, tag="fti",
                   assets=[TIN_NPZ],
                   over=dict(dataset="tinyimagenet_lt", num_classes=200, imbalance_ratio=100.0)),

    # ---- from-scratch protocol -----------------------------------------------------------------
    "S_c100lt": dict(arms=TMLR_ARMS, seeds=[0, 1, 2], ks=[8], hours=10.6, tag="scl",
                     over=dict(SCRATCH200, dataset="cifar100lt", num_classes=100,
                               imbalance_ratio=100.0)),
    "S_mini": dict(arms=TMLR_ARMS, seeds=[0, 1, 2], ks=[8], hours=10.6, tag="smi",
                   assets=[MIN_NPZ],
                   over=dict(SCRATCH200, dataset="miniimagenet_lt", num_classes=100,
                             imbalance_ratio=100.0)),
    "S_tiny": dict(arms=TMLR_ARMS, seeds=[0, 1, 2], ks=[8], hours=20.7, tag="sti",
                   assets=[TIN_NPZ],
                   over=dict(SCRATCH200, dataset="tinyimagenet_lt", num_classes=200,
                             imbalance_ratio=100.0)),
    # imbalance_ratio=1.0 is not a claim that ImageNet-LT is balanced -- its imbalance is already
    # in the official split file, so _long_tail must be a no-op (see data.PRESPLIT).
    "S_inet": dict(arms=TMLR_ARMS, seeds=[0, 1, 2], ks=[8], hours=39.1, tag="sin",
                   assets=[INL_NPZ],
                   over=dict(SCRATCH60, dataset="imagenet_lt", num_classes=1000,
                             imbalance_ratio=1.0)),

    # ---- AUGMENTATION DIAGNOSTIC ---------------------------------------------------------------
    # Every run in this benchmark reaches train CE ~0.00 -- flip + pad-4 crop does not regularise
    # a 23M-param ResNet-50 on 9,207 images at all, so the entire dense-vs-MoE comparison has been
    # measured in a MEMORISING regime. These two batches are a matched 2x5 factorial on the
    # cheapest corpus: identical arms, identical 120 epochs, differing only in augmentation.
    # The quantity of interest is the DIFFERENCE IN PENALTY between them -- if the dense head's
    # advantage is an artefact of memorisation it should shrink under "strong".
    # warmup/ramp are scaled 5->10 and 10->20 so both arms keep the 60-epoch schedule fractions.
    "A_weak": dict(arms=AUG_ARMS, seeds=[0, 1, 2], ks=[8], hours=3.3, tag="awk",
                   over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0,
                             epochs=120, warmup_epochs=10, ramp_epochs=20,
                             augment="per_batch")),
    "A_strong": dict(arms=AUG_ARMS, seeds=[0, 1, 2], ks=[8], hours=3.6, tag="ast",
                     over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0,
                               epochs=120, warmup_epochs=10, ramp_epochs=20,
                               augment="strong", rrc_min_scale=0.35, jitter_strength=0.4)),

    # ---- assignment null + per-K-tuned Switch, on FRESH seeds ----------------------------------
    # 36 configs = 3 methods x 4 balance_alphas x 3 K, all in one job per seed so every pair is
    # within-job. Seeds 3-5 so this is an independent replication of the ksweep (seeds 0-2), not a
    # reanalysis of it. See arms.TUNE_ARMS for what the two questions are.
    "U_tune": dict(arms=ARMS.TUNE_ARMS, seeds=[3, 4, 5], ks=[8], hours=16.2, tag="utn",
                   over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0)),

    # ---- PREREGISTRATION.md Amendment 1 and the confirmatory program ---------------------------
    # Hours from measured U_tune timings on A10G: ~5.0 / 5.8 / 7.3 min per run at K = 8 / 16 / 32
    # fine-tuned; balanced CIFAR-100 runs ~3.9x longer (F_c100bal); from-scratch 200-epoch
    # Mini-ImageNet-LT ~26 min per run at K = 32 (K_mini).
    # Stage 1b: SWI only, ONE job for all three seeds so the anchors pair with the new alphas.
    "U_tune1b": dict(arms=ARMS.U1B_ARMS, seeds=[3, 4, 5], ks=[8], hours=3.3, tag="u1b",
                     over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0)),
    # Stage 2: fresh seeds 6-17 (n = 12). Pack three seeds per job across four A10G types.
    "C2_confirm": dict(arms=ARMS.C2_ARMS, seeds=list(range(6, 18)), ks=[8], hours=16.1, tag="c2",
                       over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0)),
    "B_break": dict(arms=ARMS.B_ARMS, seeds=list(range(6, 12)), ks=[8], hours=10.6, tag="bbk",
                    over=dict(dataset="cifar100", num_classes=100, imbalance_ratio=1.0)),
    # Conditional follow-ups: launch only if the stage-2 outcome triggers them.
    "R_mini_conf": dict(arms=ARMS.RM_ARMS, seeds=list(range(6, 14)), ks=[8], hours=15.7,
                        tag="rmc", assets=[MIN_NPZ],
                        over=dict(SCRATCH200, dataset="miniimagenet_lt", num_classes=100,
                                  imbalance_ratio=100.0)),
    "S_sens": dict(arms=ARMS.SN_ARMS, seeds=[6, 7, 8], ks=[8], hours=3.0, tag="sns",
                   over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0)),

    # ---- the paper's uniform grid: 10 routers x K in {8, 16, 32} x 3 fresh seeds -----------------
    # CIFAR-100-LT fine-tuned (CIFAR is not in ImageNet); Mini- and Tiny-ImageNet-LT from scratch
    # (both are built from ImageNet train). The dense reference rides in the K = 32 batch.
    # Hours: measured per-run minutes x 30 runs (+3 dense), before the ~20% overrun seen so far.
    **{f"P_c100lt_K{K}": dict(arms=ARMS.paper_arms(K, False, K == 32), seeds=[20, 21, 22], ks=[8],
                              hours=h, tag=f"pc{K}",
                              over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0))
       for K, h in ((8, 2.6), (16, 3.0), (32, 3.9))},
    **{f"P_mini_K{K}": dict(arms=ARMS.paper_arms(K, True, K == 32), seeds=[20, 21, 22], ks=[8],
                            hours=h, tag=f"pm{K}", assets=[MIN_NPZ],
                            over=dict(SCRATCH200, dataset="miniimagenet_lt", num_classes=100,
                                      imbalance_ratio=100.0))
       for K, h in ((8, 8.9), (16, 10.1), (32, 13.2))},
    **{f"P_tiny_K{K}": dict(arms=ARMS.paper_arms(K, True, K == 32), seeds=[20, 21, 22], ks=[8],
                            hours=h, tag=f"pt{K}", assets=[TIN_NPZ],
                            over=dict(SCRATCH200, dataset="tinyimagenet_lt", num_classes=200,
                                      imbalance_ratio=100.0))
       for K, h in ((8, 19.1), (16, 22.1), (32, 27.8))},
    "P_ltloss": dict(arms=ARMS.PL_ARMS, seeds=[20, 21, 22], ks=[8], hours=4.2, tag="plt",
                     over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0)),
    # post-hoc fixes (2026-10-13): 16 arms per seed, ~5-7 min each on A10G
    "FX_fix": dict(arms=ARMS.FX_ARMS, seeds=[20, 21, 22], ks=[8], hours=5.4, tag="fx",
                   over=dict(dataset="cifar100lt", num_classes=100, imbalance_ratio=100.0)),

    # ---- K crossover, the one surviving positive claim ------------------------------------------
    # Measured so far: K=8 null (-0.07pp, p=0.53, 8/15 over five corpora), K=32 +0.68pp acc /
    # +2.65pp few (6/6, p=0.031) on CIFAR-100-LT and MiniImageNet-LT. The threshold moves by
    # corpus, so this is reported as "overtakes Switch at sufficiently large K", never a fixed K.
    "K_c100lt": dict(arms=["R_gt3moe_K4", "R_switch_K4", "R_gt3moe_K16", "R_switch_K16",
                           "R_gt3moe_K32", "R_switch_K32"],
                     seeds=[0, 1, 2], ks=[8], hours=8.7, tag="kcl",
                     over=dict(SCRATCH200, dataset="cifar100lt", num_classes=100,
                               imbalance_ratio=100.0)),
    "K_mini": dict(arms=["R_gt3moe_K4", "R_switch_K4", "R_gt3moe_K16", "R_switch_K16",
                         "R_gt3moe_K32", "R_switch_K32"],
                   seeds=[0, 1, 2], ks=[8], hours=8.7, tag="kmi", assets=[MIN_NPZ],
                   over=dict(SCRATCH200, dataset="miniimagenet_lt", num_classes=100,
                             imbalance_ratio=100.0)),
    # RIDE-core + a FRESH within-job dense control. The control is not optional: the same config
    # across two jobs has a paired sd of 0.21pp, so a cross-job comparison against the stored
    # 0.7112 could manufacture a win out of nothing.
    "ride": dict(arms=["T_dense_bal", "M_ride_bal", "M_ride_ce", "R_dense"],
                 seeds=[0, 1, 2], ks=[8], hours=1.3),
    # The K sweep, kept as its own job so the headline comparative claim does not share a runtime
    # budget with the congestion arms. GT3MoE and Switch at K = 16 and K = 32, paired within-job,
    # because "+9.1pp few-shot over Switch" is asserted at K = 32 and has never been reproduced.
    # K = 8 for both is already in the baselines and e7 jobs, so the sweep completes at three Ks.
    # NOTE the Switch arms here are top_k=1, i.e. Switch as published -- the e7 job showed that is
    # +1.08pp over the top-2 version and beats GT3MoE at K=8, so the K sweep has to use the
    # faithful one or it re-runs the same fidelity mistake at three more values of K.
    "ksweep": dict(arms=["R_gt3moe_K4", "R_switch_K4", "R_gt3moe_K16", "R_switch_K16",
                         "R_gt3moe_K32", "R_switch_K32"],
                   seeds=[0, 1, 2], ks=[8], hours=2.4),
    "probe": dict(arms=["D_probe"], seeds=[0, 1, 2], ks=[8], hours=0.5),
    "combo": dict(arms=["B0_base", "B1_js", "B2_align_hinge", "v2", "D_splithalf"],
                  seeds=[0, 1, 2], ks=[8], hours=1.9),
    "interact": dict(arms=["A_frozen_aug", "A_frozen_nopcgrad", "v2", "E4e_no_pcgrad",
                           "L_js", "L_no_comp", "E4d_random_groups"],
                     seeds=[0, 1, 2], ks=[8], hours=5.0),
}


def build_payload(root: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("gt3moe", "sagemaker"):
            for dirpath, dirnames, filenames in os.walk(os.path.join(root, sub)):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for fn in filenames:
                    if fn.endswith((".pyc",)):
                        continue
                    full = os.path.join(dirpath, fn)
                    tar.add(full, arcname=os.path.relpath(full, root))
    return buf.getvalue()


def arm_specs(batch: dict):
    """Return [(name, overrides_json)] for every (arm, K, seed) in the batch."""
    out = []
    over = batch.get("over", {})
    tag = batch.get("tag", "")
    for job, o in ARMS.expand(batch["arms"], batch["seeds"], batch["ks"]):
        o = dict(o)
        # Batch-level overrides (the dataset switch) are applied AFTER the arm, so one arm set
        # serves every corpus and the cross-dataset comparison is arm-for-arm identical.
        o.update(over)
        preset = o.pop("preset", "v2")
        o["preset"] = preset
        name = f"{tag}-{job}" if tag else job
        out.append((name.replace("_", "-"), json.dumps(o, separators=(",", ":"))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=sorted(BATCHES))
    ap.add_argument("--blocks", default="",
                    help="pack several (batch, seed) blocks into ONE job, e.g. "
                         "'F_mini:0,S_tiny:2'. The GPU quota is 1 CONCURRENT JOB PER INSTANCE "
                         "TYPE, so a 33-block program cannot be 33 jobs without a live scheduler "
                         "feeding slots as they free. Packing each slot's blocks into one job "
                         "that runs them sequentially needs no supervision. A block is never "
                         "split, so every arm-pair still shares one job on one GPU.")
    ap.add_argument("--label", default="", help="job-name suffix when using --blocks")
    ap.add_argument("--instance", default="ml.g5.xlarge", choices=sorted(PRICE))
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--max_run", type=int, default=0, help="0 -> derived from the batch estimate")
    ap.add_argument("--seeds", default="", help="comma list overriding the batch's seeds, e.g. 0")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--profile", default=PROFILE)
    a = ap.parse_args()
    if not (a.batch or a.blocks):
        ap.error("need --batch or --blocks")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arm_names = []
    if a.blocks:
        specs, assets, hours = [], [], 0.0
        for spec in a.blocks.split(","):
            name, _, sd = spec.partition(":")
            b = dict(BATCHES[name.strip()])
            seeds = [int(sd)] if sd else list(b["seeds"])
            hours += b["hours"] * len(seeds) / max(1, len(b["seeds"]))
            b["seeds"] = seeds
            specs += arm_specs(b)
            arm_names += list(b["arms"])
            assets += [x for x in b.get("assets", []) if x not in assets]
        batch = dict(assets=assets)
        a.batch = a.label or "pack"
    else:
        batch = dict(BATCHES[a.batch])
        hours = batch["hours"]
        arm_names = list(batch["arms"])
        if a.seeds:
            # ``hours`` is quoted for the batch as DEFINED. Launching a subset of its seeds must
            # scale the estimate down, or max_run is several times longer than the job and a hung
            # run bills for it.
            want = [int(x) for x in a.seeds.split(",")]
            hours *= len(want) / max(1, len(batch["seeds"]))
            batch["seeds"] = want
        specs = arm_specs(batch)
    # floor of 45 min covers container pull, dataset unpack/resize and the final eval pass
    max_run = a.max_run or int(min(5 * 24 * 3600, max(2700, hours * 3600 * 1.6 + 1800)))
    usd = hours * PRICE[a.instance]

    print(f"\n  batch={a.batch}  instance={a.instance}  spot={a.spot}")
    print(f"  {len(specs)} arms in ONE job (GPU training-job quota on this account is 1)")
    print(f"  estimated {hours:.1f} GPU-h -> ${usd:.2f} on-demand"
          f"{f', ~${usd * 0.35:.2f} on spot' if a.spot else ''}")
    print(f"  max_run={max_run}s ({max_run / 3600:.1f}h)")
    for n, c in specs:
        print(f"    {n}")
    provisional = sorted(set(arm_names) & ARMS.PROVISIONAL_ARMS)
    if provisional:
        print(f"\n  PROVISIONAL: {len(provisional)} arms use stage-1 alphas because "
              f"selected_alphas.json has no 'final' block yet (PREREGISTRATION.md A1.6).")
        if not a.dry_run:
            print("  ABORT: stage-2-family arms cannot launch until stage 1b has set 'final'.\n")
            return 4
    if a.dry_run:
        print("\n  --dry_run: nothing launched.\n")
        return 0

    if not BUCKET or not ROLE:
        print("\n  ABORT: set GT3MOE_S3_BUCKET and GT3MOE_SAGEMAKER_ROLE_ARN to launch.\n")
        return 2
    import boto3
    sess = boto3.Session(profile_name=a.profile, region_name=REGION)
    s3 = sess.client("s3")
    sm = sess.client("sagemaker")

    # Refuse to launch if the slot for THIS instance type is already occupied. SageMaker training
    # quotas are per instance type -- "ml.g5.xlarge for training job usage" and
    # "ml.g5.2xlarge for training job usage" are distinct quotas, each 1 on this account -- so a
    # job on a different type does not block us, but a job on the same type would fail with
    # ResourceLimitExceeded.
    busy = sm.list_training_jobs(StatusEquals="InProgress", MaxResults=50)["TrainingJobSummaries"]
    same, other = [], []
    for b in busy:
        d = sm.describe_training_job(TrainingJobName=b["TrainingJobName"])
        it = d["ResourceConfig"]["InstanceType"]
        (same if it == a.instance else other).append((b["TrainingJobName"], it))
    for n, it in other:
        print(f"  note: {n} is running on {it} (different quota, does not block {a.instance})")
    if same:
        print(f"\n  ABORT: the {a.instance} training-job quota is 1 and it is already in use:")
        for n, it in same:
            print(f"    {n}  {it}")
        print("  Wait for it, or pick another instance type with --instance.\n")
        return 3

    stamp = time.strftime("%Y%m%d-%H%M%S")
    key = f"{PREFIX}/{stamp}/{a.batch}/payload.tar.gz"
    payload = build_payload(root)
    s3.put_object(Bucket=BUCKET, Key=key, Body=payload)
    run_key = f"{PREFIX}/{stamp}/{a.batch}/run.sh"
    s3.put_object(Bucket=BUCKET, Key=run_key,
                  Body=open(os.path.join(root, "sagemaker", "run.sh"), "rb").read()
                  .replace(b"\r\n", b"\n"))
    # Stage the dataset into this batch's channel prefix (S3->S3 copy: seconds, no egress).
    # Without it the container downloads CIFAR-100 from cs.toronto.edu, which consumed the entire
    # 30-minute budget of the first calibration run before a single epoch ran.
    # A batch names any prebuilt .npz cache it needs in "assets"; only that one is copied, so a
    # CIFAR job does not drag the 1 GB Tiny-ImageNet array through its channel.
    for fname in ["cifar-100-python.tar.gz"] + list(batch.get("assets", [])):
        asset = f"{PREFIX}/assets/{fname}"
        try:
            s3.head_object(Bucket=BUCKET, Key=asset)
            s3.copy_object(Bucket=BUCKET, Key=f"{PREFIX}/{stamp}/{a.batch}/{fname}",
                           CopySource={"Bucket": BUCKET, "Key": asset})
            print(f"  staged {fname} from s3://{BUCKET}/{asset}")
        except Exception as e:
            print(f"  WARNING: asset {fname} missing ({e}).\n"
                  f"    aws s3 cp data/{fname} s3://{BUCKET}/{asset} "
                  f"--profile {a.profile} --region {REGION}")
    print(f"\n  uploaded payload ({len(payload) / 1024:.0f} KB) -> s3://{BUCKET}/{key}")

    # SageMaker job names match [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}: no underscores. Batch keys
    # may contain them (D_tiny), so translate here rather than renaming the batches.
    job = f"{PREFIX}-{a.batch}-{stamp}".replace("_", "-")[:63]
    ckpt_s3 = f"s3://{BUCKET}/{PREFIX}/{stamp}/{a.batch}/checkpoints"
    # SageMaker caps each Environment value at 512 chars, so the arm list travels as a TSV in
    # the channel instead. Names never contain tabs; the JSON is emitted without spaces.
    tsv = "".join("{}\t{}\n".format(n, c) for n, c in specs)
    s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{stamp}/{a.batch}/arms.tsv",
                  Body=tsv.encode())
    env = {"GT3MOE_BATCH": a.batch}

    req = dict(
        TrainingJobName=job,
        RoleArn=ROLE,
        AlgorithmSpecification=dict(
            TrainingImage=IMAGE, TrainingInputMode="File",
            ContainerEntrypoint=["/bin/bash", "/opt/ml/input/data/data/run.sh"],
        ),
        InputDataConfig=[dict(
            ChannelName="data",
            DataSource=dict(S3DataSource=dict(
                S3DataType="S3Prefix",
                S3Uri=f"s3://{BUCKET}/{PREFIX}/{stamp}/{a.batch}/",
                S3DataDistributionType="FullyReplicated")),
        )],
        OutputDataConfig=dict(S3OutputPath=f"s3://{BUCKET}/{PREFIX}/{stamp}/{a.batch}/output"),
        ResourceConfig=dict(InstanceType=a.instance, InstanceCount=1, VolumeSizeInGB=60),
        StoppingCondition=dict(MaxRuntimeInSeconds=max_run),
        CheckpointConfig=dict(S3Uri=ckpt_s3, LocalPath="/opt/ml/checkpoints"),
        Environment=env,
        EnableManagedSpotTraining=bool(a.spot),
    )
    if a.spot:
        req["StoppingCondition"]["MaxWaitTimeInSeconds"] = min(5 * 24 * 3600, max_run * 3)

    sm.create_training_job(**req)
    print(f"\n  LAUNCHED {job}")
    print(f"  checkpoints: {ckpt_s3}")
    print(f"  logs: aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job} "
          f"--follow --profile {a.profile} --region {REGION}")
    print(f"  status: aws sagemaker describe-training-job --training-job-name {job} "
          f"--profile {a.profile} --region {REGION} --query TrainingJobStatus\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
