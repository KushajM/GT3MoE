"""Experiment arms, as data.

Each arm is a dict of config overrides. ``SEEDS`` is shared across arms deliberately: common
random numbers make every comparison paired, which is free statistical power.

The order below is the order to run in. Stop after E1 if E1 ties -- if a frozen assignment matches
the full two-timescale loop, the method section has to be rewritten before any further spend.
"""
from __future__ import annotations

import json
import os

SEEDS = [0, 1, 2]
SEEDS5 = [0, 1, 2, 3, 4]

BASE = dict(dataset="cifar100lt", epochs=60, num_experts=8)

ARMS = {
    # ---- 0. calibration and provenance -------------------------------------------------
    # MUST span the warmup boundary: with warmup_epochs=5 a 5-epoch calibration measures only the
    # cheap pre-warmup path and never executes PCGrad, which is the dominant cost.
    "E0_calib": dict(preset="v2", epochs=9, arm="E0_calib", measure_flops=True),

    # ---- A. the augmentation bug: run this FIRST, it moves every other margin -----------
    # The original applied RandomCrop/Flip once at cache time, so 60 epochs trained on a
    # single fixed augmented copy. If fixing it moves Overall by several points, every
    # pre-registered equivalence margin below has to be re-derived.
    "A_frozen_aug": dict(preset="v2", arm="A_frozen_aug", augment="frozen_once"),
    "A_per_batch": dict(preset="v2", arm="A_per_batch", augment="per_batch"),
    # The other two cells of the augmentation x PCGrad 2x2. Hypothesis worth testing rather than
    # dismissing: group-mean averaging is an implicit class-balanced reweighting, which would
    # regularise a badly-overfitting frozen-augmentation run and become useless (or harmful) once
    # real augmentation removes the overfitting. That would EXPLAIN the paper's +2.5pp instead of
    # merely contradicting it. All four cells must be in ONE job -- cross-job noise is 0.60pp.
    "A_frozen_nopcgrad": dict(preset="v2", arm="A_frozen_nopcgrad", augment="frozen_once",
                              pcgrad=False),

    # ---- reference points ---------------------------------------------------------------
    "paper_v1": dict(preset="paper_v1", arm="paper_v1"),
    "v2": dict(preset="v2", arm="v2"),

    # ---- E1. THE decisive control: is the slow timescale live past warmup? --------------
    "E1_frozen": dict(preset="v2", arm="E1_frozen", assign_mode="frozen_at",
                      assign_freeze_epoch=6),
    # ---- E2. is the k-means init doing the work? ----------------------------------------
    "E2_kmeans_only": dict(preset="v2", arm="E2_kmeans_only", assign_mode="kmeans_only"),
    # ---- E3. does COMPETENCE matter, or just having any balanced partition? -------------
    "E3_random_balanced": dict(preset="v2", arm="E3_random_balanced",
                               assign_mode="random_balanced"),
    # stronger null than E3: preserves every marginal of the cost matrix, destroys only the
    # expert-class correspondence
    "E3b_column_shuffled": dict(preset="v2", arm="E3b_column_shuffled",
                                assign_mode="column_shuffled"),

    # ---- E4. PCGrad decomposition, one variable per arm ---------------------------------
    "E4a_full_batch": dict(preset="v2", arm="E4a_full_batch", pcgrad_project=False,
                           backbone_ce="full_batch"),
    "E4b_reweight_only": dict(preset="v2", arm="E4b_reweight_only", pcgrad_project=False,
                              backbone_ce="group_mean"),
    "E4c_size_weighted": dict(preset="v2", arm="E4c_size_weighted", pcgrad_project=True,
                              backbone_ce="size_weighted"),
    "E4d_random_groups": dict(preset="v2", arm="E4d_random_groups", pcgrad_groups="random"),
    "E4e_no_pcgrad": dict(preset="v2", arm="E4e_no_pcgrad", pcgrad=False),
    "E4f_keep_singletons": dict(preset="v2", arm="E4f_keep_singletons",
                                pcgrad_singleton="merge_to_residual", pcgrad_min_group=1),

    # ---- E5. the estimator ---------------------------------------------------------------
    "E5_train_topk": dict(preset="v2", arm="E5_train_topk", cost_source="train_topk",
                          impute=True),
    "E5_train_dense": dict(preset="v2", arm="E5_train_dense", cost_source="train_dense"),
    "E5_probe": dict(preset="v2", arm="E5_probe", cost_source="probe",
                     probe_split="train_eval"),
    "E5_probe_val": dict(preset="v2", arm="E5_probe_val", cost_source="probe",
                         probe_split="val"),   # uses balanced held-out labels: ablation only

    # ---- E6. solver ----------------------------------------------------------------------
    "E6_greedy": dict(preset="v2", arm="E6_greedy", assign_solver="greedy", cap_lo=0),
    "E6_exact": dict(preset="v2", arm="E6_exact", assign_solver="exact"),
    "E6_exact_hyst": dict(preset="v2", arm="E6_exact_hyst", assign_solver="exact",
                          assign_hysteresis=0.1),
    "E6_no_floor": dict(preset="v2", arm="E6_no_floor", assign_solver="exact", cap_lo=0),

    # ---- E7. capacity / congestion -------------------------------------------------------
    # Both sit on the BEST MEASURED config (pcgrad off, cosine term off) rather than on the v2
    # preset. Two reasons, and the second is the real one: (a) the v2 preset carries PCGrad, which
    # costs 3.3x the wall-clock for an effect measured at -0.17pp, p=0.22; (b) the question E7
    # asks is "does the congestion term help the model we would actually ship?", and answering it
    # on top of a component already shown to be net-negative answers a different question.
    #
    # E7_congestion adds the (lambda/2)(2s+1) dilution term to the exact assignment cost, which is
    # what makes the potential a WEIGHTED congestion game rather than a plain semi-assignment:
    # an expert's cost now rises with the number of classes on it. E7_mass_weighted goes further
    # and sets w_c = n_c, so congestion is measured in training examples, not classes -- the form
    # the manuscript's framing actually requires under Harks & Klimm, since affine per-player costs
    # with player-specific weights are what buy an exact potential.
    "E7_congestion": dict(preset="v2", arm="E7_congestion", pcgrad=False, comp_loss="none",
                          congestion_lambda=0.02),
    "E7_mass_weighted": dict(preset="v2", arm="E7_mass_weighted", pcgrad=False, comp_loss="none",
                             assign_solver="br_weighted", congestion_lambda=0.02,
                             assign_hysteresis=0.05),
    # lambda is unswept and 0.02 was a guess. One order of magnitude either side, so the claim is
    # "the term does/doesn't help" rather than "it doesn't help at the one value we tried".
    "E7_cong_lo": dict(preset="v2", arm="E7_cong_lo", pcgrad=False, comp_loss="none",
                       congestion_lambda=0.005),
    "E7_cong_hi": dict(preset="v2", arm="E7_cong_hi", pcgrad=False, comp_loss="none",
                       congestion_lambda=0.08),

    # ---- E8. controls --------------------------------------------------------------------
    "E8_dense_head": dict(preset="v2", arm="E8_dense_head", dense_head=True, pcgrad=False),
    "E8_K1": dict(preset="v2", arm="E8_K1", num_experts=1, top_k=1, pcgrad=False,
                  comp_loss="none"),

    # ---- losses ---------------------------------------------------------------------------
    "L_js": dict(preset="v2", arm="L_js", comp_loss="js_confusion", comp_beta=0.005),
    "L_hinge_cos": dict(preset="v2", arm="L_hinge_cos", comp_loss="centred_cosine_hinge"),
    "L_no_comp": dict(preset="v2", arm="L_no_comp", comp_loss="none"),
    "L_align_hinge": dict(preset="v2", arm="L_align_hinge", align_loss="hinge_topk"),
    "L_align_balanced": dict(preset="v2", arm="L_align_balanced", align_class_balanced=True),
    "L_no_align": dict(preset="v2", arm="L_no_align", align_loss="none"),

    # ---- building on the best measured config: PCGrad off (+1.03pp) and the cosine term gone --
    "B0_base": dict(preset="v2", arm="B0_base", pcgrad=False, comp_loss="none"),
    "B1_js": dict(preset="v2", arm="B1_js", pcgrad=False, comp_loss="js_confusion",
                  comp_beta=0.005),
    # routing is now the binding constraint: oracle-vs-deployed is 13.8pp once the cosine term
    # stops crippling the experts, so the alignment loss is the lever, not the diversity loss.
    "B2_align_hinge": dict(preset="v2", arm="B2_align_hinge", pcgrad=False, comp_loss="none",
                           align_loss="hinge_topk"),
    # Is the measured conflict signal or sampling noise? Short run, PCGrad ON so groups exist.
    "D_splithalf": dict(preset="v2", arm="D_splithalf", epochs=15, splithalf_every=25,
                        eval_every=99),

    # Window-averaged conflict probe after a short train, both BN modes. Window-averaging over W
    # steps multiplies the gradient SNR by W, lifting attenuation from ~0.01 to ~0.84, so it
    # measures cos(mu_i, mu_j) almost directly instead of 1% of it.
    "D_probe": dict(preset="v2", arm="D_probe", epochs=20, eval_every=99,
                    conflict_probe_steps=800, conflict_probe_window=50, measure_flops=False),

    # ---- BASELINE ROUTERS, v2 numerical path -------------------------------------------
    # None of these existed in v2 until now; the paper's comparative table has no evidence behind
    # it (Table 2 matches no executed run, the backbone was DenseNet-201, and 4 of 6 baselines
    # could not be constructed from the shipped code). GT3MoE here uses its best measured config
    # (PCGrad off, cosine term removed) so the comparison is method-vs-method, not bug-vs-bug.
    "R_switch": dict(preset="v2", arm="R_switch", router="switch", pcgrad=False),
    "R_vmoe": dict(preset="v2", arm="R_vmoe", router="vmoe", pcgrad=False),
    "R_expert_choice": dict(preset="v2", arm="R_expert_choice", router="expert_choice",
                            pcgrad=False),
    "R_stablemoe": dict(preset="v2", arm="R_stablemoe", router="stablemoe", pcgrad=False),
    "R_softmoe": dict(preset="v2", arm="R_softmoe", router="softmoe", pcgrad=False),
    "R_hash": dict(preset="v2", arm="R_hash", router="hash", pcgrad=False),
    "R_gt3moe": dict(preset="v2", arm="R_gt3moe", pcgrad=False, comp_loss="none"),
    "R_dense": dict(preset="v2", arm="R_dense", dense_head=True, pcgrad=False),
    # Two recent (2024-25) routers. Both are LLM/transformer FFN-MoE methods
    # adapted to a head-level vision MoE; the adaptation is documented in routers.py and must
    # be stated in the paper rather than cited as a drop-in reproduction.
    "R_adamoe": dict(preset="v2", arm="R_adamoe", router="adamoe", pcgrad=False,
                     adamoe_null_experts=4),
    "R_dynmoe": dict(preset="v2", arm="R_dynmoe", router="dynmoe", pcgrad=False,
                     dynmoe_tau_init=0.0, dynmoe_sparsity_beta=0.01),

    # ---- parameter-matched 2x2: is dense winning on PARAMETERS or on DATA-PER-HEAD? ------
    # R_dense beat every MoE router by +1.77pp acc / +3.45pp few on 3/3 seeds while carrying 1/8
    # the head parameters (1.1M vs 8.8M), which leaves two explanations tangled together:
    #   (a) data-per-head -- a sparse router trains each head on ~1/8 of 9,207 images, whereas the
    #       dense head sees all of it. Supporting evidence: the ordering by data-per-head
    #       (dense 100% > Soft-Attention ~100% weighted > sparse routers ~12-25%) reproduces the
    #       accuracy ordering exactly (0.668 > 0.655 > 0.650-0.642).
    #   (b) parameters -- 8.8M head parameters simply overfit 9,207 training images.
    # The widths below make the two factors orthogonal to within 0.06%:
    #        head params |   1 head (dense)        |  8 heads (MoE)
    #        1.1M        |   R_dense       (512)   |  R_gt3moe_narrow (64)
    #        8.8M        |   R_dense_wide (4096)   |  R_gt3moe        (512)
    # If R_dense_wide still wins, MoE adds nothing at this scale and the setting is the problem,
    # not the router. If it loses to R_dense, 1.1M is simply the right capacity here and the MoE's
    # deficit is capacity, not routing. Either answer is worth more than the accuracy deltas.
    "R_dense_wide": dict(preset="v2", arm="R_dense_wide", dense_head=True, pcgrad=False,
                         expert_hidden=4096),
    "R_gt3moe_narrow": dict(preset="v2", arm="R_gt3moe_narrow", pcgrad=False, comp_loss="none",
                            expert_hidden=64),

    # ---- K sweep: a direct test of the ceiling-vs-routing trade-off ----------------------
    # Measured on the baselines job, GT3MoE and the learned routers land at the same accuracy by
    # opposite routes. GT3MoE's experts are genuinely exclusive -- averaging all K gives 5-13%,
    # against 37-42% for Switch/V-MoE/Expert-Choice -- so its best-expert oracle is LOWER (0.71 vs
    # 0.77) while the gap it actually leaves on the table is HALF theirs (6-7pp vs 12-13pp). The
    # hard class partition makes routing easy and the ceiling low, in the same move.
    #
    # If that diagnosis is right, K is the dial: at K=4 each expert owns ~25 classes and is more of
    # a generalist, so the oracle should RISE and acc_uniform_ensemble should rise with it; at
    # K=16 each owns ~6 and both should fall, with the oracle gap shrinking further. A K sweep is
    # the cheapest falsifiable test of the mechanism claim, and the paper wants one anyway.
    # The K sweep also carries the paper's single strongest comparative claim, which is stated at
    # K = 32, not K = 8: "improves over Switch (the strongest baseline on that cell) by 9.1pp on
    # few-shot accuracy" on CIFAR-100-LT. At K = 8 the measured few-shot gap to Switch is about
    # 1.6pp, so either the effect is strongly K-dependent or the claim does not survive. Both
    # arms are run at each K so the comparison stays paired and within-job.
    # K=4 runs the sweep in the direction the MECHANISM predicts, which K=16/32 do not. If
    # GT3MoE's deficit comes from each expert seeing only C/K classes and therefore failing to be a
    # complete classifier, then shrinking K must shrink the deficit: at K=4 an expert owns ~25
    # classes instead of ~12, and at K=1 GT3MoE IS the dense head. Switch is run at the same K so
    # the comparison stays paired -- Switch routes on a learned gate, so ITS experts see all C
    # classes at every K and the prediction is that its advantage is roughly K-independent.
    "R_gt3moe_K4": dict(preset="v2", arm="R_gt3moe_K4", pcgrad=False, comp_loss="none",
                        num_experts=4),
    "R_switch_K4": dict(preset="v2", arm="R_switch_K4", router="switch", pcgrad=False,
                        top_k=1, num_experts=4),
    "R_gt3moe_K16": dict(preset="v2", arm="R_gt3moe_K16", pcgrad=False, comp_loss="none",
                         num_experts=16),
    "R_gt3moe_K32": dict(preset="v2", arm="R_gt3moe_K32", pcgrad=False, comp_loss="none",
                         num_experts=32),
    # top_k=1: Switch as published. At K=8 that is +1.08pp over the top-2 version and enough to
    # beat GT3MoE, so sweeping K against the top-2 version would repeat the same fidelity mistake
    # at three more values of K and manufacture a win.
    "R_switch_K16": dict(preset="v2", arm="R_switch_K16", router="switch", pcgrad=False,
                         top_k=1, num_experts=16),
    "R_switch_K32": dict(preset="v2", arm="R_switch_K32", router="switch", pcgrad=False,
                         top_k=1, num_experts=32),

    # ---- routability-aware assignment: the measured bottleneck, addressed -----------------
    # B0_base has the HIGHEST best-expert oracle of anything measured (0.8028, against 0.763-0.775
    # for every learned router) and the LARGEST gap to it (15.0pp, against 12.1-13.2pp). So its
    # experts are not the problem; its routing is. The cause is structural rather than a tuning
    # miss: pi groups classes by which expert is CHEAPEST for them, with nothing requiring the
    # groups to be separable in feature space, and the router is then asked to realise that
    # partition with a linear map. It manages only ~83% top-1 on an 8-way problem.
    #
    # Adding mu * (-log p_router(i|c)) to the assignment cost makes pi prefer partitions the router
    # can express, which closes the two-timescale loop in the direction it was missing. The risk is
    # self-confirmation -- at large mu the assignment just ratifies whatever the router already
    # does, and performance-awareness is lost -- so mu is swept rather than picked, and the
    # capacity floor bounds how far the degenerate solution can go. mu=0 IS B0_base, so the sweep
    # contains its own null.
    "B3_route_lo": dict(preset="v2", arm="B3_route_lo", pcgrad=False, comp_loss="none",
                        routability_mu=0.05),
    "B3_route_mid": dict(preset="v2", arm="B3_route_mid", pcgrad=False, comp_loss="none",
                         routability_mu=0.2),
    "B3_route_hi": dict(preset="v2", arm="B3_route_hi", pcgrad=False, comp_loss="none",
                        routability_mu=0.6),
    # Measured: mu=0.05 gives +0.40pp acc (p=0.025) / +0.88pp few (p=0.013), 3/3 seeds, while
    # mu=0.2 and mu=0.6 give nothing. A gain that appears only at the SMALLEST value swept is not
    # yet a result -- it cannot be distinguished from a monotone decline out of an optimum below
    # the grid, and one positive point in a sweep is the shape a forking-paths artefact takes.
    # These two extend the grid downward; they are run with a replication of mu=0.05 at fresh
    # seeds so the single positive triple is not load-bearing. Report the whole curve, not the peak.
    "B3_route_vlo": dict(preset="v2", arm="B3_route_vlo", pcgrad=False, comp_loss="none",
                         routability_mu=0.02),
    "B3_route_vvlo": dict(preset="v2", arm="B3_route_vvlo", pcgrad=False, comp_loss="none",
                          routability_mu=0.01),

    # ---- OVERLAPPING ASSIGNMENT: the m-sweep ---------------------------------------------
    # Why this and not another router fix. The deployed router already puts the assigned expert in
    # the top-2 for 90.9% of samples (97.4 many / 92.5 medium / 83.2 few). Routing errors cost
    # 5.98pp; samples where routing was CORRECT and the answer was still wrong cost 23.09pp -- a
    # 3.9x ratio. And averaging all K experts, i.e. making no routing decision at all, scores
    # 0.5255 against a dense head's 0.7112. The binding constraint is that at C/K = 12.5 each
    # expert is a 12-way classifier impersonating a 100-way one. A perfect, label-cheating router
    # would reach 0.7375 overall but only ~0.568 on few-shot against dense's 0.5623 -- i.e. even
    # its ceiling does not deliver on the tail, which is the paper's thesis.
    #
    # m raises C/K directly: each class is owned by m experts, each owner trains on all of that
    # class's samples, and a routing miss that lands on another owner is harmless. top_k = m so
    # the router can reach every owner. m=1 is the original method (12.5 classes/expert); m=8=K
    # gives every expert every class, which IS a deep ensemble and IS the RIDE-style complete-expert
    # baseline. So one sweep provides the RIDE comparison, tests the
    # completeness mechanism, and is the only remaining configuration that could beat
    # dense+BalSoftmax. All arms carry Balanced Softmax, because that is the bar (0.7112).
    "M_m1_bal": dict(preset="v2", arm="M_m1_bal", pcgrad=False, comp_loss="none",
                     base_loss="balsoftmax", class_multiplicity=1, top_k=1),
    "M_m2_bal": dict(preset="v2", arm="M_m2_bal", pcgrad=False, comp_loss="none",
                     base_loss="balsoftmax", class_multiplicity=2, top_k=2),
    "M_m4_bal": dict(preset="v2", arm="M_m4_bal", pcgrad=False, comp_loss="none",
                     base_loss="balsoftmax", class_multiplicity=4, top_k=4),
    "M_m8_bal": dict(preset="v2", arm="M_m8_bal", pcgrad=False, comp_loss="none",
                     base_loss="balsoftmax", class_multiplicity=8, top_k=8),

    # ---- TRUNK EXPERTS: experts that own a layer4 and so learn their own representation -----
    # Every head-level result in this project is bounded by the same fact: an expert is 0.5M params
    # and 0.14% of the forward, reading the SAME 2048-d vector as every other expert, so experts can
    # differ only in decision boundary and sparse routing saves 0.87%. Replicating layer4 changes
    # both: 137.1M params against a dense ResNet-50's 24.6M at 0.760 GFLOP vs 0.752 -- 5.6x the
    # capacity for +1.1% compute -- and the routed-vs-all-experts saving becomes 57.8%.
    #
    # Each router keeps its own published top_k. Expect heavy overfitting: 137M parameters on 7,826
    # images that a 24.6M dense model already interpolates (train CE 0.0015). That is the test.
    "X_trunk_gt3_bal": dict(preset="v2", arm="X_trunk_gt3_bal", trunk_experts=True, top_k=2,
                            pcgrad=False, comp_loss="none", base_loss="balsoftmax"),
    "X_trunk_switch_bal": dict(preset="v2", arm="X_trunk_switch_bal", trunk_experts=True,
                               router="switch", top_k=1, pcgrad=False, base_loss="balsoftmax"),
    "X_trunk_ride_bal": dict(preset="v2", arm="X_trunk_ride_bal", trunk_experts=True,
                             router="ride", expert_loss="per_expert", top_k=8, pcgrad=False,
                             comp_loss="none", base_loss="balsoftmax"),

    # ---- concurrency probe: 3 epochs, just enough to prove a job STARTED on its own quota ----
    "Z_tiny": dict(preset="v2", arm="Z_tiny", epochs=3, pcgrad=False, comp_loss="none",
                   eval_every=99, eval_at_end=False, measure_flops=False, save_cost_trace=False),

    # ---- RIDE-core: a multi-expert long-tailed baseline ---------------------------------------
    # NOT the same as M_m8_bal. That arm trains a gate-weighted mixture on ONE blended logit, so its
    # members score 0.291 alone on the 100-way test set (max 0.466) against a dense head's 0.713 --
    # they are residual contributors, not classifiers -- and its align/comp losses are identically
    # zero, so it contains no GT3MoE content either. RIDE's defining property is that EVERY expert
    # carries its OWN full loss on EVERY sample at equal weight, with a uniform logit average at
    # inference. That is `expert_loss="per_expert"` + `router="ride"`.
    #
    # Prediction, pre-registered: a full 3-model DEEP ensemble (independent backbones AND heads)
    # buys only +1.46pp here (0.7109 -> 0.7255), and head-only experts share one nearly-frozen trunk
    # so they are far more correlated. Expect RIDE-core to land within about +-1pp of dense. The
    # value is the answer, not the win: if a named multi-expert LT method also fails to beat one
    # head, the finding generalises from "GT3MoE is bad" to "this regime cannot reward multi-expert
    # heads at all".
    #
    # Omitted from RIDE and to be stated as such in the paper: the distribution-aware diversity
    # loss, dynamic expert allocation, and self-distillation. Hence "RIDE-core".
    "M_ride_bal": dict(preset="v2", arm="M_ride_bal", router="ride", expert_loss="per_expert",
                       base_loss="balsoftmax", pcgrad=False, comp_loss="none", top_k=8),
    "M_ride_ce": dict(preset="v2", arm="M_ride_ce", router="ride", expert_loss="per_expert",
                      pcgrad=False, comp_loss="none", top_k=8),

    # ---- the baselines AS PUBLISHED ------------------------------------------------------
    # Switch and StableMoE are top-1 routers -- that is the defining claim of both papers, not an
    # implementation detail. Running them at top_k=2 turns Switch into Shazeer's top-k gating with
    # an aux loss, which is a different method. The R_* arms above hold top_k=2 so the expert-FLOP
    # budget matches GT3MoE; these hold the ROUTER faithful instead. The tension is not real: a
    # second expert head is 2.2 MFLOPs against a 1519 MFLOP forward, i.e. 0.14%, so neither choice
    # can be attacked on compute. Report both; cite the canonical ones.
    #
    # balance_alpha is the paper's 0.1 throughout, which is 10x Switch's own 0.01. GT3MoE ramps its
    # balance term DOWN after warmup while the baselines carry it at full strength for 60 epochs,
    # so "we used the same hyperparameter for everyone" is NOT the neutral choice it sounds like.
    # R_switch_canon uses 0.01; R_switch_top1 isolates the coefficient from the top-k change.
    "R_switch_canon": dict(preset="v2", arm="R_switch_canon", router="switch", pcgrad=False,
                           top_k=1, balance_alpha=0.01),
    "R_switch_top1": dict(preset="v2", arm="R_switch_top1", router="switch", pcgrad=False,
                          top_k=1),
    "R_vmoe_canon": dict(preset="v2", arm="R_vmoe_canon", router="vmoe", pcgrad=False,
                         balance_alpha=0.01),
    "R_stablemoe_top1": dict(preset="v2", arm="R_stablemoe_top1", router="stablemoe",
                             pcgrad=False, top_k=1),

    # ---- LT-LOSS ORTHOGONALITY: the paper claims it "likely compounds" with no experiment ----
    # Each LT loss is run BOTH on GT3MoE and on a dense head, same loss / backbone / seeds, so the
    # comparison is paired. The authors' own unreported run went the wrong way
    # (Dense LDAM-DRW 0.5826 vs Folk+LDAM-DRW 0.5617), which is what this settles.
    "T_gt3_bal": dict(preset="v2", arm="T_gt3_bal", pcgrad=False, comp_loss="none",
                      base_loss="balsoftmax"),
    "T_dense_bal": dict(preset="v2", arm="T_dense_bal", dense_head=True, pcgrad=False,
                        base_loss="balsoftmax"),
    "T_gt3_ldam": dict(preset="v2", arm="T_gt3_ldam", pcgrad=False, comp_loss="none",
                       base_loss="ldam_drw"),
    "T_dense_ldam": dict(preset="v2", arm="T_dense_ldam", dense_head=True, pcgrad=False,
                         base_loss="ldam_drw"),
    "T_gt3_cb": dict(preset="v2", arm="T_gt3_cb", pcgrad=False, comp_loss="none",
                     base_loss="cb_ce"),
    "T_dense_cb": dict(preset="v2", arm="T_dense_cb", dense_head=True, pcgrad=False,
                       base_loss="cb_ce"),

    # ---- sampler / batch ------------------------------------------------------------------
    "S_group_strat": dict(preset="v2", arm="S_group_strat", sampler="group_stratified"),
    "S_batch64": dict(preset="v2", arm="S_batch64", batch_size=64),
}

# --------------------------------------------------------- the assignment null, tuned per K
# Two objections to the K-crossover claim, answered with one grid.
#
#   1. Does the ASSIGNMENT do any work? ``random_balanced`` keeps GT3MoE's architecture, capacity
#      constraint and frozen-partition schedule and replaces only the competence-driven partition
#      with a random balanced one. If it matches GT3MoE, then the cost matrix, the exact LAP
#      solver and the whole game-theoretic apparatus buy nothing and the paper has no mechanism.
#      It has never been run above K=8.
#   2. Is Switch's degradation with K real, or just an untuned baseline? Switch's balance weight
#      was fixed at the paper's 0.1 (canonical is 0.01) at every K. If larger K simply needs more
#      balance pressure, GT3MoE's "win" at K=32 is an artefact of not tuning the baseline --
#      an identical tuning budget for the methods being compared.
#
# So all three methods get the SAME four-point sweep over balance_alpha at every K. Selection is
# on VALIDATION accuracy, reported on test, which is what makes the budgets comparable.
BAL_ALPHAS = (0.01, 0.03, 0.1, 0.3)
TUNE_KS = (8, 16, 32)


def _tune_grid():
    out = {}
    for K in TUNE_KS:
        for a in BAL_ALPHAS:
            sfx = f"K{K}_a{str(a).replace('.', '')}"
            base = dict(preset="v2", pcgrad=False, comp_loss="none", num_experts=K,
                        balance_alpha=a)
            out[f"U_gt3_{sfx}"] = dict(base, arm=f"U_gt3_{sfx}")
            # identical to U_gt3 except the partition is random-but-balanced
            out[f"U_rnd_{sfx}"] = dict(base, arm=f"U_rnd_{sfx}", assign_mode="random_balanced")
            # Switch as published: top-1
            out[f"U_swi_{sfx}"] = dict(base, arm=f"U_swi_{sfx}", router="switch", top_k=1)
    return out


ARMS.update(_tune_grid())
TUNE_ARMS = sorted(_tune_grid())


# --------------------------------------------------------- stage 1b (PREREGISTRATION.md A1.2-A1.4)
# The SWI alpha = 0.001 that stage 1 omitted, plus 1.0 past the upper edge Switch selected at K = 32.
# Each K carries an ANCHOR re-run of its stage-1 selection, all in ONE job, so a new alpha is only
# ever compared with the anchor inside the job (cross-job noise on 3-seed means is ~0.28pp).
# alpha = 0 is a diagnostic (does the balance loss cause the near-uniform gate?) and never selected.
U1B_ANCHOR = {8: 0.03, 16: 0.01, 32: 0.3}
U1B_NEW = (0.001, 1.0)
U1B_DIAG_KS = (8, 32)


def _stage1b_grid():
    out = {}
    base = dict(preset="v2", pcgrad=False, comp_loss="none", router="switch", top_k=1)
    for K in TUNE_KS:
        for a in U1B_NEW + (U1B_ANCHOR[K],):
            name = f"U1b_swi_K{K}_a{str(a).replace('.', '')}"
            out[name] = dict(base, arm=name, num_experts=K, balance_alpha=a)
    for K in U1B_DIAG_KS:
        name = f"U1b_swi_K{K}_a00"
        out[name] = dict(base, arm=name, num_experts=K, balance_alpha=0.0)
    return out


ARMS.update(_stage1b_grid())
U1B_ARMS = sorted(_stage1b_grid())
U1B_DIAGNOSTIC = sorted(f"U1b_swi_K{K}_a00" for K in U1B_DIAG_KS)


# --------------------------------------------------------- stage 2 and its follow-ups
# Alphas come from analysis/selected_alphas.json: "selected" (stage 1) for GT3MoE and RND, overridden
# by "final" once stage 1b has settled Switch. Until "final" exists every arm below is built from the
# stage-1 values and listed in PROVISIONAL_ARMS, which the launcher will dry-run but not launch.
# The container never imports this file (it reads resolved configs from arms.tsv), so a missing
# JSON there is harmless.
ALPHAS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "analysis", "selected_alphas.json")
PROVISIONAL_ARMS: set = set()


def _load_alphas():
    """Return ({"gt3_K32": 0.01, ...}, provisional) or (None, True) if the file is absent."""
    try:
        with open(ALPHAS_PATH) as f:
            sel = json.load(f)
    except (OSError, ValueError):
        return None, True
    alphas = {k: v["balance_alpha"] for k, v in sel.get("selected", {}).items()}
    final = sel.get("final") or {}
    alphas.update(final)
    return alphas, not final


def _stage2_grid(alphas):
    base = dict(preset="v2", pcgrad=False, comp_loss="none")

    def gt3(K, arm):
        return dict(base, arm=arm, num_experts=K, balance_alpha=alphas[f"gt3_K{K}"])

    def rnd(K, arm):
        return dict(base, arm=arm, num_experts=K, balance_alpha=alphas[f"rnd_K{K}"],
                    assign_mode="random_balanced")

    def swi(K, arm):
        return dict(base, arm=arm, num_experts=K, balance_alpha=alphas[f"swi_K{K}"],
                    router="switch", top_k=1)

    def ec(K, arm):
        # as R_expert_choice: its own top_k, and it does not read balance_alpha
        return dict(base, arm=arm, num_experts=K, router="expert_choice")

    def km(K, arm):
        return dict(gt3(K, arm), assign_mode="kmeans_only")

    def dense(arm):
        return dict(base, arm=arm, dense_head=True)

    out = {}
    for K in (16, 32):
        for tag, fn in (("gt3", gt3), ("rnd", rnd), ("swi", swi), ("ec", ec)):
            out[f"C2_{tag}_K{K}"] = fn(K, f"C2_{tag}_K{K}")
    out["C2_km_K32"] = km(32, "C2_km_K32")
    out["C2_dense"] = dense("C2_dense")
    # PREREGISTRATION.md A2.2 (exploratory) and A3.2 (H7)
    out["C2_gt3val_K32"] = dict(gt3(32, "C2_gt3val_K32"), probe_split="val")
    out["C2_gt3cong_K32"] = dict(gt3(32, "C2_gt3cong_K32"), congestion_lambda=0.02)

    # break test: balanced CIFAR-100 is set by the batch override, the arms are the K = 32 ones
    for tag, fn in (("gt3", gt3), ("rnd", rnd), ("swi", swi)):
        out[f"B_{tag}_K32"] = fn(32, f"B_{tag}_K32")
    out["B_dense"] = dense("B_dense")

    # replication: from-scratch Mini-ImageNet-LT is set by the batch override; alphas NOT re-tuned
    for tag, fn in (("gt3", gt3), ("rnd", rnd), ("swi", swi), ("ec", ec)):
        out[f"RM_{tag}_K32"] = fn(32, f"RM_{tag}_K32")
    out["RM_dense"] = dense("RM_dense")

    # sensitivity: one factor at a time from the stage-2 GT3MoE K = 32 config, with its own base arm
    # in the same job so every difference is within-job
    sens = {"base": {}, "delta05": dict(ema_delta=0.5), "delta09": dict(ema_delta=0.9),
            "warm2": dict(warmup_epochs=2), "warm10": dict(warmup_epochs=10),
            "ramp5": dict(ramp_epochs=5), "ramp20": dict(ramp_epochs=20), "top1": dict(top_k=1)}
    for tag, over in sens.items():
        out[f"SN_{tag}"] = dict(gt3(32, f"SN_{tag}"), **over)
    return out


def _paper_grid(alphas):
    """The uniform grid behind the paper's main table: every router at K = 8, 16, 32 on each corpus.

    GT3MoE uses lambda = 0: under the tight capacity bounds floor(C/K) <= n_i <= ceil(C/K) the
    congestion term is constant over feasible assignments, so lambda has no effect. The frozen
    k-means and random partitions are its ablations. Switch and the partitions use the
    validation-selected balance weights; the other routers keep their standard settings.
    """
    base = dict(preset="v2", pcgrad=False, comp_loss="none")
    out = {}
    for K in (8, 16, 32):
        a_gt3, a_rnd, a_swi = alphas[f"gt3_K{K}"], alphas[f"rnd_K{K}"], alphas[f"swi_K{K}"]
        arms = {
            "gt3": dict(balance_alpha=a_gt3),
            "km": dict(balance_alpha=a_gt3, assign_mode="kmeans_only"),          # pretrained trunk
            "kmw": dict(balance_alpha=a_gt3, assign_mode="kmeans_at_warmup"),    # from scratch
            "rnd": dict(balance_alpha=a_rnd, assign_mode="random_balanced"),
            "swi": dict(balance_alpha=a_swi, router="switch", top_k=1),
            "ec": dict(router="expert_choice"),
            "vmoe": dict(router="vmoe"),
            "stable": dict(router="stablemoe", top_k=1),
            "soft": dict(router="softmoe"),
            "ada": dict(router="adamoe", adamoe_null_experts=4),
            "dyn": dict(router="dynmoe", dynmoe_tau_init=0.0, dynmoe_sparsity_beta=0.01),
        }
        for tag, over in arms.items():
            name = f"P_{tag}_K{K}"
            out[name] = dict(base, arm=name, num_experts=K, **over)
    out["P_dense"] = dict(base, arm="P_dense", dense_head=True)
    # long-tailed losses across MoE routers (K = 32), dense head as a reference row
    for loss, tag in (("balsoftmax", "bal"), ("cb_ce", "cb"), ("ldam_drw", "ldam")):
        for r, over in (("gt3", dict(balance_alpha=alphas["gt3_K32"])),
                        ("swi", dict(balance_alpha=alphas["swi_K32"], router="switch", top_k=1)),
                        ("ec", dict(router="expert_choice"))):
            name = f"PL_{r}_{tag}_K32"
            out[name] = dict(base, arm=name, num_experts=32, base_loss=loss, **over)
        out[f"PL_dense_{tag}"] = dict(base, arm=f"PL_dense_{tag}", dense_head=True, base_loss=loss)
    return out


def paper_arms(K, scratch, with_dense):
    tags = ["gt3", "kmw" if scratch else "km", "rnd", "swi", "ec", "vmoe", "stable", "soft",
            "ada", "dyn"]
    return [f"P_{t}_K{K}" for t in tags] + (["P_dense"] if with_dense else [])


PL_ARMS = [f"PL_{r}_{t}_K32" for t in ("bal", "cb", "ldam") for r in ("gt3", "swi", "ec")] + \
    [f"PL_dense_{t}" for t in ("bal", "cb", "ldam")]

def _fix_grid(alphas):
    """Post-hoc variants after the 3-seed grid (2026-10-13), CIFAR-100-LT, with in-job references.

    (a) the experts learn the initial (k-means) partition before the first competence solve
        (``align_from_start``), and/or the solve keeps the incumbent unless another expert is better
        by more than 1 nat (``assign_hysteresis``);
    (b) Balanced Softmax on each selected expert's own logits rather than on the gated mixture.
    """
    base = dict(preset="v2", pcgrad=False, comp_loss="none")
    out = {}
    for K in (8, 32):
        g = dict(base, num_experts=K, balance_alpha=alphas[f"gt3_K{K}"])
        variants = {
            "gt3": {}, "km": dict(assign_mode="kmeans_only"),
            "afs": dict(align_from_start=True), "hys": dict(assign_hysteresis=1.0),
            "afshys": dict(align_from_start=True, assign_hysteresis=1.0),
        }
        if K == 32:
            bal = dict(base_loss="balsoftmax")
            variants.update({
                "gt3bal": bal, "kmbal": dict(bal, assign_mode="kmeans_only"),
                "afsbal": dict(bal, align_from_start=True),
                "pebal": dict(bal, expert_loss="per_expert_routed"),
                "afspebal": dict(bal, align_from_start=True, expert_loss="per_expert_routed"),
            })
        for tag, over in variants.items():
            name = f"FX_{tag}_K{K}"
            out[name] = dict(g, arm=name, **over)
    out["FX_swibal_K32"] = dict(base, arm="FX_swibal_K32", num_experts=32, router="switch", top_k=1,
                                balance_alpha=alphas["swi_K32"], base_loss="balsoftmax")
    return out


FX_ARMS = sorted(f"FX_{t}_K8" for t in ("gt3", "km", "afs", "hys", "afshys")) + \
    sorted(f"FX_{t}_K32" for t in ("gt3", "km", "afs", "hys", "afshys", "gt3bal", "kmbal", "afsbal",
                                   "pebal", "afspebal", "swibal"))

_ALPHAS, _PROVISIONAL = _load_alphas()
if _ALPHAS and not _PROVISIONAL:
    ARMS.update(_paper_grid(_ALPHAS))
    ARMS.update(_fix_grid(_ALPHAS))
_STAGE2 = _stage2_grid(_ALPHAS) if _ALPHAS else {}
ARMS.update(_STAGE2)
if _PROVISIONAL:
    PROVISIONAL_ARMS.update(_STAGE2)
C2_ARMS = [f"C2_{t}_K{K}" for K in (16, 32) for t in ("gt3", "rnd", "swi", "ec")] + \
    ["C2_km_K32", "C2_dense", "C2_gt3val_K32", "C2_gt3cong_K32"]
B_ARMS = ["B_gt3_K32", "B_rnd_K32", "B_swi_K32", "B_dense"]
RM_ARMS = [f"RM_{t}_K32" for t in ("gt3", "rnd", "swi", "ec")] + ["RM_dense"]
SN_ARMS = [f"SN_{t}" for t in ("base", "delta05", "delta09", "warm2", "warm10", "ramp5", "ramp20",
                               "top1")]

#: Minimum viable package: the four questions that decide whether the story is true.
MVP = ["E0_calib", "A_frozen_aug", "A_per_batch", "E1_frozen", "E2_kmeans_only",
       "E3_random_balanced", "E4b_reweight_only", "E4c_size_weighted", "E4d_random_groups",
       "E4e_no_pcgrad", "E5_train_topk", "E8_dense_head"]

#: K sweep for the headline cells only.
K_SWEEP = [8, 32]


def expand(names=None, seeds=None, ks=(8,)):
    """Yield (job_name, overrides) for every (arm, seed, K)."""
    names = names or MVP
    seeds = seeds or SEEDS
    for name in names:
        base = dict(BASE)
        base.update(ARMS[name])
        for K in ks:
            for s in seeds:
                o = dict(base)
                # An arm that names its own num_experts keeps it; otherwise the sweep sets it.
                # (This used to be an `if name == "E8_K1"` special case, which silently broke any
                # other arm whose whole point is a specific K.)
                o["num_experts"] = ARMS[name]["num_experts"] if "num_experts" in ARMS[name] else K
                o["seed"] = s
                if o.get("dense_head"):
                    o["num_experts"] = 1
                job = f"{name}-K{o['num_experts']}-s{s}"
                yield job, o
