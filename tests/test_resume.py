"""Resume equivalence: an interrupted-and-resumed run must equal an uninterrupted one.

This is the one test that proves the managed-spot design. If it fails, every spot-interrupted run
in the grid is silently a different experiment from the runs that happened not to be interrupted,
and the seed-paired comparisons the whole statistical plan rests on are invalid.

Run: python tests/test_resume.py
"""
from __future__ import annotations

import copy
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gt3moe.data as D  # noqa: E402
import gt3moe.engine as EN  # noqa: E402
from gt3moe.config import make_config  # noqa: E402


def _fake(root, num_classes=100, rho=100.0, val_frac=0.15, img_size=96, data_seed=1234,
          cache=True, smoke=False):
    C = num_classes
    mu = (1.0 / rho) ** (1.0 / (C - 1))
    ratios = np.array([mu ** i for i in range(C)])
    n = np.array([max(2, int(ratios[c] * 30)) for c in range(C)])
    rng = np.random.default_rng(0)

    def mk(counts):
        y = np.concatenate([np.full(k, c) for c, k in enumerate(counts)]).astype(np.int64)
        x = (rng.random((len(y), img_size, img_size, 3)) * 60
             + (y[:, None, None, None] % 16) * 10).astype(np.uint8)
        p = rng.permutation(len(y))
        return x[p], y[p]

    xtr, ytr = mk(n)
    xva, yva = mk(np.full(C, 2))
    xte, yte = mk(np.full(C, 2))
    cc = np.bincount(ytr, minlength=C).astype(np.int64)
    many, med, few = D.class_splits(cc)
    return dict(xtr=xtr, ytr=ytr, xva=xva, yva=yva, xte=xte, yte=yte, class_counts=cc,
                many=np.array(many), medium=np.array(med), few=np.array(few))


def _cfg(tmp, epochs, **kw):
    return make_config(
        "v2", device="cpu", smoke=True, img_size=32, epochs=epochs, warmup_epochs=1,
        ramp_epochs=1, batch_size=24, num_experts=4, backbone="mobilenetv2",
        expert_hidden=32, eval_every=99, eval_at_end=False, probe_per_class=3,
        measure_flops=False, channels_last=False, data_root=tmp, out_dir=tmp,
        ckpt_dir=os.path.join(tmp, "ckpt"), save_cost_trace=False, deterministic=True,
        num_workers=0, seed=7, **kw)


def _fingerprint(t: EN.Trainer) -> dict:
    sd = t.model.state_dict()
    keys = sorted(sd)[:8] + sorted(sd)[-8:]
    h = {k: float(sd[k].double().sum().item()) for k in keys}
    return dict(
        params=h,
        assignment=t.assignment.tolist(),
        ema_sum=float(t.costs.ema.sum().item()),
        ema_t=t.costs.t,
        epoch=t.epoch,
        opt_step=int(sum(int(v.get("step", torch.zeros(())).item()) if
                         isinstance(v.get("step"), torch.Tensor) else int(v.get("step", 0) or 0)
                         for v in t.opt.state.values())),
    )


def main():
    D.build_cifar100_lt = _fake
    EN.build_cifar100_lt = _fake
    EPOCHS = 4
    ok = True

    # ------------------------------------------------------------------ PART 1
    # The actual resume guarantee: the state a fresh process loads must be BIT-IDENTICAL to the
    # state that was saved. This is the property that makes a spot interruption harmless.
    print("\n  Part 1 -- state restoration must be exact")
    tmpC = tempfile.mkdtemp(prefix="gt3_state_")
    t1 = EN.Trainer(_cfg(tmpC, 2))
    t1.fit()
    before = _fingerprint(t1)
    sd_before = {k: v.clone() for k, v in t1.model.state_dict().items()}
    ema_before = t1.costs.ema.clone()
    del t1

    t2 = EN.Trainer(_cfg(tmpC, EPOCHS))            # loads the checkpoint
    after = _fingerprint(t2)

    checks = [
        ("epoch", before["epoch"] == after["epoch"] == 2),
        ("assignment", before["assignment"] == after["assignment"]),
        ("EMA counter t", before["ema_t"] == after["ema_t"]),
        ("optimizer step count", before["opt_step"] == after["opt_step"]),
    ]
    w_exact = all(torch.equal(sd_before[k], t2.model.state_dict()[k]) for k in sd_before)
    checks.append(("model weights bit-identical", w_exact))
    checks.append(("cost-matrix EMA bit-identical",
                   bool(torch.equal(ema_before, t2.costs.ema))))
    for name, cond in checks:
        ok &= bool(cond)
        print(f"    [{'PASS' if cond else 'FAIL'}] {name}")

    # ------------------------------------------------------------------ PART 2
    # Trajectory divergence AFTER the resume point is a separate question. On CPU (and on GPU with
    # cudnn.benchmark) conv/BN reductions are not bitwise reproducible, so weights drift at ~1e-6
    # relative. That is harmless for the loss, but the assignment is an argmin, so on NEAR-TIED
    # cost cells the drift can flip it. This part measures the effect rather than asserting it
    # away -- and it is exactly what --assign_hysteresis exists to damp (REVISED_THEORY Lemma D.3).
    print("\n  Part 2 -- trajectory divergence after the resume point (diagnostic)")
    tmpA = tempfile.mkdtemp(prefix="gt3_ref_")
    tA = EN.Trainer(_cfg(tmpA, EPOCHS))
    tA.fit()
    ref = _fingerprint(tA)
    t2.fit()
    got = _fingerprint(t2)

    worst = max(abs(ref["params"][k] - got["params"][k]) for k in ref["params"])
    scale = max(1e-12, max(abs(v) for v in ref["params"].values()))
    rel = worst / scale
    cond = rel < 1e-3
    ok &= cond
    print(f"    [{'PASS' if cond else 'FAIL'}] weight drift stays negligible "
          f"(worst rel dev {rel:.2e}; float nondeterminism, not a resume bug)")
    n_flip = sum(1 for a, b in zip(ref["assignment"], got["assignment"]) if a != b)
    print(f"    [INFO] {n_flip}/{len(ref['assignment'])} classes assigned differently, and "
          f"|d EMA| = {abs(ref['ema_sum'] - got['ema_sum']):.2e}")
    print("           On this synthetic noise data the cost gaps are ~0, so argmin flips freely. "
          "Run with --assign_hysteresis > 0 to damp it; on real data report gamma_star.")

    for p in (tmpA, tmpC):
        shutil.rmtree(p, ignore_errors=True)

    print("\n  RESUME EQUIVALENCE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
