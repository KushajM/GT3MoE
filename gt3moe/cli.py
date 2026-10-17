"""Command-line entry point.

Every experiment arm in the plan is reachable without editing code:

    # the paper, reproduced (including its defects)
    python -m gt3moe.cli --preset paper_v1 --num_experts 8 --seed 0

    # v2 defaults
    python -m gt3moe.cli --preset v2 --num_experts 8 --seed 0

    # E1  the decisive control: assignment frozen at the end of warmup
    python -m gt3moe.cli --arm E1_frozen --assign_mode frozen_at --assign_freeze_epoch 5

    # E4  PCGrad decomposition, one variable per arm
    python -m gt3moe.cli --arm E4b --pcgrad_project false --backbone_ce group_mean
    python -m gt3moe.cli --arm E4c --pcgrad_project true  --backbone_ce size_weighted
    python -m gt3moe.cli --arm E4d --pcgrad_groups random

    # E8  dense single-head control on the identical numerical path
    python -m gt3moe.cli --arm E8_dense --dense_head true
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields

from .config import PRESETS, Config, make_config


def _coerce(v: str, typ):
    if typ is bool:
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")
    if typ is int:
        return int(v)
    if typ is float:
        return float(v)
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("gt3moe", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="v2", choices=sorted(PRESETS))
    p.add_argument("--config_json", default="", help="JSON file or literal with overrides")
    skip = {"class_counts", "many", "medium", "few", "feat_dim", "git_sha", "pkg_versions"}
    for f in fields(Config):
        if f.name in skip:
            continue
        typ = f.type if not isinstance(f.type, str) else None
        # resolve the annotation for the simple cases we care about
        default = getattr(Config(), f.name)
        base = type(default) if default is not None else str
        if base not in (bool, int, float, str):
            continue
        p.add_argument(f"--{f.name}", default=None, type=str,
                       help=f"({base.__name__}, default {default!r})")
    return p


def config_from_args(argv=None) -> Config:
    p = build_parser()
    ns = p.parse_args(argv)
    overrides = {}
    preset = ns.preset
    if ns.config_json:
        blob = ns.config_json
        if os.path.exists(blob):
            blob = open(blob).read()
        loaded = json.loads(blob)
        # allow the preset to travel inside the JSON, so a batch runner can ship one string per arm
        preset = loaded.pop("preset", preset)
        overrides.update(loaded)
    proto = Config()
    for k, v in vars(ns).items():
        if k in ("preset", "config_json") or v is None:
            continue
        cur = getattr(proto, k)
        overrides[k] = _coerce(v, type(cur) if cur is not None else str)
    return make_config(preset, **overrides)


def main(argv=None):
    from .engine import Trainer
    cfg = config_from_args(argv)
    print("=" * 78)
    print(f"  GT3MoE v2 | arm={cfg.arm} | K={cfg.num_experts} | seed={cfg.seed} "
          f"| {cfg.epochs}ep | hash={cfg.config_hash()}")
    print(f"  amp={cfg.amp} solver={cfg.assign_solver} cost={cfg.cost_source} "
          f"impute={cfg.impute} augment={cfg.augment} pcgrad={cfg.pcgrad}/{cfg.pcgrad_impl}")
    print("=" * 78, flush=True)
    t = Trainer(cfg)
    res = t.fit()
    if "test" in res:
        m = res["test"]
        print(f"\n  TEST acc={m['acc']:.4f} many={m['many']:.4f} med={m['medium']:.4f} "
              f"few={m['few']:.4f} f1={m['f1']:.4f}")
        print(f"       top1_only={m.get('acc_top1_only', float('nan')):.4f} "
              f"uniform_ens={m.get('acc_uniform_ensemble', float('nan')):.4f} "
              f"oracle_route={m.get('acc_oracle_routing', float('nan')):.4f} "
              f"router_group_acc={m.get('router_group_acc', float('nan')):.4f}")
        print(f"       util_cv={m.get('util_cv', float('nan')):.4f} "
              f"dead={m.get('dead_experts_abs1pct')} "
              f"rEnt={m.get('mean_routing_entropy', float('nan')):.3f}")
    print(f"  billed_seconds={res['billed_seconds']:.0f} backwards={res['backward_count']}")
    return res


if __name__ == "__main__":
    main()
