"""Run one configuration from the paper locally, on a single GPU, exactly as the SageMaker jobs did.

A batch (sagemaker/launch_boto3.py: BATCHES) fixes the dataset, schedule and seeds; an arm
(sagemaker/arms.py: ARMS) fixes the router and its settings. This script expands a batch into its
(arm, K, seed) runs and executes the one you ask for through gt3moe.cli.

    python run_arm.py --batch P_c100lt_K32 --list
    python run_arm.py --batch P_c100lt_K32 --arm P_gt3_K32 --seed 20
    python run_arm.py --batch P_tiny_K32 --arm P_swi_K32 --seed 21 --data_root ./data
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sagemaker"))

import launch_boto3 as L  # noqa: E402
from gt3moe.cli import main as cli_main  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, choices=sorted(L.BATCHES))
    ap.add_argument("--arm", help="arm name, e.g. P_gt3_K32; omit to list the batch")
    ap.add_argument("--seed", type=int, help="one of the batch's seeds")
    ap.add_argument("--list", action="store_true", help="list the runs in the batch and exit")
    ap.add_argument("--data_root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--ckpt_dir", default=os.path.join(ROOT, "ckpt"))
    a = ap.parse_args()

    batch = dict(L.BATCHES[a.batch])
    if a.seed is not None:
        batch["seeds"] = [a.seed]
    runs = [(name, json.loads(cfg)) for name, cfg in L.arm_specs(batch)]

    if a.list or not a.arm:
        for name, cfg in runs:
            print(f"{cfg.get('arm', '?'):24s} K={cfg.get('num_experts')}  seed={cfg.get('seed')}  ({name})")
        return 0

    picked = [(n, c) for n, c in runs if c.get("arm") == a.arm]
    if not picked:
        print(f"arm {a.arm!r} is not in batch {a.batch!r}; use --list to see its runs")
        return 2
    for name, cfg in picked:
        ckpt = os.path.join(a.ckpt_dir, name)
        os.makedirs(ckpt, exist_ok=True)
        os.makedirs(a.out_dir, exist_ok=True)
        print(f"=== {name}")
        cli_main(["--config_json", json.dumps(cfg), "--ckpt_dir", ckpt, "--out_dir", a.out_dir,
                  "--data_root", a.data_root])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
