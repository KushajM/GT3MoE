"""Launch the grid on SageMaker Training with managed spot.

    # dry run: print the plan and the cost estimate, launch nothing
    python sagemaker/launch.py --dry_run

    # calibration first -- ALWAYS do this before committing the grid
    python sagemaker/launch.py --arms E0_calib --seeds 0 --ks 8

    # the minimum viable package
    python sagemaker/launch.py --arms MVP --seeds 0,1,2 --ks 8 --spot

Instance choice (verify current pricing in the console before launching; SageMaker *Training*
rates run above the matching EC2 on-demand rate and vary by region):

    ml.g5.xlarge    1x A10G 24GB   4 vCPU    ~$1.52/hr   <-- recommended
    ml.g4dn.xlarge  1x T4   16GB   4 vCPU    ~$0.74/hr   no bf16; use --amp fp16
    ml.g6.xlarge    1x L4   24GB   4 vCPU    ~$1.13/hr   ~1.4x slower than A10G, worse per unit work

All are under the $3/hr ceiling. Managed spot typically saves 60-70% on g5.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms as ARMS  # noqa: E402

PRICE = {"ml.g5.xlarge": 1.515, "ml.g4dn.xlarge": 0.736, "ml.g6.xlarge": 1.127,
         "ml.g5.2xlarge": 1.69}

#: Wall-clock model, in g5.xlarge hours for a 60-epoch CIFAR-100-LT run. Derived from the
#: authors' RTX 4090 timings (GT3MoE 67.7 min at K=8, 158.7 min at K=32; dense 15.6 min) times a
#: 1.7x planning multiplier, then reduced for the v2 changes: the GPU-resident data pipeline, the
#: batched expert stack, channels_last + bf16, and eval_every=5 instead of 1.
#: These are ESTIMATES. Run E0_calib and rescale.
HOURS = {8: 1.25, 16: 2.1, 32: 2.9, 1: 0.45}


def estimate(jobs, instance):
    tot = 0.0
    for _name, o in jobs:
        K = 1 if o.get("dense_head") else int(o.get("num_experts", 8))
        h = HOURS.get(K, 1.5)
        if o.get("epochs", 60) != 60:
            h *= o["epochs"] / 60.0
        if not o.get("pcgrad", True):
            h *= 0.45                      # no per-group backward passes
        tot += h
    return tot, tot * PRICE[instance]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="MVP", help="'MVP', 'ALL', or a comma-separated arm list")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ks", default="8")
    ap.add_argument("--instance", default="ml.g5.xlarge", choices=sorted(PRICE))
    ap.add_argument("--spot", action="store_true", help="managed spot (recommended)")
    ap.add_argument("--max_run", type=int, default=6 * 3600)
    ap.add_argument("--role", default=os.environ.get("SAGEMAKER_ROLE_ARN", ""))
    ap.add_argument("--bucket", default=os.environ.get("GT3MOE_BUCKET", ""))
    ap.add_argument("--prefix", default="gt3moe-v2")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--framework_version", default="2.5.1")
    ap.add_argument("--py_version", default="py311")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    names = (ARMS.MVP if a.arms == "MVP" else
             sorted(ARMS.ARMS) if a.arms == "ALL" else
             [s.strip() for s in a.arms.split(",") if s.strip()])
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    ks = [int(s) for s in a.ks.split(",") if s.strip()]
    jobs = list(ARMS.expand(names, seeds, ks))
    if a.limit:
        jobs = jobs[:a.limit]

    hours, usd = estimate(jobs, a.instance)
    disc = 0.35 if a.spot else 1.0
    print(f"\n  {len(jobs)} jobs | {a.instance} @ ${PRICE[a.instance]:.3f}/hr"
          f"{' (spot, ~65% off)' if a.spot else ''}")
    print(f"  estimated {hours:.1f} GPU-hours, ${usd:.2f} on-demand, "
          f"~${usd * disc:.2f} on spot\n")
    for n, o in jobs:
        print(f"    {n:38s} K={o.get('num_experts')} seed={o.get('seed')} "
              f"arm={o.get('arm')}")
    if a.dry_run:
        print("\n  --dry_run: nothing launched.\n")
        return 0
    if not a.role or not a.bucket:
        print("\n  ERROR: --role and --bucket are required to launch "
              "(or set SAGEMAKER_ROLE_ARN / GT3MOE_BUCKET).\n")
        return 2

    import sagemaker
    from sagemaker.pytorch import PyTorch

    sess = sagemaker.Session()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    metric_defs = [{"Name": k, "Regex": rf"metric_{k}=([0-9.eE+-]+)"} for k in
                   ("acc", "many", "medium", "few", "f1", "acc_top1_only",
                    "acc_uniform_ensemble", "acc_oracle_routing", "router_group_acc",
                    "util_cv", "mean_routing_entropy", "billed_seconds", "backward_count")]

    for name, o in jobs:
        hp = {k: v for k, v in o.items() if k != "preset"}
        hp["preset"] = o.get("preset", "v2")
        job = f"{a.prefix}-{name}-{stamp}".replace("_", "-")[:62]
        # UNIQUE checkpoint prefix per job. Reusing one makes a fresh run resume stale state.
        ckpt = f"s3://{a.bucket}/{a.prefix}/{stamp}/{name}/checkpoints"
        est = PyTorch(
            entry_point="entrypoint.py",
            source_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            role=a.role,
            instance_type=a.instance,
            instance_count=1,
            framework_version=a.framework_version,
            py_version=a.py_version,
            hyperparameters=hp,
            environment={"GT3MOE_MAX_RUN": str(a.max_run)},
            checkpoint_s3_uri=ckpt,
            checkpoint_local_path="/opt/ml/checkpoints",
            output_path=f"s3://{a.bucket}/{a.prefix}/{stamp}/output",
            use_spot_instances=a.spot,
            max_run=a.max_run,
            max_wait=(3 * a.max_run) if a.spot else None,
            keep_alive_period_in_seconds=None,
            metric_definitions=metric_defs,
            sagemaker_session=sess,
            disable_profiler=True,
            debugger_hook_config=False,
        )
        est.fit(wait=False, job_name=job)
        print(f"  launched {job}  -> {ckpt}")
    print(f"\n  {len(jobs)} jobs launched. Watch with:  aws sagemaker "
          f"list-training-jobs --name-contains {a.prefix} --region {a.region}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
