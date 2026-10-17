"""SageMaker Training entry point.

Run one (arm, seed, K) per job. One job per run beats an inner arm loop here, and not because of
per-job overhead: with managed spot, a single long job that is interrupted late loses the whole
tail, whereas independent jobs lose at most one run, and each job's checkpoint prefix stays clean.

Paths that matter:
  * ``/opt/ml/checkpoints``  -- continuously synced to ``checkpoint_s3_uri``. Everything you want
    to survive a spot interruption goes here, INCLUDING the results JSON. ``/opt/ml/model`` is only
    uploaded on clean exit, so a budget-exit or an interruption loses it.
  * ``/opt/ml/output/data``  -- uploaded on clean exit; fine for a duplicate copy.
  * ``SM_CHANNEL_TRAIN``     -- the input channel, if the dataset is staged from S3 rather than
    downloaded. CIFAR-100 is 170 MB so downloading in-container is simpler and is the default.

Gotchas encoded below:
  * ``checkpoint_s3_uri`` MUST be unique per job. Reusing a prefix across jobs makes a fresh run
    silently resume someone else's state, which is the single easiest way to corrupt a grid.
  * Do not rely on the SIGTERM handler under the framework estimator; the 120-second warning is
    not guaranteed to reach the training process. The real protection is per-epoch atomic
    checkpointing plus the wall-clock budget guard, both of which are in ``engine.py``.
  * ``budget_seconds`` is derived from ``max_run`` minus a margin, and the trainer counts
    *cumulative billed seconds across restarts*, not process uptime, so a job that is interrupted
    three times still stops before it is billed past the cap.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gt3moe.cli import main as cli_main   # noqa: E402


def _hyperparameters() -> dict:
    """SageMaker writes hyperparameters to a JSON file and also passes them as CLI args."""
    p = "/opt/ml/input/config/hyperparameters.json"
    if os.path.exists(p):
        with open(p) as f:
            raw = json.load(f)
        out = {}
        for k, v in raw.items():
            if isinstance(v, str):
                s = v.strip()
                if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
                    s = s[1:-1]
                out[k] = s
            else:
                out[k] = v
        return out
    return {}


def main():
    hp = _hyperparameters()
    ckpt = "/opt/ml/checkpoints" if os.path.isdir("/opt/ml") else "./ckpt"
    outd = "/opt/ml/output/data" if os.path.isdir("/opt/ml/output") else "./outputs"
    data = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train") \
        if os.path.isdir("/opt/ml") else "./data"
    os.makedirs(ckpt, exist_ok=True)
    os.makedirs(outd, exist_ok=True)
    os.makedirs(data, exist_ok=True)

    argv = []
    preset = hp.pop("preset", "v2")
    argv += ["--preset", str(preset)]
    hp.setdefault("ckpt_dir", ckpt)
    hp.setdefault("out_dir", outd)
    hp.setdefault("data_root", data)
    # max_run comes through as an env var we set at launch; leave 8 minutes of headroom for the
    # final evaluation pass, the FLOPs measurement and the results write-out.
    if "budget_seconds" not in hp:
        mr = float(os.environ.get("GT3MOE_MAX_RUN", "0") or 0)
        if mr > 0:
            hp["budget_seconds"] = max(600.0, mr - 480.0)
    for k, v in hp.items():
        argv += [f"--{k}", str(v)]

    print("[entrypoint] argv:", " ".join(argv), flush=True)
    res = cli_main(argv)

    # Emit the primary metrics in a form the CloudWatch metric_definitions regexes can scrape.
    if isinstance(res, dict) and "test" in res:
        t = res["test"]
        for k in ("acc", "many", "medium", "few", "f1", "acc_top1_only",
                  "acc_uniform_ensemble", "acc_oracle_routing", "router_group_acc",
                  "util_cv", "mean_routing_entropy"):
            if k in t and isinstance(t[k], (int, float)):
                print(f"metric_{k}={t[k]:.6f}", flush=True)
        print(f"metric_billed_seconds={res.get('billed_seconds', 0):.1f}", flush=True)
        print(f"metric_backward_count={res.get('backward_count', 0)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
