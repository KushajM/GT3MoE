# GT3MoE: A Game-Theoretic Two-Timescale Routing for Mixture-of-Experts

Code, configurations and result files for the paper (under review).

GT3MoE is a class-level router for sparse mixture-of-experts (MoE) heads. On a slow, per-epoch
timescale it measures how well every expert classifies every class and assigns classes to experts
under capacity constraints, by solving a congestion game among the classes exactly as a single linear
assignment. On a fast, per-batch timescale it trains a standard sparse router, through an alignment
loss, to send each input to the expert that owns its class. The paper evaluates it against eight
sparse routers on CIFAR-100-LT, Mini-ImageNet-LT and Tiny-ImageNet-LT with 8, 16 and 32 experts.

## Repository layout

```
gt3moe/            the library
  model.py           backbone, expert heads, router
  routers.py         GT3MoE gating and the baselines (Switch, V-MoE, Expert Choice, StableMoE,
                     Soft-attention MoE, AdaMoE, DynMoE)
  assign.py          the slow step: exact capacity-constrained assignment (Hungarian), greedy
                     baseline, k-means and random partitions
  costs.py           the per-expert, per-class competence probe and its EMA
  engine.py          the two-timescale training loop
  evaluate.py        accuracy, per-shot accuracy and routing diagnostics
  data.py            long-tailed CIFAR-100, Mini-ImageNet and Tiny-ImageNet
  config.py, cli.py  every setting is a Config field and a command-line flag
sagemaker/
  arms.py            every configuration ("arm") used in the paper
  launch_boto3.py    the batches (dataset, schedule, seeds) and the SageMaker launcher
  run.sh, entrypoint.py   container side
run_arm.py         run any (batch, arm, seed) of the paper locally
analysis/
  data/              one result JSON per training run (see below)
  make_paper_artifacts.py   every table and figure of the paper from analysis/data
  make_overview_figure.py   Figure 1
  pt1_solver_replay.py      exact-versus-greedy assignment, replayed from saved competence traces
  select_alpha.py           balance-weight selection on validation
  selected_alphas.json      the selected weights
  paper_numbers.json        every number quoted in the paper text
tests/             invariants of the assignment step, the routers and checkpoint resume
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The paper's runs used one GPU per run (NVIDIA A10G, L4 or L40S) with bf16 autocast.

## Data

Datasets are read from `--data_root` (default `./data`).

- **CIFAR-100-LT** and the balanced **CIFAR-100** control: downloaded automatically by torchvision.
- **Tiny-ImageNet-LT**: place the official `tiny-imagenet-200.zip` in the data root. Its labelled
  validation split (50 images per class) is used as the test set.
- **Mini-ImageNet-LT**: place the `train-*.parquet` and `validation-*.parquet` shards of the
  `timm/mini-imagenet` mirror in `<data_root>/mini/`. This mirror splits all 100 classes by image
  (500 training and 100 held-out images per class); the held-out images are the test set.

Every benchmark is built the same way: 15% of each class's training images (75) are held out as a
balanced validation set, the remaining 425 per class are subsampled with the exponential profile
n_c = floor(425 * 100^(-c/(C-1))), and the test set stays balanced. All images are resized to 96x96.
The first run on each dataset writes a cached `.npz` of the split, which later runs reuse.

## Reproducing a run

A *batch* fixes the dataset, schedule and seeds; an *arm* fixes the router. `run_arm.py` expands a
batch and runs the arm and seed you choose, with exactly the configuration of the paper's runs.

```bash
python run_arm.py --batch P_c100lt_K32 --list                    # the runs in a batch
python run_arm.py --batch P_c100lt_K32 --arm P_gt3_K32 --seed 20 # GT3MoE, CIFAR-100-LT, K=32
python run_arm.py --batch P_tiny_K32   --arm P_swi_K32 --seed 21 # Switch, Tiny-ImageNet-LT
```

| Batch | Paper |
|---|---|
| `P_c100lt_K8`, `P_c100lt_K16`, `P_c100lt_K32` | CIFAR-100-LT, fine-tuned (main table, partitions, diagnostics) |
| `P_mini_K8`, `P_mini_K16`, `P_mini_K32` | Mini-ImageNet-LT, from scratch |
| `P_tiny_K8`, `P_tiny_K16`, `P_tiny_K32` | Tiny-ImageNet-LT, from scratch |
| `P_ltloss` | long-tailed losses at K=32 |
| `B_break` | balanced CIFAR-100 control (the paper reports seeds 6-8) |
| `U_tune`, `U_tune1b` | balance-weight selection on validation (seeds 3-5) |
| `C2_confirm` | held-out competence probe (`C2_gt3val_K32` vs `C2_gt3_K32`) |

Arm names: `P_gt3` GT3MoE, `P_km` fixed k-means partition, `P_kmw` k-means at the end of warm-up
(from scratch), `P_rnd` fixed random partition, `P_swi` Switch, `P_vmoe` V-MoE, `P_ec` Expert
Choice, `P_stable` StableMoE, `P_soft` Soft-attention MoE, `P_ada` AdaMoE, `P_dyn` DynMoE,
`P_dense` dense head.

Any single setting can also be changed directly, for example
`python -m gt3moe.cli --preset v2 --dataset cifar100lt --num_experts 16 --router gt3moe --seed 0`.

## Running the grid on SageMaker

```bash
export AWS_PROFILE=...                      # optional
export AWS_REGION=us-west-2
export GT3MOE_S3_BUCKET=...                 # payloads, dataset caches, checkpoints
export GT3MOE_SAGEMAKER_ROLE_ARN=arn:aws:iam::<account>:role/...
python sagemaker/launch_boto3.py --batch P_c100lt_K32 --dry_run
python sagemaker/launch_boto3.py --batch P_c100lt_K32 --instance ml.g5.2xlarge
```

All arms of a batch run sequentially in one job on one GPU, so that paired comparisons share data
order, initialisation and hardware. The Mini- and Tiny-ImageNet batches expect their cached `.npz`
splits in the bucket; the launcher prints the upload command if one is missing.

## Reproducing the paper's tables and figures

The result files of every run in the paper are in `analysis/data`, so no training is needed:

```bash
python analysis/make_paper_artifacts.py    # tables/*.tex, figures/*.pdf, analysis/paper_numbers.json
python analysis/make_overview_figure.py    # Figure 1
python analysis/pt1_solver_replay.py       # exact vs greedy (analysis/paper_tables/solver_replay.csv)
python analysis/select_alpha.py            # balance-weight selection
```

Outputs go to `./paper_artifacts` (override with `GT3MOE_PAPER_DIR`).

## Result files

Each run writes one JSON file. Its main fields are:

- `config`: every setting of the run, including the arm, seed, dataset and class counts;
- `test` and `val_final`: overall, many-, medium- and few-shot accuracy, macro-F1, routing
  entropy, expert--class entropy, expert utilisation, the class-by-expert routing affinity, each
  expert's standalone accuracy on every class (`competence_acc`), and, for GT3MoE, how often the
  router selects the owning expert (`router_group_acc`);
- `history`: per-epoch training losses, validation metrics and slow-loop statistics (potential,
  number of reassigned classes);
- `assignment`: the final class-to-expert assignment;
- `precision`, `params`, `flops`, `billed_seconds`: hardware, model size and cost.

`analysis/data/_costtrace` holds the per-epoch competence matrices used by the exact-versus-greedy
replay.

## Tests

```bash
pip install pytest
pytest tests
```

## License

Apache License 2.0; see `LICENSE`.
