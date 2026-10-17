#!/bin/bash
# SageMaker container entrypoint for GT3MoE v2.
#
# GPU quota on this account is ONE training instance, so a batch of arms runs sequentially inside
# a single job rather than as parallel jobs. Each arm checkpoints independently under
# /opt/ml/checkpoints (S3-synced), so a spot interruption costs at most one epoch of one arm and
# the batch resumes where it stopped.
#
# Trap this file exists to avoid: `python ... | tee log | tail -n` reports SUCCESS on a crash,
# because tail's exit code wins the pipeline. Every invocation below checks PIPESTATUS[0].

set -uo pipefail

DATA_DIR=/opt/ml/input/data/data
CKPT=/opt/ml/checkpoints
OUT=/opt/ml/output/data
MODEL=/opt/ml/model
mkdir -p "$CKPT" "$OUT" "$MODEL" /opt/ml/data /opt/ml/code_src

echo "=== GT3MoE v2 container start: $(date -u +%FT%TZ) ==="
nvidia-smi || true
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# Seed the dataset from the channel. torchvision verifies the md5 and skips the download, so no
# job depends on cs.toronto.edu -- which is what burned the first calibration run (30 min of a
# 30 min budget spent on the download).
if [ -f "$DATA_DIR/cifar-100-python.tar.gz" ]; then
  echo "--- seeding CIFAR-100 from channel ---"
  cp "$DATA_DIR/cifar-100-python.tar.gz" /opt/ml/data/
fi

# Tiny-ImageNet / MiniImageNet arrive as prebuilt .npz caches: decoding 110k JPEGs out of a zip
# or 6.4 GB of parquet would burn 10+ minutes of GPU time per job for a byte-identical result,
# and it would also make the long-tail draw depend on the container's library versions.
if ls "$DATA_DIR"/*.npz >/dev/null 2>&1; then
  echo "--- seeding prebuilt dataset caches from channel ---"
  cp "$DATA_DIR"/*.npz /opt/ml/data/
fi
ls -la /opt/ml/data/

cd "$DATA_DIR"
if [ -f payload.tar.gz ]; then
  echo "--- unpacking payload ---"
  tar xzf payload.tar.gz -C /opt/ml/code_src
fi
cd /opt/ml/code_src

echo "--- deps ---"
pip install --quiet --no-input "scipy>=1.11" "scikit-learn>=1.3" 2>&1 | tail -3

# Arms come from a TSV in the channel (name <TAB> json), NOT an env var: SageMaker caps each
# Environment map value at 512 characters, which a batch of six 60-epoch arms blows past.
ARMS_TSV="$DATA_DIR/arms.tsv"
if [ ! -s "$ARMS_TSV" ]; then
  echo "FATAL: $ARMS_TSV missing or empty"; exit 2
fi
echo "--- $(wc -l < "$ARMS_TSV") arms queued ---"

FAILED=0
while IFS=$'\t' read -r NAME CFG; do
  [ -z "$NAME" ] && continue
  LOG="$OUT/${NAME}.log"
  echo ""
  echo "############################################################"
  echo "#  ARM ${NAME}   $(date -u +%FT%TZ)"
  echo "############################################################"
  # Per-arm checkpoint subdirectory: reusing one prefix across arms would make a fresh arm
  # silently resume the previous arm's state.
  ARM_CKPT="$CKPT/$NAME"
  mkdir -p "$ARM_CKPT"

  # </dev/null is load-bearing: inside a `while read` loop whose stdin is the arm list, a child
  # that reads stdin would swallow the remaining arms and silently shorten the batch.
  python -u -m gt3moe.cli \
      --config_json "$CFG" \
      --ckpt_dir "$ARM_CKPT" \
      --out_dir "$OUT" \
      --data_root /opt/ml/data \
      < /dev/null 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}          # <-- tee must not mask a crash
  if [ "$RC" -ne 0 ]; then
    echo "!!! ARM ${NAME} FAILED rc=${RC}"
    FAILED=$((FAILED+1))
  else
    echo "--- ARM ${NAME} ok ---"
  fi
  # copy results out per arm, so a later crash cannot lose earlier arms
  cp -f "$OUT"/*.json "$ARM_CKPT"/ 2>/dev/null || true
  cp -f "$LOG" "$ARM_CKPT"/ 2>/dev/null || true
done < "$ARMS_TSV"

cp -rf "$OUT"/. "$MODEL"/ 2>/dev/null || true
echo ""
echo "=== done: $(date -u +%FT%TZ)  failed_arms=${FAILED} ==="
exit "$FAILED"
