#!/usr/bin/env bash
set -euo pipefail

# Compare NDR between CRC and PIR under matched (dataset, backbone, pred_len).
# Hyperparameters are delegated to:
#   - run_crc_batch.sh (CRC)
#   - run_pir_batch.sh (PIR)
#
# Usage:
#   bash run_ndr_compare_crc_pir.sh --dataset Electricity --backbone TimesNet
#   bash run_ndr_compare_crc_pir.sh --datasets Electricity,Weather --backbones TimesNet,PatchTST
#   bash run_ndr_compare_crc_pir.sh --datasets Electricity --backbones TimesNet --pred_len 96,192,336,720
#
# Output:
#   - Runs CRC and PIR (separately) and prints a summary table of NDR.

# ------------------ defaults (edit here) ------------------
# You can edit these arrays directly to control what runs by default.
# CLI args (--datasets/--backbones/--pred_len) can still override them.
DATASETS=(ETTm2)
BACKBONES=(TimesNet TimeXer PatchTST DLinear)
PRED_LENS=(192 336 720)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset|--datasets)
      shift
      IFS=',' read -r -a DATASETS <<< "${1:-}"
      ;;
    --backbone|--backbones|--baseline|--baselines)
      shift
      IFS=',' read -r -a BACKBONES <<< "${1:-}"
      ;;
    --pred_len|--pred_lens)
      shift
      IFS=',' read -r -a PRED_LENS <<< "${1:-}"
      ;;
    -h|--help)
      echo "Usage: bash run_ndr_compare_crc_pir.sh --datasets <DS1,DS2,...> --backbones <B1,B2,...> [--pred_len 96,192,336,720]"
      exit 0;;
  esac
  shift
done

if [[ ${#DATASETS[@]} -eq 0 || ${#BACKBONES[@]} -eq 0 ]]; then
  echo "Error: DATASETS/BACKBONES are empty. Edit defaults or pass --datasets/--backbones." >&2
  exit 1
fi

# Default horizons if not specified
if [[ ${#PRED_LENS[@]} -eq 0 ]]; then
  PRED_LENS=(96 192 336 720)
fi

TS="$(date +'%Y%m%d_%H%M%S')"
ROOT_LOG_DIR="./logs/ndr_compare_${TS}"
mkdir -p "${ROOT_LOG_DIR}"

declare -A CRC_NDR
declare -A PIR_NDR

for ds in "${DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    for pl in "${PRED_LENS[@]}"; do
      tag="${ds,,}_${bb,,}_pl${pl}"
      # Use a single LOG_DIR so all runs append to one master.log.
      # Individual per-run logs (tagged .log files) still exist, but are colocated.
      LOG_DIR="${ROOT_LOG_DIR}"

      echo "[${TS}] Running CRC (dataset=${ds}, baseline=${bb}, pred_len=${pl})"
      bash run_crc_batch.sh \
        --datasets "${ds}" \
        --baselines "${bb}" \
        --pred_lens "${pl}" \
        --log_dir "${LOG_DIR}"

      # Extract the latest NDR after CRC finishes (CRC prints NDR during test()).
      CRC_NDR["${tag}"]="$(grep -aE "NDR \\(非恶化率\\):" "${LOG_DIR}/master.log" | tail -n 1 | sed -E 's/.*NDR \\(非恶化率\\):\\s*([0-9.]+)%.*/\\1/' || true)"

      echo "[${TS}] Running PIR (dataset=${ds}, backbone=${bb}, pred_len=${pl})"
      bash run_pir_batch.sh \
        --datasets "${ds}" \
        --baselines "${bb}" \
        --pred_lens "${pl}" \
        --log_dir "${LOG_DIR}"

      # Extract the latest NDR after PIR finishes.
      PIR_NDR["${tag}"]="$(grep -aE "NDR \\(非恶化率\\):" "${LOG_DIR}/master.log" | tail -n 1 | sed -E 's/.*NDR \\(非恶化率\\):\\s*([0-9.]+)%.*/\\1/' || true)"
    done
  done
done

echo
echo "=== NDR summary (CRC vs PIR) ==="
printf "%-12s | %-10s | %-8s | %-10s | %-10s\n" "dataset" "backbone" "pred_len" "CRC_NDR(%)" "PIR_NDR(%)"
for ds in "${DATASETS[@]}"; do
  for bb in "${BACKBONES[@]}"; do
    for pl in "${PRED_LENS[@]}"; do
      tag="${ds,,}_${bb,,}_pl${pl}"
      printf "%-12s | %-10s | %-8s | %-10s | %-10s\n" "${ds}" "${bb}" "${pl}" "${CRC_NDR[${tag}]:-NA}" "${PIR_NDR[${tag}]:-NA}"
    done
  done
done
echo
echo "Logs saved under: ${ROOT_LOG_DIR}"
echo "Master log: ${ROOT_LOG_DIR}/master.log"

