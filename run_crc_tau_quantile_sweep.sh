#!/usr/bin/env bash
set -euo pipefail

# Sweep --crc_tau_quantile at training time (stage-2 retrain for each value).
# Replaces deprecated run_crc_tau_scale_sweep.sh (Phase 4 / A6).

DATASET="ETTh1"
ROOT_PATH="./dataset/ETT-small/"
DATA_PATH="ETTh1.csv"
MODEL="CRC"
BASELINE="DLinear"
PRED_LEN=96
DMODEL=512
CRC_ENCODER_LAYERS=3
CRC_VAL_RATIO=1.0
TAU_QUANTILES=(0.10 0.30 0.50 0.70 0.90)

log_ts() { date '+%Y-%m-%d %H:%M:%S'; }

LOG_DIR="./logs/tau_quantile_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"

for TQ in "${TAU_QUANTILES[@]}"; do
  LOG_FILE="${LOG_DIR}/ETTh1_pl${PRED_LEN}_tq${TQ}.log"
  echo "[$(log_ts)] ▶️ CRC tau_quantile sweep: tq=${TQ} (log: ${LOG_FILE})" | tee -a "${MASTER_LOG}"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model "${MODEL}" \
    --baseline_model "${BASELINE}" \
    --model_id "crc_tq_sweep_pl${PRED_LEN}_tq${TQ}" \
    --data ETTh1 \
    --root_path "${ROOT_PATH}" \
    --data_path "${DATA_PATH}" \
    --features M \
    --seq_len 96 --label_len 48 --pred_len "${PRED_LEN}" \
    --enc_in 7 --dec_in 7 --c_out 7 \
    --d_model "${DMODEL}" \
    --e_layers 2 --d_layers 1 --factor 3 \
    --batch_size 32 --learning_rate 1e-4 --train_epochs 50 --patience 3 \
    --crc_encoder_layers "${CRC_ENCODER_LAYERS}" \
    --crc_tau_quantile "${TQ}" \
    --crc_val_ratio "${CRC_VAL_RATIO}" \
    --warm_start --use_gpu True --gpu 0 \
    --des "CRC_tq_sweep_${TQ}" \
    2>&1 | tee "${LOG_FILE}" >> "${MASTER_LOG}"

  echo "[$(log_ts)] ✅ done tq=${TQ}" | tee -a "${MASTER_LOG}"
done

echo "[$(log_ts)] All sweeps finished. Logs: ${LOG_DIR}" | tee -a "${MASTER_LOG}"
