#!/usr/bin/env bash
set -euo pipefail

# DEPRECATED (Phase 3 / A6): tau_scale inference sweeps removed.
# Tune safety boundary via --crc_tau_quantile and retrain stage-2 instead.
# See spec/ablation_results.md (A6 REMOVABLE).
#
# Sweep tau_scale at inference time for a fixed q_val=3.
# NOTE: tau_scale is applied only in HybridTrainer.predict; training always uses the
# original tau_map estimated with quantile_q mapped from q_val=3.
#
# CRC Hybrid is trained once: checkpoint lives in CRC_CKPT_DIR. Later tau values use
# --crc_skip_retrain so only baseline warm_start + test with the new tau_scale run.

DATASET="Electricity"           # or Weather
ROOT_PATH="./dataset/electricity/"
DATA_PATH="electricity.csv"
MODEL="CRC"
BASELINE="TimesNet"
PRED_LEN=96
DMODEL=32
CRC_VAL_RATIO=0.10               # fix validation ratio for CRC
Q_VAL=3                          # fix safety quantile mapping
# Sweep tau_scale from very conservative to aggressive
TAU_SCALES=(0.25 0.5 1.0 2.0 4.0)

VAL_SUFFIX="$(python3 -c "print(int(round(${CRC_VAL_RATIO} * 100)))")"
# Stable path shared by all tau values in this sweep (must match CRC_VAL_RATIO suffix above)
CRC_CKPT_DIR="./checkpoints/crc_tau_sweep_${DATASET,,}_pl${PRED_LEN}_dm${DMODEL}_q${Q_VAL}_val${VAL_SUFFIX}"
CRC_CKPT_FILE="${CRC_CKPT_DIR}/crc_corrector_val${VAL_SUFFIX}.pth"
MODEL_ID="${DATASET,,}_crc_timesnet_pl${PRED_LEN}_dm${DMODEL}"

log_ts() { date '+%Y-%m-%d %H:%M:%S'; }

LOG_DIR="./logs/tau_scale_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"

for TS in "${TAU_SCALES[@]}"; do
  LOG_FILE="${LOG_DIR}/${MODEL_ID}_tau${TS}.log"
  SKIP_CRC=()
  if [[ -f "${CRC_CKPT_FILE}" ]]; then
    SKIP_CRC=(--crc_skip_retrain)
    echo "[$(log_ts)] ⏭️  已存在 CRC checkpoint，跳过阶段二重训：${CRC_CKPT_FILE}" | tee -a "${MASTER_LOG}"
  fi

  echo "[$(log_ts)] ▶️ CRC Tau-Scale Sweep: ds=${DATASET}, q_val=${Q_VAL}, tau_scale=${TS} (log: ${LOG_FILE})" | tee -a "${MASTER_LOG}"
  CMD=(
    python -u run.py
      --task_name long_term_forecast
      --is_training 1
      --model "${MODEL}"
      --baseline_model "${BASELINE}"
      --model_id "${MODEL_ID}"
      --data custom
      --root_path "${ROOT_PATH}"
      --data_path "${DATA_PATH}"
      --features M
      --seq_len 96
      --label_len 48
      --pred_len "${PRED_LEN}"
      --enc_in 321
      --dec_in 321
      --c_out 321
      --d_model "${DMODEL}"
      --n_heads 8
      --e_layers 2
      --d_layers 1
      --d_ff 32
      --patch_len 16
      --factor 3
      --top_k 5
      --moving_avg 25
      --batch_size 32
      --learning_rate 1e-4
      --train_epochs 50
      --patience 3
      --dropout 0.1
      --lradj type1
      --use_gpu True
      --gpu_type cuda
      --gpu 0
      --q_val "${Q_VAL}"
      --tau_scale "${TS}"
      --crc_val_ratio "${CRC_VAL_RATIO}"
      --crc_ckpt_dir "${CRC_CKPT_DIR}"
      "${SKIP_CRC[@]}"
      --warm_start
      --des "CRC_TimesNet_tau${TS}"
  )

  {
    echo "CMD @ $(date '+%Y-%m-%d %H:%M:%S')"
    printf '%q ' "${CMD[@]}"; echo
    "${CMD[@]}"
  } | tee "${LOG_FILE}" >> "${MASTER_LOG}"

  echo "[$(log_ts)] ✅ 完成：ds=${DATASET}, tau_scale=${TS}" | tee -a "${MASTER_LOG}"
  echo >> "${MASTER_LOG}"

done
