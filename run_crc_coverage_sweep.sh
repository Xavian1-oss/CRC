#!/usr/bin/env bash
set -euo pipefail

# Sweep q_val to study how correction coverage, accuracy, and NDR change.
# 注意：q_val 通过 exp/exp_crc.py 中的映射控制 HybridTrainer 的 quantile_q，
# quantile_q 只影响训练阶段估计的 tau_map，本实验中 tau_scale 始终为 1.0。

DATASET="Electricity"           # or Weather
ROOT_PATH="./dataset/electricity/"
DATA_PATH="electricity.csv"
MODEL="CRC"
BASELINE="TimesNet"
PRED_LEN=96
DMODEL=32
CRC_VAL_RATIO=0.10              # fix validation ratio for CRC
# q_val 在 argparse 里是 int，因此这里用整数
Q_LIST=(1 2 3 4 5)              # sweep from conservative to aggressive

LOG_DIR="./logs/coverage_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"

for Q in "${Q_LIST[@]}"; do
  MODEL_ID="${DATASET,,}_crc_timesnet_pl${PRED_LEN}_dm${DMODEL}_q${Q}"
  LOG_FILE="${LOG_DIR}/${MODEL_ID}.log"

  echo "["$(date '+%Y-%m-%d %H:%M:%S')"] ▶️ CRC Coverage Sweep: ds=${DATASET}, q_val=${Q} (log: ${LOG_FILE})" | tee -a "${MASTER_LOG}"
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
      --q_val ${Q}
      --crc_val_ratio "${CRC_VAL_RATIO}"
      --warm_start
      --des "CRC_TimesNet_q${Q}"
  )

  # Run and tee to per-run log
  {
    echo "CMD @ $(date '+%Y-%m-%d %H:%M:%S')"
    printf '%q ' "${CMD[@]}"; echo
    "${CMD[@]}"
  } | tee "${LOG_FILE}" >> "${MASTER_LOG}"

  echo "["$(date '+%Y-%m-%d %H:%M:%S')"] ✅ 完成：ds=${DATASET}, q_val=${Q}" | tee -a "${MASTER_LOG}"
  echo >> "${MASTER_LOG}"

done
