#!/usr/bin/env bash
# G2 — Ridge-only + validation gate (B1–B3)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/review/g2_b1 logs/review/g2_b2 logs/review/g2_b3

COMMON=(
  --crc_ridge_gate_only 1
  --crc_encoder_layers 3
  --crc_tau_quantile 0.5
  --crc_val_ratio 1.0
  --warm_start
  --use_gpu True
  --gpu 0
)

run_one() {
  local tag="$1"
  shift
  local log_dir="logs/review/${tag}"
  mkdir -p "$log_dir"
  echo "=== Running ${tag} @ $(date) ===" | tee "${log_dir}/run.log"
  python -u run.py "$@" 2>&1 | tee -a "${log_dir}/run.log"
}

run_one g2_b1 \
  --task_name long_term_forecast --is_training 1 \
  --model CRC --baseline_model DLinear \
  --model_id review_g2_b1 \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --e_layers 2 --d_layers 1 --factor 3 \
  "${COMMON[@]}" --des review_g2_b1

run_one g2_b2 \
  --task_name long_term_forecast --is_training 1 \
  --model CRC --baseline_model DLinear \
  --model_id review_g2_b2 \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 336 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --e_layers 2 --d_layers 1 --factor 3 \
  "${COMMON[@]}" --des review_g2_b2

run_one g2_b3 \
  --task_name long_term_forecast --is_training 1 \
  --model CRC --baseline_model TimesNet \
  --model_id review_g2_b3 \
  --data custom \
  --root_path ./dataset/weather/ --data_path weather.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 21 --dec_in 21 --c_out 21 \
  --d_model 32 --d_ff 32 --top_k 5 \
  --e_layers 2 --d_layers 1 --factor 3 \
  "${COMMON[@]}" --des review_g2_b3

echo "G2 runs finished."
