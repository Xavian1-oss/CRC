#!/usr/bin/env bash
# R1 rebuttal experiments — G4 Deploy-only (B1–B3)
# See review_spec/design.md §6, review_spec/task.md Phase R1-A
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs/review/g4_b1 logs/review/g4_b2 logs/review/g4_b3

G4_FLAGS=(
  --crc_use_mtl 0
  --crc_use_stcl 0
  --crc_use_minimal_priors 1
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
  echo "=== Done ${tag} ===" | tee -a "${log_dir}/run.log"
}

# B1
run_one g4_b1 \
  --task_name long_term_forecast --is_training 1 \
  --model CRC --baseline_model DLinear \
  --model_id review_g4_b1 \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --e_layers 2 --d_layers 1 --factor 3 \
  "${G4_FLAGS[@]}" \
  --des review_g4_b1

# B2
run_one g4_b2 \
  --task_name long_term_forecast --is_training 1 \
  --model CRC --baseline_model DLinear \
  --model_id review_g4_b2 \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 336 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 512 --e_layers 2 --d_layers 1 --factor 3 \
  "${G4_FLAGS[@]}" \
  --des review_g4_b2

# B3
run_one g4_b3 \
  --task_name long_term_forecast --is_training 1 \
  --model CRC --baseline_model TimesNet \
  --model_id review_g4_b3 \
  --data custom \
  --root_path ./dataset/weather/ --data_path weather.csv \
  --features M --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 21 --dec_in 21 --c_out 21 \
  --d_model 32 --d_ff 32 --top_k 5 \
  --e_layers 2 --d_layers 1 --factor 3 \
  "${G4_FLAGS[@]}" \
  --des review_g4_b3

echo "All G4 runs finished. Update review_spec/rebuttal_results.md from logs/review/g4_*/run.log"
