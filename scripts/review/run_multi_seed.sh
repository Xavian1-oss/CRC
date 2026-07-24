#!/usr/bin/env bash
# R14-E1 — B1 multi-seed: G5 (Full CRC) + G1 (always-on)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

COMMON=(
  --crc_encoder_layers 3
  --crc_tau_quantile 0.5
  --crc_val_ratio 1.0
  --warm_start
  --use_gpu True
  --gpu 0
)

B1=(
  --task_name long_term_forecast --is_training 1
  --model CRC --baseline_model DLinear
  --data ETTh1
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv
  --features M --seq_len 96 --label_len 48 --pred_len 96
  --enc_in 7 --dec_in 7 --c_out 7
  --d_model 512 --e_layers 2 --d_layers 1 --factor 3
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

for s in 2021 2022 2023; do
  run_one "r14_g5_b1_seed${s}" \
    "${B1[@]}" \
    --fix_seed "${s}" \
    --model_id "review_r14_g5_b1_seed${s}" \
    "${COMMON[@]}" \
    --des "r14_g5_b1_seed${s}"

  run_one "r14_g1_b1_seed${s}" \
    "${B1[@]}" \
    --fix_seed "${s}" \
    --crc_use_blend 0 \
    --model_id "review_r14_g1_b1_seed${s}" \
    "${COMMON[@]}" \
    --des "r14_g1_b1_seed${s}"
done

echo "R14-E1 finished. Summarize with scripts/review/summarize_review_logs.py"
