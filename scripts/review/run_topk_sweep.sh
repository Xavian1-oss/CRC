#!/usr/bin/env bash
# RMK-E1 — B1 Full CRC, graph top_k ∈ {3,5,7}
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

COMMON=(
  --fix_seed 2021
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
}

for k in 3 5 7; do
  run_one "rmk_topk${k}_b1" \
    "${B1[@]}" \
    --top_k "${k}" \
    --model_id "review_rmk_topk${k}_b1" \
    "${COMMON[@]}" \
    --des "rmk_topk${k}_b1"
done

echo "RMK-E1 finished."
