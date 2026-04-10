#!/usr/bin/env bash
set -euo pipefail

# 实验：验证集比例敏感性 (Validation Sensitivity)
# - val ratio: 5% / 10% / 20%
# - 不重训 backbone（TimesNet 基线），只重新跑 CRC（带不同 crc_val_ratio）

export CUDA_VISIBLE_DEVICES=0

Q_VAL=3
INVERSE=''      # 如需还原尺度评估，设为 '--inverse'
USE_DTW=''      # 如需计算 DTW，设为 '--use_dtw'

# ===== 实验设置：可以按需修改 =====
# 支持一次跑多个数据集：在这里列出即可
DATASETS=("Electricity" "Weather")   # 可改为只包含一个，例如 ("Electricity")
PRED_LEN=96
BASELINE="TimesNet"

# 验证集比例列表（字符串形式）；用于 --crc_val_ratio
VAL_RATIOS=(0.05 0.10 0.20)

DATA_ARG="custom"

# 其他超参沿用 run_crc_batch.sh 中 Weather + TimesNet + pl=96 的查表默认
DMODEL=32
NHEADS=8
ELAYERS=2
DLAYERS=1
DFF=32
PATCH_LEN=16
FACTOR=3
TOP_K=5
MOVING_AVG=25
SEQ_LEN=96
LABEL_LEN=48
EPOCHS=50
BATCH=32
LR=1e-4
PATIENCE=3
DROPOUT=0.1
LRADJ=type1
USE_AMP=""   # 如需 AMP，设为 '--use_amp'

TS="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="./logs/val_sensitivity_${TS}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"
log() { echo "[$(date +'%F %T')] $*" | tee -a "${MASTER_LOG}"; }

for DATASET in "${DATASETS[@]}"; do
  # ---- 根据数据集设定通道数 ----
  case "$DATASET" in
    Weather)     ROOT="./dataset/weather/";       CSV="weather.csv";      ENC_IN=21;  DEC_IN=21;  C_OUT=21;;
    Traffic)     ROOT="./dataset/traffic/";       CSV="traffic.csv";      ENC_IN=862; DEC_IN=862; C_OUT=862;;
    Electricity) ROOT="./dataset/electricity/";   CSV="electricity.csv";  ENC_IN=321; DEC_IN=321; C_OUT=321;;
    Exchange)    ROOT="./dataset/exchange_rate/"; CSV="exchange_rate.csv";ENC_IN=8;   DEC_IN=8;   C_OUT=8;;
    *) echo "暂不支持的数据集: $DATASET"; exit 1;;
  esac

  for VR in "${VAL_RATIOS[@]}"; do
  # 去掉小数点用于 tag
  VR_TAG="${VR//./}"
  MODEL_ID="${DATASET}_crc_${BASELINE,,}_pl${PRED_LEN}_dm${DMODEL}_vr${VR_TAG}"
  DES="CRC_${BASELINE}_vr${VR_TAG}"
  LOG_FILE="${LOG_DIR}/${MODEL_ID}.log"

  log "▶️ CRC Validation Sensitivity: ds=${DATASET}, baseline=${BASELINE}, pl=${PRED_LEN}, val_ratio=${VR} (log: ${LOG_FILE})"

  {
    echo "CMD @ $(date +'%F %T')"
    cat <<EOF
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model CRC \
  --baseline_model ${BASELINE} \
  --model_id ${MODEL_ID} \
  --data ${DATA_ARG} \
  --root_path ${ROOT} \
  --data_path ${CSV} \
  --features M \
  --seq_len ${SEQ_LEN} \
  --label_len ${LABEL_LEN} \
  --pred_len ${PRED_LEN} \
  --enc_in ${ENC_IN} \
  --dec_in ${DEC_IN} \
  --c_out ${C_OUT} \
  --d_model ${DMODEL} \
  --n_heads ${NHEADS} \
  --e_layers ${ELAYERS} \
  --d_layers ${DLAYERS} \
  --d_ff ${DFF} \
  --patch_len ${PATCH_LEN} \
  --factor ${FACTOR} \
  --top_k ${TOP_K} \
  --moving_avg ${MOVING_AVG} \
  --batch_size ${BATCH} \
  --learning_rate ${LR} \
  --train_epochs ${EPOCHS} \
  --patience ${PATIENCE} \
  --dropout ${DROPOUT} \
  --lradj ${LRADJ} \
  --use_gpu True \
  --gpu_type cuda \
  --gpu 0 \
  ${USE_DTW} ${INVERSE} \
  ${USE_AMP} \
  --q_val ${Q_VAL} \
  --crc_val_ratio ${VR} \
  --warm_start \
  --des ${DES}
EOF
  } | tee -a "${LOG_FILE}" >> "${MASTER_LOG}"

  set +e
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model CRC \
    --baseline_model "${BASELINE}" \
    --model_id "${MODEL_ID}" \
    --data "${DATA_ARG}" \
    --root_path "${ROOT}" \
    --data_path "${CSV}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${PRED_LEN}" \
    --enc_in "${ENC_IN}" \
    --dec_in "${DEC_IN}" \
    --c_out "${C_OUT}" \
    --d_model "${DMODEL}" \
    --n_heads "${NHEADS}" \
    --e_layers "${ELAYERS}" \
    --d_layers "${DLAYERS}" \
    --d_ff "${DFF}" \
    --patch_len "${PATCH_LEN}" \
    --factor "${FACTOR}" \
    --top_k "${TOP_K}" \
    --moving_avg "${MOVING_AVG}" \
    --batch_size "${BATCH}" \
    --learning_rate "${LR}" \
    --train_epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --dropout "${DROPOUT}" \
    --lradj "${LRADJ}" \
    --use_gpu True \
    --gpu_type cuda \
    --gpu 0 \
    ${USE_DTW} ${INVERSE} \
    ${USE_AMP} \
    --q_val "${Q_VAL}" \
    --crc_val_ratio "${VR}" \
    --warm_start \
    --des "${DES}" \
    2> >(tee -a "${LOG_FILE}" >> "${MASTER_LOG}" >&2) \
    | tee -a "${LOG_FILE}" >> "${MASTER_LOG}"
  ecode=${PIPESTATUS[0]}
  set -e

  if [ $ecode -eq 0 ]; then
    log "✅ 完成：ds=${DATASET}, val_ratio=${VR}"
  else
    log "❌ 失败：ds=${DATASET}, val_ratio=${VR}，退出码=${ecode}（详见 ${LOG_FILE}）"
    exit $ecode
  fi

  done
done

log "🎉 验证敏感性实验全部完成。日志目录：${LOG_DIR}"
