#!/usr/bin/env bash
set -euo pipefail

# ================== 基础设置 ==================
export CUDA_VISIBLE_DEVICES=0

# --- 以下参数由 GDEF 字典统一管理 (用于Backbone) ---
# ... (与您的脚本相同, e.g., SEQ_LEN, DMODEL, NHEADS...) ...

# --- 以下参数为 PIR 模型的全局设置 ---
# 这是“精炼训练”阶段的参数
REFINE_EPOCHS=10          # 精炼阶段的Epochs (Stage 3)
REFINE_LR=1e-4            # 精炼阶段的学习率
REFINE_DMODEL=128         # Refiner的模型维度
REFINE_DFF=128            # Refiner的FFN维度
REFINE_LAYERS=1           # Refiner的层数
RETRIEVAL_NUM=10          # 检索的K值
RETRIEVAL_STRIDE=1        # 检索数据库的步幅 (通常为1)
INCLUDING_TIME=1          # Refiner是否包含时间特征 (1为是, 0为否)

# --- 以下参数为 PIR 实验流程的控制参数 ---
# 0: 从头预训练Backbone (Stage 1) -> 构建索引 (Stage 2) -> 训练Refiner (Stage 3)
# 1: 加载完整PIR模型 (用于--is_training 0)
# 2: 加载预训练Backbone (跳过Stage 1) -> 构建索引 (Stage 2) -> 训练Refiner (Stage 3)
LOAD_BACKBONE=0           # 默认为0，执行完整的三阶段训练
BACKBONE_CKPT_DIR="./checkpoints/backbone/" # 预训练Backbone的(保存/加载)目录

INVERSE=''                # 若要还原尺度评估，设为 '--inverse'

# ================== 想批量跑的组合 ==================
# (与您的脚本保持一致)
DATASETS=(Weather)
PRED_LENS=(96)
BASELINES=(TimesNet) 

# ================== 数据集路径与通道数 ==================
# (与您的脚本保持一致，无需改动)
dataset_setup () {
  local ds="$1"
  case "$ds" in
    ETTh1)       ROOT="./dataset/ETT-small/";     CSV="ETTh1.csv";        ENC_IN=7;   DEC_IN=7;   C_OUT=7;;
    ETTh2)       ROOT="./dataset/ETT-small/";     CSV="ETTh2.csv";        ENC_IN=7;   DEC_IN=7;   C_OUT=7;;
    ETTm1)       ROOT="./dataset/ETT-small/";     CSV="ETTm1.csv";        ENC_IN=7;   DEC_IN=7;   C_OUT=7;;
    ETTm2)       ROOT="./dataset/ETT-small/";     CSV="ETTm2.csv";        ENC_IN=7;   DEC_IN=7;   C_OUT=7;;
    Weather)     ROOT="./dataset/weather/";       CSV="weather.csv";      ENC_IN=21;  DEC_IN=21;  C_OUT=21;;
    Traffic)     ROOT="./dataset/traffic/";       CSV="traffic.csv";      ENC_IN=862; DEC_IN=862; C_OUT=862;;
    Electricity) ROOT="./dataset/electricity/";   CSV="electricity.csv";  ENC_IN=321; DEC_IN=321; C_OUT=321;;
    Exchange)    ROOT="./dataset/exchange_rate/"; CSV="exchange_rate.csv";ENC_IN=8;   DEC_IN=8;   C_OUT=8;;
    *) echo "未知数据集: $ds"; exit 1;;
  esac
}

# ================== 三层默认 + 查表覆盖 ==================
# (与您的脚本保持一致，无需改动)
# 0) 全局缺省 (与 run.py 的 argparse default 保持一致)
declare -A GDEF=(
  [LR]="1e-4" [BATCH]="32" [PATIENCE]="3"
  [DMODEL]="512" [NHEADS]="8" [ELAYERS]="2" [DLAYERS]="1" [DFF]="2048"
  [PATCH_LEN]="16" [FACTOR]="1" [TOP_K]="5" [MOVING_AVG]="25"
  [SEQ_LEN]="96"
  [LABEL_LEN]="48"
  [EPOCHS]="50"
  [DROPOUT]="0.1"
  [LRADJ]="type1"
  [USE_AMP]=""
)

# 1) baseline × dataset 覆盖（只写与全局不同的）
declare -A BD
BD["TimeXer|Weather|FACTOR"]="3"
BD["TimeXer|Weather|BATCH"]="4"
BD["TimesNet|Weather|FACTOR"]="3"
BD["TimesNet|Weather|DMODEL"]="32"
BD["TimesNet|Weather|DFF"]="32"
BD["PatchTST|Weather|FACTOR"]="3"
BD["PatchTST|Weather|EPOCHS"]="3"

# 2) baseline × dataset × horizon 覆盖（只写确实因 horizon 不同而变化的项）
declare -A BDH
BDH["TimeXer|Weather|96|ELAYERS"]="1"
BDH["TimeXer|Weather|96|DMODEL"]="256"
BDH["TimeXer|Weather|96|DFF"]="512"
BDH["TimeXer|Weather|192|ELAYERS"]="3"
BDH["TimeXer|Weather|192|DMODEL"]="128"
BDH["TimeXer|Weather|192|DFF"]="1024"
BDH["TimeXer|Weather|336|ELAYERS"]="1"
BDH["TimeXer|Weather|336|DMODEL"]="256"
BDH["TimeXer|Weather|720|ELAYERS"]="1"
BDH["TimeXer|Weather|720|DMODEL"]="128"
BDH["TimesNet|Weather|192|EPOCHS"]="1" 
BDH["TimesNet|Weather|720|EPOCHS"]="1" 
BDH["PatchTST|Weather|96|NHEADS"]="4"
BDH["PatchTST|Weather|192|NHEADS"]="16"
BDH["PatchTST|Weather|336|NHEADS"]="4"
BDH["PatchTST|Weather|336|BATCH"]="128"
BDH["PatchTST|Weather|720|NHEADS"]="4"
BDH["PatchTST|Weather|720|BATCH"]="128"

# 3) 查表函数：horizon 覆盖 > (baseline,dataset) 覆盖 > 全局缺省
# (与您的脚本保持一致，无需改动)
getp () {
  local base="$1" ds="$2" h="$3" key="$4"
  local v=""
  v="${BDH[$base|$ds|$h|$key]:-}"
  [[ -z "$v" ]] && v="${BD[$base|$ds|$key]:-}"
  [[ -z "$v" ]] && v="${GDEF[$key]}"
  echo "$v"
}

# ================== 日志 ==================
# (与您的脚本保持一致，无需改动)
TS="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="./logs/multi_PIR_${TS}" # 更改了日志目录名以作区分
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/master.log"
log() { echo "[$(date +'%F %T')] $*" | tee -a "${MASTER_LOG}"; }
trap 'ecode=$?; [ $ecode -ne 0 ] && log "❌ 脚本异常退出，退出码=${ecode}"; exit $ecode' EXIT

# ================== 单次运行 (已魔改为PIR) ==================
run_with_log () {
  local baseline="$1"   # Backbone: PatchTST / TimesNet / ...
  local ds="$2"         # 数据集
  local pred="$3"       # 预测步长

  dataset_setup "${ds}"

  local data_arg="${ds}"
  if [[ "${ds}" == "Weather" || "${ds}" == "Traffic" || "${ds}" == "Exchange" || "${ds}" == "Electricity" ]]; then
    data_arg="custom"
  fi

  # —— 取这次要用的超参（查表合并） ——
  # 这些参数将用于PIR的 *Backbone* (e.g., TimeXer)
  local LRv       ; LRv=$(getp "$baseline" "$ds" "$pred" "LR")
  local BATCHv    ; BATCHv=$(getp "$baseline" "$ds" "$pred" "BATCH")
  local PATIENCEv ; PATIENCEv=$(getp "$baseline" "$ds" "$pred" "PATIENCE")
  local DMODELv   ; DMODELv=$(getp "$baseline" "$ds" "$pred" "DMODEL")
  local NHEADSv   ; NHEADSv=$(getp "$baseline" "$ds" "$pred" "NHEADS")
  local ELAYERSv  ; ELAYERSv=$(getp "$baseline" "$ds" "$pred" "ELAYERS")
  local DLAYERSv  ; DLAYERSv=$(getp "$baseline" "$ds" "$pred" "DLAYERS")
  local DFFv      ; DFFv=$(getp "$baseline" "$ds" "$pred" "DFF")
  local PATCHLENv ; PATCHLENv=$(getp "$baseline" "$ds" "$pred" "PATCH_LEN")
  local FACTORv   ; FACTORv=$(getp "$baseline" "$ds" "$pred" "FACTOR")
  local TOP_Kv    ; TOP_Kv=$(getp "$baseline" "$ds" "$pred" "TOP_K")
  local MOVAVGv   ; MOVAVGv=$(getp "$baseline" "$ds" "$pred" "MOVING_AVG")
  local SEQLENv   ; SEQLENv=$(getp "$baseline" "$ds" "$pred" "SEQ_LEN")
  local LABELLENv ; LABELLENv=$(getp "$baseline" "$ds" "$pred" "LABEL_LEN")
  local EPOCHSv   ; EPOCHSv=$(getp "$baseline" "$ds" "$pred" "EPOCHS") # <-- 这将是 "预训练" (Stage 1) 的 Epochs
  local DROPOUTv  ; DROPOUTv=$(getp "$baseline" "$ds" "$pred" "DROPOUT")
  local LRADJv    ; LRADJv=$(getp "$baseline" "$ds" "$pred" "LRADJ")
  local USEAMPv   ; USEAMPv=$(getp "$baseline" "$ds" "$pred" "USE_AMP")

  local tag="PIR_${ds}_${baseline,,}_pl${pred}_dm${DMODELv}_sl${SEQLENv}"
  local log_file="${LOG_DIR}/${tag}.log"

  # 为预训练Backbone创建一个唯一的保存路径
  local backbone_save_path="${BACKBONE_CKPT_DIR}/${tag}/"
  if [ $LOAD_BACKBONE -eq 0 ]; then
    log "ℹ️  Backbone (Stage 1) 将被训练并保存到: ${backbone_save_path}"
    mkdir -p "${backbone_save_path}"
  fi

  log "▶️  model=PIR / backbone=${baseline} / ds=${ds} / pred=${pred} (log: ${log_file})"

  # ------- 回显命令 -------
  {
    echo "CMD @ $(date +'%F %T')"
    cat <<EOF
python -u run_pir.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model PIR \
  --backbone ${baseline} \
  --model_id ${tag} \
  --data ${data_arg} \
  --root_path ${ROOT} \
  --data_path ${CSV} \
  --features M \
  --seq_len ${SEQLENv} \
  --label_len ${LABELLENv} \
  --pred_len ${pred} \
  --enc_in ${ENC_IN} \
  --dec_in ${DEC_IN} \
  --c_out ${C_OUT} \
  \
  # --- Backbone (TimeXer/PatchTST/...) 的参数 (来自查表) ---
  --d_model ${DMODELv} \
  --n_heads ${NHEADSv} \
  --e_layers ${ELAYERSv} \
  --d_layers ${DLAYERSv} \
  --d_ff ${DFFv} \
  --patch_len "${PATCHLENv}" \
  --factor ${FACTORv} \
  --top_k ${TOP_Kv} \
  --moving_avg ${MOVAVGv} \
  --dropout ${DROPOUTv} \
  \
  # --- 优化器 (Stage 1, 预训练) 的参数 (来自查表) ---
  --batch_size ${BATCHv} \
  --learning_rate ${LRv} \
  --train_epochs ${EPOCHSv} \
  --patience ${PATIENCEv} \
  --lradj ${LRADJv} \
  \
  # --- PIR 专属参数 (全局) ---
  --refine_epochs ${REFINE_EPOCHS} \
  --refine_lr ${REFINE_LR} \
  --refine_d_model ${REFINE_DMODEL} \
  --refine_d_ff ${REFINE_DFF} \
  --refine_layers ${REFINE_LAYERS} \
  --retrieval_num ${RETRIEVAL_NUM} \
  --retrieval_stride ${RETRIEVAL_STRIDE} \
  --including_time_features ${INCLUDING_TIME} \
  --load_pretrained_backbone ${LOAD_BACKBONE} \
  --bakcbone_checkpoints "${backbone_save_path}" \
  \
  # --- GPU 和其他设置 ---
  --use_gpu True \
  --gpu 0 \
  ${INVERSE} \
  ${USEAMPv} \
  --des "PIR_${baseline}_M_d${DMODELv}"
EOF
  } | tee -a "${log_file}" >> "${MASTER_LOG}"

  # ------- 真正执行 -------
  # 假设您上一轮的脚本文件名为 'run_pir.py'
  set +e
  python -u run_pir.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model PIR \
    --backbone "${baseline}" \
    --model_id "${tag}" \
    --data "${data_arg}" \
    --root_path "${ROOT}" \
    --data_path "${CSV}" \
    --features M \
    --seq_len "${SEQLENv}" \
    --label_len "${LABELLENv}" \
    --pred_len "${pred}" \
    --enc_in "${ENC_IN}" \
    --dec_in "${DEC_IN}" \
    --c_out "${C_OUT}" \
    --d_model "${DMODELv}" \
    --n_heads "${NHEADSv}" \
    --e_layers "${ELAYERSv}" \
    --d_layers "${DLAYERSv}" \
    --d_ff "${DFFv}" \
    --patch_len "${PATCHLENv}" \
    --factor "${FACTORv}" \
    --top_k "${TOP_Kv}" \
    --moving_avg "${MOVAVGv}" \
    --dropout "${DROPOUTv}" \
    --batch_size "${BATCHv}" \
    --learning_rate "${LRv}" \
    --train_epochs "${EPOCHSv}" \
    --patience "${PATIENCEv}" \
    --lradj "${LRADJv}" \
    --refine_epochs "${REFINE_EPOCHS}" \
    --refine_lr "${REFINE_LR}" \
    --refine_d_model "${REFINE_DMODEL}" \
    --refine_d_ff "${REFINE_DFF}" \
    --refine_layers "${REFINE_LAYERS}" \
    --retrieval_num "${RETRIEVAL_NUM}" \
    --retrieval_stride "${RETRIEVAL_STRIDE}" \
    --including_time_features "${INCLUDING_TIME}" \
    --load_pretrained_backbone "${LOAD_BACKBONE}" \
    --bakcbone_checkpoints "${backbone_save_path}" \
    --use_gpu True \
    --gpu 0 \
    ${INVERSE} \
    ${USEAMPv} \
    --des "PIR_${baseline}_M_d${DMODELv}" \
    2> >(tee -a "${log_file}" >> "${MASTER_LOG}" >&2) \
    | tee -a "${log_file}" >> "${MASTER_LOG}"
  ecode=${PIPESTATUS[0]}
  set -e

  if [ $ecode -eq 0 ]; then
    log "✅ 完成：PIR(backbone=${baseline}) / ${ds} / pred_len=${pred}"
  else
    log "❌ 失败：PIR(backbone=${baseline}) / ${ds} / pred_len=${pred}，退出码=${ecode}（详见 ${log_file}）"
    return $ecode
  fi
}

# ================== 批量调度 ==================
# (与您的脚本保持一致，无需改动)
for ds in "${DATASETS[@]}"; do
  for pl in "${PRED_LENS[@]}"; do
    for base in "${BASELINES[@]}"; do
      run_with_log "${base}" "${ds}" "${pl}"
    done
  done
done

log "🎉 全部PIR批量运行完成。日志目录：${LOG_DIR}"