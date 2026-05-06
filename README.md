# 时间序列预测 · CRC（Causal Residual Corrector）

本仓库在 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 风格的 **长期预测** 代码框架上，实现了 **CRC**：先训练 **基线预测器**，再训练 **混合残差校正器** 两阶段流程。

**入口**：`python run.py`，设置 `--model CRC`，并用 `--baseline_model` 指定基线（如 `DLinear`、`PatchTST` 等）。

---

## 仓库结构（与 CRC 相关）

| 路径 | 说明 |
|------|------|
| `run.py` | 主入口：`--model CRC` 时走 CRC 实验流程。 |
| `exp/exp_crc.py` | CRC 训练与评测逻辑（两阶段：基线 → 校正器）。 |
| `models/CRC.py` | CRC 模型与混合训练相关实现。 |
| `exp/exp_long_term_forecasting.py` | 长期预测基类（CRC 实验继承并扩展）。 |
| `data_provider/` | 数据集与 DataLoader。 |
| `utils/` | 指标（如 MSE/MAE）、可选 DTW 等。 |
| `scripts/` | CRC 相关分析脚本（如 \(\tau\) 扫描、验证比例灵敏度等）。 |
| `logs/` | 实验日志（可自行汇总表格或图）。 |
| `checkpoints/` | 保存的权重（运行时生成）。 |

---

## 环境

- Python 3.8+；PyTorch（建议配置 CUDA）。

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib tqdm einops psutil
```

本仓库未必提供固定版 `requirements.txt`，请按本机 CUDA 从 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择安装命令。

---

## 数据

将数据放在 `--root_path` / `--data_path` 指向的目录；格式与 Time-Series-Library 一致（多为带 `date` 列的 CSV）。具体默认路径以 `run.py` 中参数为准。

---

## 快速开始（CRC）

使用 **`--model CRC`**，并通过 **`--baseline_model`** 选择基线模型名称（与仓库内已注册的预测模型一致）。

### CRC 常用参数

| 参数 | 含义 |
|------|------|
| `--baseline_model` | 基线预测器名称（如 `DLinear`、`PatchTST`）。 |
| `--q_val` | CRC 中与 STCL / 图相关的超参数。 |
| `--tau_scale` | 推理时对安全边界 \(\tau\) 的全局缩放。 |
| `--crc_val_ratio` | 用于混合阶段训练的验证窗口比例。 |
| `--crc_skip_retrain` | 跳过第二阶段训练（例如在用已有 checkpoint 做 \(\tau\) 扫描时）。 |
| `--crc_ckpt_dir` | 校正器 checkpoint 目录（可选）。 |

更多参数见 `run.py` 中与 `CRC` 相关的选项，以及 `exp/exp_crc.py` 中的训练 / 测试流程。

### 示例命令

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id my_crc_run \
  --model CRC \
  --baseline_model DLinear \
  --data ETTh1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 512 \
  --des CRC_exp
```

---

## 评测说明

- 指标在实验类与 `utils/metrics.py` 中计算；可选用 **`--use_dtw`** 计算 DTW。
- 若 `test()` 中实现，日志中可能出现 **NDR** 等额外指标。

---

## Checkpoint 与日志

- **`--checkpoints`**（默认 `./checkpoints/`）：基线与 CRC 相关权重。
- **`logs/`**：运行日志；可按需整理实验汇总。

---

## 脚本

`scripts/` 下为 CRC 相关的辅助分析（如 \(\tau\) 或验证集比例扫描）。若脚本支持，可使用：

```bash
python scripts/<脚本名>.py --help
```

---

## 引用与许可证

- 若使用上游 Time-Series-Library 代码栈，请按其原项目说明引用。
- 若使用本仓库中的 **CRC** 实现，请在论文或正式发布可用后引用你的工作。
- 许可证遵循上游 Time-Series-Library；若本仓库根目录另有 `LICENSE`，以该文件为准。
