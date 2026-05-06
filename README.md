# Time-Series Forecasting with CRC (Causal Residual Corrector)

This repository implements **CRC** on top of a [Time-Series-Library](https://github.com/thuml/Time-Series-Library)-style **long-term forecasting** codebase. CRC is a **two-stage** pipeline: first train a **baseline forecaster**, then train a **hybrid residual corrector**.

**Entry point**: `python run.py` with `--model CRC`, and set **`--baseline_model`** to the backbone name (e.g. `DLinear`, `PatchTST`).

---

## Repository layout (CRC-related)

| Path | Role |
|------|------|
| `run.py` | Main launcher; `--model CRC` runs the CRC experiment path. |
| `exp/exp_crc.py` | CRC training and evaluation (baseline → corrector). |
| `models/CRC.py` | CRC model and hybrid training logic. |
| `exp/exp_long_term_forecasting.py` | Long-term forecasting base class (extended by CRC). |
| `data_provider/` | Datasets and dataloaders. |
| `utils/` | Metrics (e.g. MSE/MAE), optional DTW, etc. |
| `scripts/` | CRC-oriented analysis (e.g. \(\tau\) sweeps, validation-ratio sensitivity). |
| `logs/` | Experiment logs; you can aggregate tables or figures here. |
| `checkpoints/` | Saved weights (created at runtime). |

---

## Environment

- Python 3.8+; PyTorch (CUDA recommended for training).

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib tqdm einops psutil
```

A pinned `requirements.txt` may not be provided; install PyTorch to match your CUDA using the [official instructions](https://pytorch.org/get-started/locally/).

---

## Data

Place data under the paths given by `--root_path` and `--data_path`, using the same layout as Time-Series-Library (typically CSV with a `date` column). Default paths follow the arguments in `run.py`.

---

## Quick start (CRC)

Set **`--model CRC`** and choose the baseline via **`--baseline_model`** (any forecaster name registered in this repo).

### Common CRC flags

| Flag | Meaning |
|------|---------|
| `--baseline_model` | Baseline forecaster name (e.g. `DLinear`, `PatchTST`). |
| `--q_val` | Hyperparameter related to STCL / graph structure in CRC. |
| `--tau_scale` | Global scaling of the safety boundary \(\tau\) at inference. |
| `--crc_val_ratio` | Fraction of validation windows used for hybrid training. |
| `--crc_skip_retrain` | Skip stage-2 training (e.g. \(\tau\) sweeps with saved checkpoints). |
| `--crc_ckpt_dir` | Directory for corrector checkpoints (optional). |

See all CRC-related options in `run.py` and the train/test flow in `exp/exp_crc.py`.

### Example command

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

## Evaluation

- Metrics are computed in the experiment classes and `utils/metrics.py`; optional **DTW** via **`--use_dtw`**.
- Logs may report extra quantities such as **NDR** when implemented in `test()`.

---

## Checkpoints and logs

- **`--checkpoints`** (default `./checkpoints/`): baseline and CRC-related weights.
- **`logs/`**: run logs; organize summaries as needed.

---

## Scripts

`scripts/` contains CRC-oriented utilities (e.g. \(\tau\) or validation-ratio sweeps). If supported:

```bash
python scripts/<script_name>.py --help
```

---

## Citation and license

- If you use the upstream Time-Series-Library stack, cite it as in the original project.
- If you use the **CRC** implementation from this fork, cite your paper or release when available.
- Licensing follows the upstream Time-Series-Library unless a separate `LICENSE` file is provided in this repository.
