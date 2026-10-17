# CRC — NeurIPS 2026 Official Code

**Robust Residual Correction via Selective Deployment for Time Series Forecasting**

Implementation of **CRC** (validation-calibrated selective residual correction) on a [Time-Series-Library](https://github.com/thuml/Time-Series-Library)-style long-horizon forecasting codebase: frozen backbone → direction-aware encoder + hybrid corrector (ridge floor + MLP) → frozen deployment rule with NDR diagnostics.

- **Citation:** [`CITATION.bib`](CITATION.bib) (NeurIPS 2026)

---

## Quick start

**Entry point:** `python run.py` with `--model CRC` and `--baseline_model` (e.g. `DLinear`, `PatchTST`, `TimesNet`, `TimeXer`).

**Paper-scale benchmarks:**

```bash
bash run_crc_batch.sh \
  --datasets Electricity,Weather \
  --baselines TimesNet,DLinear \
  --pred_lens 96,192,336,720 \
  --log_dir ./logs/crc_run
```

Edit `DATASETS`, `BASELINES`, and `PRED_LENS` at the top of `run_crc_batch.sh` for the full grid. Data paths and channel counts are defined in `dataset_setup()` (under `./dataset/`).

---

## Core paths

| Path | Role |
|------|------|
| `run.py` | Launcher; CRC via `--model CRC` |
| `exp/exp_crc.py` | Two-stage CRC train/test |
| `models/CRC.py` | Encoder, priors, `HybridTrainer` |
| `run_crc_batch.sh` | Batch CRC experiments |
| `scripts/summarize_crc_logs.py` | Batch log → CSV (optional) |

---

## Environment

Python 3.8+, PyTorch (CUDA recommended):

```bash
pip install -r requirements.txt
# Then install PyTorch for your platform if needed.
```

Dataset paths and channel counts are in `dataset_setup()` inside `run_crc_batch.sh` (under `./dataset/`).

---

## CRC flags (summary)

| Flag | Meaning |
|------|---------|
| `--baseline_model` | Backbone name |
| `--q_val` | STCL depth **and** mapped safety quantile for τ (default `3` → quantile **0.50**) |
| `--top_k` | K-NN graph degree (default **5**) |
| `--tau_scale` | Scale τ at **inference** only |
| `--crc_val_ratio` | Fraction of val windows for hybrid training |
| `--crc_skip_retrain` / `--crc_ckpt_dir` | Load corrector for sweeps |
| `--warm_start` | Reuse baseline checkpoint when available (used in batch script) |

Test output includes **MSE/MAE**, relative **improvement (%)**, and **NDR (%)**.

---

## Citation

```bibtex
@inproceedings{xie2026crc,
  title     = {Robust Residual Correction via Selective Deployment for Time Series Forecasting},
  author    = {Xie, Jianxiang and Hua, Yuncheng and Cheng, Mingyue and Salim, Flora D. and Xue, Hao},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

Also cite Time-Series-Library when using shared data loaders and backbones.
