# Time-Series Library (CRC / PIR Extensions)

This repository extends a **time-series forecasting** codebase (based on the popular [Time-Series-Library](https://github.com/thuml/Time-Series-Library) style layout) with two additional directions:

- **CRC (Causal Residual Corrector)** — a two-stage pipeline: train a **baseline** forecaster, then train a **hybrid corrector** on top. Entry: `run.py` with `--model CRC`.
- **PIR** — retrieval-augmented refinement with a configurable **backbone**. Entry: `run_pir.py` with `--model PIR`.

Standard tasks (long/short-term forecast, imputation, anomaly detection, classification) and many baseline models are still available through `run.py` when `--model` is not `CRC`.

---

## Repository layout

| Path | Role |
|------|------|
| `run.py` | Main launcher: baseline models + **CRC** (`--model CRC`). |
| `run_pir.py` | **PIR** training and evaluation. |
| `exp/` | Experiment classes (`exp_long_term_forecasting.py`, `exp_crc.py`, `exp_long_term_forecasting_pir.py`, …). |
| `models/` | Model implementations (`CRC.py`, `PIR.py`, `DLinear.py`, `PatchTST.py`, …). |
| `data_provider/` | Datasets and dataloaders (`data_factory.py`, `data_loader.py`, …). |
| `utils/` | Metrics, losses, tools. |
| `scripts/` | Auxiliary plotting / analysis scripts (e.g. CRC sweeps). |
| `logs/` | Experiment logs and exported tables/figures (user-generated). |
| `checkpoints/` | Saved weights (created at runtime; not shipped by default). |

---

## Environment

- **Python** 3.8+ recommended.
- **PyTorch** with CUDA (optional but typical for training).

Install dependencies according to your CUDA version, for example:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib tqdm einops psutil
```

There is no pinned `requirements.txt` in this repo; align PyTorch with your [official install instructions](https://pytorch.org/get-started/locally/).

---

## Data

Place datasets under paths you pass to `--root_path` / `--data_path`. Defaults in code vary by script (e.g. `./data/ETT/` vs `dataset/ETT-small/`). Use the same layout as the original Time-Series-Library expectations (CSV time series with a `date` column where applicable).

---

## Quick start

### Long-term forecasting (standard model)

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id test \
  --model DLinear \
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
  --des Exp
```

### CRC (baseline + corrector)

Use `--model CRC` and select the backbone via **`--baseline_model`** (e.g. `DLinear`, `PatchTST`). CRC-specific flags include:

| Flag | Meaning |
|------|---------|
| `--baseline_model` | Baseline forecaster name (same registry as other models). |
| `--q_val` | STCL / graph-related hyperparameter in CRC. |
| `--tau_scale` | Global scale for the safety boundary \(\tau\) at inference. |
| `--crc_val_ratio` | Fraction of validation windows for hybrid training. |
| `--crc_skip_retrain` | Skip stage-2 training (e.g. for \(\tau\) sweeps with saved checkpoints). |
| `--crc_ckpt_dir` | Directory for corrector checkpoints (optional). |

Example skeleton:

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

See `run.py` and `exp/exp_crc.py` for the full argument list and training flow.

### PIR

```bash
python run_pir.py \
  --task_name long_term_forecast \
  --model PIR \
  --backbone PatchTST \
  --data ETTh1 \
  --root_path dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 96 \
  --pred_len 96 \
  ...
```

PIR-specific options (`--refine_epochs`, `--retrieval_num`, backbone checkpoints, etc.) are defined in `run_pir.py`.

---

## Evaluation notes

- Metrics (e.g. MSE, MAE) and optional **DTW** (`--use_dtw`) are handled in the experiment classes and `utils/metrics.py`.
- Test logs often include extra indicators such as **NDR** when implemented in the corresponding `test()` path.

---

## Checkpoints and logs

- **`--checkpoints`** (default `./checkpoints/`): baseline and experiment checkpoints.
- **`logs/`**: suitable for saving run logs; some workflows export LaTeX tables or figures next to log bundles.

---

## Scripts

Under `scripts/` you may find utilities for sensitivity or sweep plots (e.g. CRC \(\tau\) / validation-ratio analysis). Run them with `python scripts/<name>.py --help` if available.

---

## Citation

If you use the upstream Time-Series-Library stack, cite their work as in the original project. For CRC/PIR contributions from this fork, cite your paper or release when published.

---

## License

Follow the license of the upstream Time-Series-Library unless this repository states otherwise. Add or replace this section if you attach a dedicated `LICENSE` file.
