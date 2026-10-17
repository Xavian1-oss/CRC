#!/usr/bin/env python3
"""Aggregate CRC batch .log files into results_summary_detailed.csv."""
import argparse
import os
import re

import pandas as pd


def extract_metrics_from_log(file_path):
    metrics = {
        "baseline_mse": None,
        "baseline_mae": None,
        "corrected_mse": None,
        "corrected_mae": None,
    }
    baseline_pattern = re.compile(
        r"(?:Baseline \([^)]+\)|基线模型.*)-> MSE: (\d+\.\d+).*MAE: (\d+\.\d+)"
    )
    corrected_pattern = re.compile(
        r"(?:CRC corrected|CRC 纠偏后).*-> MSE: (\d+\.\d+).*MAE: (\d+\.\d+)"
    )

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                baseline_match = baseline_pattern.search(line)
                if baseline_match:
                    metrics["baseline_mse"] = float(baseline_match.group(1))
                    metrics["baseline_mae"] = float(baseline_match.group(2))

                corrected_match = corrected_pattern.search(line)
                if corrected_match:
                    metrics["corrected_mse"] = float(corrected_match.group(1))
                    metrics["corrected_mae"] = float(corrected_match.group(2))
        return metrics
    except OSError as e:
        print(f"Error reading '{file_path}': {e}")
        return None


def parse_filename(filename):
    try:
        parts = filename.replace(".log", "").split("_")
        dataset = parts[0]
        model = parts[2]
        horizon = None
        for part in parts:
            if part.startswith("pl"):
                horizon = int(part[2:])
                break
        return dataset, model, horizon
    except (ValueError, IndexError) as e:
        print(f"  - Skip '{filename}': parse error ({e})")
        return None, None, None


def process_log_directory(root_directory):
    if not os.path.isdir(root_directory):
        print(f"Directory not found: '{root_directory}'")
        return 1

    all_results = []
    for filename in os.listdir(root_directory):
        if not filename.endswith(".log"):
            continue
        print(f"Processing: {filename}...")
        dataset, model, horizon = parse_filename(filename)
        if not dataset:
            continue

        full_path = os.path.join(root_directory, filename)
        metrics = extract_metrics_from_log(full_path)
        if not metrics:
            continue

        all_results.append(
            {
                "dataset": dataset,
                "model": model,
                "horizon": horizon,
                "baseline_mse": metrics["baseline_mse"],
                "baseline_mae": metrics["baseline_mae"],
                "corrected_mse": metrics["corrected_mse"],
                "corrected_mae": metrics["corrected_mae"],
                "filename": filename,
            }
        )

    if not all_results:
        print("\nNo valid logs or metrics found.")
        return 1

    df = pd.DataFrame(all_results)
    cols_order = [
        "dataset",
        "model",
        "horizon",
        "baseline_mse",
        "baseline_mae",
        "corrected_mse",
        "corrected_mae",
        "filename",
    ]
    df = df[[c for c in cols_order if c in df.columns]]

    print("\n" + "=" * 30 + " SUMMARY " + "=" * 30)
    print(df)

    output_csv_path = os.path.join(root_directory, "results_summary_detailed.csv")
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {output_csv_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Summarize CRC run logs into CSV.")
    ap.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory containing *.log from run_crc_batch.sh (e.g. ./logs/multi_20251120_131506)",
    )
    args = ap.parse_args()
    raise SystemExit(process_log_directory(args.log_dir))


if __name__ == "__main__":
    main()
