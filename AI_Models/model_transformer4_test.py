# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from model_transformer4 import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUT_DIR,
    load_artifacts_transformer4,
    predict_dataframe_transformer4,
    read_csv_safely,
    run_training_transformer4,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--target", type=str, default="낙찰가")
    p.add_argument("--features", nargs="*", default=None)

    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--tune_repeats", type=int, default=2)
    p.add_argument("--max_rows_tune", type=int, default=80000)
    p.add_argument("--max_rows_train", type=int, default=0)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--ensemble_seeds", type=int, default=1)

    p.add_argument("--strong", action="store_true")     # more accurate but slower
    p.add_argument("--open_plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data) if args.data else DEFAULT_DATA_PATH
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR

    df = read_csv_safely(data_path)

    if args.features is None or len(args.features) == 0:
        base_features = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]
    else:
        base_features = args.features

    if args.strong:
        trials = max(18, args.trials)
        tune_repeats = max(2, args.tune_repeats)
        epochs = max(140, args.epochs)
        patience = max(14, args.patience)
        ensemble_seeds = max(3, args.ensemble_seeds)
        max_rows_tune = max(120000, args.max_rows_tune)
        max_rows_train = args.max_rows_train  # 0이면 전체 사용
    else:
        trials = args.trials
        tune_repeats = args.tune_repeats
        epochs = args.epochs
        patience = args.patience
        ensemble_seeds = args.ensemble_seeds
        max_rows_tune = args.max_rows_tune
        max_rows_train = args.max_rows_train

    res = run_training_transformer4(
        df=df,
        base_feature_cols=base_features,
        target_col=args.target,
        output_dir=str(out_dir),
        target_log=True,
        test_ratio=0.20,
        val_ratio=0.10,
        seed=42,
        deterministic=True,
        max_rows_tune=int(max_rows_tune),
        max_rows_train=int(max_rows_train),
        trials=int(trials),
        tune_repeats=int(tune_repeats),
        epochs=int(epochs),
        patience=int(patience),
        ensemble_seeds=int(ensemble_seeds),
        show_tqdm=True,
        verbose=True,
    )

    print("\n[Summary]")
    print(f"Artifacts: {res['output_dir']}")
    print(f"Ensemble TEST RMSE: {res['ensemble_test_metrics']['RMSE']:.3f}")
    print(f"Best single seed: {res['best_single_seed']}")

    # quick inference sanity check
    model, preproc, y_scaler, target_log, meta = load_artifacts_transformer4(Path(res["output_dir"]))
    sample = df.sample(5, random_state=0).copy()
    pred = predict_dataframe_transformer4(model, preproc, y_scaler, target_log, sample, batch_size=2048)

    out = sample[[args.target]].copy()
    out["예측낙찰가"] = pred
    out["오차(예측-실제)"] = out["예측낙찰가"] - out[args.target]

    print("\n[Sample predictions]")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.round(2).to_string(index=False))

    if args.open_plots:
        for name in ["prediction_scatter.png", "learning_curve_loss.png", "learning_curve_val_rmse.png"]:
            pth = Path(res["output_dir"]) / name
            if pth.exists():
                os.startfile(str(pth))


if __name__ == "__main__":
    main()
