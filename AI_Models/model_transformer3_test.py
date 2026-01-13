# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from model_transformer3 import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUT_DIR,
    load_artifacts_transformer3,
    predict_dataframe_transformer3,
    read_csv_safely,
    run_training_transformer3,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default=None)

    p.add_argument("--target", type=str, default="낙찰가")
    p.add_argument("--features", nargs="*", default=None)

    p.add_argument("--fe_mode", type=str, default="light", choices=["none", "light", "full"])
    p.add_argument("--full", action="store_true")  # accuracy-ish preset (still anti-overfit)
    p.add_argument("--max_rows", type=int, default=200000)  # 0 => use all

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=10)

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

    required = list(base_features) + [args.target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {missing}\n현재 컬럼: {df.columns.tolist()}")

    if args.full:
        # still "anti-overfit", but less aggressive downsampling and slightly more epochs
        fe_mode = args.fe_mode if args.fe_mode else "light"
        max_rows = None if args.max_rows == 0 else min(int(args.max_rows), 400000)
        epochs = max(args.epochs, 100)
        batch = max(256, args.batch)
        patience = max(args.patience, 12)
        wd = max(args.wd, 5e-4)
    else:
        fe_mode = args.fe_mode
        max_rows = None if args.max_rows == 0 else int(args.max_rows)
        epochs = args.epochs
        batch = args.batch
        patience = args.patience
        wd = args.wd

    res = run_training_transformer3(
        df=df,
        base_feature_cols=base_features,
        target_col=args.target,
        output_dir=str(out_dir),
        target_log=True,
        fe_mode=fe_mode,
        max_rows=max_rows,
        epochs=epochs,
        batch_size=batch,
        lr=args.lr,
        weight_decay=wd,
        patience=patience,
        verbose=True,
        show_tqdm=True,
    )

    print("\n[Best VAL metrics]")
    for k, v in res.best_val_metrics.items():
        print(f"{k:>6}: {v:.6f}")

    print("\n[TEST metrics]")
    for k, v in res.test_metrics.items():
        print(f"{k:>6}: {v:.6f}")

    print(f"\nArtifacts saved to: {res.output_dir}")

    # reload and sample prediction
    model, preproc, y_scaler, target_log, cfg = load_artifacts_transformer3(Path(res.output_dir))

    sample = df.sample(5, random_state=0).copy()
    pred = predict_dataframe_transformer3(
        model=model,
        preproc=preproc,
        y_scaler=y_scaler,
        target_log=target_log,
        df=sample,
        batch_size=2048,
    )

    out = sample[[args.target]].copy()
    out["예측낙찰가"] = pred
    out["오차(예측-실제)"] = out["예측낙찰가"] - out[args.target]

    print("\n[Sample predictions]")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.round(2).to_string(index=False))

    if args.open_plots:
        scatter = Path(res.output_dir) / "prediction_scatter.png"
        loss = Path(res.output_dir) / "learning_curve_loss.png"
        rmse = Path(res.output_dir) / "learning_curve_val_rmse.png"
        for p in [scatter, loss, rmse]:
            if p.exists():
                os.startfile(str(p))


if __name__ == "__main__":
    main()
