# -*- coding: utf-8 -*-
"""
AI_Models/model_transformer_test.py

Train + evaluate + small inference demo (relative paths).

Examples:
  py model_transformer_test.py --quick
  py model_transformer_test.py --epochs 200 --batch 256 --lr 3e-4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model_transformer import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUT_DIR,
    load_model_artifacts,
    predict_dataframe,
    read_csv_safely,
    run_training_transformer,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--no_onecycle", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data) if args.data else DEFAULT_DATA_PATH
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR

    df = read_csv_safely(data_path)

    feature_cols = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]
    target_col = "낙찰가"

    required = feature_cols + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {missing}\n현재 컬럼: {df.columns.tolist()}")

    if args.quick:
        # 파이프라인이 정상인지 빠르게 확인용(속도 우선)
        epochs = 30
        batch = max(128, args.batch)
        lr = args.lr
        patience = 8
        d_model = 64
        nhead = 4
        num_layers = 2
        dim_ff = 256
        dropout = 0.10
        noise_std = 0.005
    else:
        epochs = args.epochs
        batch = args.batch
        lr = args.lr
        patience = args.patience
        d_model = 128
        nhead = 4
        num_layers = 3
        dim_ff = 512
        dropout = 0.15
        noise_std = 0.01

    res = run_training_transformer(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        target_log=True,
        output_dir=str(out_dir),
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        patience=patience,
        split_strategy="stratified",
        loss_name="huber",
        onecycle=not args.no_onecycle,
        ema_decay=0.0 if args.no_ema else 0.999,
        feature_noise_std=noise_std,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=dropout,
        verbose=True,   # 진행 로그 필수
    )

    print("\n[Best VAL metrics]")
    for k, v in res.best_val.items():
        print(f"{k:>6}: {v:.6f}")

    print("\n[TEST metrics]")
    for k, v in res.test.items():
        print(f"{k:>6}: {v:.6f}")

    print(f"\nArtifacts saved to: {res.output_dir}")

    # Reload artifacts and run a small inference demo
    model, x_scaler, y_scaler, target_log, feat_cols = load_model_artifacts(Path(res.output_dir))

    sample = df.sample(5, random_state=0).copy()
    pred = predict_dataframe(
        model=model,
        df=sample,
        feature_cols=feat_cols,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        target_log=target_log,
        amp=True,  # CUDA면 자동 사용, CPU면 내부에서 사실상 off
    )

    out = sample[[target_col]].copy()
    out["예측낙찰가"] = pred
    out["오차(예측-실제)"] = out["예측낙찰가"] - out[target_col]

    print("\n[Sample predictions]")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
