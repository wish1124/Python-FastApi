# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model_transformer2 import (
    DEFAULT_DATA_PATH,
    DEFAULT_OUT_DIR,
    load_single_artifacts,
    predict_dataframe_single,
    read_csv_safely,
    train_kfold_ensemble,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default=None)

    p.add_argument("--target", type=str, default="낙찰가")
    p.add_argument("--features", nargs="*", default=None)

    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test_ratio", type=float, default=0.20)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=20)

    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data) if args.data else DEFAULT_DATA_PATH
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR

    df = read_csv_safely(data_path)

    # base features
    if args.features is not None and len(args.features) > 0:
        base_features = args.features
    else:
        base_features = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]

    required = list(base_features) + [args.target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV에 필요한 컬럼이 없습니다: {missing}\n현재 컬럼: {df.columns.tolist()}")

    if args.quick:
        # 빠른 검증(정확도보다 파이프라인 확인 목적)
        folds = 2
        epochs = 30
        batch = max(256, args.batch)
        lr = args.lr
        patience = 8
        d_model = 128
        nhead = 8
        layers = 3
        ff = 512
        dropout = 0.10
        noise = 0.005
    else:
        folds = args.folds
        epochs = args.epochs
        batch = args.batch
        lr = args.lr
        patience = args.patience
        d_model = 256
        nhead = 8
        layers = 4
        ff = 1024
        dropout = 0.12
        noise = 0.01

    manifest = train_kfold_ensemble(
        df=df,
        base_feature_cols=base_features,
        target_col=args.target,
        target_log=True,
        use_feature_engineering=True,
        clip_q_low=0.005,
        clip_q_high=0.995,
        test_ratio=args.test_ratio,
        folds=folds,
        seed=42,
        deterministic=True,
        epochs=epochs,
        batch_size=batch,
        lr=lr,
        weight_decay=args.wd,
        patience=patience,
        loss_name="huber",
        huber_beta=1.0,
        d_model=d_model,
        nhead=nhead,
        num_layers=layers,
        dim_feedforward=ff,
        dropout=dropout,
        amp=True,
        grad_accum_steps=1,
        feature_noise_std=noise,
        ema_decay=0.999,
        onecycle=True,
        clip_grad=1.0,
        num_workers=0,
        output_dir=str(out_dir),
        verbose=True,
    )

    print("\n[Summary]")
    print(f"Best fold: {manifest['best_fold']}  val_rmse={manifest['best_fold_val_rmse']:.3f}")
    print(f"Ensemble holdout RMSE: {manifest['ensemble_rmse']:.3f}")
    print(f"Artifacts: {manifest['output_dir']}")

    # Convenience: load root-level (best fold copied as best_model.pt/preprocess.json/target_scaler.json)
    model, preproc, y_scaler, target_log, cfg = load_single_artifacts(Path(manifest["output_dir"]))

    sample = df.sample(5, random_state=0).copy()
    pred = predict_dataframe_single(
        model=model,
        preproc=preproc,
        y_scaler=y_scaler,
        target_log=target_log,
        df=sample,
        base_feature_cols=cfg["base_feature_cols"],
        batch_size=2048,
    )

    out = sample[[args.target]].copy()
    out["예측낙찰가"] = pred
    out["오차(예측-실제)"] = out["예측낙찰가"] - out[args.target]

    print("\n[Sample predictions]")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
