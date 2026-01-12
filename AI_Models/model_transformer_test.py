import numpy as np
import pandas as pd
import torch

from model_transformer import run_training_transformer


def main():
    csv_path = "test_bid_4features.csv"
    df = pd.read_csv(csv_path)

    required = ["기초금액", "추정가격", "예가범위", "낙찰하한율", "낙찰가"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"CSV에 필수 컬럼이 없습니다: {missing}")

    res = run_training_transformer(
        df=df,
        feature_cols=("기초금액", "추정가격", "예가범위", "낙찰하한율"),
        target_col="낙찰가",
        target_log=True,
        epochs=200,
        patience=20,
        batch_size=256,
        lr=3e-4,
        weight_decay=1e-4,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )

    print("\nBest VAL:", res.best_val)
    print("TEST:", res.test)

    feature_cols = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]
    sample = df.sample(5, random_state=0).copy()

    X = sample[feature_cols].to_numpy(np.float32)
    X_s = res.x_scaler.transform(X)

    x_t = torch.from_numpy(X_s).reshape(len(sample), -1, 1)

    device = next(res.model.parameters()).device
    res.model.eval()
    with torch.no_grad():
        pred_s = res.model(x_t.to(device)).cpu().numpy()

    pred_log = res.y_scaler.inverse_transform(pred_s)
    pred_amt = np.expm1(pred_log)

    out = sample[["낙찰가"]].copy()
    out["예측낙찰가"] = pred_amt
    out["오차(예측-실제)"] = out["예측낙찰가"] - out["낙찰가"]

    print("\n[Sample predictions]")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
