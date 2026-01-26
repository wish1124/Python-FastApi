# -*- coding: utf-8 -*-
"""
Evaluate the saved 4-feature Transformer on a held-out test CSV.
- Model weights: results_transformer_4feat/transformer_4feat.pt
- Scalers/config: results_transformer_4feat/scalers.json, train_config.json
- Test data: dataset/test_26_01.csv
Outputs: metrics JSON, prediction CSV, scatter plot under results_transformer_4feat/eval_test_26_01
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model_transformer import TransformerRegressor, load_scalers_json, read_csv_safely

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_model(config: dict, num_features: int) -> TransformerRegressor:
    return TransformerRegressor(
        num_features=num_features,
        d_model=config.get("d_model", 64),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 2),
        dim_feedforward=config.get("dim_feedforward", 256),
        dropout=config.get("dropout", 0.1),
    )


def load_model(model_path: Path, config: dict, num_features: int) -> TransformerRegressor:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model = build_model(config, num_features)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-12
    r2 = float(1 - ss_res / ss_tot)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def main():
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent

    model_path = project_root / "results_transformer_4feat" / "transformer_4feat.pt"
    config_path = project_root / "results_transformer_4feat" / "train_config.json"
    scalers_path = project_root / "results_transformer_4feat" / "scalers.json"
    test_csv = project_root / "dataset" / "test_26_01.csv"
    output_dir = project_root / "results_transformer_4feat" / "eval_test_26_01"

    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    config = load_config(config_path)
    feature_cols = config.get("feature_cols", [])
    target_col = config.get("target_col", "낙찰가")

    df = read_csv_safely(test_csv)
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in test CSV: {missing_cols}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_true = df[target_col].to_numpy(dtype=np.float32)

    x_scaler, y_scaler, target_log = load_scalers_json(scalers_path)
    X_scaled = x_scaler.transform(X)
    x_tensor = torch.from_numpy(X_scaled).unsqueeze(-1)  # (B, F, 1)

    model = load_model(model_path, config, num_features=len(feature_cols))

    with torch.no_grad():
        pred_scaled = model(x_tensor.to(DEVICE)).cpu().numpy()

    pred_denorm = y_scaler.inverse_transform(pred_scaled)
    y_pred = np.expm1(pred_denorm) if target_log else pred_denorm
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    metrics = calculate_metrics(y_true, y_pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    preds_path = output_dir / "predictions.csv"
    plot_path = output_dir / "scatter_plot.png"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pred_df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred,
    })
    pred_df["Error"] = pred_df["Predicted"] - pred_df["Actual"]
    pred_df["ErrorPct"] = np.where(
        pred_df["Actual"] != 0,
        pred_df["Error"] / pred_df["Actual"] * 100.0,
        np.nan,
    )
    pred_df.to_csv(preds_path, index=False)

    plt.figure(figsize=(7, 7))
    plt.scatter(pred_df["Actual"], pred_df["Predicted"], alpha=0.4, s=10, color="blue")
    mn = min(pred_df["Actual"].min(), pred_df["Predicted"].min())
    mx = max(pred_df["Actual"].max(), pred_df["Predicted"].max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect Fit")
    plt.title(f"Actual vs Predicted (n={len(pred_df):,})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    print("\n=== Transformer 4-Feat Test ===")
    print(f"Test samples: {len(df):,}")
    print(f"Metrics saved to: {metrics_path}")
    for k, v in metrics.items():
        print(f"{k}: {v:.6e}")
    print(f"R2 Score (friendly): {metrics.get('R2', float('nan')):.6f}")
    print(f"Predictions saved to: {preds_path}")
    print(f"Scatter plot saved to: {plot_path}")

    # Sample preview
    preview = pred_df.head(5).copy()
    pd.options.display.float_format = '{:,.2f}'.format
    print("\n[Sample predictions]")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
