import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model_rnn import run_training_rnn
from model_1dcnn import run_training_cnn1d
from model_transformer import run_training_transformer


FEATURE_COLS = ("기초금액", "추정가격", "예가범위", "낙찰하한율")
TARGET_COL = "낙찰가"

DATA_BASENAME = "test_bid_4features" 

SEED = 42
TEST_RATIO = 0.20
VAL_RATIO = 0.10
TARGET_LOG = True

PLOTS_DIR = "plots"


def load_data_auto(basename: str) -> pd.DataFrame:
    candidates = [
        f"{basename}.csv",
        f"{basename}.xlsx",
        f"{basename}.xls",
        basename,
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        if ext == "":
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.read_excel(path)

    raise FileNotFoundError(
        f"데이터 파일을 찾을 수 없습니다. 시도한 후보: {candidates}\n현재 폴더: {os.getcwd()}"
    )


def split_indices(n: int, test_ratio: float, val_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    test_cut = int(n * (1.0 - test_ratio))
    trainval_idx, test_idx = idx[:test_cut], idx[test_cut:]

    val_cut = int(len(trainval_idx) * (1.0 - val_ratio))
    train_idx, val_idx = trainval_idx[:val_cut], trainval_idx[val_cut:]
    return train_idx, val_idx, test_idx


@torch.no_grad()
def predict_amount(res, X_raw: np.ndarray, target_log: bool = True) -> np.ndarray:
    X_s = res.x_scaler.transform(X_raw.astype(np.float32))
    x_t = torch.from_numpy(X_s).reshape(len(X_s), -1, 1)

    device = next(res.model.parameters()).device
    res.model.eval()
    pred_s = res.model(x_t.to(device)).cpu().numpy()

    pred_p = res.y_scaler.inverse_transform(pred_s)
    if target_log:
        pred_amt = np.expm1(pred_p)
    else:
        pred_amt = pred_p
    return pred_amt.astype(np.float64)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    return {"MSE": float(mse), "RMSE": rmse, "MAE": mae, "MAPE": mape}


def plot_bar(results_df: pd.DataFrame, metric: str, save_path: str = None):
    plt.figure()
    plt.bar(results_df["model"], results_df[metric])
    plt.title(f"TEST {metric} Comparison (same split)")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_parity_overlay(y_true: np.ndarray, preds: dict, save_path: str = None, max_points: int = 2000):
    y_true = np.asarray(y_true, dtype=np.float64)

    n = len(y_true)
    if n > max_points:
        rng = np.random.default_rng(0)
        pick = rng.choice(n, size=max_points, replace=False)
        y_true_p = y_true[pick]
        preds_p = {k: np.asarray(v, dtype=np.float64)[pick] for k, v in preds.items()}
    else:
        y_true_p = y_true
        preds_p = {k: np.asarray(v, dtype=np.float64) for k, v in preds.items()}

    plt.figure()
    for name, y_pred in preds_p.items():
        plt.scatter(y_true_p, y_pred, s=10, alpha=0.55, label=name)

    lo = float(min(y_true_p.min(), min(v.min() for v in preds_p.values())))
    hi = float(max(y_true_p.max(), max(v.max() for v in preds_p.values())))
    plt.plot([lo, hi], [lo, hi])

    plt.title("Parity Plot (TEST): Actual vs Predicted")
    plt.xlabel("Actual (낙찰가)")
    plt.ylabel("Predicted (예측낙찰가)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def _has_history_key(res, key: str) -> bool:
    return hasattr(res, "history") and isinstance(res.history, dict) and (key in res.history)


def plot_train_vs_val_loss(res_dict: dict, save_path: str = None):
    """
    한 그래프에서:
      - 모델별 train_loss vs val_loss (둘 다 y_scaled MSE)
    """
    plt.figure()
    any_plotted = False

    for name, res in res_dict.items():
        if not _has_history_key(res, "train_loss"):
            print(f"[WARN] {name}: history['train_loss'] 없음 -> 제외")
            continue
        if not _has_history_key(res, "val_loss"):
            print(f"[WARN] {name}: history['val_loss'] 없음 -> 제외 (model 파일에서 val_loss 저장 필요)")
            continue

        tr = res.history["train_loss"]
        va = res.history["val_loss"]

        plt.plot(range(1, len(tr) + 1), tr, label=f"{name} train")
        plt.plot(range(1, len(va) + 1), va, linestyle="--", label=f"{name} val")
        any_plotted = True

    if not any_plotted:
        print("[WARN] train/val loss를 그릴 수 있는 모델이 없습니다.")
        return

    plt.title("Train vs Val Loss (MSE on scaled y)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def plot_val_rmse_curves(res_dict: dict, save_path: str = None):
    """
    모델별 validation RMSE(금액 스케일) 곡선 비교
    """
    plt.figure()
    any_plotted = False

    for name, res in res_dict.items():
        if not _has_history_key(res, "val_rmse"):
            print(f"[WARN] {name}: history['val_rmse'] 없음 -> 제외")
            continue
        vr = res.history["val_rmse"]
        plt.plot(range(1, len(vr) + 1), vr, label=f"{name} val_RMSE")
        best_ep = int(np.argmin(vr)) + 1
        plt.scatter([best_ep], [vr[best_ep - 1]], s=35)

        any_plotted = True

    if not any_plotted:
        print("[WARN] val_rmse 곡선을 그릴 수 있는 모델이 없습니다.")
        return

    plt.title("Validation RMSE Curves (amount scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Val RMSE")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
    plt.show()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data_auto(DATA_BASENAME)

    required = list(FEATURE_COLS) + [TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"데이터에 필수 컬럼이 없습니다: {missing}")

    _, _, test_idx = split_indices(len(df), TEST_RATIO, VAL_RATIO, SEED)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    print("\n=== Train: RNN ===")
    res_rnn = run_training_rnn(
        df=df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        deterministic=True,
        target_log=TARGET_LOG,
        epochs=200,
        patience=20,
        batch_size=256,
        lr=1e-2,
    )

    print("\n=== Train: 1D-CNN ===")
    res_cnn = run_training_cnn1d(
        df=df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        deterministic=True,
        target_log=TARGET_LOG,
        epochs=200,
        patience=20,
        batch_size=256,
        lr=1e-2,
        hidden=64,
        dropout=0.1,
    )

    print("\n=== Train: Transformer ===")
    res_tr = run_training_transformer(
        df=df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=SEED,
        deterministic=True,
        target_log=TARGET_LOG,
        epochs=250,
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

    X_test = df_test[list(FEATURE_COLS)].to_numpy(np.float32)
    y_test = df_test[TARGET_COL].to_numpy(np.float64)

    pred_rnn = predict_amount(res_rnn, X_test, target_log=TARGET_LOG)
    pred_cnn = predict_amount(res_cnn, X_test, target_log=TARGET_LOG)
    pred_tr = predict_amount(res_tr, X_test, target_log=TARGET_LOG)

    m_rnn = regression_metrics(y_test, pred_rnn)
    m_cnn = regression_metrics(y_test, pred_cnn)
    m_tr = regression_metrics(y_test, pred_tr)

    results_df = pd.DataFrame([
        {"model": "RNN", **m_rnn},
        {"model": "1D-CNN", **m_cnn},
        {"model": "Transformer", **m_tr},
    ])

    print("\n=== TEST metrics (same split) ===")
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(PLOTS_DIR, "compare_test_metrics.csv"), index=False, encoding="utf-8-sig")

    plot_bar(results_df, "RMSE", save_path=os.path.join(PLOTS_DIR, "cmp_test_rmse.png"))
    plot_bar(results_df, "MAE", save_path=os.path.join(PLOTS_DIR, "cmp_test_mae.png"))
    plot_bar(results_df, "MAPE", save_path=os.path.join(PLOTS_DIR, "cmp_test_mape.png"))

    plot_parity_overlay(
        y_true=y_test,
        preds={"RNN": pred_rnn, "1D-CNN": pred_cnn, "Transformer": pred_tr},
        save_path=os.path.join(PLOTS_DIR, "cmp_parity_overlay.png"),
        max_points=2000,
    )

    res_dict = {"RNN": res_rnn, "1D-CNN": res_cnn, "Transformer": res_tr}

    plot_train_vs_val_loss(
        res_dict,
        save_path=os.path.join(PLOTS_DIR, "cmp_train_vs_val_loss.png")
    )

    plot_val_rmse_curves(
        res_dict,
        save_path=os.path.join(PLOTS_DIR, "cmp_val_rmse_curves.png")
    )

    print("\nSaved outputs to ./plots/")
    print(" - compare_test_metrics.csv")
    print(" - cmp_test_rmse.png / cmp_test_mae.png / cmp_test_mape.png")
    print(" - cmp_parity_overlay.png")
    print(" - cmp_train_vs_val_loss.png (train vs val loss)")
    print(" - cmp_val_rmse_curves.png (val RMSE curves)")


if __name__ == "__main__":
    main()