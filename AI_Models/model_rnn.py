import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Reproducibility

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# StandardScaler

class StandardScaler:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=np.float32)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler is not fitted.")
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("StandardScaler is not fitted.")
        x = np.asarray(x, dtype=np.float32)
        return x * self.std_ + self.mean_


# y용 스케일러

class TargetScaler:
    def __init__(self):
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, y: np.ndarray) -> "TargetScaler":
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = float(y.mean())
        self.std_ = float(y.std())
        if self.std_ < 1e-12:
            self.std_ = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetScaler is not fitted.")
        y = np.asarray(y, dtype=np.float32)
        return (y - self.mean_) / self.std_

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetScaler is not fitted.")
        y = np.asarray(y, dtype=np.float32)
        return y * self.std_ + self.mean_


# Dataset

class FeatureSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D (N,F). Got {self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1D (N,). Got {self.y.shape}")
        if len(self.X) != len(self.y):
            raise ValueError("X and y must have same length.")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].reshape(-1, 1)          # (F,1)
        y = float(self.y[idx])                  # already processed
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# Model

class RNNRegressor(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.rnn(x)
        last = h_n[-1]
        return self.head(last).squeeze(-1)


# Metrics

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    return {"MSE": float(mse), "RMSE": rmse, "MAE": mae, "MAPE": mape}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        y_hat = model(x).detach().cpu().numpy()
        ys.extend(y.numpy().tolist())
        preds.extend(y_hat.tolist())
    return np.asarray(ys, dtype=np.float64), np.asarray(preds, dtype=np.float64)


def train_one_epoch(model: nn.Module, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        y_hat = model(x)
        loss = nn.functional.mse_loss(y_hat, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * x.size(0)
    return total / max(1, len(loader.dataset))


def split_indices(n: int, test_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    test_cut = int(n * (1.0 - test_ratio))
    trainval_idx, test_idx = idx[:test_cut], idx[test_cut:]

    val_cut = int(len(trainval_idx) * (1.0 - val_ratio))
    train_idx, val_idx = trainval_idx[:val_cut], trainval_idx[val_cut:]
    return train_idx, val_idx, test_idx


@dataclass
class TrainResult:
    model: nn.Module
    x_scaler: StandardScaler
    y_scaler: TargetScaler
    best_val: Dict[str, float]
    test: Dict[str, float]


def run_training_rnn(
    df: pd.DataFrame,
    feature_cols: Sequence[str] = ("기초금액", "추정가격", "예가범위", "낙찰하한율"),
    target_col: str = "낙찰가",
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    seed: int = 42,
    deterministic: bool = True,
    target_log: bool = True,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 1e-2,
    patience: int = 30,
):
    seed_everything(seed, deterministic=deterministic)

    use_cols = list(feature_cols) + [target_col]
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    work = df[use_cols].copy()
    for c in feature_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=use_cols).reset_index(drop=True)

    if len(work) < 50:
        raise ValueError(f"유효 데이터가 너무 적습니다: {len(work)} rows")

    X = work[list(feature_cols)].to_numpy(np.float32)
    y_raw = work[target_col].to_numpy(np.float32)

    train_idx, val_idx, test_idx = split_indices(len(work), test_ratio, val_ratio, seed)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train_raw, y_val_raw, y_test_raw = y_raw[train_idx], y_raw[val_idx], y_raw[test_idx]

    # X scale
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    # y preprocess: log1p then standardize
    if target_log:
        y_train_p = np.log1p(y_train_raw)
        y_val_p = np.log1p(y_val_raw)
        y_test_p = np.log1p(y_test_raw)
    else:
        y_train_p, y_val_p, y_test_p = y_train_raw, y_val_raw, y_test_raw

    y_scaler = TargetScaler().fit(y_train_p)
    y_train_s = y_scaler.transform(y_train_p)
    y_val_s = y_scaler.transform(y_val_p)
    y_test_s = y_scaler.transform(y_test_p)

    train_loader = DataLoader(FeatureSeqDataset(X_train_s, y_train_s), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatureSeqDataset(X_val_s, y_val_s), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FeatureSeqDataset(X_test_s, y_test_s), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RNNRegressor(hidden_size=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_state = None
    best_val_rmse = float("inf")
    best_val_metrics = {}
    bad = 0

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, device)

        yv_true_s, yv_pred_s = predict(model, val_loader, device)
        # inverse y scaling
        yv_true_p = y_scaler.inverse_transform(yv_true_s)
        yv_pred_p = y_scaler.inverse_transform(yv_pred_s)

        # if log, convert to amount
        if target_log:
            yv_true = np.expm1(yv_true_p)
            yv_pred = np.expm1(yv_pred_p)
        else:
            yv_true, yv_pred = yv_true_p, yv_pred_p

        val_m = regression_metrics(yv_true, yv_pred)

        print(
            f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | "
            f"val_RMSE={val_m['RMSE']:.4f} | val_MAE={val_m['MAE']:.4f} | val_MAPE={val_m['MAPE']:.2f}"
        )

        if val_m["RMSE"] < best_val_rmse:
            best_val_rmse = val_m["RMSE"]
            best_val_metrics = val_m
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping (patience={patience}).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final test
    yt_true_s, yt_pred_s = predict(model, test_loader, device)
    yt_true_p = y_scaler.inverse_transform(yt_true_s)
    yt_pred_p = y_scaler.inverse_transform(yt_pred_s)
    if target_log:
        yt_true = np.expm1(yt_true_p)
        yt_pred = np.expm1(yt_pred_p)
    else:
        yt_true, yt_pred = yt_true_p, yt_pred_p

    test_m = regression_metrics(yt_true, yt_pred)
    print(f"[FINAL TEST] RMSE={test_m['RMSE']:.4f} | MAE={test_m['MAE']:.4f} | MAPE={test_m['MAPE']:.2f}")

    return TrainResult(model=model, x_scaler=x_scaler, y_scaler=y_scaler, best_val=best_val_metrics, test=test_m)