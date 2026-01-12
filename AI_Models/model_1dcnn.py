import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import matplotlib.pyplot as plt



# Reproducibility

def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Scalers

class StandardScaler:
    """Feature scaler for X (2D: N x F). Fit on train only."""
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


class TargetScaler:
    """Scaler for y (1D). Fit on train only."""
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
        x = self.X[idx].reshape(-1, 1)  # (F,1)
        y = float(self.y[idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# 1D CNN Regressor

class CNN1DRegressor(nn.Module):
    def __init__(self, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden, kernel_size=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=2),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B, hidden, 1)
        self.head = nn.Sequential(
            nn.Flatten(),                     # (B, hidden)
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, 1) -> (B, 1, F)
        x = x.transpose(1, 2)
        z = self.conv(x)               # (B, hidden, L')
        z = self.pool(z)               # (B, hidden, 1)
        y_hat = self.head(z).squeeze(-1)  # (B,)
        return y_hat


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
    ys: List[float] = []
    preds: List[float] = []
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
    if test_ratio <= 0 or val_ratio < 0 or (test_ratio + val_ratio) >= 1.0:
        raise ValueError("Invalid split ratios. Require: test_ratio>0, val_ratio>=0, test_ratio+val_ratio<1.")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    test_cut = int(n * (1.0 - test_ratio))
    trainval_idx, test_idx = idx[:test_cut], idx[test_cut:]

    val_cut = int(len(trainval_idx) * (1.0 - val_ratio))
    train_idx, val_idx = trainval_idx[:val_cut], trainval_idx[val_cut:]
    return train_idx, val_idx, test_idx


# Runner

@dataclass
class TrainResult:
    model: nn.Module
    x_scaler: StandardScaler
    y_scaler: TargetScaler
    best_val: Dict[str, float]
    test: Dict[str, float]
    history: Dict[str, list]


def run_training_cnn1d(
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
    weight_decay: float = 1e-5,
    patience: int = 20,
    hidden: int = 64,
    dropout: float = 0.1,
    output_dir: str = "./results"  # [추가] 결과 저장 경로
) -> TrainResult:
    
    # 결과 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    seed_everything(seed, deterministic=deterministic)

    # 1) 데이터 로드 및 전처리
    use_cols = list(feature_cols) + [target_col]
    work = df[use_cols].copy()
    for c in use_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna().reset_index(drop=True)

    X = work[list(feature_cols)].to_numpy(np.float32)
    y_raw = work[target_col].to_numpy(np.float32)

    # 타겟 로그 변환 (옵션)
    if target_log:
        y_raw = np.log1p(y_raw)

    # 2) 데이터 분할
    train_idx, val_idx, test_idx = split_indices(len(X), test_ratio, val_ratio, seed)

    # 3) 스케일링 (Train 기준으로 Fit)
    x_scaler = StandardScaler().fit(X[train_idx])
    X_scaled = x_scaler.transform(X)

    y_scaler = TargetScaler().fit(y_raw[train_idx])
    y_scaled = y_scaler.transform(y_raw)

    # 4) 데이터셋 및 로더 생성
    train_ds = FeatureSeqDataset(X_scaled[train_idx], y_scaled[train_idx])
    val_ds = FeatureSeqDataset(X_scaled[val_idx], y_scaled[val_idx])
    test_ds = FeatureSeqDataset(X_scaled[test_idx], y_scaled[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 5) 모델 초기화
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DRegressor(hidden=hidden, dropout=dropout).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 6) 학습 루프
    best_loss = float("inf")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"Start Training on {device}...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optim, device)
        
        # Validation 평가
        model.eval()
        val_ys, val_preds = predict(model, val_loader, device)
        val_mse = np.mean((val_ys - val_preds) ** 2)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_mse)

        if val_mse < best_loss:
            best_loss = val_mse
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | Val: {val_mse:.4f}")

    # 7) [추가] 모델 저장 (.pt 파일)
    if best_state is not None:
        model.load_state_dict(best_state)
        save_path = os.path.join(output_dir, "best_model.pt")
        torch.save(best_state, save_path)
        print(f"Saved best model to {save_path}")

    # 8) [추가] 학습 결과 시각화 1: Train/Val Loss 그래프
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()

    # 9) 최종 테스트 평가
    test_ys, test_preds = predict(model, test_loader, device)
    
    # 스케일 역변환 (원래 가격 단위로 복구)
    real_y = y_scaler.inverse_transform(test_ys)
    pred_y = y_scaler.inverse_transform(test_preds)
    
    if target_log:
        real_y = np.expm1(real_y)
        pred_y = np.expm1(pred_y)

    metrics = regression_metrics(real_y, pred_y)
    print(f"Test Metrics: {metrics}")

    # 10) [추가] 학습 결과 시각화 2: Prediction vs Actual (Confusion Matrix 대체)
    plt.figure(figsize=(8, 8))
    plt.scatter(real_y, pred_y, alpha=0.5, s=10)
    
    # 기준선 (Perfect Prediction Line)
    min_val = min(real_y.min(), pred_y.min())
    max_val = max(real_y.max(), pred_y.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    
    plt.title(f'Actual vs Predicted (Test Set)\nMAE: {metrics["MAE"]:.2f}, RMSE: {metrics["RMSE"]:.2f}')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'))
    plt.close()

    return TrainResult(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        best_val={"MSE": best_loss},
        test=metrics,
        history=history
    )



if __name__ == "__main__":
    pass
