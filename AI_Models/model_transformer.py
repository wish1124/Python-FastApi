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


# Transformer Regressor

class TransformerRegressor(nn.Module):
    def __init__(
        self,
        num_features: int = 4,
        d_model: int = 512,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.num_features = num_features
        self.d_model = d_model

        # project scalar value -> d_model
        self.value_proj = nn.Linear(1, d_model)

        # feature id embedding (0..F-1)
        self.feature_emb = nn.Embedding(num_features, d_model)

        # positional embedding (CLS + F tokens)
        self.pos_emb = nn.Embedding(num_features + 1, d_model)

        # learned CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # init
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, 1)
        returns: (B,)
        """
        B, F, _ = x.shape
        if F != self.num_features:
            raise ValueError(f"Expected num_features={self.num_features}, got F={F}")

        # value projection
        v = self.value_proj(x)  # (B, F, d_model)

        # feature id embedding
        feat_ids = torch.arange(F, device=x.device).unsqueeze(0).expand(B, F)  # (B, F)
        v = v + self.feature_emb(feat_ids)  # (B, F, d_model)

        # positions: 1..F for feature tokens (0 reserved for CLS)
        pos_ids = torch.arange(1, F + 1, device=x.device).unsqueeze(0).expand(B, F)  # (B,F)
        v = v + self.pos_emb(pos_ids)

        # CLS token at position 0
        cls = self.cls_token.expand(B, 1, self.d_model) + self.pos_emb(
            torch.zeros((B, 1), device=x.device, dtype=torch.long)
        )  # (B,1,d_model)

        tokens = torch.cat([cls, v], dim=1)  # (B, 1+F, d_model)

        z = self.encoder(tokens)             # (B, 1+F, d_model)
        cls_out = z[:, 0, :]                 # (B, d_model)

        y_hat = self.head(cls_out).squeeze(-1)  # (B,)
        return y_hat


# Metrics

# Metrics
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)
    
    # [추가] R2 Score 계산
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # 분모가 0이 되는 것을 방지하기 위해 eps 더함
    r2 = 1 - (ss_res / (ss_tot + eps))

    return {"MSE": float(mse), "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": float(r2)}



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
    history: Dict[str, List[float]]  # [추가] 학습 기록 저장

def run_training_transformer(
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
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 20,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    output_dir: str = "./results_transformer"  # [추가] 결과 저장 경로
) -> TrainResult:
    
    # 결과 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    seed_everything(seed, deterministic=deterministic)

    # 1) load + numeric
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
        raise ValueError(f"유효 데이터가 너무 적습니다: {len(work)} rows (권장: 최소 수십~수백 이상)")

    X = work[list(feature_cols)].to_numpy(np.float32)  # (N, F)
    y_raw = work[target_col].to_numpy(np.float32)      # (N,)

    # 2) split
    train_idx, val_idx, test_idx = split_indices(len(work), test_ratio, val_ratio, seed)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train_raw, y_val_raw, y_test_raw = y_raw[train_idx], y_raw[val_idx], y_raw[test_idx]

    # 3) X scale
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    # 4) y preprocess: log1p -> standardize
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

    # 5) Dataset/Loader
    train_loader = DataLoader(FeatureSeqDataset(X_train_s, y_train_s), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatureSeqDataset(X_val_s, y_val_s), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FeatureSeqDataset(X_test_s, y_test_s), batch_size=batch_size, shuffle=False)

    # 6) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor(
        num_features=len(feature_cols),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 7) Train loop
    best_state = None
    best_val_rmse = float("inf")
    best_val_metrics: Dict[str, float] = {}
    bad = 0
    history = {"train_loss": [], "val_loss": []} # [추가] Loss 기록

    print(f"Start Transformer Training on {device}...")

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optim, device)

        # Validation Prediction (Scaled 상태)
        yv_true_s, yv_pred_s = predict(model, val_loader, device)
        
        # [추가] Loss Plot을 위해 Scaled 상태에서의 Val MSE 계산 (Train Loss와 비교 가능하도록)
        val_loss_scaled = np.mean((yv_true_s - yv_pred_s) ** 2)
        
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss_scaled)

        # Metrics 계산을 위한 역변환
        yv_true_p = y_scaler.inverse_transform(yv_true_s)
        yv_pred_p = y_scaler.inverse_transform(yv_pred_s)

        if target_log:
            yv_true = np.expm1(yv_true_p)
            yv_pred = np.expm1(yv_pred_p)
        else:
            yv_true, yv_pred = yv_true_p, yv_pred_p

        val_m = regression_metrics(yv_true, yv_pred)

        print(
            f"Epoch {epoch:03d} | train_loss={tr_loss:.6f} | val_loss(scaled)={val_loss_scaled:.6f} | "
            f"val_RMSE={val_m['RMSE']:.4f}"
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

    # 8) [추가] 모델 저장 및 시각화
    
    # Best Model 저장
    if best_state is not None:
        model.load_state_dict(best_state)
        save_path = os.path.join(output_dir, "best_transformer.pt")
        torch.save(best_state, save_path)
        print(f"Saved best model to {save_path}")

    # Loss Graph 저장
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss (Scaled MSE)')
    plt.plot(history['val_loss'], label='Val Loss (Scaled MSE)')
    plt.title('Transformer Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_history.png'))
    plt.close()

    # 9) Final Test & Scatter Plot
    yt_true_s, yt_pred_s = predict(model, test_loader, device)
    yt_true_p = y_scaler.inverse_transform(yt_true_s)
    yt_pred_p = y_scaler.inverse_transform(yt_pred_s)

    if target_log:
        yt_true = np.expm1(yt_true_p)
        yt_pred = np.expm1(yt_pred_p)
    else:
        yt_true, yt_pred = yt_true_p, yt_pred_p

    test_m = regression_metrics(yt_true, yt_pred)
    print(f"[FINAL TEST] RMSE={test_m['RMSE']:.4f} | MAE={test_m['MAE']:.4f} | MAPE={test_m['MAPE']:.2f} | R2={test_m['R2']:.4f}")


    # Scatter Plot 저장 (Prediction vs Actual)
    plt.figure(figsize=(8, 8))
    plt.scatter(yt_true, yt_pred, alpha=0.5, s=10)
    
    # Perfect Fit Line
    min_val = min(yt_true.min(), yt_pred.min())
    max_val = max(yt_true.max(), yt_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Fit')
    
    plt.title(f'Transformer: Actual vs Predicted\nRMSE: {test_m["RMSE"]:.2f}, MAE: {test_m["MAE"]:.2f}, R2: {test_m["R2"]:.4f}')
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
        best_val=best_val_metrics,
        test=test_m,
        history=history
    )
