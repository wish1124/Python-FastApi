import random
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Sequence, Tuple
from copy import deepcopy
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ==============================================================================
# 1. Reproducibility & Utility
# ==============================================================================
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==============================================================================
# 2. Scalers
# ==============================================================================
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
        if self.mean_ is None: raise RuntimeError("Scaler not fitted")
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean_) / self.std_

class TargetScaler:
    def __init__(self):
        self.mean_: float = 0.0
        self.std_: float = 1.0

    def fit(self, y: np.ndarray) -> "TargetScaler":
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = float(y.mean())
        self.std_ = float(y.std())
        if self.std_ < 1e-12: self.std_ = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y) - self.mean_) / self.std_

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y) * self.std_ + self.mean_

# ==============================================================================
# 3. Dataset
# ==============================================================================
class FeatureSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
    def __len__(self) -> int: return len(self.y)
    def __getitem__(self, idx: int):
        x = self.X[idx].reshape(1, -1) # (1, F)
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

# ==============================================================================
# 4. Models (CNN & ResNet)
# ==============================================================================
class CNN1DRegressor(nn.Module):
    def __init__(self, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden, out_channels=hidden, kernel_size=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = self.pool(z)
        return self.head(z).squeeze(-1)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.downsample is not None: identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet1DRegressor(nn.Module):
    def __init__(self, layers: List[int] = [2, 2, 2, 2], base_filters: int = 64):
        super().__init__()
        self.in_channels = base_filters
        self.conv1 = nn.Conv1d(1, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(base_filters, layers[0])
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 8, 1)
    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.squeeze(-1)

# ==============================================================================
# 5. Training Functions
# ==============================================================================
@dataclass
class TrainResult:
    model: nn.Module
    x_scaler: StandardScaler
    y_scaler: TargetScaler
    best_val: Dict[str, float]
    test: Dict[str, float]
    history: Dict[str, List[float]]  # [추가됨] Loss 기록용

def _prepare_data(df, feature_cols, target_col, test_ratio, val_ratio, seed, target_log):
    seed_everything(seed)
    use_cols = list(feature_cols) + [target_col]
    data = df[use_cols].dropna().reset_index(drop=True)
    X = data[feature_cols].to_numpy(np.float32)
    y = data[target_col].to_numpy(np.float32)
    if target_log: y = np.log1p(y)
    n = len(data)
    idx = np.arange(n)
    np.random.shuffle(idx)
    test_cut = int(n * (1 - test_ratio))
    trainval_idx, test_idx = idx[:test_cut], idx[test_cut:]
    val_cut = int(len(trainval_idx) * (1 - val_ratio))
    train_idx, val_idx = trainval_idx[:val_cut], trainval_idx[val_cut:]
    x_scaler = StandardScaler().fit(X[train_idx])
    y_scaler = TargetScaler().fit(y[train_idx])
    return (X, y, train_idx, val_idx, test_idx, x_scaler, y_scaler)

def _train_loop(model, train_loader, val_loader, test_loader, epochs, lr, patience, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_loss = float('inf')
    bad_epochs = 0
    best_state = None
    
    # [추가됨] Loss 기록 리스트
    history = {"train_loss": [], "val_loss": []}
    
    print(f"Start Training (Device: {device})...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                val_loss += criterion(pred, by).item() * bx.size(0)
        val_loss /= len(val_loader.dataset)
        
        # 기록
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            bad_epochs = 0
            best_state = deepcopy(model.state_dict())
            if epoch % 10 == 0: print(f"Epoch {epoch}: Train {train_loss:.5f} | Val {val_loss:.5f} (*)")
        else:
            bad_epochs += 1
            if epoch % 10 == 0: print(f"Epoch {epoch}: Train {train_loss:.5f} | Val {val_loss:.5f}")
        if bad_epochs >= patience:
            print(f"Early Stopping at epoch {epoch}")
            break
            
    if best_state: model.load_state_dict(best_state)
    return model, best_loss, history

def _evaluate(model, test_loader, device, y_scaler, target_log):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            p = model(bx).cpu().numpy()
            preds.extend(p)
            actuals.extend(by.numpy())
    preds = y_scaler.inverse_transform(np.array(preds))
    actuals = y_scaler.inverse_transform(np.array(actuals))
    if target_log:
        preds = np.expm1(preds)
        actuals = np.expm1(actuals)
    mse = np.mean((actuals - preds)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actuals - preds))
    if np.sum((actuals - np.mean(actuals))**2) < 1e-8: r2 = 0.0
    else: r2 = 1 - (np.sum((actuals - preds)**2) / np.sum((actuals - np.mean(actuals))**2))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def run_training_cnn1d(
    df: pd.DataFrame, feature_cols: Sequence[str], target_col: str = "낙찰가",
    test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 42,
    epochs: int = 200, batch_size: int = 256, lr: float = 1e-2, patience: int = 20,
    hidden: int = 64, dropout: float = 0.1, target_log: bool = True, output_dir: str = "./results_cnn"
) -> TrainResult:
    os.makedirs(output_dir, exist_ok=True)
    X, y, train_idx, val_idx, test_idx, x_scaler, y_scaler = _prepare_data(
        df, feature_cols, target_col, test_ratio, val_ratio, seed, target_log
    )
    X_tr_s = x_scaler.transform(X[train_idx])
    y_tr_s = y_scaler.transform(y[train_idx])
    X_val_s = x_scaler.transform(X[val_idx])
    y_val_s = y_scaler.transform(y[val_idx])
    X_te_s = x_scaler.transform(X[test_idx])
    y_te_s = y_scaler.transform(y[test_idx])
    
    train_loader = DataLoader(FeatureSeqDataset(X_tr_s, y_tr_s), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatureSeqDataset(X_val_s, y_val_s), batch_size=batch_size*2)
    test_loader = DataLoader(FeatureSeqDataset(X_te_s, y_te_s), batch_size=batch_size*2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1DRegressor(hidden=hidden, dropout=dropout).to(device)
    
    model, best_loss, history = _train_loop(model, train_loader, val_loader, test_loader, epochs, lr, patience, device)
    test_metrics = _evaluate(model, test_loader, device, y_scaler, target_log)
    print(f"[CNN1D] Final Test: {test_metrics}")
    
    return TrainResult(model=model, x_scaler=x_scaler, y_scaler=y_scaler, best_val={"MSE": best_loss}, test=test_metrics, history=history)

def run_training_resnet1d(
    df: pd.DataFrame, feature_cols: Sequence[str], target_col: str = "낙찰가",
    test_ratio: float = 0.2, val_ratio: float = 0.1, seed: int = 42,
    epochs: int = 100, batch_size: int = 256, lr: float = 1e-3, patience: int = 15,
    target_log: bool = True, output_dir: str = "./results_resnet"
) -> TrainResult:
    os.makedirs(output_dir, exist_ok=True)
    X, y, train_idx, val_idx, test_idx, x_scaler, y_scaler = _prepare_data(
        df, feature_cols, target_col, test_ratio, val_ratio, seed, target_log
    )
    X_tr_s = x_scaler.transform(X[train_idx])
    y_tr_s = y_scaler.transform(y[train_idx])
    X_val_s = x_scaler.transform(X[val_idx])
    y_val_s = y_scaler.transform(y[val_idx])
    X_te_s = x_scaler.transform(X[test_idx])
    y_te_s = y_scaler.transform(y[test_idx])
    
    train_loader = DataLoader(FeatureSeqDataset(X_tr_s, y_tr_s), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatureSeqDataset(X_val_s, y_val_s), batch_size=batch_size*2)
    test_loader = DataLoader(FeatureSeqDataset(X_te_s, y_te_s), batch_size=batch_size*2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet1DRegressor(layers=[2, 2, 2, 2], base_filters=64).to(device)
    
    model, best_loss, history = _train_loop(model, train_loader, val_loader, test_loader, epochs, lr, patience, device)
    test_metrics = _evaluate(model, test_loader, device, y_scaler, target_log)
    print(f"[ResNet1D] Final Test: {test_metrics}")
    
    return TrainResult(model=model, x_scaler=x_scaler, y_scaler=y_scaler, best_val={"MSE": best_loss}, test=test_metrics, history=history)
