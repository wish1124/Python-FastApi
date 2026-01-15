# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt


# ---------- Relative paths ----------
BASE_DIR = Path(__file__).resolve().parent       # .../BidAssitance/AI_Models
PROJECT_ROOT = BASE_DIR.parent                   # .../BidAssitance
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset" / "preprocessed_dataset.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results_transformer2"


# =========================
# Utils
# =========================
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def _safe_expm1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50.0, 50.0)
    return np.expm1(x)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / (ss_tot + eps))

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": float(r2)}


# =========================
# AMP (new API + safe fallback)
# =========================
def _amp_available_for_device(device: torch.device, amp_flag: bool) -> bool:
    return bool(amp_flag and device.type == "cuda")


def _get_autocast(device: torch.device, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.autocast(enabled=enabled)


def _get_grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device_type=device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


# =========================
# Scalers / Preprocessor
# =========================
class StandardScaler:
    def __init__(self) -> None:
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
    def __init__(self) -> None:
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, y: np.ndarray) -> "TargetScaler":
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = float(y.mean())
        std = float(y.std())
        self.std_ = std if std > 1e-12 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetScaler is not fitted.")
        y = np.asarray(y, dtype=np.float32)
        return (y - float(self.mean_)) / float(self.std_)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetScaler is not fitted.")
        y = np.asarray(y, dtype=np.float32)
        return y * float(self.std_) + float(self.mean_)


def build_engineered_features(X: np.ndarray, base_names: Sequence[str], eps: float = 1e-6) -> Tuple[np.ndarray, List[str]]:
    """
    Accuracy-first feature engineering for numeric base features.
    X: (N,F)
    Returns: X_eng (N, F_eng), names
    """
    X = np.asarray(X, dtype=np.float32)
    N, F = X.shape
    names: List[str] = []
    feats: List[np.ndarray] = []

    # 1) raw
    feats.append(X)
    names.extend([f"{c}" for c in base_names])

    # 2) per-feature transforms
    X_pos = np.clip(X, 0.0, None)
    feats.append(np.log1p(X_pos).astype(np.float32))
    names.extend([f"log1p({c})" for c in base_names])

    feats.append(np.sqrt(X_pos + 1e-6).astype(np.float32))
    names.extend([f"sqrt({c})" for c in base_names])

    # 3) row-level aggregates
    row_sum = np.sum(X, axis=1, keepdims=True)
    row_mean = np.mean(X, axis=1, keepdims=True)
    row_min = np.min(X, axis=1, keepdims=True)
    row_max = np.max(X, axis=1, keepdims=True)
    feats.extend([row_sum, row_mean, row_min, row_max])
    names.extend(["row_sum", "row_mean", "row_min", "row_max"])

    # 4) pairwise interactions
    #   +, -, *, ratio (both directions)
    for i in range(F):
        for j in range(i + 1, F):
            a = X[:, i]
            b = X[:, j]
            feats.append((a + b).reshape(N, 1).astype(np.float32))
            names.append(f"({base_names[i]}+{base_names[j]})")

            feats.append((a - b).reshape(N, 1).astype(np.float32))
            names.append(f"({base_names[i]}-{base_names[j]})")

            feats.append((a * b).reshape(N, 1).astype(np.float32))
            names.append(f"({base_names[i]}*{base_names[j]})")

            feats.append((a / (np.abs(b) + eps)).reshape(N, 1).astype(np.float32))
            names.append(f"({base_names[i]}/abs({base_names[j]}))")

            feats.append((b / (np.abs(a) + eps)).reshape(N, 1).astype(np.float32))
            names.append(f"({base_names[j]}/abs({base_names[i]}))")

    X_eng = np.concatenate(feats, axis=1).astype(np.float32)
    return X_eng, names


class TabularPreprocessor:
    """
    - numeric coercion
    - base feature clipping by quantiles (fit on train only)
    - engineered features (optional)
    - standard scaling on engineered feature space (fit on train only)
    """
    def __init__(
        self,
        base_feature_cols: Sequence[str],
        use_feature_engineering: bool = True,
        clip_q_low: float = 0.005,
        clip_q_high: float = 0.995,
    ) -> None:
        self.base_feature_cols = list(base_feature_cols)
        self.use_feature_engineering = bool(use_feature_engineering)
        self.clip_q_low = float(clip_q_low)
        self.clip_q_high = float(clip_q_high)

        self.clip_lo_: Optional[np.ndarray] = None
        self.clip_hi_: Optional[np.ndarray] = None
        self.x_scaler = StandardScaler()
        self.feature_names_: Optional[List[str]] = None

    def _to_base_matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.base_feature_cols].copy()
        for c in self.base_feature_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.to_numpy(np.float32)
        return X

    def fit(self, df_train: pd.DataFrame) -> "TabularPreprocessor":
        X = self._to_base_matrix(df_train)

        # clip thresholds on base features
        lo = np.quantile(X, self.clip_q_low, axis=0)
        hi = np.quantile(X, self.clip_q_high, axis=0)
        lo = lo.astype(np.float32)
        hi = hi.astype(np.float32)

        # protect degenerate columns
        hi = np.where(hi - lo < 1e-6, lo + 1.0, hi).astype(np.float32)

        self.clip_lo_ = lo
        self.clip_hi_ = hi

        Xc = np.clip(X, self.clip_lo_, self.clip_hi_).astype(np.float32)

        if self.use_feature_engineering:
            Xeng, names = build_engineered_features(Xc, self.base_feature_cols)
        else:
            Xeng = Xc
            names = [f"{c}" for c in self.base_feature_cols]

        self.feature_names_ = list(names)
        self.x_scaler.fit(Xeng)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.clip_lo_ is None or self.clip_hi_ is None:
            raise RuntimeError("Preprocessor not fitted.")

        X = self._to_base_matrix(df)
        Xc = np.clip(X, self.clip_lo_, self.clip_hi_).astype(np.float32)

        if self.use_feature_engineering:
            Xeng, _ = build_engineered_features(Xc, self.base_feature_cols)
        else:
            Xeng = Xc

        return self.x_scaler.transform(Xeng)

    def to_json(self) -> Dict:
        if self.clip_lo_ is None or self.clip_hi_ is None:
            raise RuntimeError("Preprocessor not fitted.")
        if self.x_scaler.mean_ is None or self.x_scaler.std_ is None:
            raise RuntimeError("Scaler not fitted.")
        return {
            "base_feature_cols": self.base_feature_cols,
            "use_feature_engineering": self.use_feature_engineering,
            "clip_q_low": self.clip_q_low,
            "clip_q_high": self.clip_q_high,
            "clip_lo": self.clip_lo_.tolist(),
            "clip_hi": self.clip_hi_.tolist(),
            "x_mean": self.x_scaler.mean_.tolist(),
            "x_std": self.x_scaler.std_.tolist(),
            "engineered_feature_names": self.feature_names_,
        }

    @staticmethod
    def from_json(payload: Dict) -> "TabularPreprocessor":
        pp = TabularPreprocessor(
            base_feature_cols=payload["base_feature_cols"],
            use_feature_engineering=payload.get("use_feature_engineering", True),
            clip_q_low=float(payload.get("clip_q_low", 0.005)),
            clip_q_high=float(payload.get("clip_q_high", 0.995)),
        )
        pp.clip_lo_ = np.asarray(payload["clip_lo"], dtype=np.float32)
        pp.clip_hi_ = np.asarray(payload["clip_hi"], dtype=np.float32)
        pp.feature_names_ = list(payload.get("engineered_feature_names", []))

        pp.x_scaler.mean_ = np.asarray(payload["x_mean"], dtype=np.float32)
        pp.x_scaler.std_ = np.asarray(payload["x_std"], dtype=np.float32)
        return pp


def save_preprocessor(path: Path, preproc: TabularPreprocessor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preproc.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_preprocessor(path: Path) -> TabularPreprocessor:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TabularPreprocessor.from_json(payload)


def save_target_scaler(path: Path, y_scaler: TargetScaler, target_log: bool) -> None:
    if y_scaler.mean_ is None or y_scaler.std_ is None:
        raise RuntimeError("Target scaler not fitted.")
    payload = {
        "y_mean": float(y_scaler.mean_),
        "y_std": float(y_scaler.std_),
        "target_log": bool(target_log),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_target_scaler(path: Path) -> Tuple[TargetScaler, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ys = TargetScaler()
    ys.mean_ = float(payload["y_mean"])
    ys.std_ = float(payload["y_std"])
    return ys, bool(payload.get("target_log", True))


# =========================
# Dataset (torch tensors)
# =========================
class TensorSeqDataset(Dataset):
    """
    X: torch.float32 (N,F)
    y: torch.float32 (N,)
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor, noise_std: float = 0.0) -> None:
        if X.ndim != 2:
            raise ValueError("X must be 2D tensor (N,F).")
        if y.ndim != 1:
            raise ValueError("y must be 1D tensor (N,).")
        if X.size(0) != y.size(0):
            raise ValueError("X/y length mismatch.")
        self.X = X
        self.y = y
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return self.y.size(0)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x.unsqueeze(-1), self.y[idx]


# =========================
# Model
# =========================
class TransformerRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.12,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.num_features = int(num_features)
        self.d_model = int(d_model)

        self.value_proj = nn.Linear(1, d_model)
        self.feature_emb = nn.Embedding(self.num_features, d_model)
        self.pos_emb = nn.Embedding(self.num_features + 1, d_model)  # 0 reserved for CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=bool(norm_first),
        )
        try:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,F,1)
        B, F, _ = x.shape
        if F != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {F}")

        v = self.value_proj(x)  # (B,F,d)
        feat_ids = torch.arange(F, device=x.device).unsqueeze(0).expand(B, F)
        v = v + self.feature_emb(feat_ids)

        pos_ids = torch.arange(1, F + 1, device=x.device).unsqueeze(0).expand(B, F)
        v = v + self.pos_emb(pos_ids)

        cls = self.cls_token.expand(B, 1, self.d_model)
        cls = cls + self.pos_emb(torch.zeros((B, 1), device=x.device, dtype=torch.long))

        tokens = torch.cat([cls, v], dim=1)  # (B,1+F,d)
        z = self.encoder(tokens)
        return self.head(z[:, 0, :]).squeeze(-1)


# =========================
# EMA
# =========================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.detach().clone()
                p.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.copy_(self.backup[name])
        self.backup = {}


# =========================
# Split helpers (stratified)
# =========================
def make_target_bins(y: np.ndarray, n_bins: int = 12) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y, qs))
    if len(edges) < 4:
        # fallback: equal-width
        edges = np.linspace(float(y.min()), float(y.max()) + 1e-6, n_bins + 1)
    bins = np.digitize(y, edges[1:-1], right=True)
    return bins.astype(np.int64)


def stratified_holdout_indices(y: np.ndarray, test_ratio: float, seed: int, n_bins: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns trainval_idx, test_idx
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    bins = make_target_bins(y, n_bins=n_bins)

    trainval: List[int] = []
    test: List[int] = []

    for b in np.unique(bins):
        idx_b = np.where(bins == b)[0]
        rng.shuffle(idx_b)

        n_te = int(round(len(idx_b) * test_ratio))
        test.extend(idx_b[:n_te].tolist())
        trainval.extend(idx_b[n_te:].tolist())

    return np.asarray(trainval, dtype=np.int64), np.asarray(test, dtype=np.int64)


def stratified_kfold_indices(y: np.ndarray, k: int, seed: int, n_bins: int = 12) -> List[np.ndarray]:
    """
    Returns list of fold indices (each is an array of indices into y)
    """
    rng = np.random.default_rng(seed)
    bins = make_target_bins(y, n_bins=n_bins)

    folds: List[List[int]] = [[] for _ in range(k)]
    for b in np.unique(bins):
        idx_b = np.where(bins == b)[0]
        rng.shuffle(idx_b)
        for i, ix in enumerate(idx_b):
            folds[i % k].append(int(ix))

    return [np.asarray(f, dtype=np.int64) for f in folds]


# =========================
# Training / Prediction
# =========================
@torch.no_grad()
def predict_scaled(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    use_amp = _amp_available_for_device(device, amp)
    ys: List[float] = []
    ps: List[float] = []
    for x, y in loader:
        x = x.to(device)
        with _get_autocast(device, enabled=use_amp):
            pred = model(x)
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(pred.detach().cpu().numpy().tolist())
    return np.asarray(ys, dtype=np.float64), np.asarray(ps, dtype=np.float64)


def inverse_target(y_scaled: np.ndarray, y_scaler: TargetScaler, target_log: bool) -> np.ndarray:
    y_t = y_scaler.inverse_transform(np.asarray(y_scaled, dtype=np.float32))
    return _safe_expm1(y_t) if target_log else np.asarray(y_t, dtype=np.float64)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    amp: bool,
    grad_accum: int,
    clip_grad: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    ema: Optional[EMA],
) -> float:
    model.train()
    use_amp = _amp_available_for_device(device, amp)
    grad_accum = max(1, int(grad_accum))

    total = 0.0
    optim.zero_grad(set_to_none=True)

    if not use_amp:
        for i, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y) / grad_accum
            loss.backward()

            step_now = (i % grad_accum == 0) or (i == len(loader))
            if step_now:
                if clip_grad and clip_grad > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
                optim.step()
                optim.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

            total += float(loss.item()) * x.size(0) * grad_accum

        return total / max(1, len(loader.dataset))

    scaler = _get_grad_scaler(device, enabled=True)
    for i, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        with _get_autocast(device, enabled=True):
            pred = model(x)
            loss = loss_fn(pred, y) / grad_accum

        scaler.scale(loss).backward()

        step_now = (i % grad_accum == 0) or (i == len(loader))
        if step_now:
            scaler.unscale_(optim)
            if clip_grad and clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        total += float(loss.item()) * x.size(0) * grad_accum

    return total / max(1, len(loader.dataset))


@dataclass
class TrainConfig:
    base_feature_cols: List[str]
    target_col: str
    target_log: bool
    use_feature_engineering: bool
    clip_q_low: float
    clip_q_high: float

    seed: int
    deterministic: bool
    test_ratio: float
    folds: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    loss_name: str
    huber_beta: float

    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float

    amp: bool
    grad_accum_steps: int
    feature_noise_std: float
    ema_decay: float
    onecycle: bool
    clip_grad: float
    num_workers: int


def _plot_curves(history: Dict[str, List[float]], out_dir: Path) -> None:
    e = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(e, history["train_loss"], label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_loss.png")
    plt.close()

    plt.figure()
    plt.plot(e, history["val_rmse"], label="val_rmse")
    plt.xlabel("epoch")
    plt.ylabel("rmse")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "learning_curve_val_rmse.png")
    plt.close()


def _plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.4)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("actual")
    plt.ylabel("predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_scatter.png")
    plt.close()


def train_single_fold(
    df: pd.DataFrame,
    base_feature_cols: Sequence[str],
    target_col: str,
    target_log: bool,
    use_feature_engineering: bool,
    clip_q_low: float,
    clip_q_high: float,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    deterministic: bool,
    out_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    loss_name: str,
    huber_beta: float,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_feedforward: int,
    dropout: float,
    amp: bool,
    grad_accum_steps: int,
    feature_noise_std: float,
    ema_decay: float,
    onecycle: bool,
    clip_grad: float,
    num_workers: int,
    verbose: bool = True,
) -> Dict:
    seed_everything(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    else:
        amp = False

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build train/val/test frames
    work = df.reset_index(drop=True)
    df_tr = work.iloc[train_idx]
    df_va = work.iloc[val_idx]
    df_te = work.iloc[test_idx]

    # Preprocess fit on train only
    preproc = TabularPreprocessor(
        base_feature_cols=base_feature_cols,
        use_feature_engineering=use_feature_engineering,
        clip_q_low=clip_q_low,
        clip_q_high=clip_q_high,
    ).fit(df_tr)

    X_tr = preproc.transform(df_tr)
    X_va = preproc.transform(df_va)
    X_te = preproc.transform(df_te)

    y_tr_raw = pd.to_numeric(df_tr[target_col], errors="coerce").to_numpy(np.float32)
    y_va_raw = pd.to_numeric(df_va[target_col], errors="coerce").to_numpy(np.float32)
    y_te_raw = pd.to_numeric(df_te[target_col], errors="coerce").to_numpy(np.float32)

    if target_log:
        y_tr_t = np.log1p(np.clip(y_tr_raw, 0.0, None))
        y_va_t = np.log1p(np.clip(y_va_raw, 0.0, None))
        y_te_t = np.log1p(np.clip(y_te_raw, 0.0, None))
    else:
        y_tr_t, y_va_t, y_te_t = y_tr_raw, y_va_raw, y_te_raw

    y_scaler = TargetScaler().fit(y_tr_t)
    y_tr = y_scaler.transform(y_tr_t)
    y_va = y_scaler.transform(y_va_t)
    y_te = y_scaler.transform(y_te_t)

    # Torch tensors
    Xtr_t = torch.from_numpy(X_tr).float()
    Xva_t = torch.from_numpy(X_va).float()
    Xte_t = torch.from_numpy(X_te).float()
    ytr_t = torch.from_numpy(y_tr).float()
    yva_t = torch.from_numpy(y_va).float()
    yte_t = torch.from_numpy(y_te).float()

    ds_tr = TensorSeqDataset(Xtr_t, ytr_t, noise_std=feature_noise_std)
    ds_va = TensorSeqDataset(Xva_t, yva_t, noise_std=0.0)
    ds_te = TensorSeqDataset(Xte_t, yte_t, noise_std=0.0)

    pin = device.type == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=max(256, batch_size), shuffle=False, num_workers=num_workers, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=max(256, batch_size), shuffle=False, num_workers=num_workers, pin_memory=pin)

    model = TransformerRegressor(
        num_features=X_tr.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        norm_first=True,
    ).to(device)

    if loss_name.lower() == "huber":
        loss_fn: nn.Module = nn.SmoothL1Loss(beta=float(huber_beta))
    else:
        loss_fn = nn.MSELoss()

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = None
    if onecycle:
        steps_per_epoch = max(1, math.ceil(len(dl_tr) / max(1, int(grad_accum_steps))))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            div_factor=25.0,
            final_div_factor=1000.0,
        )

    ema = EMA(model, decay=ema_decay) if ema_decay and ema_decay > 0 else None

    history: Dict[str, List[float]] = {"train_loss": [], "val_rmse": [], "lr": [], "epoch_sec": []}
    best_rmse = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad = 0

    if verbose:
        print(f"[Fold] out={out_dir}")
        print(f"[Device] {device}")
        print(f"[Sizes] train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)} | feats={X_tr.shape[1]}")
        print(f"[HP] epochs={epochs} batch={batch_size} lr={lr} amp={amp} onecycle={onecycle} ema={ema is not None}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model=model,
            loader=dl_tr,
            optim=optim,
            device=device,
            loss_fn=loss_fn,
            amp=amp,
            grad_accum=grad_accum_steps,
            clip_grad=clip_grad,
            scheduler=scheduler,
            ema=ema,
        )

        # validation (EMA weights if enabled)
        if ema is not None:
            ema.apply_to(model)
        y_va_true_s, y_va_pred_s = predict_scaled(model, dl_va, device, amp=amp)
        if ema is not None:
            ema.restore(model)

        y_va_true = inverse_target(y_va_true_s, y_scaler, target_log)
        y_va_pred = inverse_target(y_va_pred_s, y_scaler, target_log)
        val_m = regression_metrics(y_va_true, y_va_pred)

        dt = time.time() - t0
        history["train_loss"].append(float(tr_loss))
        history["val_rmse"].append(float(val_m["RMSE"]))
        history["lr"].append(float(optim.param_groups[0]["lr"]))
        history["epoch_sec"].append(float(dt))

        if verbose:
            print(f"Epoch {epoch:4d}/{epochs} | loss={tr_loss:.6f} | val_RMSE={val_m['RMSE']:.3f} | lr={optim.param_groups[0]['lr']:.2e} | {dt:.1f}s")

        if val_m["RMSE"] < best_rmse:
            best_rmse = float(val_m["RMSE"])
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # also save EMA version if enabled (usually better)
            if ema is not None:
                ema.apply_to(model)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                ema.restore(model)

            torch.save(best_state, out_dir / "best_model.pt")
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[EarlyStop] no improvement for {patience} epochs. best_val_rmse={best_rmse:.3f}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, out_dir / "best_model.pt")

    # load best + evaluate test
    model.load_state_dict(best_state)
    model.to(device)

    y_te_true_s, y_te_pred_s = predict_scaled(model, dl_te, device, amp=amp)
    y_te_true = inverse_target(y_te_true_s, y_scaler, target_log)
    y_te_pred = inverse_target(y_te_pred_s, y_scaler, target_log)
    test_m = regression_metrics(y_te_true, y_te_pred)

    # save artifacts
    save_preprocessor(out_dir / "preprocess.json", preproc)
    save_target_scaler(out_dir / "target_scaler.json", y_scaler, target_log=target_log)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    _plot_curves(history, out_dir)
    _plot_scatter(y_te_true, y_te_pred, out_dir)

    fold_report = {
        "best_val_rmse": best_rmse,
        "test_metrics": test_m,
        "num_features_after_fe": int(X_tr.shape[1]),
        "device": str(device),
    }
    (out_dir / "fold_report.json").write_text(json.dumps(fold_report, ensure_ascii=False, indent=2), encoding="utf-8")

    return fold_report


def train_kfold_ensemble(
    df: pd.DataFrame,
    base_feature_cols: Sequence[str],
    target_col: str,
    target_log: bool = True,
    use_feature_engineering: bool = True,
    clip_q_low: float = 0.005,
    clip_q_high: float = 0.995,
    test_ratio: float = 0.20,
    folds: int = 5,
    seed: int = 42,
    deterministic: bool = True,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 2e-4,
    patience: int = 20,
    loss_name: str = "huber",
    huber_beta: float = 1.0,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.12,
    amp: bool = True,
    grad_accum_steps: int = 1,
    feature_noise_std: float = 0.01,
    ema_decay: float = 0.999,
    onecycle: bool = True,
    clip_grad: float = 1.0,
    num_workers: int = 0,
    output_dir: str = str(DEFAULT_OUT_DIR),
    verbose: bool = True,
) -> Dict:
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Basic cleanup: numeric coercion + dropna
    need = list(base_feature_cols) + [target_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    work = df[need].copy()
    for c in need:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=need).reset_index(drop=True)
    if len(work) < 200:
        raise ValueError(f"Not enough valid rows: {len(work)}")

    y_raw = work[target_col].to_numpy(np.float32)

    # Holdout test
    trainval_idx, test_idx = stratified_holdout_indices(y_raw, test_ratio=test_ratio, seed=seed, n_bins=12)
    y_trainval = y_raw[trainval_idx]

    # Stratified K-fold on trainval
    fold_sets_local = stratified_kfold_indices(y_trainval, k=folds, seed=seed, n_bins=12)

    fold_reports: List[Dict] = []
    best_fold = -1
    best_val_rmse = float("inf")

    if verbose:
        print(f"[Ensemble] rows={len(work)} | base_feats={len(base_feature_cols)} | holdout_test={len(test_idx)} | folds={folds}")
        print(f"[Output] {out_root}")

    # Save config once
    cfg = TrainConfig(
        base_feature_cols=list(base_feature_cols),
        target_col=target_col,
        target_log=bool(target_log),
        use_feature_engineering=bool(use_feature_engineering),
        clip_q_low=float(clip_q_low),
        clip_q_high=float(clip_q_high),
        seed=int(seed),
        deterministic=bool(deterministic),
        test_ratio=float(test_ratio),
        folds=int(folds),
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        patience=int(patience),
        loss_name=str(loss_name),
        huber_beta=float(huber_beta),
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dim_feedforward=int(dim_feedforward),
        dropout=float(dropout),
        amp=bool(amp),
        grad_accum_steps=int(grad_accum_steps),
        feature_noise_std=float(feature_noise_std),
        ema_decay=float(ema_decay),
        onecycle=bool(onecycle),
        clip_grad=float(clip_grad),
        num_workers=int(num_workers),
    )
    (out_root / "train_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

    # Train folds
    for fi in range(folds):
        val_local = fold_sets_local[fi]
        train_local = np.concatenate([fold_sets_local[j] for j in range(folds) if j != fi], axis=0)

        # Map local indices -> global indices
        tr_idx = trainval_idx[train_local]
        va_idx = trainval_idx[val_local]

        fold_dir = out_root / f"fold_{fi}"
        rep = train_single_fold(
            df=work,
            base_feature_cols=base_feature_cols,
            target_col=target_col,
            target_log=target_log,
            use_feature_engineering=use_feature_engineering,
            clip_q_low=clip_q_low,
            clip_q_high=clip_q_high,
            train_idx=tr_idx,
            val_idx=va_idx,
            test_idx=test_idx,
            seed=seed + fi * 1009,
            deterministic=deterministic,
            out_dir=fold_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            loss_name=loss_name,
            huber_beta=huber_beta,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            amp=amp,
            grad_accum_steps=grad_accum_steps,
            feature_noise_std=feature_noise_std,
            ema_decay=ema_decay,
            onecycle=onecycle,
            clip_grad=clip_grad,
            num_workers=num_workers,
            verbose=verbose,
        )
        rep["fold"] = fi
        fold_reports.append(rep)

        if rep["best_val_rmse"] < best_val_rmse:
            best_val_rmse = float(rep["best_val_rmse"])
            best_fold = fi

    # Build ensemble prediction on holdout test by averaging fold predictions
    ens = evaluate_ensemble_on_holdout(
        df=work,
        base_feature_cols=base_feature_cols,
        target_col=target_col,
        target_log=target_log,
        out_root=out_root,
        test_idx=test_idx,
        folds=folds,
    )

    manifest = {
        "output_dir": str(out_root),
        "best_fold": int(best_fold),
        "best_fold_val_rmse": float(best_val_rmse),
        "folds": int(folds),
        "fold_reports": fold_reports,
        "holdout_test_size": int(len(test_idx)),
        "ensemble_test_metrics": ens["ensemble_test_metrics"],
        "ensemble_rmse": float(ens["ensemble_test_metrics"]["RMSE"]),
    }
    (out_root / "ensemble_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Copy best fold artifacts to root for convenience (single-model inference compatibility)
    _copy_best_fold_as_root(out_root, best_fold)

    if verbose:
        print(f"[BestFold] fold={best_fold} val_rmse={best_val_rmse:.3f}")
        print(f"[EnsembleHoldout] RMSE={ens['ensemble_test_metrics']['RMSE']:.3f}  MAE={ens['ensemble_test_metrics']['MAE']:.3f}")

    return manifest


def _copy_best_fold_as_root(out_root: Path, best_fold: int) -> None:
    import shutil
    fold_dir = out_root / f"fold_{best_fold}"
    for name in ["best_model.pt", "preprocess.json", "target_scaler.json", "fold_report.json"]:
        src = fold_dir / name
        if src.exists():
            shutil.copy2(src, out_root / name)


@torch.no_grad()
def _predict_holdout_with_fold(
    df: pd.DataFrame,
    base_feature_cols: Sequence[str],
    target_col: str,
    fold_dir: Path,
    test_idx: np.ndarray,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preproc = load_preprocessor(fold_dir / "preprocess.json")
    y_scaler, target_log = load_target_scaler(fold_dir / "target_scaler.json")

    # rebuild model with correct feature count
    cfg = json.loads((fold_dir.parent / "train_config.json").read_text(encoding="utf-8"))
    X_test = preproc.transform(df.iloc[test_idx])

    model = TransformerRegressor(
        num_features=X_test.shape[1],
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        norm_first=True,
    ).to(device)

    state = torch.load(fold_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    Xt = torch.from_numpy(X_test).float()
    dummy_y = torch.zeros((Xt.size(0),), dtype=torch.float32)
    ds = TensorSeqDataset(Xt, dummy_y, noise_std=0.0)
    dl = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    use_amp = _amp_available_for_device(device, True)
    preds_s: List[float] = []
    for x, _ in dl:
        x = x.to(device)
        with _get_autocast(device, enabled=use_amp):
            p = model(x)
        preds_s.extend(p.detach().cpu().numpy().tolist())

    pred = inverse_target(np.asarray(preds_s, dtype=np.float64), y_scaler, target_log)
    return pred


def evaluate_ensemble_on_holdout(
    df: pd.DataFrame,
    base_feature_cols: Sequence[str],
    target_col: str,
    target_log: bool,
    out_root: Path,
    test_idx: np.ndarray,
    folds: int,
) -> Dict:
    y_true = df.iloc[test_idx][target_col].to_numpy(np.float64)
    preds_all = []

    for fi in range(folds):
        fold_dir = out_root / f"fold_{fi}"
        pred = _predict_holdout_with_fold(df, base_feature_cols, target_col, fold_dir, test_idx)
        preds_all.append(pred)

    ens_pred = np.mean(np.stack(preds_all, axis=0), axis=0)
    m = regression_metrics(y_true, ens_pred)

    report = {"ensemble_test_metrics": m}
    (out_root / "ensemble_holdout_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


# =========================
# Inference APIs
# =========================
def load_single_artifacts(artifacts_dir: Path) -> Tuple[nn.Module, TabularPreprocessor, TargetScaler, bool, Dict]:
    artifacts_dir = Path(artifacts_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = json.loads((artifacts_dir / "train_config.json").read_text(encoding="utf-8"))
    preproc = load_preprocessor(artifacts_dir / "preprocess.json")
    y_scaler, target_log = load_target_scaler(artifacts_dir / "target_scaler.json")

    # need feature count from preproc
    # Build dummy to infer dimension reliably
    # (Using scaler mean length is safe)
    feat_dim = len(preproc.x_scaler.mean_) if preproc.x_scaler.mean_ is not None else len(preproc.base_feature_cols)

    model = TransformerRegressor(
        num_features=feat_dim,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        norm_first=True,
    ).to(device)

    state = torch.load(artifacts_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, preproc, y_scaler, target_log, cfg


@torch.no_grad()
def predict_dataframe_single(
    model: nn.Module,
    preproc: TabularPreprocessor,
    y_scaler: TargetScaler,
    target_log: bool,
    df: pd.DataFrame,
    base_feature_cols: Sequence[str],
    batch_size: int = 2048,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = df[list(base_feature_cols)].copy()
    for c in base_feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    valid_mask = ~X.isna().any(axis=1)
    out = np.full((len(df),), np.nan, dtype=np.float64)

    if valid_mask.sum() == 0:
        return out

    Xv = preproc.transform(df.loc[valid_mask])
    Xt = torch.from_numpy(Xv).float()
    dummy_y = torch.zeros((Xt.size(0),), dtype=torch.float32)
    ds = TensorSeqDataset(Xt, dummy_y, noise_std=0.0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    use_amp = _amp_available_for_device(device, True)
    preds_s: List[float] = []
    for x, _ in dl:
        x = x.to(device)
        with _get_autocast(device, enabled=use_amp):
            p = model(x)
        preds_s.extend(p.detach().cpu().numpy().tolist())

    pred = inverse_target(np.asarray(preds_s, dtype=np.float64), y_scaler, target_log)
    out[valid_mask.to_numpy()] = pred
    return out


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default=None)

    p.add_argument("--target", type=str, default="낙찰가")
    p.add_argument("--features", nargs="*", default=None)

    p.add_argument("--no_log_target", action="store_true")
    p.add_argument("--no_fe", action="store_true")
    p.add_argument("--clip_low", type=float, default=0.005)
    p.add_argument("--clip_high", type=float, default=0.995)

    p.add_argument("--test_ratio", type=float, default=0.20)
    p.add_argument("--folds", type=int, default=5)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=20)

    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.12)

    p.add_argument("--no_onecycle", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--noise", type=float, default=0.01)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data) if args.data else DEFAULT_DATA_PATH
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR

    df = read_csv_safely(data_path)

    # features: default numeric candidates (but keep your known baseline columns as first choice)
    if args.features is not None and len(args.features) > 0:
        base_features = args.features
    else:
        base_features = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]

    manifest = train_kfold_ensemble(
        df=df,
        base_feature_cols=base_features,
        target_col=args.target,
        target_log=not args.no_log_target,
        use_feature_engineering=not args.no_fe,
        clip_q_low=args.clip_low,
        clip_q_high=args.clip_high,
        test_ratio=args.test_ratio,
        folds=args.folds,
        seed=args.seed,
        deterministic=True,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.wd,
        patience=args.patience,
        loss_name="huber",
        huber_beta=1.0,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        amp=True,
        grad_accum_steps=1,
        feature_noise_std=args.noise,
        ema_decay=0.0 if args.no_ema else args.ema_decay,
        onecycle=not args.no_onecycle,
        clip_grad=1.0,
        num_workers=0,
        output_dir=str(out_dir),
        verbose=bool(args.verbose),
    )

    print("\n[Done]")
    print(f"Best fold: {manifest['best_fold']}  val_rmse={manifest['best_fold_val_rmse']:.3f}")
    print(f"Ensemble holdout RMSE: {manifest['ensemble_rmse']:.3f}")
    print(f"Artifacts: {manifest['output_dir']}")


if __name__ == "__main__":
    main()
