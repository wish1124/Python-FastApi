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

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ---------- Relative paths ----------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "dataset" / "preprocessed_dataset.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results_transformer3"


# =========================
# Repro / IO
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


# =========================
# Metrics / transforms
# =========================
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
# AMP helpers (safe)
# =========================
def _amp_available(device: torch.device, amp_flag: bool) -> bool:
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
            raise RuntimeError("StandardScaler not fitted.")
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
            raise RuntimeError("TargetScaler not fitted.")
        y = np.asarray(y, dtype=np.float32)
        return (y - float(self.mean_)) / float(self.std_)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("TargetScaler not fitted.")
        y = np.asarray(y, dtype=np.float32)
        return y * float(self.std_) + float(self.mean_)


def inverse_target(y_scaled: np.ndarray, y_scaler: TargetScaler, target_log: bool) -> np.ndarray:
    y_t = y_scaler.inverse_transform(np.asarray(y_scaled, dtype=np.float32))
    return _safe_expm1(y_t) if target_log else np.asarray(y_t, dtype=np.float64)


def build_features_light(X: np.ndarray, base_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Light FE to keep dimension small (faster + less overfit):
    - raw
    - log1p(clip>=0)
    - sqrt(clip>=0)
    - row stats: sum/mean/min/max
    """
    X = np.asarray(X, dtype=np.float32)
    X_pos = np.clip(X, 0.0, None)

    feats: List[np.ndarray] = [X]
    names: List[str] = [f"{c}" for c in base_names]

    feats.append(np.log1p(X_pos).astype(np.float32))
    names += [f"log1p({c})" for c in base_names]

    feats.append(np.sqrt(X_pos + 1e-6).astype(np.float32))
    names += [f"sqrt({c})" for c in base_names]

    row_sum = np.sum(X, axis=1, keepdims=True)
    row_mean = np.mean(X, axis=1, keepdims=True)
    row_min = np.min(X, axis=1, keepdims=True)
    row_max = np.max(X, axis=1, keepdims=True)
    feats += [row_sum, row_mean, row_min, row_max]
    names += ["row_sum", "row_mean", "row_min", "row_max"]

    Xf = np.concatenate(feats, axis=1).astype(np.float32)
    return Xf, names


def build_features_full(X: np.ndarray, base_names: Sequence[str], eps: float = 1e-6) -> Tuple[np.ndarray, List[str]]:
    """
    Full FE (potentially better, but slower + more overfit risk):
    light + pairwise (+, -, *, ratio both ways)
    """
    X = np.asarray(X, dtype=np.float32)
    N, F = X.shape

    X_light, names = build_features_light(X, base_names)
    feats: List[np.ndarray] = [X_light]

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

    Xf = np.concatenate(feats, axis=1).astype(np.float32)
    return Xf, names


class TabularPreprocessor:
    """
    - numeric coercion
    - quantile clipping (fit on train only)
    - feature engineering: none/light/full
    - standard scaling (fit on train only)
    """
    def __init__(
        self,
        base_feature_cols: Sequence[str],
        fe_mode: str = "light",
        clip_q_low: float = 0.005,
        clip_q_high: float = 0.995,
    ) -> None:
        self.base_feature_cols = list(base_feature_cols)
        self.fe_mode = str(fe_mode).lower()  # none/light/full
        self.clip_q_low = float(clip_q_low)
        self.clip_q_high = float(clip_q_high)

        self.clip_lo_: Optional[np.ndarray] = None
        self.clip_hi_: Optional[np.ndarray] = None
        self.scaler = StandardScaler()
        self.feature_names_: Optional[List[str]] = None

    def _to_base_matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.base_feature_cols].copy()
        for c in self.base_feature_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X.to_numpy(np.float32)

    def fit(self, df_train: pd.DataFrame) -> "TabularPreprocessor":
        X = self._to_base_matrix(df_train)
        lo = np.quantile(X, self.clip_q_low, axis=0).astype(np.float32)
        hi = np.quantile(X, self.clip_q_high, axis=0).astype(np.float32)
        hi = np.where(hi - lo < 1e-6, lo + 1.0, hi).astype(np.float32)

        self.clip_lo_ = lo
        self.clip_hi_ = hi

        Xc = np.clip(X, self.clip_lo_, self.clip_hi_).astype(np.float32)
        Xf, names = self._fe(Xc)
        self.feature_names_ = names
        self.scaler.fit(Xf)
        return self

    def _fe(self, Xc: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        if self.fe_mode == "none":
            return Xc.astype(np.float32), [f"{c}" for c in self.base_feature_cols]
        if self.fe_mode == "full":
            return build_features_full(Xc, self.base_feature_cols)
        return build_features_light(Xc, self.base_feature_cols)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.clip_lo_ is None or self.clip_hi_ is None:
            raise RuntimeError("Preprocessor not fitted.")
        X = self._to_base_matrix(df)
        Xc = np.clip(X, self.clip_lo_, self.clip_hi_).astype(np.float32)
        Xf, _ = self._fe(Xc)
        return self.scaler.transform(Xf)

    def to_json(self) -> Dict:
        if self.clip_lo_ is None or self.clip_hi_ is None:
            raise RuntimeError("Preprocessor not fitted.")
        if self.scaler.mean_ is None or self.scaler.std_ is None:
            raise RuntimeError("Scaler not fitted.")
        return {
            "base_feature_cols": self.base_feature_cols,
            "fe_mode": self.fe_mode,
            "clip_q_low": self.clip_q_low,
            "clip_q_high": self.clip_q_high,
            "clip_lo": self.clip_lo_.tolist(),
            "clip_hi": self.clip_hi_.tolist(),
            "x_mean": self.scaler.mean_.tolist(),
            "x_std": self.scaler.std_.tolist(),
            "engineered_feature_names": self.feature_names_,
        }

    @staticmethod
    def from_json(payload: Dict) -> "TabularPreprocessor":
        pp = TabularPreprocessor(
            base_feature_cols=payload["base_feature_cols"],
            fe_mode=payload.get("fe_mode", "light"),
            clip_q_low=float(payload.get("clip_q_low", 0.005)),
            clip_q_high=float(payload.get("clip_q_high", 0.995)),
        )
        pp.clip_lo_ = np.asarray(payload["clip_lo"], dtype=np.float32)
        pp.clip_hi_ = np.asarray(payload["clip_hi"], dtype=np.float32)
        pp.feature_names_ = list(payload.get("engineered_feature_names", []))
        pp.scaler.mean_ = np.asarray(payload["x_mean"], dtype=np.float32)
        pp.scaler.std_ = np.asarray(payload["x_std"], dtype=np.float32)
        return pp


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


# =========================
# Split (stratified by target quantiles)
# =========================
def make_target_bins(y: np.ndarray, n_bins: int = 12) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y, qs))
    if len(edges) < 4:
        edges = np.linspace(float(y.min()), float(y.max()) + 1e-6, n_bins + 1)
    return np.digitize(y, edges[1:-1], right=True).astype(np.int64)


def stratified_split_indices(
    y: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    n_bins: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns train_idx, val_idx, test_idx (all stratified by y bins).
    """
    rng = np.random.default_rng(seed)
    bins = make_target_bins(y, n_bins=n_bins)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for b in np.unique(bins):
        idx_b = np.where(bins == b)[0]
        rng.shuffle(idx_b)

        n = len(idx_b)
        n_test = int(round(n * test_ratio))
        test_part = idx_b[:n_test]
        rest = idx_b[n_test:]

        n_val = int(round(len(rest) * val_ratio))
        val_part = rest[:n_val]
        train_part = rest[n_val:]

        test_idx.extend(test_part.tolist())
        val_idx.extend(val_part.tolist())
        train_idx.extend(train_part.tolist())

    return (
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(val_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
    )


# =========================
# Dataset
# =========================
class TensorSeqDataset(Dataset):
    """
    X: (N,F) float32 -> returns (F,1) for transformer tokens
    y: (N,) float32
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, noise_std: float = 0.0) -> None:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or y.ndim != 1:
            raise ValueError("X must be (N,F) and y must be (N,).")
        if len(X) != len(y):
            raise ValueError("X/y length mismatch.")
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.noise_std = float(noise_std)

    def __len__(self) -> int:
        return int(self.y.size(0))

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x.unsqueeze(-1), self.y[idx]


# =========================
# Model (small + regularized)
# =========================
class TransformerRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.25,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        self.num_features = int(num_features)
        self.d_model = int(d_model)

        self.value_proj = nn.Linear(1, d_model)
        self.feature_emb = nn.Embedding(self.num_features, d_model)
        self.pos_emb = nn.Embedding(self.num_features + 1, d_model)
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

        tokens = torch.cat([cls, v], dim=1)
        z = self.encoder(tokens)
        return self.head(z[:, 0, :]).squeeze(-1)


# =========================
# Training
# =========================
@torch.no_grad()
def predict_scaled(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    use_amp = _amp_available(device, amp)
    ys: List[float] = []
    ps: List[float] = []

    it = loader
    for x, y in it:
        x = x.to(device)
        with _get_autocast(device, enabled=use_amp):
            pred = model(x)
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(pred.detach().cpu().numpy().tolist())

    return np.asarray(ys, dtype=np.float64), np.asarray(ps, dtype=np.float64)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module,
    amp: bool,
    clip_grad: float,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    show_tqdm: bool,
) -> float:
    model.train()
    use_amp = _amp_available(device, amp)
    total = 0.0
    n = 0

    if use_amp:
        scaler = _get_grad_scaler(device, enabled=True)
    else:
        scaler = None

    it = loader
    if show_tqdm and tqdm is not None:
        it = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)

    optim.zero_grad(set_to_none=True)
    for x, y in it:
        x, y = x.to(device), y.to(device)

        if use_amp:
            with _get_autocast(device, enabled=True):
                pred = model(x)
                loss = loss_fn(pred, y)
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            if clip_grad and clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            scaler.step(optim)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            if clip_grad and clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(clip_grad))
            optim.step()

        optim.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        bs = int(x.size(0))
        total += float(loss.item()) * bs
        n += bs

        if show_tqdm and tqdm is not None:
            lr = float(optim.param_groups[0]["lr"])
            it.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

    return total / max(1, n)


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


@dataclass
class TrainConfig:
    base_feature_cols: List[str]
    target_col: str
    target_log: bool
    fe_mode: str
    clip_q_low: float
    clip_q_high: float

    test_ratio: float
    val_ratio: float
    seed: int
    deterministic: bool

    max_rows: int

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    min_delta: float
    loss_name: str
    huber_beta: float

    d_model: int
    nhead: int
    num_layers: int
    dim_feedforward: int
    dropout: float

    amp: bool
    feature_noise_std: float
    onecycle: bool
    clip_grad: float
    num_workers: int


@dataclass
class TrainResult:
    output_dir: str
    best_val_rmse: float
    best_val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    history: Dict[str, List[float]]


def run_training_transformer3(
    df: pd.DataFrame,
    base_feature_cols: Sequence[str],
    target_col: str,
    output_dir: str = str(DEFAULT_OUT_DIR),
    target_log: bool = True,
    fe_mode: str = "light",
    clip_q_low: float = 0.005,
    clip_q_high: float = 0.995,
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    seed: int = 42,
    deterministic: bool = True,
    max_rows: Optional[int] = 200000,  # fast default; set None to use all
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,        # stronger reg
    patience: int = 10,
    min_delta: float = 0.0,
    loss_name: str = "huber",
    huber_beta: float = 1.0,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.25,
    amp: bool = True,
    feature_noise_std: float = 0.01,   # mild noise => regularization
    onecycle: bool = True,
    clip_grad: float = 1.0,
    num_workers: int = 0,
    verbose: bool = True,
    show_tqdm: bool = True,
) -> TrainResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        amp = False
    else:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

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

    # optional downsample for speed (stratified by target bins)
    if max_rows is not None and len(work) > int(max_rows):
        y_tmp = work[target_col].to_numpy(np.float64)
        bins = make_target_bins(y_tmp, n_bins=12)
        rng = np.random.default_rng(seed)
        keep: List[int] = []
        per_bin = max(1, int(max_rows) // max(1, len(np.unique(bins))))
        for b in np.unique(bins):
            idx_b = np.where(bins == b)[0]
            rng.shuffle(idx_b)
            keep.extend(idx_b[:per_bin].tolist())
        keep = keep[: int(max_rows)]
        work = work.iloc[keep].reset_index(drop=True)

    y_raw = work[target_col].to_numpy(np.float32)
    tr_idx, va_idx, te_idx = stratified_split_indices(
        y_raw, test_ratio=test_ratio, val_ratio=val_ratio, seed=seed, n_bins=12
    )

    df_tr = work.iloc[tr_idx]
    df_va = work.iloc[va_idx]
    df_te = work.iloc[te_idx]

    preproc = TabularPreprocessor(
        base_feature_cols=base_feature_cols,
        fe_mode=fe_mode,
        clip_q_low=clip_q_low,
        clip_q_high=clip_q_high,
    ).fit(df_tr)

    X_tr = preproc.transform(df_tr)
    X_va = preproc.transform(df_va)
    X_te = preproc.transform(df_te)

    y_tr_raw = df_tr[target_col].to_numpy(np.float32)
    y_va_raw = df_va[target_col].to_numpy(np.float32)
    y_te_raw = df_te[target_col].to_numpy(np.float32)

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

    ds_tr = TensorSeqDataset(X_tr, y_tr, noise_std=feature_noise_std)
    ds_va = TensorSeqDataset(X_va, y_va, noise_std=0.0)
    ds_te = TensorSeqDataset(X_te, y_te, noise_std=0.0)

    pin = device.type == "cuda"
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=max(1024, batch_size), shuffle=False, num_workers=num_workers, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=max(1024, batch_size), shuffle=False, num_workers=num_workers, pin_memory=pin)

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
        steps_per_epoch = max(1, len(dl_tr))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.15,
            div_factor=25.0,
            final_div_factor=2000.0,
        )

    cfg = TrainConfig(
        base_feature_cols=list(base_feature_cols),
        target_col=target_col,
        target_log=bool(target_log),
        fe_mode=str(fe_mode),
        clip_q_low=float(clip_q_low),
        clip_q_high=float(clip_q_high),
        test_ratio=float(test_ratio),
        val_ratio=float(val_ratio),
        seed=int(seed),
        deterministic=bool(deterministic),
        max_rows=int(max_rows) if max_rows is not None else -1,
        epochs=int(epochs),
        batch_size=int(batch_size),
        lr=float(lr),
        weight_decay=float(weight_decay),
        patience=int(patience),
        min_delta=float(min_delta),
        loss_name=str(loss_name),
        huber_beta=float(huber_beta),
        d_model=int(d_model),
        nhead=int(nhead),
        num_layers=int(num_layers),
        dim_feedforward=int(dim_feedforward),
        dropout=float(dropout),
        amp=bool(amp),
        feature_noise_std=float(feature_noise_std),
        onecycle=bool(onecycle),
        clip_grad=float(clip_grad),
        num_workers=int(num_workers),
    )

    save_json(out_dir / "train_config.json", asdict(cfg))
    save_json(out_dir / "preprocess.json", preproc.to_json())
    save_json(out_dir / "target_scaler.json", {"y_mean": y_scaler.mean_, "y_std": y_scaler.std_, "target_log": target_log})

    history: Dict[str, List[float]] = {"train_loss": [], "val_rmse": [], "lr": [], "epoch_sec": []}

    best_val_rmse = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad = 0

    if verbose:
        print(f"[Device] {device}")
        print(f"[Rows] total={len(work)} | train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")
        print(f"[Features] base={len(base_feature_cols)} -> used={X_tr.shape[1]} (fe_mode={fe_mode})")
        if max_rows is not None:
            print(f"[Speed] max_rows={max_rows}")
        print(f"[HP] epochs={epochs} batch={batch_size} lr={lr} wd={weight_decay} dropout={dropout} amp={amp}")

    epoch_iter = range(1, epochs + 1)
    if show_tqdm and tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="epochs", dynamic_ncols=True)

    for epoch in epoch_iter:
        t0 = time.time()
        tr_loss = train_one_epoch(
            model=model,
            loader=dl_tr,
            optim=optim,
            device=device,
            loss_fn=loss_fn,
            amp=amp,
            clip_grad=clip_grad,
            scheduler=scheduler,
            show_tqdm=bool(show_tqdm),
        )

        y_va_true_s, y_va_pred_s = predict_scaled(model, dl_va, device, amp=amp)
        y_va_true = inverse_target(y_va_true_s, y_scaler, target_log)
        y_va_pred = inverse_target(y_va_pred_s, y_scaler, target_log)
        val_m = regression_metrics(y_va_true, y_va_pred)

        dt = time.time() - t0
        history["train_loss"].append(float(tr_loss))
        history["val_rmse"].append(float(val_m["RMSE"]))
        history["lr"].append(float(optim.param_groups[0]["lr"]))
        history["epoch_sec"].append(float(dt))

        if show_tqdm and tqdm is not None:
            epoch_iter.set_postfix(loss=f"{tr_loss:.4f}", val_rmse=f"{val_m['RMSE']:.1f}", lr=f"{optim.param_groups[0]['lr']:.2e}")

        improved = (val_m["RMSE"] + float(min_delta)) < best_val_rmse
        if improved:
            best_val_rmse = float(val_m["RMSE"])
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, out_dir / "best_model.pt")
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[EarlyStop] patience={patience} best_val_rmse={best_val_rmse:.3f}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        torch.save(best_state, out_dir / "best_model.pt")

    model.load_state_dict(best_state)
    y_te_true_s, y_te_pred_s = predict_scaled(model, dl_te, device, amp=amp)
    y_te_true = inverse_target(y_te_true_s, y_scaler, target_log)
    y_te_pred = inverse_target(y_te_pred_s, y_scaler, target_log)
    test_m = regression_metrics(y_te_true, y_te_pred)

    # also compute best-val metrics at the end (on current best model)
    y_va_true_s, y_va_pred_s = predict_scaled(model, dl_va, device, amp=amp)
    y_va_true = inverse_target(y_va_true_s, y_scaler, target_log)
    y_va_pred = inverse_target(y_va_pred_s, y_scaler, target_log)
    best_val_metrics = regression_metrics(y_va_true, y_va_pred)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    _plot_curves(history, out_dir)
    _plot_scatter(y_te_true, y_te_pred, out_dir)

    save_json(out_dir / "metrics.json", {"best_val": best_val_metrics, "test": test_m})

    return TrainResult(
        output_dir=str(out_dir),
        best_val_rmse=float(best_val_rmse),
        best_val_metrics=best_val_metrics,
        test_metrics=test_m,
        history=history,
    )


# =========================
# Inference (single best_model.pt)
# =========================
def load_artifacts_transformer3(artifacts_dir: Path) -> Tuple[nn.Module, TabularPreprocessor, TargetScaler, bool, Dict]:
    artifacts_dir = Path(artifacts_dir)
    cfg = load_json(artifacts_dir / "train_config.json")
    preproc = TabularPreprocessor.from_json(load_json(artifacts_dir / "preprocess.json"))
    ts_payload = load_json(artifacts_dir / "target_scaler.json")

    y_scaler = TargetScaler()
    y_scaler.mean_ = float(ts_payload["y_mean"])
    y_scaler.std_ = float(ts_payload["y_std"])
    target_log = bool(ts_payload.get("target_log", True))

    feat_dim = len(preproc.scaler.mean_) if preproc.scaler.mean_ is not None else len(preproc.base_feature_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
def predict_dataframe_transformer3(
    model: nn.Module,
    preproc: TabularPreprocessor,
    y_scaler: TargetScaler,
    target_log: bool,
    df: pd.DataFrame,
    batch_size: int = 2048,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # keep only rows without NaN in base features
    Xbase = df[preproc.base_feature_cols].copy()
    for c in preproc.base_feature_cols:
        Xbase[c] = pd.to_numeric(Xbase[c], errors="coerce")
    valid = ~Xbase.isna().any(axis=1)

    out = np.full((len(df),), np.nan, dtype=np.float64)
    if valid.sum() == 0:
        return out

    Xv = preproc.transform(df.loc[valid])
    dummy_y = np.zeros((Xv.shape[0],), dtype=np.float32)
    ds = TensorSeqDataset(Xv, dummy_y, noise_std=0.0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    use_amp = _amp_available(device, True)
    preds_s: List[float] = []
    for x, _ in dl:
        x = x.to(device)
        with _get_autocast(device, enabled=use_amp):
            p = model(x)
        preds_s.extend(p.detach().cpu().numpy().tolist())

    pred = inverse_target(np.asarray(preds_s, dtype=np.float64), y_scaler, target_log)
    out[valid.to_numpy()] = pred
    return out


# =========================
# CLI (optional)
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--target", type=str, default="낙찰가")
    p.add_argument("--features", nargs="*", default=None)

    p.add_argument("--fe_mode", type=str, default="light", choices=["none", "light", "full"])
    p.add_argument("--max_rows", type=int, default=200000)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=10)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data) if args.data else DEFAULT_DATA_PATH
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR

    df = read_csv_safely(data_path)

    if args.features is None or len(args.features) == 0:
        base_features = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]
    else:
        base_features = args.features

    max_rows = None if args.max_rows <= 0 else int(args.max_rows)

    res = run_training_transformer3(
        df=df,
        base_feature_cols=base_features,
        target_col=args.target,
        output_dir=str(out_dir),
        fe_mode=args.fe_mode,
        max_rows=max_rows,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.wd,
        patience=args.patience,
        verbose=True,
        show_tqdm=True,
    )

    print("\n[Best VAL]")
    for k, v in res.best_val_metrics.items():
        print(f"{k:>6}: {v:.6f}")
    print("\n[TEST]")
    for k, v in res.test_metrics.items():
        print(f"{k:>6}: {v:.6f}")
    print(f"\nArtifacts: {res.output_dir}")


if __name__ == "__main__":
    main()
