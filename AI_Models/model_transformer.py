# -*- coding: utf-8 -*-
"""
AI_Models/model_transformer.py

Tabular regression with a small Transformer encoder.

Relative paths resolved from this file:
- dataset:  ../dataset/preprocessed_dataset.csv
- outputs:  ../results_transformer
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
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
DEFAULT_OUT_DIR = PROJECT_ROOT / "results_transformer"


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


# ---------- Scalers ----------
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


def save_scalers_json(path: Path, x_scaler: StandardScaler, y_scaler: TargetScaler, target_log: bool) -> None:
    if x_scaler.mean_ is None or x_scaler.std_ is None:
        raise RuntimeError("x_scaler not fitted.")
    if y_scaler.mean_ is None or y_scaler.std_ is None:
        raise RuntimeError("y_scaler not fitted.")
    payload = {
        "x_mean": x_scaler.mean_.tolist(),
        "x_std": x_scaler.std_.tolist(),
        "y_mean": float(y_scaler.mean_),
        "y_std": float(y_scaler.std_),
        "target_log": bool(target_log),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_scalers_json(path: Path) -> Tuple[StandardScaler, TargetScaler, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    xs = StandardScaler()
    ys = TargetScaler()
    xs.mean_ = np.asarray(payload["x_mean"], dtype=np.float32)
    xs.std_ = np.asarray(payload["x_std"], dtype=np.float32)
    ys.mean_ = float(payload["y_mean"])
    ys.std_ = float(payload["y_std"])
    return xs, ys, bool(payload.get("target_log", True))


# ---------- Dataset ----------
class FeatureSeqDataset(Dataset):
    """
    X: (N,F) -> x: (F,1) tokens
    y: (N,)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, noise_std: float = 0.0) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.noise_std = float(noise_std)
        if self.X.ndim != 2 or self.y.ndim != 1:
            raise ValueError("X must be (N,F) and y must be (N,).")
        if len(self.X) != len(self.y):
            raise ValueError("X and y length mismatch.")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.noise_std > 0:
            x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
        x = x.reshape(-1, 1)  # (F,1)
        return torch.from_numpy(x), torch.tensor(float(self.y[idx]), dtype=torch.float32)


# ---------- Model ----------
class TransformerRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.15,
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

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=bool(norm_first),
        )

        # PyTorch 버전에 따라 enable_nested_tensor 인자가 없을 수 있어 안전 처리
        try:
            self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,F,1) -> y: (B,)
        B, F, _ = x.shape
        if F != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {F}")

        v = self.value_proj(x)
        feat_ids = torch.arange(F, device=x.device).unsqueeze(0).expand(B, F)
        v = v + self.feature_emb(feat_ids)

        pos_ids = torch.arange(1, F + 1, device=x.device).unsqueeze(0).expand(B, F)
        v = v + self.pos_emb(pos_ids)

        cls = self.cls_token.expand(B, 1, self.d_model)
        cls = cls + self.pos_emb(torch.zeros((B, 1), device=x.device, dtype=torch.long))

        tokens = torch.cat([cls, v], dim=1)
        z = self.encoder(tokens)
        return self.head(z[:, 0, :]).squeeze(-1)


# ---------- EMA ----------
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


# ---------- Metrics / transforms ----------
def _safe_expm1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -50.0, 50.0)
    return np.expm1(x)


def inverse_target(y_scaled: np.ndarray, y_scaler: TargetScaler, target_log: bool) -> np.ndarray:
    y_t = y_scaler.inverse_transform(np.asarray(y_scaled, dtype=np.float32))
    return _safe_expm1(y_t) if target_log else np.asarray(y_t, dtype=np.float64)


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


# ---------- AMP helpers (new API) ----------
def _amp_available_for_device(device: torch.device, amp_flag: bool) -> bool:
    return bool(amp_flag and device.type == "cuda")


def _get_autocast(device: torch.device, enabled: bool):
    # torch.amp.autocast(device_type="cuda") 형태 지원
    try:
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    except Exception:
        # fallback (아주 오래된 torch)
        return torch.cuda.amp.autocast(enabled=enabled)


def _get_grad_scaler(device: torch.device, enabled: bool):
    try:
        return torch.amp.GradScaler(device_type=device.type, enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


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
        ys.extend(y.numpy().tolist())
        ps.extend(pred.detach().cpu().numpy().tolist())
    return np.asarray(ys, dtype=np.float64), np.asarray(ps, dtype=np.float64)


# ---------- Split ----------
def split_indices_random(n: int, test_ratio: float, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut_test = int(n * (1.0 - test_ratio))
    trainval, test = idx[:cut_test], idx[cut_test:]
    cut_val = int(len(trainval) * (1.0 - val_ratio))
    train, val = trainval[:cut_val], trainval[cut_val:]
    return train, val, test


def split_indices_stratified_by_target(
    y: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 50:
        return split_indices_random(n, test_ratio, val_ratio, seed)
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 4:
        return split_indices_random(n, test_ratio, val_ratio, seed)

    bins = np.digitize(y, edges[1:-1], right=True)
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for b in np.unique(bins):
        idx_b = np.where(bins == b)[0]
        rng.shuffle(idx_b)
        nb = len(idx_b)
        n_te = int(round(nb * test_ratio))
        n_tv = nb - n_te
        n_va = int(round(n_tv * val_ratio))
        n_tr = n_tv - n_va
        tr.extend(idx_b[:n_tr].tolist())
        va.extend(idx_b[n_tr:n_tr + n_va].tolist())
        te.extend(idx_b[n_tr + n_va:].tolist())
    return np.asarray(tr), np.asarray(va), np.asarray(te)


# ---------- Train core ----------
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
    scaler = _get_grad_scaler(device, enabled=use_amp)
    grad_accum = max(1, int(grad_accum))

    total = 0.0
    optim.zero_grad(set_to_none=True)

    for i, (x, y) in enumerate(loader, start=1):
        x, y = x.to(device), y.to(device)
        with _get_autocast(device, enabled=use_amp):
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
    feature_cols: List[str]
    target_col: str
    target_log: bool
    test_ratio: float
    val_ratio: float
    split_strategy: str
    seed: int
    deterministic: bool
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


@dataclass
class TrainResult:
    model: nn.Module
    x_scaler: StandardScaler
    y_scaler: TargetScaler
    target_log: bool
    feature_cols: List[str]
    best_val: Dict[str, float]
    test: Dict[str, float]
    history: Dict[str, List[float]]
    config: TrainConfig
    output_dir: str


def run_training_transformer(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = ("기초금액", "추정가격", "예가범위", "낙찰하한율"),
    target_col: str = "낙찰가",
    target_log: bool = True,
    test_ratio: float = 0.20,
    val_ratio: float = 0.10,
    split_strategy: str = "stratified",
    seed: int = 42,
    deterministic: bool = True,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 2e-4,
    patience: int = 20,
    loss_name: str = "huber",
    huber_beta: float = 1.0,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 3,
    dim_feedforward: int = 512,
    dropout: float = 0.15,
    amp: bool = True,
    grad_accum_steps: int = 1,
    feature_noise_std: float = 0.01,
    ema_decay: float = 0.999,
    onecycle: bool = True,
    clip_grad: float = 1.0,
    num_workers: int = 0,
    output_dir: str = str(DEFAULT_OUT_DIR),
    save_model_name: str = "best_model.pt",
    save_scalers_name: str = "scalers.json",
    save_config_name: str = "train_config.json",
    verbose: bool = True,
) -> TrainResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed, deterministic=deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 성능 옵션 (CUDA에서만)
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
        # CPU면 AMP는 의미가 없거나 오히려 느려질 수 있어 자동 off
        amp = False

    if target_col not in df.columns:
        raise KeyError(f"target_col not found: {target_col}")

    if feature_cols is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric if c != target_col]
        if not feature_cols:
            raise ValueError("feature_cols=None but no numeric features found.")

    feature_cols = list(feature_cols)
    need = feature_cols + [target_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    work = df[need].copy()
    for c in need:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=need).reset_index(drop=True)
    if len(work) < 50:
        raise ValueError(f"Not enough valid rows: {len(work)}")

    X = work[feature_cols].to_numpy(np.float32)
    y_raw = work[target_col].to_numpy(np.float32)

    if split_strategy.lower() == "stratified":
        tr, va, te = split_indices_stratified_by_target(y_raw, test_ratio, val_ratio, seed)
    else:
        tr, va, te = split_indices_random(len(work), test_ratio, val_ratio, seed)

    X_tr, X_va, X_te = X[tr], X[va], X[te]
    y_tr_raw, y_va_raw, y_te_raw = y_raw[tr], y_raw[va], y_raw[te]

    if target_log:
        y_tr_t = np.log1p(np.clip(y_tr_raw, 0.0, None))
        y_va_t = np.log1p(np.clip(y_va_raw, 0.0, None))
        y_te_t = np.log1p(np.clip(y_te_raw, 0.0, None))
    else:
        y_tr_t, y_va_t, y_te_t = y_tr_raw, y_va_raw, y_te_raw

    x_scaler = StandardScaler().fit(X_tr)
    y_scaler = TargetScaler().fit(y_tr_t)

    X_tr_s = x_scaler.transform(X_tr)
    X_va_s = x_scaler.transform(X_va)
    X_te_s = x_scaler.transform(X_te)

    y_tr_s = y_scaler.transform(y_tr_t)
    y_va_s = y_scaler.transform(y_va_t)
    y_te_s = y_scaler.transform(y_te_t)

    ds_tr = FeatureSeqDataset(X_tr_s, y_tr_s, noise_std=feature_noise_std)
    ds_va = FeatureSeqDataset(X_va_s, y_va_s, noise_std=0.0)
    ds_te = FeatureSeqDataset(X_te_s, y_te_s, noise_std=0.0)

    pin = device.type == "cuda"
    # Windows에서 num_workers>0는 spawn 오버헤드가 있을 수 있어 기본 0 유지(필요시 test에서 올리면 됨)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    dl_va = DataLoader(ds_va, batch_size=max(128, batch_size), shuffle=False, num_workers=num_workers, pin_memory=pin)
    dl_te = DataLoader(ds_te, batch_size=max(128, batch_size), shuffle=False, num_workers=num_workers, pin_memory=pin)

    model = TransformerRegressor(
        num_features=len(feature_cols),
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

    config = TrainConfig(
        feature_cols=feature_cols,
        target_col=target_col,
        target_log=bool(target_log),
        test_ratio=float(test_ratio),
        val_ratio=float(val_ratio),
        split_strategy=str(split_strategy),
        seed=int(seed),
        deterministic=bool(deterministic),
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

    (out_dir / save_config_name).write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")
    save_scalers_json(out_dir / save_scalers_name, x_scaler, y_scaler, target_log=target_log)

    history: Dict[str, List[float]] = {"train_loss": [], "val_rmse": [], "lr": [], "epoch_sec": []}
    best_rmse = float("inf")
    best_state = None
    bad = 0

    if verbose:
        print(f"[Device] {device}")
        print(f"[Data] rows={len(work)} | features={len(feature_cols)} | train={len(tr)} val={len(va)} test={len(te)}")
        print(f"[Config] epochs={epochs} batch={batch_size} lr={lr} amp={amp} onecycle={onecycle} ema={ema is not None}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(
            model, dl_tr, optim, device, loss_fn,
            amp=amp, grad_accum=grad_accum_steps, clip_grad=clip_grad,
            scheduler=scheduler, ema=ema
        )

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
            best_state = deepcopy(model.state_dict())
            if ema is not None:
                ema.apply_to(model)
                best_state = deepcopy(model.state_dict())
                ema.restore(model)
            torch.save(best_state, out_dir / save_model_name)
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[EarlyStop] no improvement for {patience} epochs. best_val_rmse={best_rmse:.3f}")
                break

    if best_state is None:
        best_state = deepcopy(model.state_dict())
        torch.save(best_state, out_dir / save_model_name)

    model.load_state_dict(best_state)

    y_te_true_s, y_te_pred_s = predict_scaled(model, dl_te, device, amp=amp)
    y_te_true = inverse_target(y_te_true_s, y_scaler, target_log)
    y_te_pred = inverse_target(y_te_pred_s, y_scaler, target_log)
    test_m = regression_metrics(y_te_true, y_te_pred)

    y_va_true_s, y_va_pred_s = predict_scaled(model, dl_va, device, amp=amp)
    y_va_true = inverse_target(y_va_true_s, y_scaler, target_log)
    y_va_pred = inverse_target(y_va_pred_s, y_scaler, target_log)
    best_val_m = regression_metrics(y_va_true, y_va_pred)

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    _plot_curves(history, out_dir)
    _plot_scatter(y_te_true, y_te_pred, out_dir)

    return TrainResult(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        target_log=target_log,
        feature_cols=feature_cols,
        best_val=best_val_m,
        test=test_m,
        history=history,
        config=config,
        output_dir=str(out_dir),
    )


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
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("actual")
    plt.ylabel("predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_scatter.png")
    plt.close()


# ---------- Inference helpers ----------
def load_model_artifacts(
    artifacts_dir: Path,
    device: Optional[torch.device] = None,
    model_name: str = "best_model.pt",
    scalers_name: str = "scalers.json",
    config_name: str = "train_config.json",
) -> Tuple[nn.Module, StandardScaler, TargetScaler, bool, List[str]]:
    artifacts_dir = Path(artifacts_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = json.loads((artifacts_dir / config_name).read_text(encoding="utf-8"))
    feature_cols = list(cfg["feature_cols"])
    target_log = bool(cfg.get("target_log", True))

    x_scaler, y_scaler, tlog_scaler = load_scalers_json(artifacts_dir / scalers_name)
    target_log = bool(cfg.get("target_log", tlog_scaler))

    model = TransformerRegressor(
        num_features=len(feature_cols),
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        norm_first=True,
    ).to(device)

    state = torch.load(artifacts_dir / model_name, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, x_scaler, y_scaler, target_log, feature_cols


@torch.no_grad()
def predict_dataframe(
    model: nn.Module,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    x_scaler: StandardScaler,
    y_scaler: TargetScaler,
    target_log: bool,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
    amp: bool = False,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = df[list(feature_cols)].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna(subset=list(feature_cols))
    X_s = x_scaler.transform(X.to_numpy(np.float32))

    ds = FeatureSeqDataset(X_s, np.zeros(len(X_s), dtype=np.float32), noise_std=0.0)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    use_amp = _amp_available_for_device(device, amp)
    preds_s: List[float] = []
    model.eval()
    for x, _ in dl:
        x = x.to(device)
        with _get_autocast(device, enabled=use_amp):
            y_hat = model(x)
        preds_s.extend(y_hat.detach().cpu().numpy().tolist())

    return inverse_target(np.asarray(preds_s, dtype=np.float64), y_scaler, target_log)


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--target", type=str, default="낙찰가")
    p.add_argument("--features", nargs="*", default=None)
    p.add_argument("--no_log_target", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data) if args.data else DEFAULT_DATA_PATH
    out_dir = Path(args.out) if args.out else DEFAULT_OUT_DIR

    df = read_csv_safely(data_path)

    feature_cols = args.features
    if feature_cols is None:
        feature_cols = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]

    res = run_training_transformer(
        df=df,
        feature_cols=feature_cols,
        target_col=args.target,
        target_log=not args.no_log_target,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
        output_dir=str(out_dir),
        verbose=bool(args.verbose),
    )

    print("\n[Best VAL]")
    for k, v in res.best_val.items():
        print(f"{k:>6}: {v:.6f}")
    print("\n[TEST]")
    for k, v in res.test.items():
        print(f"{k:>6}: {v:.6f}")
    print(f"\nArtifacts: {res.output_dir}")


if __name__ == "__main__":
    main()
