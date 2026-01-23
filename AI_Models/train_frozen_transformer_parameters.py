# -*- coding: utf-8 -*-
"""
train_frozen_transformer_parameters.py

Train only parameters (a, b) while keeping transformer model frozen.
Uses pre-trained transformer_4feat.pt to generate M, then optimizes (a, b).
Includes train/validation split, loss curves, and R2 score like model_transformer_4feat_train.py
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from model_transformer import read_csv_safely, load_scalers_json, TransformerRegressor
from scipy.optimize import least_squares
import json
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 1e-10


def load_frozen_model(model_path: Path, device=DEVICE):
    """Load transformer model and freeze all parameters"""
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state = checkpoint['model_state']
    else:
        state = checkpoint
    
    num_features = 4
    for key in state.keys():
        if 'feature_emb' in key and 'weight' in key:
            num_features = state[key].shape[0]
            break
    
    model = TransformerRegressor(
        num_features=num_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    # ✅ FREEZE ALL PARAMETERS
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    model.eval()
    
    print("✅ 모델 파라미터 동결 (Model parameters FROZEN)")
    print(f"   학습 가능한 파라미터: 0개 (Trainable parameters: 0)")
    
    return model


def calculate_r2_score(y_actual, y_pred):
    """Calculate R2 score"""
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))
    return float(r2)


def main():
    """Main training"""
    
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    csv_path = PROJECT_ROOT / "dataset" / "dataset_feature_selected.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "dataset" / "preprocessed_dataset.csv"
    
    print(f"📂 데이터 로드 중: {csv_path}")
    df = read_csv_safely(csv_path)
    
    # ============================================================
    # 2. LOAD FROZEN TRANSFORMER MODEL
    # ============================================================
    model_path = PROJECT_ROOT / "results_transformer_4feat" / "transformer_4feat.pt"
    scalers_path = PROJECT_ROOT / "results_transformer_4feat" / "scalers.json"
    
    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    if not scalers_path.exists():
        print(f"❌ Scalers 파일을 찾을 수 없습니다: {scalers_path}")
        return
    
    print(f"\n🔒 동결된 Transformer 모델 로드 중: {model_path}")
    model = load_frozen_model(model_path, DEVICE)
    x_scaler, y_scaler, target_log = load_scalers_json(scalers_path)
    print("✅ Scaler 로드 완료")
    
    # ============================================================
    # 3. PREPARE DATA & GET PREDICTIONS M
    # ============================================================
    feature_cols = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]
    target_col = "낙찰가"
    
    print(f"\n📊 사용될 입력 피처 ({len(feature_cols)}개): {feature_cols}")
    print("🚀 Transformer 모델 파라미터 최적화 시작...")
    
    X = df[feature_cols].to_numpy(dtype=np.float32)
    X_scaled = x_scaler.transform(X)
    x_tensor = torch.from_numpy(X_scaled).unsqueeze(-1)
    
    with torch.no_grad():
        pred_scaled = model(x_tensor.to(DEVICE)).cpu().numpy()
    
    pred_log = y_scaler.inverse_transform(pred_scaled)
    M = np.expm1(pred_log).flatten()
    y_actual = df[target_col].to_numpy(dtype=np.float32)
    
    # Split into train/val
    train_idx, val_idx = train_test_split(
        np.arange(len(M)), 
        test_size=0.1, 
        random_state=42
    )
    
    M_train, M_val = M[train_idx], M[val_idx]
    y_train, y_val = y_actual[train_idx], y_actual[val_idx]
    
    print(f"✅ 데이터 분할: Train={len(M_train)}, Val={len(M_val)}")
    
    # ============================================================
    # 4. MODEL 1: DIRECT TRANSFORMER OUTPUT M (NO TRANSFORMATION)
    # ============================================================
    print("\n" + "="*80)
    print("🔵 모델 1: 직접 출력 (MODEL 1 - DIRECT OUTPUT): y = M")
    print("="*80)
    
    # Use M directly
    pred_train_m = M_train
    pred_val_m = M_val
    pred_m = M
    
    # Train metrics
    mse_train_m = np.mean((y_train - pred_train_m) ** 2)
    rmse_train_m = np.sqrt(mse_train_m)
    mae_train_m = np.mean(np.abs(y_train - pred_train_m))
    r2_train_m = calculate_r2_score(y_train, pred_train_m)
    
    # Val metrics
    mse_val_m = np.mean((y_val - pred_val_m) ** 2)
    rmse_val_m = np.sqrt(mse_val_m)
    mae_val_m = np.mean(np.abs(y_val - pred_val_m))
    r2_val_m = calculate_r2_score(y_val, pred_val_m)
    
    # Full metrics
    mse_m = np.mean((y_actual - pred_m) ** 2)
    rmse_m = np.sqrt(mse_m)
    mae_m = np.mean(np.abs(y_actual - pred_m))
    r2_m = calculate_r2_score(y_actual, pred_m)
    
    print(f"\n📌 모델: y = M (변환 없음 / No transformation)")
    
    print(f"\n📊 학습 세트 성능 (Train Metrics):")
    print(f"   MSE  = {mse_train_m:.6e}")
    print(f"   RMSE = {rmse_train_m:.6e}")
    print(f"   MAE  = {mae_train_m:.6e}")
    print(f"   R²   = {r2_train_m:.6f}")
    
    print(f"\n📊 검증 세트 성능 (Val Metrics):")
    print(f"   MSE  = {mse_val_m:.6e}")
    print(f"   RMSE = {rmse_val_m:.6e}")
    print(f"   MAE  = {mae_val_m:.6e}")
    print(f"   R²   = {r2_val_m:.6f}")
    
    print(f"\n📊 전체 성능 (Full Metrics):")
    print(f"   MSE  = {mse_m:.6e}")
    print(f"   RMSE = {rmse_m:.6e}")
    print(f"   MAE  = {mae_m:.6e}")
    print(f"   R²   = {r2_m:.6f}")
    
    # ============================================================
    # 5. MODEL 2: LOG-LINEAR WITH 3 PARAMETERS: y = a*log(b*M) + c
    # ============================================================
    print("\n" + "="*80)
    print("🟠 모델 2: 로그 변환 (MODEL 2 - LOG TRANSFORMATION): y = a*log(b*M) + c")
    print("="*80)
    
    # Prepare data for fitting y = a*log(b*M) + c
    # We need to find a, b, c such that y = a*log(b*M) + c
    # This can be rewritten as: log(b*M) = (y - c) / a
    # We'll use optimization or numerical methods
    
    # Use non-linear least squares
    from scipy.optimize import least_squares
    
    def model_func(params, M_data, y_data):
        a, b, c = params
        # Ensure b*M is positive for log
        bM = np.clip(b * M_data, 1e-12, None)
        pred = a * np.log(bM) + c
        return y_data - pred
    
    # Initial guess
    initial_guess = [1.0, 1.0, np.mean(y_train)]
    
    # Fit on training data
    result = least_squares(model_func, initial_guess, args=(M_train, y_train))
    a_log, b_log, c_log = result.x
    
    # Predictions
    bM = np.clip(b_log * M, 1e-12, None)
    pred_loglinear2 = a_log * np.log(bM) + c_log
    
    bM_train = np.clip(b_log * M_train, 1e-12, None)
    pred_train_loglinear2 = a_log * np.log(bM_train) + c_log
    
    bM_val = np.clip(b_log * M_val, 1e-12, None)
    pred_val_loglinear2 = a_log * np.log(bM_val) + c_log
    
    # Train metrics
    mse_train_loglinear2 = np.mean((y_train - pred_train_loglinear2) ** 2)
    rmse_train_loglinear2 = np.sqrt(mse_train_loglinear2)
    mae_train_loglinear2 = np.mean(np.abs(y_train - pred_train_loglinear2))
    r2_train_loglinear2 = calculate_r2_score(y_train, pred_train_loglinear2)
    
    # Val metrics
    mse_val_loglinear2 = np.mean((y_val - pred_val_loglinear2) ** 2)
    rmse_val_loglinear2 = np.sqrt(mse_val_loglinear2)
    mae_val_loglinear2 = np.mean(np.abs(y_val - pred_val_loglinear2))
    r2_val_loglinear2 = calculate_r2_score(y_val, pred_val_loglinear2)
    
    # Full metrics
    mse_loglinear2 = np.mean((y_actual - pred_loglinear2) ** 2)
    rmse_loglinear2 = np.sqrt(mse_loglinear2)
    mae_loglinear2 = np.mean(np.abs(y_actual - pred_loglinear2))
    r2_loglinear2 = calculate_r2_score(y_actual, pred_loglinear2)
    
    print(f"\n📌 최적화된 매개변수 (Optimized Parameters):")
    print(f"   a = {a_log:.12e}")
    print(f"   b = {b_log:.12e}")
    print(f"   c = {c_log:.12e}")
    
    print(f"\n📊 학습 세트 성능 (Train Metrics):")
    print(f"   MSE  = {mse_train_loglinear2:.6e}")
    print(f"   RMSE = {rmse_train_loglinear2:.6e}")
    print(f"   MAE  = {mae_train_loglinear2:.6e}")
    print(f"   R²   = {r2_train_loglinear2:.6f}")
    
    print(f"\n📊 검증 세트 성능 (Val Metrics):")
    print(f"   MSE  = {mse_val_loglinear2:.6e}")
    print(f"   RMSE = {rmse_val_loglinear2:.6e}")
    print(f"   MAE  = {mae_val_loglinear2:.6e}")
    print(f"   R²   = {r2_val_loglinear2:.6f}")
    
    print(f"\n📊 전체 성능 (Full Metrics):")
    print(f"   MSE  = {mse_loglinear2:.6e}")
    print(f"   RMSE = {rmse_loglinear2:.6e}")
    print(f"   MAE  = {mae_loglinear2:.6e}")
    print(f"   R²   = {r2_loglinear2:.6f}")
    
    # ============================================================
    # 6. THRESHOLD-BASED ROUTING (1e-10)
    # ============================================================
    print("\n" + "="*80)
    print("🎯 임계값 기반 라우팅 (THRESHOLD-BASED ROUTING): 1e-10")
    print("="*80)
    
    # Calculate difference between two models
    difference = np.abs(pred_m - pred_loglinear2)
    
    # Routing: if difference < 1e-10, use M; otherwise use a*log(b*M)+c
    use_m = difference < THRESHOLD
    use_loglinear = difference >= THRESHOLD
    
    pred_threshold = np.where(use_m, pred_m, pred_loglinear2)
    
    mse_threshold = np.mean((y_actual - pred_threshold) ** 2)
    rmse_threshold = np.sqrt(mse_threshold)
    mae_threshold = np.mean(np.abs(y_actual - pred_threshold))
    r2_threshold = calculate_r2_score(y_actual, pred_threshold)
    
    n_m = np.sum(use_m)
    n_loglinear = np.sum(use_loglinear)
    
    print(f"\n📊 라우팅 통계 (Routing Statistics):")
    print(f"   임계값: {THRESHOLD:.0e}")
    print(f"   모델 M 사용: {n_m} 샘플 ({n_m/len(M)*100:.1f}%)")
    print(f"   모델 a*log(b*M)+c 사용: {n_loglinear} 샘플 ({n_loglinear/len(M)*100:.1f}%)")
    
    print(f"\n📊 성능 지표 (Performance Metrics):")
    print(f"   MSE  = {mse_threshold:.6e}")
    print(f"   RMSE = {rmse_threshold:.6e}")
    print(f"   MAE  = {mae_threshold:.6e}")
    print(f"   R²   = {r2_threshold:.6f}")
    
    # ============================================================
    # 7. SAMPLE PREDICTIONS
    # ============================================================
    sample_size = min(10, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    print("\n" + "="*80)
    print("[샘플 예측 결과 (Sample predictions)]")
    print("="*80)
    
    sample_df = pd.DataFrame({
        '실제낙찰가': y_actual[sample_indices],
        '모델M': pred_m[sample_indices],
        '모델alog(bM)+c': pred_loglinear2[sample_indices],
        '임계값선택': pred_threshold[sample_indices],
    })
    
    pd.options.display.float_format = '{:,.2f}'.format
    print(sample_df.to_string())
    
    # ============================================================
    # 8. MODEL COMPARISON TABLE
    # ============================================================
    print("\n" + "="*80)
    print("[모델 성능 비교 (Model Performance Comparison)]")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Model': ['M (Direct)', 'a*log(b*M)+c', 'Threshold-Routed'],
        'Train R²': [f"{r2_train_m:.6f}", f"{r2_train_loglinear2:.6f}", "-"],
        'Val R²': [f"{r2_val_m:.6f}", f"{r2_val_loglinear2:.6f}", "-"],
        'Full R²': [f"{r2_m:.6f}", f"{r2_loglinear2:.6f}", f"{r2_threshold:.6f}"],
        'MAE': [f"{mae_m:.6e}", f"{mae_loglinear2:.6e}", f"{mae_threshold:.6e}"],
    })
    
    pd.options.display.float_format = None
    print(comparison.to_string(index=False))
    
    # ============================================================
    # 9. VISUALIZATION: LOSS CURVES
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear Model - Train vs Val Loss
    axes[0, 0].plot([mse_train_m], 'o-', label='Train Loss', linewidth=2, markersize=8)
    axes[0, 0].plot([mse_val_m], 's-', label='Val Loss', linewidth=2, markersize=8)
    axes[0, 0].set_title(f'Model M (Direct Output)\nTrain R²={r2_train_m:.4f}, Val R²={r2_val_m:.4f}')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log-Linear Model - Train vs Val Loss
    axes[0, 1].plot([mse_train_loglinear2], 'o-', label='Train Loss', linewidth=2, markersize=8, color='orange')
    axes[0, 1].plot([mse_val_loglinear2], 's-', label='Val Loss', linewidth=2, markersize=8, color='darkorange')
    axes[0, 1].set_title(f'Model a*log(b*M)+c\na={a_log:.2e}, b={b_log:.2e}, c={c_log:.2e}\nTrain R²={r2_train_loglinear2:.4f}, Val R²={r2_val_loglinear2:.4f}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Actual vs Predicted - Model M
    axes[1, 0].scatter(y_actual, pred_m, alpha=0.5, s=20)
    axes[1, 0].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', label='Perfect')
    axes[1, 0].set_xlabel('Actual')
    axes[1, 0].set_ylabel('Predicted')
    axes[1, 0].set_title(f'Model M Predictions\nR² = {r2_m:.6f}, MAE = {mae_m:.2e}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Actual vs Predicted - Model a*log(b*M)+c
    axes[1, 1].scatter(y_actual, pred_loglinear2, alpha=0.5, s=20, color='orange')
    axes[1, 1].plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', label='Perfect')
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].set_title(f'Model a*log(b*M)+c Predictions\nR² = {r2_loglinear2:.6f}, MAE = {mae_loglinear2:.2e}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_fig = BASE_DIR / "frozen_transformer_training_results.png"
    plt.savefig(output_fig, dpi=100, bbox_inches='tight')
    print(f"\n✅ 그래프 저장: {output_fig}")
    
    # ============================================================
    # 10. SAVE PARAMETERS
    # ============================================================
    print("\n" + "="*80)
    print("💾 학습된 매개변수 저장 (Saving Learned Parameters)")
    print("="*80)
    
    params = {
        "model_type": "Frozen Transformer with M and a*log(b*M)+c models",
        "transformer_model": str(model_path),
        "status": "Transformer parameters FROZEN",
        "data_split": {
            "train_samples": int(len(M_train)),
            "val_samples": int(len(M_val)),
        },
        
        "model_m": {
            "formula": "y = M (no transformation)",
            "train_metrics": {
                "MSE": float(mse_train_m),
                "RMSE": float(rmse_train_m),
                "MAE": float(mae_train_m),
                "R2": float(r2_train_m),
            },
            "val_metrics": {
                "MSE": float(mse_val_m),
                "RMSE": float(rmse_val_m),
                "MAE": float(mae_val_m),
                "R2": float(r2_val_m),
            },
            "full_metrics": {
                "MSE": float(mse_m),
                "RMSE": float(rmse_m),
                "MAE": float(mae_m),
                "R2": float(r2_m),
            },
        },
        
        "model_loglinear": {
            "formula": "y = a*log(b*M) + c",
            "parameters": {
                "a": float(a_log),
                "b": float(b_log),
                "c": float(c_log),
            },
            "train_metrics": {
                "MSE": float(mse_train_loglinear2),
                "RMSE": float(rmse_train_loglinear2),
                "MAE": float(mae_train_loglinear2),
                "R2": float(r2_train_loglinear2),
            },
            "val_metrics": {
                "MSE": float(mse_val_loglinear2),
                "RMSE": float(rmse_val_loglinear2),
                "MAE": float(mae_val_loglinear2),
                "R2": float(r2_val_loglinear2),
            },
            "full_metrics": {
                "MSE": float(mse_loglinear2),
                "RMSE": float(rmse_loglinear2),
                "MAE": float(mae_loglinear2),
                "R2": float(r2_loglinear2),
            },
        },
        
        "threshold_routing": {
            "threshold": float(THRESHOLD),
            "condition": "Use M if |M - a*log(b*M)+c| < 1e-10, otherwise use a*log(b*M)+c",
            "samples_use_m": int(n_m),
            "samples_use_loglinear": int(n_loglinear),
            "metrics": {
                "MSE": float(mse_threshold),
                "RMSE": float(rmse_threshold),
                "MAE": float(mae_threshold),
                "R2": float(r2_threshold),
            },
        }
    }
    
    output_json = BASE_DIR / "frozen_transformer_parameters.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"✅ 저장 완료: {output_json}")
    
    # Save predictions
    output_csv = BASE_DIR / "frozen_transformer_predictions.csv"
    pred_df = pd.DataFrame({
        'Actual': y_actual,
        'Pred_M': pred_m,
        'Pred_LogTransform': pred_loglinear2,
        'Pred_Threshold': pred_threshold,
    })
    pred_df.to_csv(output_csv, index=False)
    print(f"✅ 저장 완료: {output_csv}")
    
    print("\n" + "="*80)
    print("✅ 학습 완료 (TRAINING COMPLETE)")
    print("="*80)


if __name__ == "__main__":
    main()


def load_frozen_model(model_path: Path, device=DEVICE):
    """Load transformer model and freeze all parameters"""
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        state = checkpoint['model_state']
    else:
        state = checkpoint
    
    num_features = 4
    for key in state.keys():
        if 'feature_emb' in key and 'weight' in key:
            num_features = state[key].shape[0]
            break
    
    model = TransformerRegressor(
        num_features=num_features,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    
    # ✅ FREEZE ALL PARAMETERS
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    model.eval()
    
    print("✅ 모델 파라미터 동결 (Model parameters FROZEN)")
    print(f"   학습 가능한 파라미터: 0개 (Trainable parameters: 0)")
    
    return model


def main():
    """Main training"""
    
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    csv_path = PROJECT_ROOT / "dataset" / "dataset_feature_selected.csv"
    if not csv_path.exists():
        csv_path = PROJECT_ROOT / "dataset" / "preprocessed_dataset.csv"
    
    print(f"📂 데이터 로드 중: {csv_path}")
    df = read_csv_safely(csv_path)
    
    # ============================================================
    # 2. LOAD FROZEN TRANSFORMER MODEL
    # ============================================================
    model_path = PROJECT_ROOT / "results_transformer_4feat" / "transformer_4feat.pt"
    scalers_path = PROJECT_ROOT / "results_transformer_4feat" / "scalers.json"
    
    if not model_path.exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    if not scalers_path.exists():
        print(f"❌ Scalers 파일을 찾을 수 없습니다: {scalers_path}")
        return
    
    print(f"\n🔒 동결된 Transformer 모델 로드 중: {model_path}")
    model = load_frozen_model(model_path, DEVICE)
    x_scaler, y_scaler, target_log = load_scalers_json(scalers_path)
    print("✅ Scaler 로드 완료")
    
    # ============================================================
    # 3. GET MODEL PREDICTIONS M (FROZEN - NO GRADIENT)
    # ============================================================
    feature_cols = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]
    target_col = "낙찰가"
    
    print(f"\n📊 사용될 입력 피처 ({len(feature_cols)}개): {feature_cols}")
    
    X = df[feature_cols].to_numpy(dtype=np.float32)
    X_scaled = x_scaler.transform(X)
    x_tensor = torch.from_numpy(X_scaled).unsqueeze(-1)
    
    print("\n🔮 Transformer 예측 (M) 계산 중...")
    with torch.no_grad():
        pred_scaled = model(x_tensor.to(DEVICE)).cpu().numpy()
    
    pred_log = y_scaler.inverse_transform(pred_scaled)
    M = np.expm1(pred_log).flatten()
    
    y_actual = df[target_col].to_numpy(dtype=np.float32)
    
    print(f"✅ M 계산 완료")
    print(f"   개수: {len(M)} 샘플")
    print(f"   평균: {M.mean():.6e}")
    print(f"   표준편차: {M.std():.6e}")
    
    # ============================================================
    # 4. TRAIN LINEAR PARAMETERS: y = a*M + b
    # ============================================================
    print("\n" + "="*80)
    print("🔵 선형 모델 학습 (LINEAR MODEL TRAINING): y = a*M + b")
    print("="*80)
    
    # Least squares fitting
    A_linear = np.column_stack([M, np.ones(len(M))])
    params_linear = np.linalg.lstsq(A_linear, y_actual, rcond=None)[0]
    a_linear, b_linear = params_linear
    
    pred_linear = a_linear * M + b_linear
    error_linear = y_actual - pred_linear
    
    mse_linear = np.mean(error_linear ** 2)
    rmse_linear = np.sqrt(mse_linear)
    mae_linear = np.mean(np.abs(error_linear))
    
    print(f"\n📌 최적화된 매개변수 (Optimized Parameters):")
    print(f"   a = {a_linear:.12e}")
    print(f"   b = {b_linear:.12e}")
    
    print(f"\n📊 성능 지표 (Performance Metrics):")
    print(f"   MSE  = {mse_linear:.6e}")
    print(f"   RMSE = {rmse_linear:.6e}")
    print(f"   MAE  = {mae_linear:.6e}")
    
    # ============================================================
    # 5. TRAIN LOG-LINEAR PARAMETERS: y = a*log(M) + b
    # ============================================================
    print("\n" + "="*80)
    print("🟠 로그선형 모델 학습 (LOG-LINEAR MODEL TRAINING): y = a*log(M) + b")
    print("="*80)
    
    M_clipped = np.clip(M, 1e-12, None)
    log_M = np.log(M_clipped)
    
    A_loglinear = np.column_stack([log_M, np.ones(len(M))])
    params_loglinear = np.linalg.lstsq(A_loglinear, y_actual, rcond=None)[0]
    a_loglinear, b_loglinear = params_loglinear
    
    pred_loglinear = a_loglinear * log_M + b_loglinear
    error_loglinear = y_actual - pred_loglinear
    
    mse_loglinear = np.mean(error_loglinear ** 2)
    rmse_loglinear = np.sqrt(mse_loglinear)
    mae_loglinear = np.mean(np.abs(error_loglinear))
    
    print(f"\n📌 최적화된 매개변수 (Optimized Parameters):")
    print(f"   a = {a_loglinear:.12e}")
    print(f"   b = {b_loglinear:.12e}")
    
    print(f"\n📊 성능 지표 (Performance Metrics):")
    print(f"   MSE  = {mse_loglinear:.6e}")
    print(f"   RMSE = {rmse_loglinear:.6e}")
    print(f"   MAE  = {mae_loglinear:.6e}")
    
    # ============================================================
    # 6. THRESHOLD-BASED ROUTING
    # ============================================================
    print("\n" + "="*80)
    print("🎯 임계값 기반 라우팅 (THRESHOLD-BASED ROUTING): 1e-10")
    print("="*80)
    
    difference = np.abs(pred_linear - pred_loglinear)
    use_linear = difference < THRESHOLD
    use_loglinear = ~use_linear
    
    pred_threshold = np.where(use_linear, pred_linear, pred_loglinear)
    error_threshold = y_actual - pred_threshold
    
    mse_threshold = np.mean(error_threshold ** 2)
    rmse_threshold = np.sqrt(mse_threshold)
    mae_threshold = np.mean(np.abs(error_threshold))
    
    n_linear = np.sum(use_linear)
    n_loglinear = np.sum(use_loglinear)
    
    print(f"\n📊 라우팅 통계 (Routing Statistics):")
    print(f"   임계값: {THRESHOLD:.0e}")
    print(f"   선형 모델 사용: {n_linear} 샘플 ({n_linear/len(M)*100:.1f}%)")
    print(f"   로그선형 모델 사용: {n_loglinear} 샘플 ({n_loglinear/len(M)*100:.1f}%)")
    
    print(f"\n📊 성능 지표 (Performance Metrics):")
    print(f"   MSE  = {mse_threshold:.6e}")
    print(f"   RMSE = {rmse_threshold:.6e}")
    print(f"   MAE  = {mae_threshold:.6e}")
    
    # ============================================================
    # 7. SAMPLE PREDICTIONS
    # ============================================================
    sample_size = min(10, len(df))
    sample_indices = np.random.choice(len(df), sample_size, replace=False)
    
    print("\n" + "="*80)
    print("[샘플 예측 결과 (Sample predictions)]")
    print("="*80)
    
    sample_df = pd.DataFrame({
        '실제낙찰가': y_actual[sample_indices],
        '선형예측(aM+b)': pred_linear[sample_indices],
        '로그선형예측(alog(M)+b)': pred_loglinear[sample_indices],
        '임계값예측': pred_threshold[sample_indices],
        '선형오차': error_linear[sample_indices],
        '로그선형오차': error_loglinear[sample_indices],
        '임계값오차': error_threshold[sample_indices],
    })
    
    pd.options.display.float_format = '{:,.2f}'.format
    print(sample_df.to_string())
    
    # ============================================================
    # 8. MODEL COMPARISON
    # ============================================================
    print("\n" + "="*80)
    print("[모델 성능 비교 (Model Performance Comparison)]")
    print("="*80)
    
    summary_df = pd.DataFrame({
        'Model': ['Linear (aM+b)', 'Log-Linear (alog(M)+b)', 'Threshold-Routed'],
        'a': [f"{a_linear:.6e}", f"{a_loglinear:.6e}", "-"],
        'b': [f"{b_linear:.6e}", f"{b_loglinear:.6e}", "-"],
        'MAE': [f"{mae_linear:.6e}", f"{mae_loglinear:.6e}", f"{mae_threshold:.6e}"],
        'RMSE': [f"{rmse_linear:.6e}", f"{rmse_loglinear:.6e}", f"{rmse_threshold:.6e}"],
    })
    
    pd.options.display.float_format = None
    print(summary_df.to_string(index=False))
    
    # ============================================================
    # 9. THRESHOLD VERIFICATION
    # ============================================================
    print("\n" + "="*80)
    print("🔍 임계값 조건 검증 (Threshold Condition Verification)")
    print("="*80)
    
    if n_linear > 0:
        error_linear_below = np.abs(pred_linear[use_linear] - y_actual[use_linear])
        error_loglinear_below = np.abs(pred_loglinear[use_linear] - y_actual[use_linear])
        print(f"\n✅ 범주 1: 차이 < {THRESHOLD:.0e}")
        print(f"   선형 모델 MAE: {np.mean(error_linear_below):.6e}")
        print(f"   로그선형 모델 MAE: {np.mean(error_loglinear_below):.6e}")
    
    if n_loglinear > 0:
        error_linear_above = np.abs(pred_linear[use_loglinear] - y_actual[use_loglinear])
        error_loglinear_above = np.abs(pred_loglinear[use_loglinear] - y_actual[use_loglinear])
        better = np.sum(error_loglinear_above < error_linear_above)
        print(f"\n✅ 범주 2: 차이 >= {THRESHOLD:.0e}")
        print(f"   선형 모델 MAE: {np.mean(error_linear_above):.6e}")
        print(f"   로그선형 모델 MAE: {np.mean(error_loglinear_above):.6e}")
        print(f"   로그선형이 더 나음: {better}/{n_loglinear} ({better/n_loglinear*100:.1f}%)")
    
    # ============================================================
    # 10. SAVE PARAMETERS
    # ============================================================
    print("\n" + "="*80)
    print("💾 학습된 매개변수 저장 (Saving Learned Parameters)")
    print("="*80)
    
    params = {
        "model_type": "Frozen Transformer with Linear/Log-Linear Parameters",
        "transformer_model": str(model_path),
        "status": "Transformer parameters FROZEN",
        
        "linear_model": {
            "formula": "y = a*M + b",
            "a": float(a_linear),
            "b": float(b_linear),
            "MSE": float(mse_linear),
            "RMSE": float(rmse_linear),
            "MAE": float(mae_linear),
        },
        
        "loglinear_model": {
            "formula": "y = a*log(M) + b",
            "a": float(a_loglinear),
            "b": float(b_loglinear),
            "MSE": float(mse_loglinear),
            "RMSE": float(rmse_loglinear),
            "MAE": float(mae_loglinear),
        },
        
        "threshold_routing": {
            "threshold": float(THRESHOLD),
            "samples_use_linear": int(n_linear),
            "samples_use_loglinear": int(n_loglinear),
            "MSE": float(mse_threshold),
            "RMSE": float(rmse_threshold),
            "MAE": float(mae_threshold),
        }
    }
    
    output_json = BASE_DIR / "frozen_transformer_parameters.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    print(f"✅ 저장 완료: {output_json}")
    
    # Save predictions
    output_csv = BASE_DIR / "frozen_transformer_predictions.csv"
    pred_df = pd.DataFrame({
        'Actual': y_actual,
        'Pred_Linear': pred_linear,
        'Pred_LogLinear': pred_loglinear,
        'Pred_Threshold': pred_threshold,
        'Error_Linear': error_linear,
        'Error_LogLinear': error_loglinear,
        'Error_Threshold': error_threshold,
    })
    pred_df.to_csv(output_csv, index=False)
    print(f"✅ 저장 완료: {output_csv}")
    
    print("\n" + "="*80)
    print("✅ 학습 완료 (TRAINING COMPLETE)")
    print("="*80)
    print(f"\n🏆 최적 모델:")
    
    models = [
        ('Linear', mae_linear),
        ('Log-Linear', mae_loglinear),
        ('Threshold-Routed', mae_threshold),
    ]
    best_name, best_mae = min(models, key=lambda x: x[1])
    
    print(f"   {best_name} (MAE: {best_mae:.6e})")


if __name__ == "__main__":
    main()
