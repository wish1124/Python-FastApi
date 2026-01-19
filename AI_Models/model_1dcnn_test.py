import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1D CNN 학습 함수 임포트
# (model_1dcnn.py에 run_training_cnn1d가 있어야 함)
from model_1dcnn import run_training_cnn1d

def main():
    csv_path = "../dataset/dataset_feature_selected.csv"
    
    if not os.path.exists(csv_path):
        # 파일이 없으면 현재 경로 기준으로 다시 확인하거나 경로 수정 필요
        # 사용자가 제공한 경로에 맞춤
        csv_path = "/home/jeonboyun/바탕화면/BidAssitance/dataset/dataset_feature_selected.csv"
        if not os.path.exists(csv_path):
            print(f"❌ 데이터 파일을 찾을 수 없습니다: {csv_path}")
            return

    df = pd.read_csv(csv_path)

    # 1. 모든 수치형 피처 자동 선택 (낙찰가 제외)
    target_col = "낙찰가"
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

    if target_col not in df.columns:
        raise KeyError(f"CSV에 타겟 컬럼 '{target_col}'이 없습니다.")
    if not feature_cols:
        raise ValueError("학습할 수치형 피처가 하나도 없습니다.")

    # 결과 저장용 폴더 경로
    output_dir = "./results_cnn_allfeat"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== [1D CNN] 학습 시작 (All Features) ===")
    print(f"전체 데이터 수: {len(df):,}개")
    print(f"타겟 컬럼: {target_col}")
    print(f"사용된 피처({len(feature_cols)}개): {feature_cols}")
    print(f"저장 경로: {output_dir}")

    # 2. 학습 실행
    res = run_training_cnn1d(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        target_log=True,
        epochs=300,
        patience=30,
        batch_size=256,
        lr=1e-2,
        hidden=64,
        dropout=0.1,
        output_dir=output_dir
    )

    print("\n" + "="*30)
    print(" [최종 학습 결과] ")
    print("="*30)
    print(f"Best Val MSE : {res.best_val['MSE']:.4f}")
    print(f"Test MSE     : {res.test['MSE']:.4f}")
    print(f"Test RMSE    : {res.test['RMSE']:.4f}")
    print(f"Test MAE     : {res.test['MAE']:.4f}")
    print(f"Test R2      : {res.test.get('R2', 0.0):.4f}")

    # =========================================================
    # [추가됨] 2-1. 학습된 모델 저장 (.pt)
    # =========================================================
    model_save_path = os.path.join(output_dir, "best_model.pt")
    checkpoint = {
        "model_state_dict": res.model.state_dict(),
        "x_scaler": res.x_scaler,
        "y_scaler": res.y_scaler,
        "feature_cols": feature_cols,
        "target_log": True
    }
    torch.save(checkpoint, model_save_path)
    print(f"💾 모델 저장 완료: {model_save_path}")

    # =========================================================
    # [추가됨] 2-2. Loss 그래프 저장
    # =========================================================
    # model_1dcnn.py 수정 후 res.history가 반환된다고 가정
    if hasattr(res, 'history') and res.history:
        plt.figure(figsize=(10, 6))
        plt.plot(res.history['train_loss'], label='Train Loss')
        plt.plot(res.history['val_loss'], label='Val Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (MSE)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        loss_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(loss_path)
        plt.close()
        print(f"📈 Loss 그래프 저장 완료: {loss_path}")

    # =========================================================
    # [추가됨] 2-3. Scatter Plot (전체 데이터 시각화 저장)
    # =========================================================
    print("📊 스캐터 차트 생성 중...")
    # 시각화용 샘플링 (1만개)
    plot_sample = df.sample(min(len(df), 10000), random_state=42).copy()
    
    X_plot = plot_sample[feature_cols].to_numpy(np.float32)
    X_plot_s = res.x_scaler.transform(X_plot)
    # (N, 1, F) 형태
    x_tensor = torch.from_numpy(X_plot_s).reshape(len(plot_sample), 1, -1)
    
    device = next(res.model.parameters()).device
    res.model.eval()
    
    with torch.no_grad():
        pred_raw = res.model(x_tensor.to(device)).cpu().numpy()
        
    pred_val = res.y_scaler.inverse_transform(pred_raw)
    pred_val = np.expm1(pred_val) # target_log=True 가정
    actual_val = plot_sample[target_col].values

    plt.figure(figsize=(8, 8))
    plt.scatter(actual_val, pred_val, alpha=0.4, s=5, color='blue')
    
    # 기준선
    mn = min(actual_val.min(), pred_val.min())
    mx = max(actual_val.max(), pred_val.max())
    plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect Fit')
    
    plt.title(f'Actual vs Predicted (n={len(plot_sample):,})')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    
    # 축 포맷
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:,.0f}'))
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    scatter_path = os.path.join(output_dir, "scatter_plot.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"📊 스캐터 차트 저장 완료: {scatter_path}")


    # 3. 샘플 예측 (Inference) - 기존 코드 유지
    sample = df.sample(5, random_state=42).copy()

    # 데이터 전처리
    X = sample[feature_cols].to_numpy(np.float32)
    X_s = res.x_scaler.transform(X)

    # (Batch, 1, Features) 형태로 변환
    x_t = torch.from_numpy(X_s).reshape(len(sample), 1, -1)

    device = next(res.model.parameters()).device
    res.model.eval()

    with torch.no_grad():
        pred_s = res.model(x_t.to(device)).cpu().numpy()

    # 로그 역변환
    pred_log = res.y_scaler.inverse_transform(pred_s)
    pred_amt = np.expm1(pred_log)

    # 결과 출력
    out = sample[[target_col]].copy()
    out["예측낙찰가"] = pred_amt
    out["오차"] = out["예측낙찰가"] - out[target_col]
    out["오차율(%)"] = (out["오차"] / out[target_col]) * 100

    print("\n[Sample Predictions]")
    pd.options.display.float_format = '{:,.2f}'.format
    print(out.to_string())

if __name__ == "__main__":
    main()
