import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt

# 같은 폴더에 model_transformer.py가 있어야 함
from model_transformer import run_training_transformer

def main():
    # 데이터 경로 설정 (상대 경로 주의)
    csv_path = "../dataset/dataset_feature_selected.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {csv_path}")
        # 혹시 현재 폴더 기준일 수도 있으니 체크
        if os.path.exists("./dataset_feature_selected.csv"):
            csv_path = "./dataset_feature_selected.csv"
            print(f"📂 현재 폴더에서 파일 발견: {csv_path}")
        else:
            return

    print(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. 타겟 및 사용할 핵심 피처 4개 정의
    target_col = "낙찰가"
    
    # [수정됨] 전체 컬럼 자동 감지 대신 4개 컬럼 고정
    feature_cols = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]

    # 2. 필수 컬럼 확인 (타겟 + 피처가 모두 있는지)
    required_cols = feature_cols + [target_col]
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        raise KeyError(f"CSV에 필수 컬럼이 없습니다: {missing}")

    print(f"📊 사용될 입력 피처 ({len(feature_cols)}개): {feature_cols}")

    # 3. 학습 실행
    print("🚀 Transformer 학습 시작 (4 Features)...")
    res = run_training_transformer(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        target_log=True,         # 타겟 로그 변환 사용
        epochs=300,               
        patience=30,             # 인내심 증가 (학습 안정화 고려)
        batch_size=128,          # 배치 사이즈 조정
        lr=1e-4,                 # 학습률
        weight_decay=1e-4,
        d_model=64,              # 입력 차원이 작으므로 모델 사이즈 축소 (512 -> 64)
        nhead=4,
        num_layers=2,            # 레이어 수 축소 (3 -> 2)
        dim_feedforward=256,     # FFN 차원 축소 (2048 -> 256)
        dropout=0.1,
        # model_transformer.py 정의에 따라 아래 인자는 존재 여부 확인 필요
        # verbose=True 
    )

    print("\n✅ Best VAL Loss:", res.best_val)
    print("✅ TEST Loss:", res.test)

    # 4. 샘플 예측 테스트
    sample_size = min(5, len(df))
    sample = df.sample(sample_size, random_state=42).copy()

    # 데이터 전처리 (Scaler 사용)
    X = sample[feature_cols].to_numpy(dtype=np.float32)
    X_s = res.x_scaler.transform(X)

    # 텐서 변환 및 3D 변환
    # (Batch, Features) -> (Batch, Features, 1)
    x_t = torch.from_numpy(X_s)
    x_t = x_t.unsqueeze(-1)  
    
    device = next(res.model.parameters()).device
    res.model.eval()
    
    with torch.no_grad():
        pred_s = res.model(x_t.to(device)).cpu().numpy()

    # 역변환 (Log -> 원래 가격)
    use_log = getattr(res, 'target_log', True)
    
    pred_log = res.y_scaler.inverse_transform(pred_s)
    pred_amt = np.expm1(pred_log) if use_log else pred_log

    # 결과 비교 출력
    out = sample[[target_col]].copy()
    out["예측낙찰가"] = pred_amt
    out["오차(예측-실제)"] = out["예측낙찰가"] - out[target_col]
    
    # 오차율 계산
    out["오차율(%)"] = 0.0
    mask = out[target_col] != 0
    out.loc[mask, "오차율(%)"] = (out.loc[mask, "오차(예측-실제)"] / out.loc[mask, target_col] * 100).abs()

    pd.options.display.float_format = '{:,.2f}'.format
    print("\n[Sample predictions (Transformer, 4 Features)]")
    print(out.to_string())

if __name__ == "__main__":
    main()
