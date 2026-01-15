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

    # 1. 타겟 컬럼 정의
    target_col = "낙찰가"

    # 2. 필수 컬럼 확인 (타겟이 있는지)
    if target_col not in df.columns:
        raise KeyError(f"CSV에 타겟 컬럼 '{target_col}'이(가) 없습니다.")

    # 3. Feature 컬럼 자동 정의 (전체 컬럼에서 타겟만 제외)
    feature_cols = [c for c in df.columns if c != target_col]
    print(f"📊 감지된 입력 피처 ({len(feature_cols)}개): {feature_cols}")

    # 4. 학습 실행
    # (주의: run_training_transformer 함수의 정의에 없는 인자는 넣으면 에러 남)
    print("🚀 학습 시작...")
    res = run_training_transformer(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        target_log=True,         # 타겟 로그 변환 사용
        epochs=50,               # 테스트용으로 Epoch 줄임 (필요시 200으로 변경)
        patience=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=1e-4,
        d_model=512,             
        nhead=4,
        num_layers=2,
        dim_feedforward=2048,     
        dropout=0.1,
        # 아래 옵션들은 함수 정의에 따라 에러가 날 수 있어 제거하거나 확인 필요
        # feature_noise_std=0.001, 
        verbose=True
    )

    print("\n✅ Best VAL Loss:", res.best_val)
    print("✅ TEST Loss:", res.test)

    # 5. 샘플 예측 테스트
    # 데이터가 적을 경우를 대비해 min 처리
    sample_size = min(5, len(df))
    sample = df.sample(sample_size, random_state=42).copy()

    # 데이터 전처리 (Scaler 사용)
    X = sample[feature_cols].to_numpy(dtype=np.float32)
    X_s = res.x_scaler.transform(X)

    # 텐서 변환
    x_t = torch.from_numpy(X_s)
    
    device = next(res.model.parameters()).device
    res.model.eval()
    
    with torch.no_grad():
        try:
            # 기본 시도: (Batch, Feature) 2D
            pred_s = res.model(x_t.to(device)).cpu().numpy()
        except RuntimeError:
            # 차원 에러 시: (Batch, Feature, 1) 3D 시도 (Sequence Length=Feature Dim인 경우)
            # 혹은 (Batch, 1, Feature)일 수도 있음
            # 에러 메시지: "expected 3D input" 등이 뜨면 unsqueeze 필요
            x_t_reshaped = x_t.unsqueeze(-1) # (Batch, Feature, 1)
            pred_s = res.model(x_t_reshaped.to(device)).cpu().numpy()

    # 역변환 (Log -> 원래 가격)
    # res 객체에 target_log 속성이 없으면 True라고 가정 (위에서 넣었으므로)
    use_log = getattr(res, 'target_log', True)
    
    pred_log = res.y_scaler.inverse_transform(pred_s)
    pred_amt = np.expm1(pred_log) if use_log else pred_log

    # 결과 비교 출력
    out = sample[[target_col]].copy()
    out["예측낙찰가"] = pred_amt
    out["오차(예측-실제)"] = out["예측낙찰가"] - out[target_col]
    
    # 0으로 나누기 방지
    out["오차율(%)"] = 0.0
    mask = out[target_col] != 0
    out.loc[mask, "오차율(%)"] = (out.loc[mask, "오차(예측-실제)"] / out.loc[mask, target_col] * 100).abs()

    pd.options.display.float_format = '{:,.0f}'.format
    print("\n[Sample predictions (Unit: KRW)]")
    print(out.to_string())

if __name__ == "__main__":
    main()
