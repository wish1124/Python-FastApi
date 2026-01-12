import numpy as np      # numpy: 수치 연산, 배열 처리
import pandas as pd     # pandas: CSV 로딩 및 테이블 데이터 처리

import torch
import torch.nn as nn   # torch.nn: 신경망 레이어, 손실 함수
from torch.utils.data import Dataset, DataLoader        # Dataset: 커스텀 데이터셋 정의용 / DataLoader: 배치 단위로 데이터를 모델에 공급

from sklearn.model_selection import train_test_split                # train_test_split : train/val/test 분리
from sklearn.preprocessing import StandardScaler     # OneHotEncoder : 범주형 → 숫자 벡터 / StandardScaler : 정규화

from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError
# PyTorch에서 MAE, MSE, MAPE를 정확하게 계산하기 위한 공식 metric 모듈

# 1) 초기 설정
CSV_PATH = "C:/Users/User/Downloads/Faker.csv"
TARGET_COL = "sucsfbidAmt"   # 예측 대상 컬럼 : ""

BATCH_SIZE = 256        # 한 번에 학습할 데이터 개수
EPOCHS = 50             # 전체 데이터 반복 횟수
LR = 1e-3               # 학습률
WEIGHT_DECAY = 1e-4     # L2 정규화 (과적합 방지)
SEED = 42               # 재현성 확보용 난수 시드

TEST_SIZE = 0.3         
VAL_SIZE = 0.3
# 데이터 분할 비율 (7:3)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# GPU 사용 가능하면 cuda, 아니면 cpu 사용

class TabularDataset(Dataset):      # numpy 배열을 PyTorch 학습용 데이터셋으로 변환
    def __init__(self, X: np.ndarray, y: np.ndarray):               # 입력 특성 X, 타겟 y를 받음
        self.X = torch.tensor(X, dtype=torch.float32)               # numpy → torch tensor 변환
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)   # view(-1,1) : 회귀 출력 shape 맞추기 (N,1)

    def __len__(self): return len(self.X)                           # 데이터 전체 길이 반환
    def __getitem__(self, idx): return self.X[idx], self.y[idx]     # idx번째 샘플 반환 (DataLoader가 호출)

class MLPRegressor(nn.Module):              # PyTorch 신경망 기본 클래스 상속
    def __init__(self, in_dim: int):        # in_dim: 입력 feature 개수
        super().__init__()                  # super() : nn.Module 초기화
        self.net = nn.Sequential(           
            nn.Linear(in_dim, 256),         # 입력 → 256 차원
            nn.ReLU(),                      # ReLU 활성화
            nn.Dropout(0.1),                # Dropout으로 과적합 방지
            nn.Linear(256, 128),            # 256 → 128 은닉층
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),              # 최종 출력: 투찰가 1개 값
        )

    def forward(self, x): return self.net(x)        # 입력 x가 들어오면 정의한 .net()를 통과

def set_seed(seed: int):    # 실험 재현성을 위해 난수 고정
    import random
    random.seed(seed)                    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python / NumPy / PyTorch / GPU 전부 고정

def train_one_epoch(model, loader, optimizer, DEVICE):      # 모델을 1 epoch 학습
    model.train()
    crit = nn.MSELoss()                             # 손실 함수: MSE
    total, n = 0.0, 0                               # 
    for X, y in loader:     # DataLoader에서 배치 단위로 데이터 로드
        X, y = X.to(DEVICE), y.to(DEVICE)           # CPU/GPU 이동
        optimizer.zero_grad(set_to_none=True)       # gradient 초기화
        pred = model(X)                             # 예측
        loss = crit(pred, y)                        # 손실 계산
        loss.backward()                             # 역전파
        optimizer.step()                            # 파라미터 업데이트
        bs = X.size(0)                              
        total += loss.item() * bs                   # 전체 데이터 기준 평균 loss 계산용 누적
        n += bs
    return total / max(n, 1)                        # epoch 평균 MSE 반환

@torch.no_grad()
def evaluate(model, loader, DEVICE):        # gradient 계산 비활성화 (속도 + 메모리)
    model.eval()                            # 평가 모드 (Dropout 비활성)
    mae_m = MeanAbsoluteError().to(DEVICE)  
    mse_m = MeanSquaredError().to(DEVICE)
    mape_m = MeanAbsolutePercentageError().to(DEVICE)
    # TorchMetrics로 MAE / MSE 계산   
    crit = nn.MSELoss()
    total, n = 0.0, 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        pred = model(X)
        loss = crit(pred, y)
        bs = X.size(0)
        total += loss.item() * bs
        n += bs
        mae_m.update(pred, y)
        mse_m.update(pred, y)
        mape_m.update(pred, y)
        # 배치 단위 누적

    mse = float(mse_m.compute().item())
    rmse = float(np.sqrt(mse))              # RMSE는 직접 sqrt로 계산
    mae = float(mae_m.compute().item())
    mape = float(mape_m.compute().item())
    avg_loss = total / max(n, 1)
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "loss_mse": avg_loss}
    # 딕셔너리 형태로 결과 반환 

def main():
    set_seed(SEED)

    # 2) CSV 로드
    df = pd.read_csv(CSV_PATH)

    # feature / target 분리
    X_df = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(float).values

    # feature = target 제외 전부(수치형만 존재하므로 그대로 사용)
    X = X_df.astype(float).values

    # 3) train/val/test split (전처리 fit은 train에서만)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=SEED
    )

    # 4) 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 5) DataLoader / 모델 학습
    train_loader = DataLoader(TabularDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TabularDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TabularDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = MLPRegressor(in_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):      # epoch 반복
        train_mse = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_metrics = evaluate(model, val_loader, DEVICE)
        # 학습 → 검증

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[{epoch:03d}/{EPOCHS}] "
            f"train_mse={train_mse} | val_mae={val_metrics['mae']} val_rmse={val_metrics['rmse']} val_mape={val_metrics['mape']*100:.2f}%"
        ) # 학습 상태 로그 출력
        
    if best_state is not None:
        model.load_state_dict(best_state)

    # 6) 테스트 평가(MAE/RMSE), 한 번만 테스트 데이터 평가
    test_metrics = evaluate(model, test_loader, DEVICE)
    print(
        f"[TEST] mae={test_metrics['mae']} rmse={test_metrics['rmse']} mse={test_metrics['mse']} mape={test_metrics['mape']*100:.2f}%"
    )

if __name__ == "__main__":  # 이 파일을 직접 실행할 때만 main 실행
    main()
