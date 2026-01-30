import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import time
warnings.filterwarnings('ignore')

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 데이터 로드
print("데이터 로딩 중...")
df = pd.read_csv("../dataset/dataset_feature_selected.csv")

# 2. 4개 핵심 피처만 선택
print("4개 핵심 피처만 선택 중...")
selected_features = ['예가범위', '낙찰하한율', '추정가격', '기초금액']
target_col = '사정율'

# 필요한 컬럼들이 존재하는지 확인
missing_cols = [col for col in selected_features if col not in df.columns]
if missing_cols:
    print(f"경고: 다음 컬럼들이 데이터에 없습니다: {missing_cols}")
    print(f"사용 가능한 컬럼: {df.columns.tolist()}")
    exit()

# 타겟과 선택된 피처만 추출
df = df[selected_features + [target_col]]
df = df.fillna(0)
print(f"선택된 피처: {selected_features}")
print(f"타겟 변수: {target_col}")
print(f"데이터 shape: {df.shape}")

# 3. 특성과 타겟 분리
feature_cols = selected_features
X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

print(f"\n특성 수: {X.shape[1]}")
print(f"샘플 수: {X.shape[0]}")

# 4. Train/Val/Test 분할
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 5. 정규화
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

# 6. PyTorch Dataset 정의
class BidDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 7. DataLoader 생성
batch_size = 1024
num_workers = min(10, os.cpu_count() or 1)  # CPU 코어 수에 맞게 자동 설정 (최대 10)
pin_memory = device.type == 'cuda'  # GPU 사용 시 pin_memory 활성화

train_dataset = BidDataset(X_train, y_train)
val_dataset = BidDataset(X_val, y_val)
test_dataset = BidDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

print(f"\nBatch size: {batch_size}")
print(f"Num workers: {num_workers}")
print(f"Pin memory: {pin_memory}")

# 8. Transformer 기반 Quantile Regression 모델 정의
class QuantileTransformerRegressor(nn.Module):
    def __init__(self, input_dim, num_quantiles=999, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super(QuantileTransformerRegressor, self).__init__()
        self.num_quantiles = num_quantiles
        
        # 입력 임베딩
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 출력 레이어 (여러 분위수 동시 예측)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, num_quantiles)
        )
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.input_embedding(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        x = x + self.pos_encoder  # Positional encoding
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        x = x.squeeze(1)  # (batch_size, d_model)
        
        # 출력
        out = self.fc_out(x)  # (batch_size, num_quantiles)
        return out

# 9. 분위수 정의 및 모델 초기화
quantiles = np.linspace(0.001, 0.999, 999).tolist()  # 0.001 ~ 0.999 (999개)
num_quantiles = len(quantiles)
input_dim = X_train.shape[1]
model = QuantileTransformerRegressor(
    input_dim=input_dim,
    num_quantiles=num_quantiles,
    d_model=128,  # 512에서 1/4로 축소
    nhead=8,
    num_layers=3,
    dim_feedforward=512,  # 2048에서 1/4로 축소
    dropout=0.1
).to(device)

print(f"\n모델 생성 완료")
print(f"총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# 10. Quantile Loss 정의
def quantile_loss(y_pred, y_true, quantiles):
    """
    Quantile Loss (Pinball Loss)
    y_pred: (batch_size, num_quantiles)
    y_true: (batch_size, 1)
    quantiles: list of quantile values
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = y_true.squeeze() - y_pred[:, i]
        loss = torch.max((q - 1) * error, q * error).mean()
        losses.append(loss)
    return torch.stack(losses).mean()

# Optimizer와 Scheduler
optimizer = optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# 11. 학습 함수
def train_epoch(model, loader, quantiles, optimizer, device):
    model.train()
    total_loss = 0
    data_time = 0
    compute_time = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for X_batch, y_batch in pbar:
        batch_start = time.time()
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        data_time += time.time() - batch_start
        
        compute_start = time.time()
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = quantile_loss(outputs, y_batch, quantiles)
        loss.backward()
        optimizer.step()
        compute_time += time.time() - compute_start
        
        total_loss += loss.item() * X_batch.size(0)
        
        # GPU 사용률 표시
        if device.type == 'cuda':
            gpu_mem = torch.cuda.memory_allocated(device) / 1024**3  # GB
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'GPU': f'{gpu_mem:.2f}GB'
            })
        else:
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    avg_loss = total_loss / len(loader.dataset)
    
    # 병목 분석
    total_time = data_time + compute_time
    if total_time > 0:
        data_ratio = (data_time / total_time) * 100
        compute_ratio = (compute_time / total_time) * 100
        
        if data_ratio > 30:
            bottleneck = f"⚠️ 데이터 로딩 병목 ({data_ratio:.1f}%)"
        elif compute_ratio > 90:
            bottleneck = f"✓ GPU 활용 우수 ({compute_ratio:.1f}%)"
        else:
            bottleneck = f"균형적 ({compute_ratio:.1f}% GPU)"
    else:
        bottleneck = ""
    
    return avg_loss, bottleneck

# 12. 검증 함수
def validate(model, loader, quantiles, device):
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc="Validation", leave=False)
    with torch.no_grad():
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = quantile_loss(outputs, y_batch, quantiles)
            total_loss += loss.item() * X_batch.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(loader.dataset)

# 13. 학습 루프
num_epochs = 300
best_val_loss = float('inf')
patience = 10
patience_counter = 0

train_losses = []
val_losses = []

print("\n학습 시작...")
os.makedirs('./results_tft_4feat', exist_ok=True)

pbar_epoch = tqdm(range(num_epochs), desc="Epoch", position=0)
for epoch in pbar_epoch:
    train_loss, bottleneck = train_epoch(model, train_loader, quantiles, optimizer, device)
    val_loss = validate(model, val_loader, quantiles, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    scheduler.step(val_loss)
    
    # GPU 메모리 상태
    if device.type == 'cuda':
        gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
        gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
        gpu_info = f' | GPU: {gpu_mem_allocated:.1f}/{gpu_mem_reserved:.1f}GB'
    else:
        gpu_info = ''
    
    pbar_epoch.set_postfix({
        'train_loss': f'{train_loss:.6f}',
        'val_loss': f'{val_loss:.6f}',
        'status': bottleneck
    })
    
    # Early stopping 및 최고 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, './results_tft_4feat/best_model.pt')
        pbar_epoch.write(f"  → 최고 모델 저장! (Val Loss: {val_loss:.6f}){gpu_info} | {bottleneck}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            pbar_epoch.write(f"\nEarly stopping at epoch {epoch+1}")
            break

pbar_epoch.close()

print("\n학습 완료!")

# 14. 최고 모델 로드
if os.path.exists('./results_tft_4feat/best_model.pt'):
    try:
        checkpoint = torch.load('./results_tft_4feat/best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"최고 모델 로드 완료 (Epoch {checkpoint['epoch']+1}, Val Loss: {checkpoint['val_loss']:.6f})")
    except RuntimeError as e:
        print(f"경고: 체크포인트 로드 실패")
        print(f"현재 학습된 모델을 사용합니다.")
else:
    print("저장된 체크포인트가 없습니다. 현재 모델을 사용합니다.")

# 15. Train/Val Loss 그래프
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, 'o-', label='Train Loss', linewidth=2, markersize=4)
plt.plot(range(1, len(val_losses)+1), val_losses, 's-', label='Validation Loss', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./results_tft_4feat/train_val_loss.png', dpi=300, bbox_inches='tight')
print("Loss 그래프 저장: ./results_tft_4feat/train_val_loss.png")
plt.show()

# 16. 테스트 평가
def predict(model, loader, device):
    model.eval()
    predictions = []  # (batch, num_quantiles)
    actuals = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch.numpy())
    
    predictions = np.concatenate(predictions, axis=0)  # (total, num_quantiles)
    actuals = np.concatenate(actuals, axis=0).flatten()
    return predictions, actuals

pred_quantiles, gt_values = predict(model, test_loader, device)  # (N, 5)

# 중간값(median, 0.5 분위수) 기준으로 메트릭 계산
median_idx = quantiles.index(0.5)
pred_values = pred_quantiles[:, median_idx]

mae = mean_absolute_error(gt_values, pred_values)
mse = mean_squared_error(gt_values, pred_values)
rmse = np.sqrt(mse)
r2 = r2_score(gt_values, pred_values)

print(f"\n===== 테스트 평가 지표 (중간값 기준) =====")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²:   {r2:.6f}")
print(f"\n분위수별 예측 (예시 첫 5개 샘플):")
print(f"{'GT':<10} {' | '.join([f'Q{q:.0%}' for q in quantiles])}")
for i in range(min(5, len(gt_values))):
    print(f"{gt_values[i]:<10.4f} | " + " | ".join([f"{pred_quantiles[i, j]:.4f}" for j in range(num_quantiles)]))

# 17. GT vs Prediction 그래프 (Quantile Regression)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Scatter plot with confidence interval
axes[0, 0].scatter(gt_values, pred_values, alpha=0.5, s=10, label='Median (Q0.5)')
sorted_idx = np.argsort(gt_values)
axes[0, 0].fill_between(np.sort(gt_values), 
                        pred_quantiles[sorted_idx, 0],  # Q0.05
                        pred_quantiles[sorted_idx, -1],  # Q0.95
                        alpha=0.2, color='blue', label='90% Confidence Interval')
axes[0, 0].plot([gt_values.min(), gt_values.max()], 
                [gt_values.min(), gt_values.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Ground Truth (사정율)', fontsize=12)
axes[0, 0].set_ylabel('Prediction (사정율)', fontsize=12)
axes[0, 0].set_title(f'GT vs Quantile Predictions\nR² = {r2:.4f}', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 샘플별 비교 (신뢰도 구간 포함)
n_samples = min(100, len(gt_values))
x_range = np.arange(n_samples)
axes[0, 1].plot(x_range, gt_values[:n_samples], 'o-', label='Ground Truth', alpha=0.7, markersize=4, linewidth=2)
axes[0, 1].plot(x_range, pred_quantiles[:n_samples, median_idx], 's-', label='Median Pred', alpha=0.7, markersize=4, linewidth=2)
axes[0, 1].fill_between(x_range, 
                        pred_quantiles[:n_samples, 0],  # Q0.05
                        pred_quantiles[:n_samples, -1],  # Q0.95
                        alpha=0.2, color='blue', label='90% CI')
axes[0, 1].set_xlabel('Sample Index', fontsize=12)
axes[0, 1].set_ylabel('사정율', fontsize=12)
axes[0, 1].set_title(f'Quantile Predictions (First {n_samples} Samples)', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Prediction Interval Width
interval_width = pred_quantiles[:, -1] - pred_quantiles[:, 0]  # Q0.95 - Q0.05
sc = axes[1, 0].scatter(pred_values, interval_width, alpha=0.5, s=10, c=gt_values, cmap='viridis')
axes[1, 0].set_xlabel('Predicted Median Value', fontsize=12)
axes[1, 0].set_ylabel('Prediction Interval Width (Q0.95 - Q0.05)', fontsize=12)
axes[1, 0].set_title('Uncertainty by Predicted Value', fontsize=14)
cbar = plt.colorbar(sc, ax=axes[1, 0])
cbar.set_label('Ground Truth', fontsize=11)
axes[1, 0].grid(True, alpha=0.3)

# 4. 분위수별 에러 분포
residuals = gt_values - pred_values
quantile_names = [f'Q{q:.0%}' for q in quantiles]
colors = plt.cm.RdYlGn_r(np.linspace(0, 1, num_quantiles))

for i, (q, name, color) in enumerate(zip(quantiles, quantile_names, colors)):
    pred_q = pred_quantiles[:, i]
    residual_q = gt_values - pred_q
    axes[1, 1].hist(residual_q, bins=30, alpha=0.4, label=name, color=color, edgecolor='black')

axes[1, 1].axvline(x=0, color='black', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Residuals (GT - Pred)', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Residual Distribution by Quantile', fontsize=14)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./results_tft_4feat/gt_vs_prediction.png', dpi=300, bbox_inches='tight')
print("예측 그래프 저장: ./results_tft_4feat/gt_vs_prediction.png")
plt.show()

# 18. 예측 결과 CSV 저장
result_dict = {'Ground_Truth': gt_values}
for i, q in enumerate(quantiles):
    result_dict[f'Q{q:.0%}'] = pred_quantiles[:, i]
result_dict['Median_Pred'] = pred_values
result_dict['Interval_Width'] = pred_quantiles[:, -1] - pred_quantiles[:, 0]
result_dict['Residual'] = gt_values - pred_values
result_dict['Absolute_Error'] = np.abs(gt_values - pred_values)

result_df = pd.DataFrame(result_dict)
result_df.to_csv('./results_tft_4feat/predictions.csv', index=False)
print("예측 결과 저장: ./results_tft_4feat/predictions.csv")

print("\n모든 작업 완료!")