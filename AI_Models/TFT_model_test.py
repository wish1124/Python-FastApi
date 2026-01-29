import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import entropy
import warnings
import random
warnings.filterwarnings('ignore')

# 랜덤 시드 고정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# GPU 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 데이터 로드 및 전처리
print("데이터 로딩 중...")
df = pd.read_csv("../dataset/dataset_feature_selected.csv")

# 컬럼 제거 (train과 동일)
print("컬럼 제거 중...")
# 20개 feature 사용 - 난이도계수, 입찰준비기간, 지역의무공동계약여부 제거
if '난이도계수' in df.columns:
    df = df.drop(columns=['난이도계수'])
if '입찰준비기간' in df.columns:
    df = df.drop(columns=['입찰준비기간'])
if '지역의무공동계약여부' in df.columns:
    df = df.drop(columns=['지역의무공동계약여부'])
print("난이도계수, 입찰준비기간, 지역의무공동계약여부 제거 완료 - 20개 feature 사용")

# 특성과 타겟 분리
df = df.fillna(0)
target_col = '사정율'
feature_cols = [col for col in df.columns if col != target_col]

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

print(f"데이터 shape: {df.shape}")
print(f"특성 수: {X.shape[1]}")

# 2. 정규화 (학습과 동일하게)
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler_X = StandardScaler()
scaler_X.fit(X_train)  # 훈련 세트로 fit
X_test = scaler_X.transform(X_test)

# 3. 모델 정의 (학습과 동일)
class QuantileTransformerRegressor(nn.Module):
    def __init__(self, input_dim, num_quantiles=5, d_model=128, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super(QuantileTransformerRegressor, self).__init__()
        self.num_quantiles = num_quantiles
        
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        x = self.input_embedding(x)
        x = x.unsqueeze(1)
        x = x + self.pos_encoder
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        out = self.fc_out(x)
        return out

# 4. 모델 로드
quantiles = np.linspace(0.001, 0.999, 999).tolist()  # 0.001 ~ 0.999 (999개)
num_quantiles = len(quantiles)
input_dim = X_train.shape[1]

model = QuantileTransformerRegressor(
    input_dim=input_dim,
    num_quantiles=num_quantiles,
    d_model=512,
    nhead=8,
    num_layers=3,
    dim_feedforward=2048,
    dropout=0.1
).to(device)

print("\n학습된 모델 로드 중...")
try:
    checkpoint = torch.load('./results_tft_20feat/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"✓ 모델 로드 완료 (Epoch {checkpoint['epoch']+1}, Val Loss: {checkpoint['val_loss']:.6f})")
except Exception as e:
    print(f"✗ 모델 로드 실패: {e}")
    exit()

# 5. 예측
model.eval()
X_test_tensor = torch.FloatTensor(X_test).to(device)

with torch.no_grad():
    pred_quantiles = model(X_test_tensor).cpu().numpy()

# 중간값(median, 0.5 분위수) 기준
median_idx = quantiles.index(0.5)
pred_values = pred_quantiles[:, median_idx]
gt_values = y_test.flatten()

# 6. 메트릭 계산
mae = mean_absolute_error(gt_values, pred_values)
mse = mean_squared_error(gt_values, pred_values)
rmse = np.sqrt(mse)
r2 = r2_score(gt_values, pred_values)

print(f"\n===== 테스트 결과 (Quantile Regression) =====")
print(f"MAE:  {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²:   {r2:.6f}")

# 7. 분포 예측 평가 함수
def evaluate_distribution(gt_values, pred_quantiles, quantiles):
    n_samples = len(gt_values)
    
    # 1. CRPS (분포 정확도)
    crps_sum = 0
    for i, q in enumerate(quantiles):
        error = gt_values - pred_quantiles[:, i]
        loss = np.maximum((q-1)*error, q*error)
        crps_sum += loss
    crps = np.mean(crps_sum) / len(quantiles)
    
    # 2. Coverage (신뢰도 검증 - 50%, 90% 구간)
    # 50% 구간: Q25 ~ Q75
    idx_25 = int(len(quantiles) * 0.25)
    idx_75 = int(len(quantiles) * 0.75)
    in_50_interval = ((gt_values >= pred_quantiles[:, idx_25]) & 
                      (gt_values <= pred_quantiles[:, idx_75]))
    coverage_50 = np.mean(in_50_interval)
    
    # 90% 구간: Q05 ~ Q95
    idx_05 = int(len(quantiles) * 0.05)
    idx_95 = int(len(quantiles) * 0.95)
    in_90_interval = ((gt_values >= pred_quantiles[:, idx_05]) & 
                      (gt_values <= pred_quantiles[:, idx_95]))
    coverage_90 = np.mean(in_90_interval)

    print(f"\n===== 분포 예측 평가 ({len(quantiles)} Quantiles) =====")
    print(f"CRPS Score (낮을수록 좋음): {crps:.4f}")
    print(f"50% 구간 실제 커버리지: {coverage_50*100:.1f}% (목표: 50%)")
    print(f"90% 구간 실제 커버리지: {coverage_90*100:.1f}% (목표: 90%)")
    
    if abs(coverage_90 - 0.9) > 0.1:
        print(">> 경고: 모델의 확률 계산이 부정확합니다 (Calibration 필요)")
    else:
        print(">> 좋음: 모델이 내놓은 확률을 신뢰할 수 있습니다.")
    
    return crps, coverage_50, coverage_90

def plot_gt_vs_pred_distribution(gt_values, pred_quantiles, pred_values):
    """
    실제 GT 분포와 예측 분포 비교
    
    Args:
        gt_values: 실제값 (shape: [n_samples])
        pred_quantiles: 예측 분위수 (shape: [n_samples, n_quantiles])
        pred_values: 중앙값 예측 (shape: [n_samples])
    """
    from scipy import stats
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. 확률 밀도 함수(PDF) 비교 - KDE 사용
    x_range = np.linspace(min(gt_values.min(), pred_values.min()), 
                          max(gt_values.max(), pred_values.max()), 200)
    
    # GT의 KDE
    gt_kde = stats.gaussian_kde(gt_values)
    gt_pdf = gt_kde(x_range)
    
    # Pred의 KDE
    pred_kde = stats.gaussian_kde(pred_values)
    pred_pdf = pred_kde(x_range)
    
    axes[0, 0].plot(x_range, gt_pdf, linewidth=2.5, label='GT Distribution (KDE)', color='blue')
    axes[0, 0].plot(x_range, pred_pdf, linewidth=2.5, label='Pred Distribution (KDE)', color='red')
    axes[0, 0].fill_between(x_range, gt_pdf, alpha=0.3, color='blue')
    axes[0, 0].fill_between(x_range, pred_pdf, alpha=0.3, color='red')
    axes[0, 0].set_xlabel('사정율', fontsize=12)
    axes[0, 0].set_ylabel('Probability Density', fontsize=12)
    axes[0, 0].set_title('Probability Density Function (PDF) - KDE', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 히스토그램으로도 표현
    axes[0, 1].hist(gt_values, bins=50, alpha=0.6, label='Ground Truth', density=True, 
                    color='blue', edgecolor='black')
    axes[0, 1].hist(pred_values, bins=50, alpha=0.6, label='Predicted Median', density=True, 
                    color='red', edgecolor='black')
    axes[0, 1].plot(x_range, gt_pdf, linewidth=2.5, color='darkblue', label='GT KDE')
    axes[0, 1].plot(x_range, pred_pdf, linewidth=2.5, color='darkred', label='Pred KDE')
    axes[0, 1].set_xlabel('사정율', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Histogram + PDF Overlay', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 누적분포함수(CDF) 비교
    gt_sorted = np.sort(gt_values)
    gt_cdf = np.arange(1, len(gt_sorted) + 1) / len(gt_sorted)
    
    pred_sorted = np.sort(pred_values)
    pred_cdf = np.arange(1, len(pred_sorted) + 1) / len(pred_sorted)
    
    axes[1, 0].plot(gt_sorted, gt_cdf, label='GT CDF', linewidth=2.5, color='blue')
    axes[1, 0].plot(pred_sorted, pred_cdf, label='Pred CDF', linewidth=2.5, color='red')
    axes[1, 0].set_xlabel('사정율', fontsize=12)
    axes[1, 0].set_ylabel('Cumulative Probability', fontsize=12)
    axes[1, 0].set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 통계 비교
    axes[1, 1].axis('off')
    
    stats_text = f"""
    ===== GT 통계 =====
    Mean: {np.mean(gt_values):.6f}
    Std Dev: {np.std(gt_values):.6f}
    Min: {np.min(gt_values):.6f}
    Max: {np.max(gt_values):.6f}
    Median: {np.median(gt_values):.6f}
    Skewness: {stats.skew(gt_values):.4f}
    Kurtosis: {stats.kurtosis(gt_values):.4f}
    
    ===== Predicted 통계 =====
    Mean: {np.mean(pred_values):.6f}
    Std Dev: {np.std(pred_values):.6f}
    Min: {np.min(pred_values):.6f}
    Max: {np.max(pred_values):.6f}
    Median: {np.median(pred_values):.6f}
    Skewness: {stats.skew(pred_values):.4f}
    Kurtosis: {stats.kurtosis(pred_values):.4f}
    
    ===== Wasserstein Distance =====
    Distance: {stats.wasserstein_distance(gt_values, pred_values):.6f}
    (두 분포 간의 차이 - 작을수록 유사)
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('./results_tft_20feat/gt_vs_pred_distribution.png', dpi=100, bbox_inches='tight')
    print("\n>> GT vs Pred 확률 분포 비교 그래프 저장: ./results_tft_20feat/gt_vs_pred_distribution.png")
    plt.show()

def plot_distribution_check(gt_values, pred_quantiles):
    """
    PIT Histogram과 Reliability Diagram으로 분포 예측 검증
    
    Args:
        gt_values: 실제값 (shape: [n_samples])
        pred_quantiles: 예측 분위수 (shape: [n_samples, n_quantiles])
    """
    n_samples = len(gt_values)
    
    # 1. PIT 값 계산 (정답이 분포의 어디쯤에 있는지)
    # 각 샘플별로 내 정답보다 작은 예측값의 비율을 구함 (Empirical CDF)
    pit_values = []
    for i in range(n_samples):
        # 정답보다 작은 예측값 개수 / 전체 예측값 개수
        cdf_prob = np.mean(pred_quantiles[i] < gt_values[i])
        pit_values.append(cdf_prob)
    
    pit_values = np.array(pit_values)

    # 2. 그래프 그리기
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # (1) PIT Histogram (분포 모양 검증)
    axes[0].hist(pit_values, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axhline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal (Uniform)')
    axes[0].set_title("PIT Histogram\n(Flat is Best)", fontsize=14)
    axes[0].set_xlabel("PIT Value (CDF of GT)", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].legend()
    
    # (2) Reliability Diagram (신뢰도 검증)
    # 이론적 분위수(X) vs 실제 포함 비율(Y)
    n_quantiles = pred_quantiles.shape[1]
    theoretical_quantiles = np.linspace(0.01, 0.99, n_quantiles)
    observed_frequencies = []
    
    for q_idx in range(n_quantiles):
        # 해당 분위수(q) 값보다 실제 정답이 작은 비율
        # (모델이 q% 확률로 작다고 했으니, 실제로도 q% 비율이어야 함)
        q_pred_values = pred_quantiles[:, q_idx]
        freq = np.mean(gt_values < q_pred_values)
        observed_frequencies.append(freq)
    
    observed_frequencies = np.array(observed_frequencies)
    
    axes[1].plot(theoretical_quantiles, observed_frequencies, 'o-', markersize=4, label='Model')
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfectly Calibrated')
    axes[1].set_title("Reliability Diagram\n(Line on Diagonal is Best)", fontsize=14)
    axes[1].set_xlabel("Predicted Probability (Quantile)", fontsize=12)
    axes[1].set_ylabel("Observed Frequency", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('./results_tft_20feat/distribution_check.png', dpi=100, bbox_inches='tight')
    print("\n>> Distribution Check 그래프 저장: ./results_tft_20feat/distribution_check.png")
    plt.show()

def compare_entropy(gt_values, pred_quantiles):
    """
    GT의 정보량(Entropy)과 예측 분포의 평균 정보량을 비교
    
    Args:
        gt_values: 실제값 (shape: [n_samples])
        pred_quantiles: 예측 분위수 (shape: [n_samples, n_quantiles])
    """
    # 1. GT의 엔트로피 (Histogram 기반 추정)
    # 실제 값들이 얼마나 퍼져있는지 확인
    hist_gt, _ = np.histogram(gt_values, bins=50, density=True)
    # 0인 빈도는 로그 계산 오류 방지를 위해 제거
    hist_gt = hist_gt[hist_gt > 0]
    entropy_gt = entropy(hist_gt)
    
    # 2. 예측 분포의 엔트로피 (각 샘플별 평균)
    pred_entropies = []
    for i in range(len(pred_quantiles)):
        # 개별 예측 분포(999개 점)의 히스토그램 생성
        hist_pred, _ = np.histogram(pred_quantiles[i], bins=50, density=True)
        hist_pred = hist_pred[hist_pred > 0]
        pred_entropies.append(entropy(hist_pred))
    
    avg_pred_entropy = np.mean(pred_entropies)
    
    print(f"\n===== 정보량(Entropy) 비교 분석 =====")
    print(f"1. 실제 시장의 혼란도 (GT Entropy): {entropy_gt:.4f}")
    print(f"2. 모델의 평균 불확실성 (Avg Pred Entropy): {avg_pred_entropy:.4f}")
    
    diff = avg_pred_entropy - entropy_gt
    print(f"\n[해석]")
    if diff > 0.5:
        print(f"⚠️ 모델이 너무 겁을 먹었습니다. (구간을 너무 넓게 잡음, +{diff:.2f})")
    elif diff < -0.5:
        print(f"⚠️ 모델이 근거 없는 자신감을 보입니다. (위험함, {diff:.2f})")
    else:
        print(f"✅ 모델이 시장의 난이도를 정확하게 파악하고 있습니다.")


# 분포 예측 평가 실행
crps, coverage_50, coverage_90 = evaluate_distribution(gt_values, pred_quantiles, quantiles)

# 예측 결과 CSV 저장
result_dict = {'Ground_Truth': gt_values}
for i, q in enumerate(quantiles):
    result_dict[f'Q{q:.0%}'] = pred_quantiles[:, i]
result_dict['Median_Pred'] = pred_values
result_dict['Residual'] = gt_values - pred_values
result_dict['Absolute_Error'] = np.abs(gt_values - pred_values)

result_df = pd.DataFrame(result_dict)
result_df.to_csv('./results_tft_20feat/test_predictions.csv', index=False)
print("예측 결과 저장: ./results_tft_20feat/test_predictions.csv")

# 9. GT vs Pred 분포 비교
print("\nGT vs Pred 분포 비교 중...")
plot_gt_vs_pred_distribution(gt_values, pred_quantiles, pred_values)

# 10. 분포 검증 (PIT Histogram & Reliability Diagram)
print("\n분포 검증 중...")
plot_distribution_check(gt_values, pred_quantiles)

# 11. 엔트로피 비교
print("\n엔트로피 분석 중...")
compare_entropy(gt_values, pred_quantiles)

print("\n테스트 완료! ✓")
