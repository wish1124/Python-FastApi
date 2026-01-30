import torch
import torch.nn as nn
import numpy as np
import os
import json  # ← 추가!
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QuantileTransformerRegressor(nn.Module):
    """Quantile Regression을 위한 Transformer 기반 모델"""

    def __init__(self, input_dim, num_quantiles=999, d_model=128, nhead=8,
                 num_layers=3, dim_feedforward=512, dropout=0.1):
        super(QuantileTransformerRegressor, self).__init__()
        self.num_quantiles = num_quantiles

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
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
        x = x.unsqueeze(1) + self.pos_encoder
        x = self.transformer_encoder(x)
        return self.fc_out(x.squeeze(1))


class ProbabilityPredictor:
    """TFT 4-Feature 모델을 사용한 확률 예측 클래스"""

    def __init__(self, model_path='./results_tft_4feat/best_model.pt'):
        self.model_path = model_path
        self.device = device
        self.quantiles = np.linspace(0.001, 0.999, 999)
        self.feature_names = ['기초금액', '추정가격', '예가범위', '낙찰하한율']  # ← 순서 변경!
        self.model = self._load_model()
        self.scaler = self._load_scaler()  # ← 스케일러 로드 추가!

    def _load_scaler(self):
        """scalers.json 파일 로드"""
        scaler_path = self.model_path.replace('best_model.pt', 'scalers.json')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler = json.load(f)
                print(f"✓ 스케일러 로드 완료: {scaler_path}")
                return scaler
        print(f"⚠️  스케일러 파일 없음: {scaler_path}")
        return None

    def _load_model(self):
        """학습된 모델 로드"""
        model = QuantileTransformerRegressor(
            input_dim=4, num_quantiles=999, d_model=128, nhead=8,
            num_layers=3, dim_feedforward=512, dropout=0.1
        ).to(self.device)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # 다양한 checkpoint 형식 처리
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            else:
                # checkpoint 자체가 state_dict인 경우
                model.load_state_dict(checkpoint, strict=False)
        else:
            # checkpoint가 직접 state_dict인 경우 (오래된 PyTorch)
            model.load_state_dict(checkpoint, strict=False)

        model.eval()

        print(f"✓ 모델 로드 완료: {self.model_path}")
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_loss', 0):.6f}")

        return model

    def _prepare_input(self, input_features):
        """입력 피처를 numpy array로 변환 및 스케일링"""
        if isinstance(input_features, dict):
            # ✅ 순서: 기초금액, 추정가격, 예가범위(%), 낙찰하한율(%)
            X = np.array([[
                input_features['기초금액'],
                input_features['추정가격'],
                input_features['예가범위'],  # 0-100 스케일 (3.5, not 0.035)
                input_features['낙찰하한율']  # 0-100 스케일 (87.74, not 0.8774)
            ]], dtype=np.float32)
        else:
            X = np.array([input_features], dtype=np.float32)
            if X.shape[1] != 4:
                raise ValueError(f"입력 피처는 4개여야 합니다. 현재: {X.shape[1]}개")

        # ✅ 스케일링 적용
        if self.scaler is not None:
            x_mean = np.array(self.scaler['x_mean'], dtype=np.float32)
            x_std = np.array(self.scaler['x_std'], dtype=np.float32)
            X = (X - x_mean) / x_std

        return X

    def _predict_quantiles(self, X):
        """999개 quantile 예측 및 역변환"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            pred = self.model(X_tensor).cpu().numpy()[0]  # 표준화된 로그 값

            # ✅ 역변환 (역표준화 + 역로그)
            if self.scaler is not None:
                y_mean = self.scaler['y_mean']
                y_std = self.scaler['y_std']
                target_log = self.scaler.get('target_log', False)

                # 역표준화: (pred * std) + mean
                pred = pred * y_std + y_mean

                # 역로그 변환: exp(pred)
                if target_log:
                    pred = np.exp(pred)

            return pred

    def _get_input_features_dict(self, X):
        """입력 피처를 dict 형태로 반환"""
        return {
            '기초금액': float(X[0, 0]) if self.scaler is None else float(
                X[0, 0] * self.scaler['x_std'][0] + self.scaler['x_mean'][0]),
            '추정가격': float(X[0, 1]) if self.scaler is None else float(
                X[0, 1] * self.scaler['x_std'][1] + self.scaler['x_mean'][1]),
            '예가범위': float(X[0, 2]) if self.scaler is None else float(
                X[0, 2] * self.scaler['x_std'][2] + self.scaler['x_mean'][2]),
            '낙찰하한율': float(X[0, 3]) if self.scaler is None else float(
                X[0, 3] * self.scaler['x_std'][3] + self.scaler['x_mean'][3])
        }

    def predict_probability(self, input_features, lower_bound, upper_bound):
        """특정 구간의 확률 예측"""
        X = self._prepare_input(input_features)
        pred_quantiles = self._predict_quantiles(X)

        # 구간 내 확률 계산
        lower_idx = np.searchsorted(pred_quantiles, lower_bound, side='left')
        upper_idx = np.searchsorted(pred_quantiles, upper_bound, side='right')
        probability = (upper_idx - lower_idx) / len(pred_quantiles)

        return {
            'probability': float(probability),
            'probability_percent': float(probability * 100),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'lower_quantile_index': int(lower_idx),
            'upper_quantile_index': int(upper_idx),
            'median_prediction': float(pred_quantiles[499]),
            'mean_prediction': float(np.mean(pred_quantiles)),
            'input_features': self._get_input_features_dict(X)
        }

    def get_prediction_intervals(self, input_features, confidence_levels=[0.5, 0.8, 0.9, 0.95]):
        """여러 신뢰구간 예측"""
        X = self._prepare_input(input_features)
        pred_quantiles = self._predict_quantiles(X)

        intervals = {}
        for conf in confidence_levels:
            lower_idx = int((1 - conf) / 2 * 999)
            upper_idx = int((1 + conf) / 2 * 999)

            intervals[f'{int(conf * 100)}%'] = {
                'lower': float(pred_quantiles[lower_idx]),
                'upper': float(pred_quantiles[upper_idx]),
                'median': float(pred_quantiles[499]),
                'width': float(pred_quantiles[upper_idx] - pred_quantiles[lower_idx])
            }

        return {
            'intervals': intervals,
            'median_prediction': float(pred_quantiles[499]),
            'mean_prediction': float(np.mean(pred_quantiles)),
            'input_features': self._get_input_features_dict(X)
        }

    def get_highest_probability_ranges(self, input_features, bin_width=0.001, top_k=3):
        """
        Quantile Function을 PDF로 변환하여 확률 밀도가 높은 구간 찾기

        수학적 원리:
        - Quantile Function: Q(τ) = y, τ ∈ [0.001, 0.999]
        - CDF: F(y) = τ (역함수 관계)
        - PDF: f(y) = dF(y)/dy = dτ/dy

        이산 근사:
        - f(y_i) ≈ Δτ / ΔQ = (τ_{i+1} - τ_{i-1}) / (Q_{i+1} - Q_{i-1})
        """
        X = self._prepare_input(input_features)
        pred_quantiles = self._predict_quantiles(X)  # Q(τ_i) for i=0..998

        # 🔍 단조성 검사
        non_monotonic = np.diff(pred_quantiles) < 0
        if np.any(non_monotonic):
            n_violations = np.sum(non_monotonic)
            print(f"⚠️  경고: Quantile Function이 {n_violations}개 구간에서 감소합니다!")
            print(f"   이는 역함수가 정의되지 않는 구간입니다.")
            violation_indices = np.where(non_monotonic)[0][:5]  # 처음 5개만
            for idx in violation_indices:
                print(f"   τ={self.quantiles[idx]:.3f}: Q={pred_quantiles[idx]:.4f} → Q={pred_quantiles[idx + 1]:.4f}")

        # 1. PDF 계산: f(y) = Δτ / ΔQ
        pdf_values = np.zeros(len(pred_quantiles))

        # 중심차분으로 PDF 계산 (양 끝 제외)
        for i in range(1, len(pred_quantiles) - 1):
            delta_tau = self.quantiles[i + 1] - self.quantiles[i - 1]  # 0.002
            delta_Q = pred_quantiles[i + 1] - pred_quantiles[i - 1]

            if abs(delta_Q) > 1e-10:  # 0으로 나누기 방지
                pdf_values[i] = delta_tau / delta_Q
                # 음수 PDF 방지 (비단조 구간)
                if pdf_values[i] < 0:
                    pdf_values[i] = 0  # 음수 확률밀도는 0으로 처리
            else:
                pdf_values[i] = 100.0  # 매우 높은 밀도 (하지만 현실적인 값)

        # 양 끝점 처리 (전진/후진 차분)
        if len(pred_quantiles) > 1:
            # 첫 점 (전진차분)
            delta_tau_0 = self.quantiles[1] - self.quantiles[0]
            delta_Q_0 = pred_quantiles[1] - pred_quantiles[0]
            if abs(delta_Q_0) > 1e-10:
                pdf_values[0] = max(0, delta_tau_0 / delta_Q_0)  # 음수 방지
            else:
                pdf_values[0] = 100.0

            # 마지막 점 (후진차분)
            delta_tau_last = self.quantiles[-1] - self.quantiles[-2]
            delta_Q_last = pred_quantiles[-1] - pred_quantiles[-2]
            if abs(delta_Q_last) > 1e-10:
                pdf_values[-1] = max(0, delta_tau_last / delta_Q_last)  # 음수 방지
            else:
                pdf_values[-1] = 100.0

        # 2. bin_width 단위로 구간을 나누고 평균 PDF 계산
        # min/max를 bin_width 단위로 정렬하여 깔끔한 경계 생성
        min_val = float(pred_quantiles.min())
        max_val = float(pred_quantiles.max())

        # bin_width 단위로 내림/올림하여 정밀도 맞춤
        min_aligned = np.floor(min_val / bin_width) * bin_width
        max_aligned = np.ceil(max_val / bin_width) * bin_width

        bins = np.arange(min_aligned, max_aligned + bin_width, bin_width)

        bin_info = []
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]

            # 이 구간에 속하는 quantile 찾기
            in_bin = (pred_quantiles >= lower) & (
                pred_quantiles < upper if i < len(bins) - 2 else pred_quantiles <= upper)
            quantile_indices = np.where(in_bin)[0]

            if len(quantile_indices) == 0:
                continue

            # 구간 내 평균 PDF (확률밀도)
            avg_pdf = float(np.mean(pdf_values[quantile_indices]))

            # 구간의 확률 ≈ ∫ f(y) dy ≈ f(y) × Δy
            probability = avg_pdf * bin_width

            bin_info.append({
                'range': f'{lower:.0f}원 ~ {upper:.0f}원',
                'lower': float(lower),
                'upper': float(upper),
                'center': float((lower + upper) / 2),
                'pdf': avg_pdf,  # 확률밀도 f(y)
                'probability': float(probability),  # P(y ∈ [lower, upper]) - 정규화 전
                'probability_percent': float(probability * 100)
            })

        # 전체 확률 정규화 (∑P = 1이 되도록)
        total_probability = sum(b['probability'] for b in bin_info)
        print(f"[DEBUG] 정규화 전 total_probability: {total_probability:.4f}")

        if total_probability > 0:
            for b in bin_info:
                old_prob = b['probability']
                b['probability'] = b['probability'] / total_probability
                b['probability_percent'] = b['probability'] * 100

        # PDF 기준으로 정렬 (확률밀도가 높은 순)
        sorted_bins = sorted(bin_info, key=lambda x: x['pdf'], reverse=True)

        return {
            'top_ranges': sorted_bins[:top_k],
            'all_ranges': sorted_bins,
            'total_bins': len(sorted_bins),
            'bin_width': bin_width,
            'prediction_range': {'min': min_val, 'max': max_val, 'range': max_val - min_val},
            'statistics': {
                'median': float(pred_quantiles[499]),
                'mean': float(np.mean(pred_quantiles)),
                'std': float(np.std(pred_quantiles)),
                'q25': float(pred_quantiles[249]),
                'q75': float(pred_quantiles[749])
            },
            'input_features': self._get_input_features_dict(X)
        }

    def get_most_probable_range(self, input_features, bin_width=0.5):
        """가장 확률 밀도가 높은 구간 1개 반환"""
        result = self.get_highest_probability_ranges(input_features, bin_width, top_k=1)

        if not result['top_ranges']:
            return None

        most_probable = result['top_ranges'][0]
        return {
            'most_probable_range': most_probable['range'],
            'lower': most_probable['lower'],
            'upper': most_probable['upper'],
            'center': most_probable['center'],
            'probability': most_probable['probability'],
            'probability_percent': most_probable['probability_percent'],
            'statistics': result['statistics'],
            'prediction_range': result['prediction_range'],
            'input_features': result['input_features']
        }

    def get_mode_and_peak_density(self, input_features, bandwidth=0.001):
        """최빈값(mode)과 peak 밀도 분석"""
        X = self._prepare_input(input_features)
        pred_quantiles = self._predict_quantiles(X)

        # 밀도 계산
        densities = np.array([
            np.sum(np.abs(pred_quantiles - q_val) <= bandwidth) / 999 / (2 * bandwidth)
            for q_val in pred_quantiles
        ])

        # 최대 밀도 인덱스
        peak_idx = np.argmax(densities)
        mode_value = float(pred_quantiles[peak_idx])
        peak_lower, peak_upper = mode_value - bandwidth, mode_value + bandwidth
        peak_count = np.sum((pred_quantiles >= peak_lower) & (pred_quantiles <= peak_upper))

        return {
            'mode': mode_value,
            'mode_quantile': float(self.quantiles[peak_idx]),
            'peak_density': float(densities[peak_idx]),
            'peak_range': {
                'lower': float(peak_lower),
                'upper': float(peak_upper),
                'probability': float(peak_count / 999),
                'probability_percent': float(peak_count / 999 * 100)
            },
            'median': float(pred_quantiles[499]),
            'mean': float(np.mean(pred_quantiles)),
            'std': float(np.std(pred_quantiles)),
            'input_features': self._get_input_features_dict(X)
        }


def main():
    """사용 예시"""
    print("=" * 80)
    print("TFT 4-Feature 모델 - 가장 확률이 높은 구간 예측")
    print("=" * 80)

    predictor = ProbabilityPredictor(model_path='./results_transformer/best_model.pt')

    # ✅ 예시 입력값 (백분율 스케일!)
    input_dict = {
        '기초금액': 50000000,
        '추정가격': 45000000,
        '예가범위': 3.5,  # ← 3.5% (not 0.035)
        '낙찰하한율': 87.74  # ← 87.74% (not 0.8774)
    }

    print(f"\n입력 피처:")
    for key, value in input_dict.items():
        print(f"  {key}: {value}")

    # 확률이 높은 상위 5개 구간
    result = predictor.get_highest_probability_ranges(input_dict, bin_width=100000, top_k=5)

    print("\n" + "=" * 80)
    print(f"모델 예측 범위: {result['prediction_range']['min']:.0f}원 ~ {result['prediction_range']['max']:.0f}원")
    print(f"중앙값: {result['statistics']['median']:.0f}원")
    print(f"평균: {result['statistics']['mean']:.0f}원")
    print("=" * 80)

    print("\n✨ 확률이 높은 상위 5개 구간:")
    for i, r in enumerate(result['top_ranges'], 1):
        print(f"  {i}위. {r['range']} (확률: {r['probability_percent']:.2f}%)")


if __name__ == "__main__":
    print(f"Using device: {device}")
    main()