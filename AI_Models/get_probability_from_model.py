import torch
import torch.nn as nn
import numpy as np
import os
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
        self.feature_names = ['안전관리비비율', '안전관리비_적용여부', '추정가격', '기초금액']
        self.model = self._load_model()
        self.scaler = None
        
    def _load_model(self):
        """학습된 모델 로드"""
        model = QuantileTransformerRegressor(
            input_dim=4, num_quantiles=999, d_model=128, nhead=8,
            num_layers=3, dim_feedforward=512, dropout=0.1
        ).to(self.device)
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        print(f"✓ 모델 로드 완료: {self.model_path}")
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}, Val Loss: {checkpoint.get('val_loss', 0):.6f}")
        
        return model
    
    def _prepare_input(self, input_features):
        """입력 피처를 numpy array로 변환"""
        if isinstance(input_features, dict):
            X = np.array([[
                input_features['안전관리비비율'],
                input_features['안전관리비_적용여부'],
                input_features['추정가격'],
                input_features['기초금액']
            ]], dtype=np.float32)
        else:
            X = np.array([input_features], dtype=np.float32)
            if X.shape[1] != 4:
                raise ValueError(f"입력 피처는 4개여야 합니다. 현재: {X.shape[1]}개")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def _predict_quantiles(self, X):
        """999개 quantile 예측"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            return self.model(X_tensor).cpu().numpy()[0]
    
    def _get_input_features_dict(self, X):
        """입력 피처를 dict 형태로 반환"""
        return {
            '안전관리비비율': float(X[0, 0]),
            '안전관리비_적용여부': float(X[0, 1]),
            '추정가격': float(X[0, 2]),
            '기초금액': float(X[0, 3])
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
            
            intervals[f'{int(conf*100)}%'] = {
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
    
    
    def get_highest_probability_ranges(self, input_features, bin_width=0.5, top_k=5):
        """999개 quantile 중 확률 밀도가 높은 구간 찾기"""
        X = self._prepare_input(input_features)
        pred_quantiles = self._predict_quantiles(X)
        
        # 히스토그램으로 밀도 계산
        min_val, max_val = float(pred_quantiles.min()), float(pred_quantiles.max())
        bins = np.arange(min_val, max_val + bin_width, bin_width)
        hist, bin_edges = np.histogram(pred_quantiles, bins=bins)
        
        # 구간별 정보 생성
        bin_info = []
        for i in range(len(hist)):
            if hist[i] == 0:
                continue
                
            lower, upper = bin_edges[i], bin_edges[i + 1]
            in_bin = (pred_quantiles >= lower) & (pred_quantiles < upper if i < len(hist) - 1 else pred_quantiles <= upper)
            quantile_indices = np.where(in_bin)[0]
            
            bin_info.append({
                'range': f'[{lower:.2f}%, {upper:.2f}%]',
                'lower': float(lower),
                'upper': float(upper),
                'center': float((lower + upper) / 2),
                'probability': float(hist[i] / 999),
                'probability_percent': float(hist[i] / 999 * 100),
                'quantile_count': int(hist[i]),
                'quantile_range': f'{self.quantiles[quantile_indices[0]]:.1f}% ~ {self.quantiles[quantile_indices[-1]]:.1f}%'
            })
        
        sorted_bins = sorted(bin_info, key=lambda x: x['probability'], reverse=True)
        
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
            'quantile_count': most_probable['quantile_count'],
            'quantile_range': most_probable['quantile_range'],
            'statistics': result['statistics'],
            'prediction_range': result['prediction_range'],
            'input_features': result['input_features']
        }
    
    def get_mode_and_peak_density(self, input_features, bandwidth=0.5):
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
    
    predictor = ProbabilityPredictor(model_path='./results_tft_4feat/best_model.pt')
    
    # 예시 입력값
    input_dict = {
        '안전관리비비율': 0.0348,
        '안전관리비_적용여부': 1,
        '추정가격': 53643620,
        '기초금액': 48279258
    }
    
    print(f"\n입력 피처:")
    for key, value in input_dict.items():
        print(f"  {key}: {value}")
    
    # 확률이 높은 상위 5개 구간
    result = predictor.get_highest_probability_ranges(input_dict, bin_width=0.001, top_k=5)
    
    print("\n" + "=" * 80)
    print(f"모델 예측 범위: {result['prediction_range']['min']*100:.2f}% ~ {result['prediction_range']['max']*100:.2f}%")
    print(f"중앙값: {result['statistics']['median']*100:.2f}%")
    print(f"평균: {result['statistics']['mean']*100:.2f}%")
    print("=" * 80)
    
    print(f"\n✨ 확률이 높은 상위 5개 구간:")
    for i, r in enumerate(result['top_ranges'], 1):
        lower_val = r['lower']*100 - 100
        upper_val = r['upper']*100 - 100
        print(f"  {i}위. {lower_val:+.1f}~{upper_val:+.1f} (확률: {r['probability_percent']:.2f}%, 999개 중 {r['quantile_count']}개)")


if __name__ == "__main__":
    print(f"Using device: {device}")
    main()
