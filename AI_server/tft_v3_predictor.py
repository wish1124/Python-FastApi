# tft_v3_predictor.py
import ast
import joblib
import numpy as np
import torch
import torch.nn as nn

MODEL_PATH = "model/tft_v3/best_model.pt"
SCALER_PATH = "model/tft_v3/scaler_X.pkl"
FEATURES_PATH = "model/tft_v3/features.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuantileTransformerRegressor(nn.Module):
    def __init__(self, input_dim, num_quantiles,
                 d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, dim_feedforward // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 4, num_quantiles),
        )

    def forward(self, x):
        x = self.input_embedding(x).unsqueeze(1) + self.pos_encoder
        x = self.transformer_encoder(x).squeeze(1)
        return self.fc_out(x)


def _load_features():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read().strip())  # features.txt가 파이썬 리스트 문자열 형태


def _build_quantiles():
    q = np.linspace(0.05, 0.95, 10).tolist()
    if 0.5 not in q:
        q.append(0.5)
        q = sorted(q)
    return q


FEATURES = _load_features()
SCALER_X = joblib.load(SCALER_PATH)

QUANTILES = _build_quantiles()
NUM_QUANTILES = len(QUANTILES)       # 보통 11
MEDIAN_IDX = QUANTILES.index(0.5)    # 보통 5

MODEL = QuantileTransformerRegressor(
    input_dim=len(FEATURES),
    num_quantiles=NUM_QUANTILES,
    d_model=512,
    nhead=8,
    num_layers=2,
    dim_feedforward=2048,
    dropout=0.1
).to(device)

ckpt = torch.load(MODEL_PATH, map_location=device)
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
MODEL.load_state_dict(state, strict=False)
MODEL.eval()


def predict_sajeong_percent(feature_dict: dict) -> float:
    """
    feature_dict: FEATURES에 있는 키들을 포함하는 dict.
    return: 사정율(%) 예측 (중앙값)
    """
    x = np.array([[float(feature_dict.get(k, 0.0)) for k in FEATURES]], dtype=np.float32)
    x_scaled = SCALER_X.transform(x)

    with torch.no_grad():
        yq = MODEL(torch.tensor(x_scaled, dtype=torch.float32, device=device)).cpu().numpy()[0]

    return float(yq[MEDIAN_IDX])
def predict_quantiles(feature_dict: dict) -> np.ndarray:
    """11개 quantile 값을 반환(중앙값 포함)"""
    x = np.array([[float(feature_dict.get(k, 0.0)) for k in FEATURES]], dtype=np.float32)
    x_scaled = SCALER_X.transform(x)

    with torch.no_grad():
        yq = MODEL(torch.tensor(x_scaled, dtype=torch.float32, device=device)).cpu().numpy()[0]
    return yq


def get_highest_probability_ranges_v3(
    feature_dict: dict,
    bin_width: float = 0.001,
    top_k: int = 3,
    n_samples: int = 20000,
    output_as_percent: bool = True,
):
    """
    quantile 출력(11개)을 이용해 inverse-CDF 샘플링 → 히스토그램 → top_k 구간 반환
    bin_width=0.001이면 '0.1%p' 단위 구간
    """
    yq = predict_quantiles(feature_dict)

    # v3 모델 출력이 배율(예: 1.0012)이라면 percent로 변환해서 구간을 만들면 보기 좋음
    values = yq * 100.0 if output_as_percent else yq

    # inverse CDF 샘플링 (quantiles: 0.05~0.95 + 0.5)
    qs = np.array(QUANTILES, dtype=float)
    # qs와 values를 정렬(안전)
    order = np.argsort(qs)
    qs = qs[order]
    values = values[order]

    u = np.random.rand(n_samples)
    samples = np.interp(u, qs, values)

    # histogram
    mn, mx = float(samples.min()), float(samples.max())
    # bin_width 간격으로 bin 생성
    edges = np.arange(np.floor(mn/bin_width)*bin_width, np.ceil(mx/bin_width)*bin_width + bin_width, bin_width)
    hist, edges = np.histogram(samples, bins=edges)
    probs = hist / hist.sum()

    # top bins
    idxs = np.argsort(probs)[::-1][:top_k]
    top_ranges = []
    for idx in idxs:
        low = float(edges[idx])
        high = float(edges[idx+1])
        center = (low + high) / 2.0
        prob = float(probs[idx])

        # 보기 형식: 89.7% ~ 89.6% 같이(소수 1자리) + center는 2자리
        # (표시 순서가 high~low로 나가길 원하면 아래처럼)
        range_display = f"{high:.1f}% ~ {low:.1f}%"
        top_ranges.append({
            "range": [low, high],
            "range_display": range_display,
            "center": round(center, 2),
            "probability": round(prob * 100.0, 2),  # %
            "lower": low,
            "upper": high
        })

    # statistics(필요하면 보고서에 같이 쓰려고)
    stats = {
        "q25": float(np.quantile(samples, 0.25)),
        "q50": float(np.quantile(samples, 0.50)),
        "q75": float(np.quantile(samples, 0.75)),
    }

    return {"top_ranges": top_ranges, "statistics": stats}
