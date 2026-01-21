import uvicorn
import torch
import re
import nest_asyncio
import os
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pyngrok import ngrok

# --- 모듈 임포트 ---
# 파일들이 같은 디렉토리에 있다고 가정합니다.
try:
    from model_transformer import TransformerRegressor
    from BidAssitanceModel import BidRAGPipeline
except ImportError as e:
    print(f"❌ 필수 모듈 로딩 실패: {e}")
    print("model_transformer.py 와 BidAssitanceModel.py 파일이 필요합니다.")
    exit(1)

# ==========================================
# 0. 유틸리티 함수
# ==========================================
def parsenumber(value: Any) -> Optional[float]:
    """다양한 형태의 숫자 문자열을 float로 변환"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    
    s = str(value).strip()
    if not s: return None
        
    s = s.replace(',', '').replace('원', '').replace('KRW', '').replace('₩', '')
    s = re.sub(r'[^0-9.\-]', '', s)
    
    if not s or s in ('-', '.', '-.'): return None
    try:
        return float(s)
    except Exception:
        return None

def load_scalers_json(path: str):
    if not os.path.exists(path):
        print(f"⚠️ 스케일러 파일 없음: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_transformer_model(model_path: str):
    print(f"🔄 모델 로딩 중: {model_path}")
    if not os.path.exists(model_path):
        print("⚠️ 모델 파일 없음. 기본값 사용.")
        return None, {"num_features": 4, "d_model": 512} 

    state_dict = torch.load(model_path, map_location='cpu')

    # --- [자동 감지 시작] ---
    config = {
        "num_features": 4, "d_model": 128, "num_layers": 2, "dim_feedforward": 512, "nhead": 4
    }

    if 'cls_token' in state_dict:
        config['d_model'] = state_dict['cls_token'].shape[2]
        
    for key, param in state_dict.items():
        if 'feature_emb.weight' in key:
            dim1, dim2 = param.shape
            config['num_features'] = min(dim1, dim2)
            if 'cls_token' not in state_dict: config['d_model'] = max(dim1, dim2)
            break

    for key, param in state_dict.items():
        if 'linear1.weight' in key:
            config['dim_feedforward'] = param.shape[0]
            break

    max_layer_idx = -1
    for key in state_dict.keys():
        match = re.search(r'layers\.(\d+)\.', key)
        if match: max_layer_idx = max(max_layer_idx, int(match.group(1)))
    if max_layer_idx != -1:
        config['num_layers'] = max_layer_idx + 1

    print(f"✅ 최종 자동 감지 설정: {config}")

    model = TransformerRegressor(
        num_features=config['num_features'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=0.1
    )
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("🎉 모델 파라미터 로드 성공!")
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
    
    model.eval()
    return model, config


# ==========================================
# 1. 모델 및 스케일러 로드
# ==========================================
MODEL_PATH = "../results_transformer_4feat/transformer_4feat.pt"
SCALER_PATH = "../results_transformer_4feat/scalers.json"

TF_MODEL, TF_CONFIG = load_transformer_model(MODEL_PATH)
SCALER_DATA = load_scalers_json(SCALER_PATH)

if SCALER_DATA is None:
    print("⚠️ 스케일러 데이터가 없어 임시값(mean=0, std=1)을 사용합니다.")
    SCALER_DATA = {
        "x_mean": [0.0] * TF_CONFIG['num_features'],
        "x_std": [1.0] * TF_CONFIG['num_features'],
        "y_mean": 0.0,
        "y_std": 1.0,
        "target_log": True
    }

# ==========================================
# 2. RAG 파이프라인 어댑터
# ==========================================
class TransformerPredictorAdapter:
    def __init__(self, model, scaler_data):
        self.model = model
        # 스케일러 정보 로드 (없으면 기본값)
        self.x_mean = np.array(scaler_data.get('x_mean', [0.0]*4))
        self.x_std = np.array(scaler_data.get('x_std', [1.0]*4))
        self.y_mean = float(scaler_data.get('y_mean', 0.0))
        self.y_std = float(scaler_data.get('y_std', 1.0))
        self.target_log = bool(scaler_data.get('target_log', False))

    def predict(self, requirements: Dict[str, Any], retrieved_context: str) -> Dict[str, Any]:
        try:
            budget = parsenumber(requirements.get('budget'))
            estimate = parsenumber(requirements.get('estimate_price'))
            pr_range = parsenumber(requirements.get('expected_price_range'))
            lower_rate = parsenumber(requirements.get('award_lower_rate'))

            features = np.array([
                budget if budget else 0.0,
                estimate if estimate else 0.0,
                pr_range if pr_range else 0.0,
                lower_rate if lower_rate else 0.0
            ])
            
            # 1. Scaling (Input)
            scaled_features = (features - self.x_mean) / self.x_std
        
            if self.model:
                # 2. Reshape & Tensor Convert
                # (1, 4) -> (1, 4, 1) : Transformer 모델 입력 shape 확인 필요
                # 보통 TransformerRegressor가 (Batch, Features, 1)을 기대한다면 아래처럼:
                input_tensor = torch.tensor(scaled_features, dtype=torch.float32).reshape(1, -1, 1)
                
                with torch.no_grad():
                    pred_s = self.model(input_tensor).item()
                
                # 3. Inverse Scaling (Output)
                pred_log = pred_s * self.y_std + self.y_mean
                
                # 4. Log Inverse (expm1)
                final_pred = np.expm1(pred_log) if self.target_log else pred_log
                
            else:
                final_pred = 0.0

            return {
                "currency": "KRW",
                "point_estimate": round(final_pred),
                "predicted_min": round(final_pred * 0.98),
                "predicted_max": round(final_pred * 1.02),
                "confidence": "high" if self.model else "low",
                "rationale": f"Transformer Model (Inputs: {features.tolist()})",
                "model_type": "TransformerRegressor"
            }
        except Exception as e:
            return {"error": str(e), "rationale": "Prediction Failed"}

# [수정됨] 어댑터 생성 시 스케일러 데이터 전달
adapter = TransformerPredictorAdapter(TF_MODEL, SCALER_DATA)

print("🚀 RAG 파이프라인 초기화...")
rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus", 
    index_dir="./rag_index",
    award_predict_fn=adapter.predict
)

# ==========================================
# 3. FastAPI 서버
# ==========================================
app = FastAPI(title="Integrated Bid Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictReq(BaseModel):
    features: List[float]

class AnalyzeReq(BaseModel):
    text: str
    thread_id: str = "default"

@app.post("/predictBase")
async def predict_base(req: PredictReq):
    if not TF_MODEL:
        return {"error": "Model not loaded", "predBid": 0}
        
    try:
        # 여기도 어댑터 로직과 비슷하게 스케일링 필요
        feat_arr = np.array(req.features)
        x_mean = np.array(SCALER_DATA['x_mean'])
        x_std = np.array(SCALER_DATA['x_std'])
        
        scaled = (feat_arr - x_mean) / x_std
        input_tensor = torch.tensor(scaled, dtype=torch.float32).reshape(1, -1, 1)
        
        with torch.no_grad():
            pred_s = TF_MODEL(input_tensor).item()
            
        y_mean = SCALER_DATA['y_mean']
        y_std = SCALER_DATA['y_std']
        target_log = SCALER_DATA.get('target_log', False)
        
        pred_log = pred_s * y_std + y_mean
        final_pred = np.expm1(pred_log) if target_log else pred_log
        
        return {"predBid": final_pred}
    except Exception as e:
        return {"error": str(e), "predBid": 0}

@app.post("/analyze")
async def analyze(req: AnalyzeReq):
    try:
        result = rag_pipeline.analyze(req.text, thread_id=req.thread_id)
        return {
            "extracted_requirements": result.get("requirements", {}),
            "prediction": result.get("prediction_result", {}),
            "report": result.get("report_markdown", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "running"}

if __name__ == "__main__":
    auth_token = "38H6WIHF5Hn1xV68lPnXu15Tutc_4PDGKRtxpJhbJuVdcUCEp"
    ngrok.set_auth_token(auth_token)
    url = ngrok.connect(9999).public_url
    print(f"🌍 Public URL: {url}")
    
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=9999)
