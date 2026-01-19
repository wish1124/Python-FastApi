import uvicorn
import torch
import re
import nest_asyncio
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- 모듈 임포트 ---
# 파일들이 같은 디렉토리에 있다고 가정합니다.
try:
    from model_transformer import TransformerRegressor
    from BidAssitanceModel import BidRAGPipeline
except ImportError as e:
    print(f"❌ 필수 모듈 로딩 실패: {e}")
    print("model_transformer.py 와 BidAssitanceModel.py 파일이 필요합니다.")
    exit(1)

from pyngrok import ngrok

# ==========================================
# 0. 유틸리티 함수 (parsenumber 직접 구현)
# ==========================================
def parsenumber(value: Any) -> Optional[float]:
    """
    다양한 형태의 숫자 문자열을 float로 변환 (BidAssitanceModel.py 로직 복사)
    예: "1,000,000원" -> 1000000.0
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    
    s = str(value).strip()
    if not s:
        return None
        
    # 통화 기호 및 콤마 제거
    s = s.replace(',', '').replace('원', '').replace('KRW', '').replace('₩', '')
    
    # 숫자, 점(.), 마이너스(-) 외의 문자 제거
    s = re.sub(r'[^0-9.\-]', '', s)
    
    if not s or s in ('-', '.', '-.'):
        return None
        
    try:
        return float(s)
    except Exception:
        return None

def load_transformer_model(model_path: str):
    print(f"🔄 모델 로딩 중: {model_path}")
    
    if not os.path.exists(model_path):
        print("⚠️ 모델 파일 없음. 기본값 사용.")
        return None, {"num_features": 4, "d_model": 512} # 더미 반환

    state_dict = torch.load(model_path, map_location='cpu')

    # --- [자동 감지 시작] ---
    config = {
        "num_features": 4,   # 기본값 (감지 실패 시)
        "d_model": 128,      # 기본값
        "num_layers": 2,     # 기본값
        "dim_feedforward": 512, # 기본값
        "nhead": 4           # 기본값 (weight shape만으로는 알 수 없음)
    }

    # 1. d_model & num_features 감지 (cls_token 또는 첫 레이어)
    if 'cls_token' in state_dict:
        # cls_token shape: [1, 1, d_model]
        config['d_model'] = state_dict['cls_token'].shape[2]
        
    for key, param in state_dict.items():
        # feature_emb.weight shape: [num_features, d_model] (또는 반대)
        # 하지만 보통 작은 값이 num_features이므로 min/max로 구분 가능
        if 'feature_emb.weight' in key:
            # 예: [4, 128] -> 4가 feature, 128이 d_model
            dim1, dim2 = param.shape
            config['num_features'] = min(dim1, dim2)
            # d_model은 위에서 cls_token으로 찾은 걸 신뢰하거나, 큰 값을 사용
            if 'cls_token' not in state_dict:
                 config['d_model'] = max(dim1, dim2)
            break

    # 2. dim_feedforward 감지 (linear1의 출력 크기)
    # 보통 'encoder.layers.0.linear1.weight' 형태로 저장됨
    for key, param in state_dict.items():
        if 'linear1.weight' in key:
            # Linear(d_model, dim_feedforward) -> weight shape: [dim_ff, d_model]
            # 따라서 shape[0]이 dim_feedforward
            config['dim_feedforward'] = param.shape[0]
            print(f"🔍 Feedforward 차원 감지됨: {config['dim_feedforward']}")
            break

    # 3. num_layers 감지
    max_layer_idx = -1
    for key in state_dict.keys():
        match = re.search(r'layers\.(\d+)\.', key)
        if match:
            max_layer_idx = max(max_layer_idx, int(match.group(1)))
    if max_layer_idx != -1:
        config['num_layers'] = max_layer_idx + 1

    print(f"✅ 최종 자동 감지 설정: {config}")
    # -----------------------

    # 모델 초기화
    model = TransformerRegressor(
        num_features=config['num_features'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        nhead=config['nhead'], # nhead는 감지 불가 (보통 4 or 8)
        dim_feedforward=config['dim_feedforward'], # ★ 자동 감지된 값 적용
        dropout=0.1
    )
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("🎉 모델 파라미터 로드 성공!")
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        # 실패 시 빈 모델 반환하지만 config는 유지
    
    model.eval()
    return model, config



# 모델 경로 설정 (실제 경로로 수정 필요)
MODEL_PATH = "../results_transformer_4feat/transformer_4feat.pt"
TF_MODEL, TF_CONFIG = load_transformer_model(MODEL_PATH)


# ==========================================
# 2. RAG 파이프라인 어댑터
# ==========================================
class TransformerPredictorAdapter:
    def __init__(self, model, input_dim):
        self.model = model
        self.input_dim = input_dim

    def predict(self, requirements: Dict[str, Any], retrieved_context: str) -> Dict[str, Any]:
        try:
            # 추출된 정보 파싱
            budget = parsenumber(requirements.get('budget'))
            estimate = parsenumber(requirements.get('estimate_price'))
            # 예가범위, 낙찰하한율은 백분율일 수 있으므로 추가 처리 필요할 수 있음
            pr_range = parsenumber(requirements.get('expected_price_range'))
            lower_rate = parsenumber(requirements.get('award_lower_rate'))

            # None이면 0.0으로 대체
            features = [
                budget if budget else 0.0,
                estimate if estimate else 0.0,
                pr_range if pr_range else 0.0,
                lower_rate if lower_rate else 0.0
            ]

            # 모델 추론
            if self.model:
                input_tensor = torch.tensor([features], dtype=torch.float32)
                with torch.no_grad():
                    pred_raw = self.model(input_tensor).item()
            else:
                pred_raw = 0.0 # 모델 없을 때

            return {
                "currency": "KRW",
                "point_estimate": round(pred_raw),
                "predicted_min": round(pred_raw * 0.98), # 단순 예시 범위
                "predicted_max": round(pred_raw * 1.02),
                "confidence": "high" if self.model else "low",
                "rationale": f"Transformer Model (Inputs: {features})",
                "model_type": "TransformerRegressor"
            }
        except Exception as e:
            return {"error": str(e), "rationale": "Prediction Failed"}

# 어댑터 및 파이프라인 생성
adapter = TransformerPredictorAdapter(TF_MODEL, TF_CONFIG['num_features'])

print("🚀 RAG 파이프라인 초기화...")
# 문서/인덱스 경로는 실제 환경에 맞게 수정해주세요
rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus", 
    index_dir="./rag_index",
    award_predict_fn=adapter.predict # ★ 어댑터 함수 주입
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
        input_tensor = torch.tensor([req.features], dtype=torch.float32)
        with torch.no_grad():
            pred = TF_MODEL(input_tensor).item()
        return {"predBid": pred}
    except Exception as e:
        return {"error": str(e), "predBid": 0}

@app.post("/analyze")
async def analyze(req: AnalyzeReq):
    try:
        # RAG 파이프라인 실행
        result = rag_pipeline.analyze(req.text, thread_id=req.thread_id)
        
        # 결과 정리
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
