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

# --- 외부 모듈 임포트 (파일 경로에 맞게 수정 필요) ---
try:
    from model_transformer import TransformerRegressor  # 모델 정의 파일
    # BidAssitanceModel이 같은 폴더에 있다고 가정
    from BidAssitanceModel import BidRAGPipeline, CallableAwardPricePredictor, parsenumber
except ImportError as e:
    print(f"❌ 필수 모듈을 찾을 수 없습니다: {e}")
    print("BidAssitanceModel.py와 model_transformer.py가 같은 폴더에 있는지 확인해주세요.")
    exit(1)

from pyngrok import ngrok

# ==========================================
# 1. Transformer 모델 로드 및 설정 (model_serving.py 로직)
# ==========================================

def load_transformer_model(model_path: str):
    print(f"🔄 모델 로딩 중: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
    state_dict = torch.load(model_path, map_location='cpu')

    # 하이퍼파라미터 자동 추론
    config = {
        "num_features": 4, 
        "d_model": 512, 
        "num_layers": 2, 
        "nhead": 4
    }
    
    # 1. 입력 차원 추론
    for key, param in state_dict.items():
        if ('input' in key or 'embedding' in key) and 'weight' in key and param.dim() == 2:
            config['num_features'] = param.shape[1]
            config['d_model'] = param.shape[0]
            break
            
    # 2. 레이어 깊이 추론
    max_layer_idx = -1
    for key in state_dict.keys():
        match = re.search(r'layers\.(\d+)\.', key)
        if match:
            max_layer_idx = max(max_layer_idx, int(match.group(1)))
    if max_layer_idx != -1:
        config['num_layers'] = max_layer_idx + 1

    print(f"✅ 감지된 모델 설정: {config}")

    # 모델 초기화
    model = TransformerRegressor(
        num_features=config['num_features'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        nhead=4,
        dim_feedforward=config['d_model'] * 4,
        dropout=0.1
    )
    
    model.load_state_dict(state_dict)
    model.eval()
    return model, config

# 전역 모델 로드
MODEL_PATH = "../results_transformer/best_model.pt" # 경로 확인 필요
TF_MODEL, TF_CONFIG = load_transformer_model(MODEL_PATH)


# ==========================================
# 2. RAG 파이프라인 연동용 어댑터 정의
# ==========================================

class TransformerPredictorAdapter:
    """
    BidRAGPipeline이 Transformer 모델을 사용할 수 있게 해주는 어댑터
    Dict[str, Any] (requirements) -> Tensor -> Dict[str, Any] (prediction result)
    """
    def __init__(self, model, input_dim):
        self.model = model
        self.input_dim = input_dim

    def predict(self, requirements: Dict[str, Any], retrieved_context: str) -> Dict[str, Any]:
        """
        RAG 파이프라인에서 호출하는 표준 예측 함수
        """
        try:
            # 1. 특성 추출 (순서 중요: budget, estimate, range, rate)
            # BidAssitanceModel.py의 CNN 로직과 동일하게 파싱
            budget = parsenumber(requirements.get('budget'))
            estimate = parsenumber(requirements.get('estimate_price'))
            # 예가범위, 하한율 처리 (간소화됨, 실제로는 정규 표현식 필요할 수 있음)
            pr_range = parsenumber(requirements.get('expected_price_range'))
            lower_rate = parsenumber(requirements.get('award_lower_rate'))

            # 결측치 처리 (기본값 또는 에러)
            features = [
                budget if budget else 0.0,
                estimate if estimate else 0.0,
                pr_range if pr_range else 0.0,
                lower_rate if lower_rate else 0.0
            ]

            # 2. 텐서 변환 및 추론
            input_tensor = torch.tensor([features], dtype=torch.float32)
            
            with torch.no_grad():
                pred_raw = self.model(input_tensor).item()

            # 3. 결과 포맷팅 (RAG 파이프라인이 기대하는 형식)
            return {
                "currency": "KRW",
                "predicted_min": None, # 범위 예측 모델이 아니므로 단일값
                "predicted_max": None,
                "point_estimate": round(pred_raw),
                "confidence": "high" if all(f > 0 for f in features) else "low",
                "rationale": f"Transformer Model Inference (Inputs: {features})",
                "model": {"type": "transformer_regressor", "features": features}
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "confidence": "low",
                "rationale": "Inference Failed"
            }

# 어댑터 인스턴스 생성
tf_adapter = TransformerPredictorAdapter(TF_MODEL, TF_CONFIG['num_features'])

# RAG 파이프라인 초기화
print("🚀 RAG 파이프라인 초기화 중...")
rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus",      # 문서 경로
    index_dir="./rag_index",     # FAISS 인덱스 경로
    award_predict_fn=tf_adapter.predict  # ★ Transformer 모델 연결
)


# ==========================================
# 3. FastAPI 서버 설정
# ==========================================

app = FastAPI(title="Bid Analytics & Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 요청 DTO ---
class PredictBaseRequest(BaseModel):
    features: List[float] = Field(..., description="[budget, estimate, range, rate] 순서의 4개 실수 리스트")

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="입찰 공고문 전체 텍스트")
    thread_id: Optional[str] = Field(default="default_thread", description="대화형 컨텍스트 유지용 ID")

# --- 엔드포인트 1: 기본 모델 추론 (model_serving.py 기능) ---
@app.post("/predictBase")
async def predict_base(req: PredictBaseRequest):
    expected = TF_CONFIG['num_features']
    if len(req.features) != expected:
        raise HTTPException(status_code=400, detail=f"입력 개수 불일치 (기대: {expected}, 실제: {len(req.features)})")
    
    try:
        input_tensor = torch.tensor([req.features], dtype=torch.float32)
        with torch.no_grad():
            pred = TF_MODEL(input_tensor).item()
        return {"predBid": pred}
    except Exception as e:
        return {"error": str(e), "predBid": 9999}

# --- 엔드포인트 2: RAG 기반 전체 분석 (BidAssitanceModel.py 기능) ---
@app.post("/analyze")
async def analyze_bid(req: AnalyzeRequest):
    """
    공고문 텍스트를 받아 RAG 검색 -> 정보 추출 -> Transformer 가격 예측 -> 보고서 생성 수행
    """
    try:
        # Pipeline의 analyze 메서드 호출
        results = rag_pipeline.analyze(req.text, thread_id=req.thread_id)
        
        return {
            "requirements": results.get("requirements"), # 추출된 정보
            "report_markdown": results.get("report_markdown"), # LLM 분석 보고서
            "prediction": results.get("prediction_result") # Transformer 예측 결과 포함됨
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"status": "online", "model": "TransformerRegressor", "pipeline": "BidRAG"}

# ==========================================
# 4. 서버 실행 (ngrok 포함)
# ==========================================

if __name__ == "__main__":
    # ngrok 설정
    AUTH_TOKEN = "38H6WIHF5Hn1xV68lPnXu15Tutc_4PDGKRtxpJhbJuVdcUCEp" # 기존 토큰 유지
    ngrok.set_auth_token(AUTH_TOKEN)
    
    port = 9999
    public_url = ngrok.connect(port).public_url
    print(f"🌍 공용 URL: {public_url}")

    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=port)
