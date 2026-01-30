import uvicorn
import torch
import re
import nest_asyncio
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pyngrok import ngrok

# 커스텀 모듈 임포트
try:
    from BidAssitanceModel import BidRAGPipeline
    from get_probability_from_model import ProbabilityPredictor
except ImportError as e:
    print(f"❌ 필수 모듈 로딩 실패: {e}")
    print("BidAssitanceModel.py, get_probability_from_model.py 파일이 필요합니다.")
    exit(1)


# ==========================================
# 1. 유틸리티 함수
# ==========================================
def parsenumber(value: Any) -> Optional[float]:
    """
    다양한 형태의 숫자 문자열을 float로 변환
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

# ==========================================
# 2. 모델 로드
# ==========================================

# TFT 모델 로드
TFT_MODEL_PATH = './results_tft_4feat/best_model.pt'
try:
    tft_predictor = ProbabilityPredictor(model_path=TFT_MODEL_PATH)
    print("✅ TFT 모델 로드 성공")
except Exception as e:
    print(f"⚠️ TFT 모델 로드 실패: {e}")
    tft_predictor = None


# ==========================================
# 3. RAG 파이프라인 어댑터
# ==========================================
class TFTPredictorAdapter:
    """RAG 파이프라인에서 사용할 TFT 모델 어댑터"""
    
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, requirements: Dict[str, Any], retrieved_context: str) -> Dict[str, Any]:
        """입찰 요구사항을 기반으로 TFT 모델로 예측 수행"""
        try:
            if not self.predictor:
                return {"error": "Model not loaded", "rationale": "Prediction Failed"}

            pr_range = parsenumber(requirements.get('expected_price_range'))
            lower_rate = parsenumber(requirements.get('award_lower_rate'))
            estimate = parsenumber(requirements.get('estimate_price'))
            budget = parsenumber(requirements.get('budget'))

            input_dict = {
                '예가범위': pr_range if pr_range else 0.0,
                '낙찰하한율': lower_rate if lower_rate else 0.0,
                '추정가격': estimate if estimate else 0.0,
                '기초금액': budget if budget else 0.0
            }
            
            result = self.predictor.get_highest_probability_ranges(input_dict, bin_width=0.001, top_k=3)
            
            if result and result.get("top_ranges"):
                top_ranges = result["top_ranges"]
                return {
                    "currency": "KRW",
                    "point_estimate": top_ranges[0]["center"],
                    "predicted_min": result["statistics"]["q25"],
                    "predicted_max": result["statistics"]["q75"],
                    "confidence": "high",
                    "top_ranges": top_ranges,
                    "rationale": f"TFT Model - Top 5 ranges computed",
                    "model_type": "QuantileTransformerRegressor"
                }
            else:
                return {"error": "Prediction failed", "rationale": "No result"}
        except Exception as e:
            return {"error": str(e), "rationale": "Prediction Failed"}

# 어댑터 초기화
adapter = TFTPredictorAdapter(tft_predictor)

print("🚀 RAG 파이프라인 초기화...")
# 문서/인덱스 경로는 실제 환경에 맞게 수정해주세요
rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus", 
    index_dir="./rag_index",
    award_predict_fn=adapter.predict # ★ 어댑터 함수 주입
)

# ==========================================
# 4. FastAPI 서버
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
    if not tft_predictor:
        return {"error": "TFT Model not loaded", "predBid": 0}
        
    try:
        # req.features는 [예가범위, 낙찰하한율, 추정가격, 기초금액] 순서로 가정
        if len(req.features) != 4:
            return {"error": "4개의 feature가 필요합니다", "predBid": 0}
        
        input_dict = {
            '예가범위': req.features[0],
            '낙찰하한율': req.features[1],
            '추정가격': req.features[2],
            '기초금액': req.features[3]
        }
        
        # TFT 모델로 확률 높은 상위 5개 구간 예측
        result = tft_predictor.get_highest_probability_ranges(input_dict, bin_width=0.001, top_k=3)
        
        if result and result.get("top_ranges"):
            top_ranges = result["top_ranges"]
            return {
                "predBid": top_ranges[0]["center"],
                "top_ranges": top_ranges,
                "median": result["statistics"]["median"],
                "mean": result["statistics"]["mean"]
            }
        else:
            return {"error": "예측 실패", "predBid": 0}
            
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
