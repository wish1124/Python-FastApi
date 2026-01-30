import math
import uvicorn
import torch
import re
import nest_asyncio
import os
import json
import numpy as np
import uuid
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fpdf import FPDF
from azure.storage.blob import BlobServiceClient, ContentSettings
from datetime import datetime
from pathlib import Path

# --- 모듈 임포트 ---
try:
    from BidAssitanceModel import BidRAGPipeline
    from get_probability_from_model import ProbabilityPredictor  # ✅ TFT 모델 사용
except ImportError as e:
    print(f"❌ 필수 모듈 로딩 실패: {e}")
    exit(1)


# ==========================================
# 0. 유틸리티 함수
# ==========================================
def parsenumber(value: Any) -> Optional[float]:
    if value is None: return None
    if isinstance(value, (int, float)): return float(value)
    s = str(value).strip()
    s = re.sub(r'[^0-9.\-]', '', s.replace(',', ''))
    try:
        return float(s)
    except:
        return None


# ==========================================
# 1. TFT 모델 초기화 (절대 경로 체크 추가)
# ==========================================
BASE_DIR = Path(__file__).parent.absolute()
TFT_MODEL_PATH = BASE_DIR / 'results_tft_4feat' / 'best_model.pt'

print("=" * 60)
print("🔍 TFT 모델 초기화 시작")
print(f"   작업 디렉토리: {os.getcwd()}")
print(f"   BASE_DIR: {BASE_DIR}")
print(f"   모델 경로: {TFT_MODEL_PATH}")
print(f"   파일 존재 여부: {TFT_MODEL_PATH.exists()}")
print("=" * 60)

tft_predictor = None

# 파일 존재 여부 체크 후 로드
if not TFT_MODEL_PATH.exists():
    print(f"❌ TFT 모델 파일을 찾을 수 없습니다! 경로를 확인하세요: {TFT_MODEL_PATH}")
else:
    try:
        print("📦 TFT 모델 로딩 중...")
        tft_predictor = ProbabilityPredictor(model_path=str(TFT_MODEL_PATH))
        print("✅ TFT 모델 로드 성공!")
    except Exception as e:
        print(f"❌ TFT 모델 로드 실패! 에러: {e}")
        import traceback

        traceback.print_exc()


# ==========================================
# 2. TFT 예측 어댑터 (기존 코드 유지)
# ==========================================
class TFTPredictorAdapter:
    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, requirements: Dict[str, Any], retrieved_context: str = "") -> Dict[str, Any]:
        try:
            if not self.predictor:
                return {"error": "Model not loaded", "point_estimate": 0, "confidence": "error"}

            input_dict = {
                '예가범위': parsenumber(requirements.get('expected_price_range')) or 0.0,
                '낙찰하한율': parsenumber(requirements.get('award_lower_rate')) or 0.0,
                '추정가격': parsenumber(requirements.get('estimate_price')) or 0.0,
                '기초금액': parsenumber(requirements.get('budget')) or 0.0
            }

            result = self.predictor.get_highest_probability_ranges(input_dict, bin_width=0.001, top_k=3)

            if result and result.get("top_ranges"):
                top_ranges = result["top_ranges"]
                return {
                    "currency": "KRW",
                    "point_estimate": int(top_ranges[0]["center"]),
                    "predicted_min": int(result["statistics"]["q25"]),
                    "predicted_max": int(result["statistics"]["q75"]),
                    "confidence": "high",
                    "top_ranges": top_ranges,
                    "statistics": result["statistics"],
                    "rationale": f"TFT Model - 상위 확률 구간 분석 완료",
                    "model_type": "QuantileTransformerRegressor"
                }
            return {"error": "Prediction failed", "point_estimate": 0, "confidence": "low"}
        except Exception as e:
            return {"error": str(e), "point_estimate": 0, "confidence": "error"}


# 어댑터 및 RAG 파이프라인 생성
adapter = TFTPredictorAdapter(tft_predictor)

print("🚀 RAG 파이프라인 초기화...")
rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus",
    index_dir="./rag_index",
    award_predict_fn=adapter.predict  # ✅ 인자명을 award_predict_fn으로 통일
)

print("=" * 60)
print("🎉 모든 초기화 완료! 서버 시작 준비됨")
print("=" * 60)

# ==========================================
# 3. FastAPI 서버 및 로직 (기존 유지)
# ==========================================
app = FastAPI(title="Integrated Bid Prediction API with TFT")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

load_dotenv()
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = "uploads"


def upload_to_azure(file_path, file_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=file_name)
        with open(file=file_path, mode="rb") as data:
            blob_client.upload_blob(data, overwrite=True, content_type="application/pdf")
        return blob_client.url
    except Exception as e:
        print(f"❌ Azure 업로드 실패: {e}")
        return str(e)


def generate_pdf(report_text, output_path):
    try:
        pdf = FPDF()
        pdf.add_page()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(current_dir, "NanumGothic-Regular.ttf")
        if not os.path.exists(font_path): raise FileNotFoundError(f"폰트 없음: {font_path}")
        pdf.add_font("Nanum", "", font_path)
        pdf.set_font("Nanum", size=11)
        clean_text = report_text.replace("#", "").replace("*", "").replace(">", "").replace("- ", "• ").strip()
        pdf.multi_cell(0, 8, txt=clean_text)
        pdf.output(output_path)
    except Exception as e:
        print(f"❌ PDF 생성 실패: {e}");
        raise e


@app.post("/analyze")
async def analyze(req: Dict[str, Any]):
    try:
        # RAG 파이프라인 분석 수행
        result = rag_pipeline.analyze(req.get("text", ""), thread_id=req.get("thread_id", "default"))

        extracted = result.get("requirements", {})
        prediction = result.get("prediction_result", {})
        report_md = result.get("report_markdown", "")

        # PDF 생성 및 업로드
        output_dir = "./output"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        pdf_filename = f"report_{uuid.uuid4().hex[:6]}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)

        final_url = None
        try:
            generate_pdf(report_md, pdf_path)
            final_url = upload_to_azure(os.path.abspath(pdf_path), pdf_filename)
        except Exception as e:
            final_url = f"PDF 생성 실패: {str(e)}"

        # ✅ 클라이언트 요구사항에 맞춘 필드 매핑 응답
        return {
            "status": "success",
            "message": "요청 성공",
            "data": {
                "bidName": extracted.get("title") or "공고명 확인 불가",
                "predictedPrice": int(prediction.get("point_estimate", 0)),
                "analysisContent": report_md,
                "pdfUrl": final_url,
                "analysisDate": datetime.now().isoformat(),
                "prediction_details": prediction  # 상세 확률 구간 포함
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"status": "running", "model": "TFT"}


if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=9999)