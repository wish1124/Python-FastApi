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
    """
    다양한 형태의 숫자 문자열을 float로 변환
    예: "1,000,000원" -> 1000000.0
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    s = re.sub(r'[^0-9.\-]', '', s.replace(',', ''))
    try:
        return float(s)
    except:
        return None


# ==========================================
# 1. TFT 모델 로드 (절대 경로 체크 및 초기화 로직)
# ==========================================
BASE_DIR = Path(__file__).parent.absolute()
TFT_MODEL_PATH = BASE_DIR / 'results_transformer_4feat' / 'transformer_4feat.pt'

print("=" * 60)
print("🔍 Transformer 모델 초기화 시작")
print(f"   작업 디렉토리: {os.getcwd()}")
print(f"   BASE_DIR: {BASE_DIR}")
print(f"   모델 경로: {TFT_MODEL_PATH}")
print(f"   파일 존재 여부: {TFT_MODEL_PATH.exists()}")
print("=" * 60)

# 모델 파일 존재 확인
if not TFT_MODEL_PATH.exists():
    print(f"❌ 모델 파일을 찾을 수 없습니다!")
    print(f"   경로: {TFT_MODEL_PATH}")
    print(f"   현재 디렉토리: {os.getcwd()}")
    print(f"   해결: 올바른 디렉토리에서 서버를 실행하거나 모델 파일 경로를 확인하세요.")
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {TFT_MODEL_PATH}")

# 모델 로드
try:
    print("📦 Transformer 모델 로딩 중...")
    tft_predictor = ProbabilityPredictor(model_path=str(TFT_MODEL_PATH))
    print("✅ Transformer 모델 로드 성공!")
except Exception as e:
    print(f"❌ 모델 로드 실패! 에러: {e}")
    import traceback
    traceback.print_exc()
    raise RuntimeError(f"모델 로드 실패: {e}")


# ==========================================
# 2. TFT 예측 어댑터 (top_ranges 지원)
# ==========================================
class TFTPredictorAdapter:
    """RAG 파이프라인에서 사용할 TFT 모델 어댑터 - top_ranges 지원"""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, requirements: Dict[str, Any], retrieved_context: str = "") -> Dict[str, Any]:
        """입찰 요구사항을 기반으로 TFT 모델로 예측 수행 - top_ranges 포함"""
        try:
            if not self.predictor:
                return {
                    "error": "Model not loaded",
                    "point_estimate": 0,
                    "confidence": "error",
                    "rationale": "TFT Model not loaded"
                }

            # 입력 데이터 파싱
            pr_range = parsenumber(requirements.get('expected_price_range')) or 0.0
            lower_rate = parsenumber(requirements.get('award_lower_rate')) or 0.0
            estimate = parsenumber(requirements.get('estimate_price')) or 0.0
            budget = parsenumber(requirements.get('budget')) or 0.0

            input_dict = {
                '예가범위': pr_range,
                '낙찰하한율': lower_rate,
                '추정가격': estimate,
                '기초금액': budget
            }

            # TFT 모델로 확률 높은 상위 3개 구간 예측
            result = self.predictor.get_highest_probability_ranges(
                input_dict,
                bin_width=0.001,
                top_k=3
            )

            if result and result.get("top_ranges"):
                top_ranges = result["top_ranges"]
                return {
                    "currency": "KRW",
                    "point_estimate": int(top_ranges[0]["center"]),  # 가장 확률 높은 구간의 중심값
                    "predicted_min": int(result["statistics"]["q25"]),  # 25% 분위수
                    "predicted_max": int(result["statistics"]["q75"]),  # 75% 분위수
                    "confidence": "high",
                    "top_ranges": top_ranges,  # ✅ 상위 확률 구간들
                    "statistics": result["statistics"],  # 추가 통계 정보
                    "rationale": f"Transformer Model - Top {len(top_ranges)} 확률 구간 분석 완료",
                    "model_type": "QuantileTransformerRegressor"
                }
            else:
                return {
                    "error": "Prediction failed",
                    "point_estimate": 0,
                    "confidence": "low",
                    "rationale": "예측 결과 없음"
                }

        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "point_estimate": 0,
                "confidence": "error",
                "rationale": f"Prediction Failed: {str(e)}"
            }


# ==========================================
# 어댑터 및 파이프라인 생성
# ==========================================

# Adapter 생성
try:
    print("🔧 TFTPredictorAdapter 초기화 중...")
    adapter = TFTPredictorAdapter(tft_predictor)
    print("✅ Adapter 초기화 완료!")
except Exception as e:
    print(f"❌ Adapter 초기화 실패: {e}")
    raise RuntimeError(f"Adapter 초기화 실패: {e}")

# RAG 파이프라인 초기화
try:
    print("🚀 RAG 파이프라인 초기화 중...")
    rag_pipeline = BidRAGPipeline(
        doc_dir="./rag_corpus",
        index_dir="./rag_index",
        award_predictor_instance=adapter  # ✅ 객체 자체를 주입!
    )
    print("✅ RAG 파이프라인 초기화 완료!")
except Exception as e:
    print(f"❌ RAG 파이프라인 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    raise RuntimeError(f"RAG 파이프라인 초기화 실패: {e}")

print("=" * 60)
print("🎉 모든 초기화 완료! 서버 시작 준비됨")
print("=" * 60)


# ==========================================
# 3. FastAPI 서버 및 PDF 생성 로직
# ==========================================
app = FastAPI(title="Integrated Bid Prediction API with Transformer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Azure Blob Storage 설정 ---
load_dotenv()
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = "uploads"

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("❌환경변수 'AZURE_STORAGE_CONNECTION_STRING'이 설정되지 않았습니다!")


def upload_to_azure(file_path, file_name):
    """Azure Blob Storage에 파일 업로드"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=file_name)

        with open(file=file_path, mode="rb") as data:
            blob_client.upload_blob(data, overwrite=True, content_type="application/pdf")

        print(f"✅ Azure 업로드 성공: {blob_client.url}")
        return blob_client.url
    except Exception as e:
        print(f"❌ Azure 업로드 실패: {e}")
        return str(e)


def generate_pdf(report_text, output_path):
    """fpdf2로 한글 PDF 생성"""
    try:
        pdf = FPDF()
        pdf.add_page()

        # 나눔고딕 폰트 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(current_dir, "NanumGothic-Regular.ttf")

        if not os.path.exists(font_path):
            raise FileNotFoundError(f"폰트 파일이 없습니다: {font_path}")

        pdf.add_font("Nanum", "", font_path)
        pdf.set_font("Nanum", size=11)

        # 텍스트 정제
        clean_text = report_text.replace("#", "").replace("*", "").replace(">", "").replace("- ", "• ").strip()

        pdf.multi_cell(0, 8, txt=clean_text)
        pdf.output(output_path)

        print(f"✅ PDF 생성 성공: {output_path}")
    except Exception as e:
        print(f"❌ PDF 생성 실패: {e}")
        raise e


@app.post("/analyze")
async def analyze(req: Dict[str, Any]):
    """입찰공고 분석 + TFT 예측 + PDF 생성 + Azure 업로드"""
    try:
        # 1. RAG 파이프라인 분석 수행
        result = rag_pipeline.analyze(
            req.get("text", ""),
            thread_id=req.get("thread_id", "default")
        )

        report_md = result.get("report_markdown", "")
        prediction_result = result.get("prediction_result", {})
        extracted = result.get("requirements", {})

        # 2. PDF 저장 폴더 준비
        output_dir = "./output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pdf_filename = f"report_{uuid.uuid4().hex[:6]}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)

        # 3. PDF 생성 및 Azure 업로드
        final_url = None
        try:
            if not report_md:
                raise ValueError("리포트 생성 실패: 마크다운 내용이 없습니다.")

            generate_pdf(report_md, pdf_path)
            full_pdf_path = os.path.abspath(pdf_path)

            final_url = upload_to_azure(full_pdf_path, pdf_filename)

        except Exception as e:
            print(f"❌ PDF/Azure 처리 실패: {e}")
            final_url = f"PDF 생성 실패: {str(e)}"

        # 4. 응답 반환 (명시적 필드 매핑 반영)
        return {
            "status": "success",
            "message": "요청 성공",
            "data": {
                "id": req.get("id", 0),
                "bidId": req.get("bidId", 0),
                "bidName": extracted.get("title") or "공고명 확인 불가",
                "predictedPrice": int(prediction_result.get("point_estimate", 0)),
                "analysisContent": report_md,
                "pdfUrl": str(final_url) if final_url else None,
                "attachmentUrls": req.get("attachmentUrls", None),
                "analysisDate": datetime.now().isoformat(),
                "extracted_requirements": extracted,
                "prediction": prediction_result
            }
        }

    except Exception as e:
        print(f"❌ /analyze 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"분석 도중 서버 에러가 발생했습니다: {str(e)}",
            "data": None
        }


@app.post("/predictBase")
async def predict_base(req: Dict[str, List[float]]):
    """직접 예측 API (Transformer 모델)"""
    try:
        features = req.get("features", [])
        if len(features) != 4:
            return {"error": "4개의 feature가 필요합니다 (예가범위, 낙찰하한율, 추정가격, 기초금액)", "predBid": 0}

        input_dict = {
            '예가범위': features[0],
            '낙찰하한율': features[1],
            '추정가격': features[2],
            '기초금액': features[3]
        }

        result = tft_predictor.get_highest_probability_ranges(input_dict, bin_width=0.001, top_k=3)

        if result and result.get("top_ranges"):
            top_ranges = result["top_ranges"]
            return {
                "predBid": top_ranges[0]["center"],
                "top_ranges": top_ranges,
                "statistics": result["statistics"]
            }
        else:
            return {"error": "예측 실패", "predBid": 0}

    except Exception as e:
        print(f"❌ /predictBase 오류: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "predBid": 0}


@app.get("/")
def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "model": "Quantile Transformer (4-features)",
        "features": ["top_ranges", "PDF generation", "Azure upload"],
        "model_path": str(TFT_MODEL_PATH)
    }


if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=9999)