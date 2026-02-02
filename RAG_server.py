import uvicorn
import torch
import re
import nest_asyncio
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fpdf import FPDF
from azure.storage.blob import BlobServiceClient

# [1] 커스텀 모듈 임포트
try:
    from BidAssitanceModel import BidRAGPipeline
    from get_probability_from_model import ProbabilityPredictor
except ImportError as e:
    print(f"❌ 필수 모듈 로딩 실패: {e}")
    exit(1)

load_dotenv()

# ==========================================
# [2] 경로 및 유틸리티 설정
# ==========================================
# 실행 파일의 부모 디렉토리를 기준으로 절대 경로 설정
BASE_DIR = Path(__file__).parent.absolute()
# 이미지에 명시된 경로 반영: results_transformer/best_model.pt
TFT_MODEL_PATH = str(BASE_DIR / 'results_transformer' / 'best_model.pt')


def parsenumber(value: Any) -> Optional[float]:
    """다양한 형태의 숫자 문자열을 float로 변환"""
    if value is None: return None
    if isinstance(value, (int, float)): return float(value)
    s = str(value).strip()
    # 숫자, 점(.), 마이너스(-)만 남기고 제거
    s = re.sub(r'[^0-9.\-]', '', s.replace(',', ''))
    try:
        return float(s)
    except:
        return None


def generate_pdf(report_md: str, prediction: dict, output_path: str):
    """상세 예측 결과가 포함된 한글 PDF 생성"""
    try:
        pdf = FPDF()
        pdf.add_page()

        # NanumGothic-Regular.ttf 파일이 /app 경로에 있어야 함
        font_path = os.path.join(os.getcwd(), "NanumGothic-Regular.ttf")
        if os.path.exists(font_path):
            pdf.add_font("Nanum", "", font_path)
            pdf.set_font("Nanum", size=11)
        else:
            # 한글 폰트가 없을 경우 기본 폰트로 대체 (한글 깨짐 주의)
            pdf.set_font("Arial", size=11)

        # 마크다운 특수문자 제거 및 본문 작성
        clean_text = report_md.replace("#", "").replace("*", "").replace(">", "").strip()
        pdf.multi_cell(0, 10, txt=clean_text)

        # 상위 확률 구간 명시적 추가
        if "top_ranges" in prediction:
            pdf.ln(10)
            if os.path.exists(font_path): pdf.set_font("Nanum", size=12)
            pdf.cell(0, 10, txt="[모델 기반 사정율 확률 구간 분석]", ln=True)
            if os.path.exists(font_path): pdf.set_font("Nanum", size=10)
            for i, r in enumerate(prediction.get("top_ranges", [])):
                pdf.cell(0, 8, txt=f"{i + 1}위: {r['range']} (확률: {r['prob'] * 100:.2f}%)", ln=True)

        pdf.output(output_path)
        return True
    except Exception as e:
        print(f"❌ PDF 생성 실패: {e}")
        return False


def upload_to_azure(file_path: str, blob_name: str):
    """Azure Blob Storage에 파일 업로드"""
    try:
        conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not conn_str:
            print("⚠️ Azure 연결 문자열이 설정되지 않았습니다.")
            return None
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        # 컨테이너 명은 'uploads'로 고정 (Azure에서 미리 생성 필요)
        blob_client = blob_service_client.get_blob_client(container="uploads", blob=blob_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True, content_type="application/pdf")
        return blob_client.url
    except Exception as e:
        print(f"❌ Azure 업로드 실패: {e}")
        return None


# ==========================================
# [3] TFTPredictorAdapter 클래스 정의
# ==========================================
class TFTPredictorAdapter:
    """RAG 파이프라인에서 사용할 Transformer 모델 어댑터"""

    def __init__(self, predictor):
        self.predictor = predictor

    def predict(self, requirements: Dict[str, Any], retrieved_context: str = "") -> Dict[str, Any]:
        """입찰 요구사항을 기반으로 Transformer 모델로 예측 수행"""
        try:
            if not self.predictor:
                return {"error": "Model not loaded", "confidence": "error"}

            # 데이터 정규화
            input_dict = {
                '예가범위': parsenumber(requirements.get('expected_price_range')) or 0.0,
                '낙찰하한율': parsenumber(requirements.get('award_lower_rate')) or 0.0,
                '추정가격': parsenumber(requirements.get('estimate_price')) or 0.0,
                '기초금액': parsenumber(requirements.get('budget')) or 0.0
            }

            # 확률 높은 상위 3개 구간 예측
            result = self.predictor.get_highest_probability_ranges(input_dict, bin_width=100000, top_k=3)

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
                    "rationale": "Transformer Model 분석 완료",
                    "model_type": "QuantileTransformer"
                }
            return {"error": "Prediction failed", "confidence": "low"}
        except Exception as e:
            return {"error": str(e), "confidence": "error"}


# ==========================================
# [4] 모델 로드 및 파이프라인 주입
# ==========================================
tft_predictor = None
adapter = None

print("=" * 60)
print(f"🔍 Transformer 모델 로드 시도 경로: {TFT_MODEL_PATH}")

try:
    if os.path.exists(TFT_MODEL_PATH):
        # Transformer 모델 로드
        tft_predictor = ProbabilityPredictor(model_path=TFT_MODEL_PATH)
        # Adapter 생성
        adapter = TFTPredictorAdapter(tft_predictor)
        print(f"✅ Transformer 모델 로드 성공!")
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {TFT_MODEL_PATH}")
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
print("=" * 60)

# RAG 파이프라인 초기화 및 어댑터 주입
# adapter가 성공적으로 생성된 경우에만 predict 함수를 주입함
rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus",
    index_dir="./rag_index",
    award_predict_fn=adapter.predict if adapter else None
)

# ==========================================
# [5] FastAPI 서버 및 엔드포인트
# ==========================================
app = FastAPI(title="Integrated Bid Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyzeReq(BaseModel):
    text: str
    id: Optional[int] = 0
    bidId: Optional[int] = 0


class PredictReq(BaseModel):
    features: List[float]  # [예가범위, 낙찰하한율, 추정가격, 기초금액]


@app.post("/predictBase")
async def predict_base(req: PredictReq):
    """직접 숫자 입력 예측 API (수치 전용)"""
    if not tft_predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        if len(req.features) != 4:
            return {"error": "4개의 피처가 필요합니다. [예가, 하한율, 추정가, 기초가]", "predBid": 0}

        input_dict = {
            '예가범위': req.features[0],
            '낙찰하한율': req.features[1],
            '추정가격': req.features[2],
            '기초금액': req.features[3]
        }

        result = tft_predictor.get_highest_probability_ranges(input_dict, bin_width=0.001, top_k=3)
        return {
            "predBid": int(result["top_ranges"][0]["center"]),
            "top_ranges": result["top_ranges"],
            "statistics": result["statistics"]
        }
    except Exception as e:
        return {"error": str(e), "predBid": 0}


@app.post("/analyze")
async def analyze(req: AnalyzeReq):
    """텍스트 분석 + Transformer 예측 + 리포트 생성 + PDF + Azure 업로드 통합"""
    try:
        # RAG 분석 실행 (추출 및 예측 수행)
        result = rag_pipeline.analyze(req.text)

        report_md = result.get("report_markdown", "")
        prediction = result.get("prediction_result", {})
        extracted = result.get("requirements", {})

        # PDF 저장 폴더 준비
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # 고유 파일명 생성
        pdf_filename = f"report_{uuid.uuid4().hex[:6]}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)

        # PDF 생성 및 Azure 업로드 로직 실행
        pdf_url = None
        if generate_pdf(report_md, prediction, pdf_path):
            pdf_url = upload_to_azure(pdf_path, pdf_filename)

        # 백엔드 규격에 맞춘 최종 데이터 반환
        return {
            "status": "success",
            "data": {
                "id": req.id,
                "bidId": req.bidId,
                "bidName": extracted.get("title", "공고명 미확인"),
                "predictedPrice": prediction.get("point_estimate", 0),
                "analysisContent": report_md,
                "pdfUrl": pdf_url,
                "analysisDate": datetime.now().isoformat(),
                "extracted_requirements": extracted,
                "prediction": prediction
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    """서버 상태 확인용 Root 엔드포인트"""
    return {
        "status": "running",
        "model_loaded": tft_predictor is not None,
        "model_path": TFT_MODEL_PATH
    }


if __name__ == "__main__":
    # Azure App Service 대응을 위해 PORT 환경 변수 처리
    port = int(os.environ.get("PORT", 9999))
    uvicorn.run(app, host="0.0.0.0", port=port)