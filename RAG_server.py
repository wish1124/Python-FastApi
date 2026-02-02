import uvicorn
import nest_asyncio
import os
import json
import uuid
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pyngrok import ngrok
from fpdf import FPDF
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

# 최신 BidAssitanceModel 임포트
try:
    from BidAssitanceModel import BidRAGPipeline
except ImportError as e:
    print(f"❌ 모듈 로딩 실패: {e}")
    exit(1)

load_dotenv()


# ==========================================
# 1. 누락된 유틸리티 (PDF & Azure)
# ==========================================
def generate_pdf(report_md: str, prediction: dict, output_path: str):
    """한글 폰트를 적용하여 상세 예측값이 포함된 PDF 생성"""
    try:
        pdf = FPDF()
        pdf.add_page()

        # 서버 환경에 NanumGothic-Regular.ttf 파일이 있어야 함
        font_path = "NanumGothic-Regular.ttf"
        if os.path.exists(font_path):
            pdf.add_font("Nanum", "", font_path)
            pdf.set_font("Nanum", size=11)
        else:
            pdf.set_font("Arial", size=11)

        # 텍스트 정제 및 본문 쓰기
        clean_text = report_md.replace("#", "").replace("*", "").replace(">", "").strip()
        pdf.multi_cell(0, 10, txt=clean_text)

        # 상위 확률 구간 명시적 추가 (누락 방지)
        if "top_ranges" in prediction:
            pdf.ln(10)
            pdf.set_font("Nanum" if os.path.exists(font_path) else "Arial", size=12)
            pdf.cell(0, 10, txt="[모델 기반 사정율 확률 구간 분석]", ln=True)
            pdf.set_font("Nanum" if os.path.exists(font_path) else "Arial", size=10)
            for i, r in enumerate(prediction["top_ranges"]):
                pdf.cell(0, 8, txt=f"{i + 1}위: {r['range']} (확률: {r['prob'] * 100:.2f}%)", ln=True)

        pdf.output(output_path)
        return True
    except Exception as e:
        print(f"❌ PDF 생성 오류: {e}")
        return False


def upload_to_azure(file_path: str, blob_name: str):
    """Azure Blob Storage 업로드"""
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = "uploads"
    if not conn_str: return None
    try:
        service_client = BlobServiceClient.from_connection_string(conn_str)
        blob_client = service_client.get_blob_client(container=container, blob=blob_name)
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True, content_type="application/pdf")
        return blob_client.url
    except Exception as e:
        print(f"❌ Azure 업로드 오류: {e}")
        return None


# ==========================================
# 2. 서버 및 파이프라인 초기화
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 파이프라인 객체 생성 (환경변수 TFT_MODEL_PATH 설정 권장)
rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus",
    index_dir="./rag_index"
)


class AnalyzeReq(BaseModel):
    text: str
    bidId: Optional[int] = 0
    id: Optional[int] = 0


@app.post("/analyze")
async def analyze_bid(req: AnalyzeReq):
    try:
        # 1. 분석 실행 (BidAssitanceModel의 analyze 호출)
        result = rag_pipeline.analyze(req.text)

        # 2. 결과 추출
        report_md = result.get("report_markdown", "")
        prediction = result.get("prediction_result", {})
        requirements = result.get("requirements", {})

        # 3. PDF 생성
        os.makedirs("./output", exist_ok=True)
        pdf_name = f"report_{uuid.uuid4().hex[:6]}.pdf"
        pdf_path = f"./output/{pdf_name}"

        pdf_ok = generate_pdf(report_md, prediction, pdf_path)

        # 4. Azure 업로드
        pdf_url = upload_to_azure(pdf_path, pdf_name) if pdf_ok else None

        # 5. 프론트엔드 반환
        return {
            "status": "success",
            "data": {
                "id": req.id,
                "bidId": req.bidId,
                "bidName": requirements.get("title", "알 수 없음"),
                "predictedPrice": prediction.get("point_estimate", 0),
                "analysisContent": report_md,
                "pdfUrl": pdf_url,
                "analysisDate": datetime.now().isoformat(),
                "extracted_requirements": requirements,
                "prediction": prediction
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # ngrok 설정
    NGROK_TOKEN = "38H6WIHF5Hn1xV68lPnXu15Tutc_4PDGKRtxpJhbJuVdcUCEp"
    ngrok.set_auth_token(NGROK_TOKEN)

    # [수정] 기존에 열려있는 모든 터널을 닫아서 중복 에러 방지
    tunnels = ngrok.get_tunnels()
    for t in tunnels:
        ngrok.disconnect(t.public_url)

    try:
        # 새로 연결 시도
        url = ngrok.connect(9999).public_url
        print(f"🌍 Public URL: {url}")
    except Exception as e:
        print(f"⚠️ ngrok 연결 실패 (로컬 접속만 가능): {e}")

    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=9999)