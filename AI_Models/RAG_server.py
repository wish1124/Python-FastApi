import math
import uvicorn
import torch
import re
import nest_asyncio
import os
import json
import numpy as np
import uuid
import tempfile
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fpdf import FPDF
from azure.storage.blob import BlobServiceClient, ContentSettings
from datetime import datetime

# --- 모듈 임포트 ---
try:
    from BidAssitanceModel import (
        BidRAGPipeline,
        extract_text_from_hwp,
        extract_text_from_hwpx,
        extract_text_from_pdf
    )
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
# 1. TFT 모델 로드
# ==========================================
TFT_MODEL_PATH = "./results_transformer_4feat/best_model.pt"
try:
    tft_predictor = ProbabilityPredictor(model_path=TFT_MODEL_PATH)
    print("✅ TFT 모델 로드 성공")
except Exception as e:
    print(f"⚠️ TFT 모델 로드 실패: {e}")
    tft_predictor = None


# ==========================================
# 2. TFT 예측 어댑터 (top_ranges 포함)
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

                # 🔍 디버그: top_ranges 상세 출력
                print("=" * 60)
                print(" [DEBUG] TFTPredictorAdapter - top_ranges 상세:")
                for i, r in enumerate(top_ranges[:3]):
                    print(f"  {i+1}순위:")
                    print(f"    range: {r.get('range')}")
                    print(f"    range_display: {r.get('range_display')}")
                    print(f"    center: {r.get('center')}")
                    print(f"    rate: {r.get('rate')}")
                    print(f"    probability: {r.get('probability')}")
                    print(f"    lower: {r.get('lower')}")
                    print(f"    upper: {r.get('upper')}")
                print("=" * 60)

                # 낙찰가 계산: 기초금액 × 배율(1+사정율) × 낙찰하한율
                # center는 배율 (1 + 사정율) 형태
                pred_multiplier = float(top_ranges[0]["center"])  # 배율
                award_price = round(budget * pred_multiplier * lower_rate) if (budget and lower_rate) else None

                return {
                    "currency": "KRW",
                    "point_estimate": award_price,  # 원 단위 낙찰가
                    "predicted_sashiritsu": abs(pred_multiplier - 1),  # 사정율 (배율에서 변환)
                    "predicted_min": abs(result["statistics"]["q25"] - 1),  # 사정율 하한
                    "predicted_max": abs(result["statistics"]["q75"] - 1),  # 사정율 상한
                    "confidence": "high",
                    "top_ranges": top_ranges,
                    "statistics": result["statistics"],
                    "rationale": f"TFT Model - Top {len(top_ranges)} 확률 구간 분석 완료",
                    "model_type": "QuantileTransformerRegressor"
                }
            else:
                return {
                    "error": "Prediction failed",
                    "point_estimate": 0,
                    "confidence": "low",
                    "rationale": "TFT 예측 결과 없음"
                }

        except Exception as e:
            print(f"❌ TFT 예측 오류: {e}")
            return {
                "error": str(e),
                "point_estimate": 0,
                "confidence": "error",
                "rationale": f"Prediction Failed: {str(e)}"
            }


# 어댑터 및 파이프라인 생성
adapter = TFTPredictorAdapter(tft_predictor)
print("🚀 RAG 파이프라인 초기화...")
rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus",
    index_dir="./rag_index",
    award_predict_fn=adapter.predict  # ✅ TFT 어댑터 주입
)

# ==========================================
# 3. FastAPI 서버 및 PDF 생성 로직
# ==========================================
app = FastAPI(title="Integrated Bid Prediction API with TFT")
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
async def analyze(request: Request):
    """입찰공고 분석 + TFT 예측 + PDF 생성 + Azure 업로드

    지원하는 요청 방식:
    1. JSON body: {"text": "공고문 내용", "thread_id": "optional"}
    2. Form + File: file 업로드 (.hwp, .hwpx, .pdf, .txt)
    """
    try:
        bid_text = ""
        thread_id = "default"

        # Content-Type 확인
        content_type = request.headers.get("content-type", "")
        print(f"📥 수신된 Content-Type: {content_type}")

        # 1. JSON 요청 처리 (기존 Spring Boot 방식)
        if "application/json" in content_type:
            try:
                body = await request.json()
                print(f"📦 JSON Body: {body}")
                bid_text = body.get("text", "")
                thread_id = body.get("thread_id", "default")
                print(f"✅ JSON 요청 수신: 텍스트 {len(bid_text)} 글자, thread_id={thread_id}")
            except Exception as e:
                print(f"❌ JSON 파싱 실패: {e}")
                raise HTTPException(status_code=400, detail=f"JSON 파싱 실패: {str(e)}")

        # 2. 파일 업로드 처리
        elif "multipart/form-data" in content_type:
            form = await request.form()

            # thread_id 가져오기
            thread_id = form.get("thread_id", "default")

            # 텍스트 직접 입력 확인
            text_input = form.get("text")
            if text_input:
                bid_text = str(text_input)
                print(f"✅ Form 텍스트 입력: {len(bid_text)} 글자")

            # 파일 업로드 확인 (여러 파일 지원)
            all_file_texts = []

            # form.getlist()나 여러 키로 파일 가져오기
            files_to_process = []
            for key in form.keys():
                value = form.get(key)
                # file, file1, file2, files 등 다양한 키 이름 지원
                if hasattr(value, "filename") and value.filename:
                    files_to_process.append(value)

            print(f"📦 업로드된 파일 개수: {len(files_to_process)}")

            for file in files_to_process:
                filename = file.filename.lower()
                print(f"📄 처리 중: {filename}")

                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
                    content = await file.read()
                    tmp_file.write(content)
                    tmp_path = tmp_file.name

                try:
                    extracted_text = ""

                    # 확장자에 따라 텍스트 추출
                    if filename.endswith('.hwp'):
                        print(f"  📄 HWP 파일 추출 중...")
                        extracted_text = extract_text_from_hwp(tmp_path)
                    elif filename.endswith('.hwpx'):
                        print(f"  📄 HWPX 파일 추출 중...")
                        extracted_text = extract_text_from_hwpx(tmp_path)
                    elif filename.endswith('.pdf'):
                        print(f"  📄 PDF 파일 추출 중...")
                        extracted_text = extract_text_from_pdf(tmp_path)
                    elif filename.endswith('.txt'):
                        print(f"  📄 TXT 파일 읽기 중...")
                        with open(tmp_path, 'r', encoding='utf-8', errors='ignore') as f:
                            extracted_text = f.read()
                    else:
                        print(f"  ⚠️ 지원하지 않는 파일 형식: {filename} (건너뜀)")
                        continue  # 지원하지 않는 파일은 건너뜀

                    if extracted_text and len(extracted_text.strip()) > 10:
                        all_file_texts.append(f"\n\n{'=' * 60}\n파일: {filename}\n{'=' * 60}\n{extracted_text}")
                        print(f"  ✅ 추출 완료: {len(extracted_text)} 글자")
                    else:
                        print(f"  ⚠️ 텍스트 추출 실패 또는 내용 없음")

                finally:
                    # 임시 파일 삭제
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

            # 모든 파일의 텍스트를 합침
            if all_file_texts:
                if bid_text:  # 직접 입력한 텍스트가 있으면 맨 앞에
                    bid_text = bid_text + "\n\n" + "\n\n".join(all_file_texts)
                else:
                    bid_text = "\n\n".join(all_file_texts)
                print(f"✅ 전체 텍스트 통합 완료: {len(bid_text)} 글자 (파일 {len(all_file_texts)}개)")

        else:
            print(f"❌ 지원하지 않는 Content-Type: {content_type}")
            raise HTTPException(
                status_code=400,
                detail="Content-Type은 'application/json' 또는 'multipart/form-data'이어야 합니다."
            )

        # 텍스트 검증
        print(f"🔍 텍스트 검증: 길이={len(bid_text)}, 공백제거 길이={len(bid_text.strip())}")
        if not bid_text or len(bid_text.strip()) < 50:
            print(f"❌ 텍스트가 너무 짧음: {len(bid_text.strip())} 글자")
            raise HTTPException(
                status_code=400,
                detail=f"텍스트가 없거나 너무 짧습니다. (현재 {len(bid_text.strip())}자, 최소 50자 이상 필요)"
            )

        # 3. RAG 파이프라인 분석 수행
        result = rag_pipeline.analyze(
            bid_text,
            thread_id=thread_id
        )

        report_md = result.get("report_markdown", "")
        prediction_result = result.get("prediction_result", {})

        # 4. PDF 저장 폴더 준비
        output_dir = "./output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pdf_filename = f"report_{uuid.uuid4().hex[:6]}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)

        # 5. PDF 생성 및 Azure 업로드
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

        # 6. 응답 반환
        return {
            "extracted_requirements": result.get("requirements", {}),
            "prediction": prediction_result,  # ✅ top_ranges 포함됨
            "report": report_md,
            "pdf_link": final_url
        }

    except HTTPException as he:
        print(f"❌ HTTPException 발생: status={he.status_code}, detail={he.detail}")
        raise
    except Exception as e:
        print(f"❌ /analyze 예상치 못한 오류: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predictBase")
async def predict_base(req: Dict[str, List[float]]):
    """직접 예측 API (TFT 모델)"""
    if not tft_predictor:
        return {"error": "TFT Model not loaded", "predBid": 0}

    try:
        features = req.get("features", [])
        if len(features) != 4:
            return {"error": "4개의 feature가 필요합니다", "predBid": 0}

        input_dict = {
            '예가범위': features[0],
            '낙찰하한율': features[1],
            '추정가격': features[2],
            '기초금액': features[3]
        }

        result = tft_predictor.get_highest_probability_ranges(input_dict, bin_width=0.001, top_k=3)

        if result and result.get("top_ranges"):
            top_ranges = result["top_ranges"]
            budget = features[3]  # 기초금액

            # 낙찰가 계산: 기초금액 × (1 + 사정율) × 낙찰하한율
            lower_rate = features[1]  # 낙찰하한율
            pred_sashiritsu = top_ranges[0]["center"]
            award_price = round(budget * (1 + pred_sashiritsu) * lower_rate)
            award_min = round(budget * (1 + result["statistics"]["q25"]) * lower_rate)
            award_max = round(budget * (1 + result["statistics"]["q75"]) * lower_rate)

            return {
                "predBid": pred_sashiritsu,  # 사정율
                "award_price": award_price,  # 원 단위 낙찰가
                "award_min": award_min,  # 낙찰가 하한 (q25)
                "award_max": award_max,  # 낙찰가 상한 (q75)
                "top_ranges": top_ranges,
                "statistics": result["statistics"]
            }
        else:
            return {"error": "예측 실패", "predBid": 0}

    except Exception as e:
        return {"error": str(e), "predBid": 0}


@app.get("/")
def root():
    """서버 상태 확인"""
    return {
        "status": "running",
        "model": "TFT (Quantile Transformer)",
        "features": ["top_ranges", "PDF generation", "Azure upload", "File upload (.hwp, .hwpx, .pdf, .txt)"]
    }


if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=9999)