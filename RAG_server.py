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
import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fpdf import FPDF
from azure.storage.blob import BlobServiceClient, ContentSettings
from datetime import datetime
from security_logger import SecurityLogMiddleware
from rate_limit import RateLimitMiddleware
from tft_v3_predictor import predict_sajeong_percent, FEATURES, get_highest_probability_ranges_v3



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
# TFT_MODEL_PATH = "./results_transformer_4feat/best_model.pt"
# try:
#     tft_predictor = ProbabilityPredictor(model_path=TFT_MODEL_PATH)
#     print("✅ TFT 모델 로드 성공")
# except Exception as e:
#     print(f"⚠️ TFT 모델 로드 실패: {e}")
#     tft_predictor = None

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
                pred_multiplier = float(top_ranges[0]["center"])

                # center가 99.xx 같은 퍼센트로 들어오는 경우 방어
                if pred_multiplier > 2:
                    pred_multiplier /= 100.0

                # 낙찰가 = 기초금액 × 투찰배율(99%)
                award_price = round(budget * pred_multiplier) if budget else None

                # 퍼센트는 금액에서 역산 → 항상 일치
                predicted_percent = (award_price / budget) * 100 if (award_price and budget) else None

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
# =========================================================
# [모델v2추가] pkl 모델 + scaler 로드 (투찰율/사정율 % 예측용)
# =========================================================
# V2_MODEL_PATH = "./model/model_v2.pkl"
# V2_SCALER_PATH = "./model/scaler2.pkl"
#
# v2_model = None
# v2_scaler = None
#
# try:
#     if os.path.exists(V2_MODEL_PATH) and os.path.exists(V2_SCALER_PATH):
#         v2_model = joblib.load(V2_MODEL_PATH)
#         v2_scaler = joblib.load(V2_SCALER_PATH)
#         print("✅ V2(pkl) 모델/스케일러 로드 성공")
#     else:
#         print("⚠️ V2(pkl) 파일 없음 → 기존 모델만 사용")
# except Exception as e:
#     print(f"⚠️ V2(pkl) 로드 실패: {e}")
#     v2_model, v2_scaler = None, None
#
# except Exception as e:
#     print(f"⚠️ V2(pkl) 로드 실패: {e}")
#     v2_model, v2_scaler = None, None


# =========================================================
# [모델v2추가] RAG 파이프라인용 예측 함수
# =========================================================
# def v2_award_predict(requirements: Dict[str, Any], retrieved_context: str = "") -> Dict[str, Any]:
#     try:
#         if v2_model is None or v2_scaler is None:
#             return {
#                 "error": "V2 model/scaler not loaded",
#                 "point_estimate": 0,
#                 "confidence": "error",
#                 "rationale": "V2 Model not loaded"
#             }
#
#         pr_range = parsenumber(requirements.get('expected_price_range')) or 0.0
#         lower_rate = parsenumber(requirements.get('award_lower_rate')) or 0.0
#         estimate = parsenumber(requirements.get('estimate_price')) or 0.0
#         budget = parsenumber(requirements.get('budget')) or 0.0
#
#         # =========================================================
#         # [보정추가] % 형태로 들어오면 소수로 변환 (89.745 -> 0.89745)
#         # =========================================================
#         if lower_rate > 1:
#             lower_rate = lower_rate / 100.0
#         if pr_range > 1:
#             pr_range = pr_range / 100.0
#
#         x = np.array([[pr_range, lower_rate, estimate, budget]], dtype=float)
#         x_scaled = v2_scaler.transform(x)
#         y_pred_transformed = float(v2_model.predict(x_scaled)[0])
#
#         pred_percent = (y_pred_transformed / 100.0) + 100.0
#         pred_multiplier = pred_percent / 100.0
#
#         award_price = round(budget * pred_multiplier * lower_rate) if (budget and lower_rate) else None
#
#         return {
#             "currency": "KRW",
#             "point_estimate": award_price,
#             "predicted_percent": pred_percent,
#             "confidence": "high",
#             "rationale": "V2(pkl) Model prediction",
#             "model_type": "v2_pkl",
#             "y_pred_transformed": y_pred_transformed  # [선택] PDF 근거에 쓰려면 유지
#         }
#     except Exception as e:
#         return {
#             "error": str(e),
#             "point_estimate": 0,
#             "confidence": "error",
#             "rationale": f"V2 Prediction Failed: {str(e)}"
#         }
# =========================================================
# [모델v3추가] PT + scaler_X + features(22) 기반 예측 함수
# - 이번 모델 출력은 0.98~0.99 같은 "배율(multiplier)"로 가정 (1을 더하지 않음)
# =========================================================
def v3_award_predict(requirements: Dict[str, Any], retrieved_context: str = "") -> Dict[str, Any]:
    try:
        pr_range = parsenumber(requirements.get('expected_price_range')) or 0.0
        lower_rate_raw = parsenumber(requirements.get('award_lower_rate')) or 0.0
        estimate = parsenumber(requirements.get('estimate_price')) or 0.0
        budget = parsenumber(requirements.get('budget')) or 0.0

        lower_rate = lower_rate_raw
        if lower_rate > 1:
            lower_rate = lower_rate / 100.0

        feat = {name: 0.0 for name in FEATURES}
        if "예가범위" in feat: feat["예가범위"] = float(pr_range)
        if "낙찰하한율" in feat: feat["낙찰하한율"] = float(lower_rate_raw)
        if "추정가격" in feat: feat["추정가격"] = float(estimate)
        if "기초금액" in feat: feat["기초금액"] = float(budget)

        # ✅ top3 확률 구간
        dist = get_highest_probability_ranges_v3(feat, bin_width=0.5, top_k=3)
        top_ranges = dist.get("top_ranges", [])
        statistics = dist.get("statistics", {})

        # ✅ 중앙값 예측(배율)
        pred_multiplier = float(predict_sajeong_percent(feat))
        if pred_multiplier > 2:
            pred_multiplier /= 100.0

        award_price = round(budget * pred_multiplier) if budget else None
        predicted_percent = (award_price / budget) * 100 if (award_price and budget) else None
        lower_bound_price = round(budget * pred_multiplier * lower_rate) if (budget and lower_rate) else None

        # ----------------------------------------------------
        # 🔥 (중요) top_ranges가 배율(1.00xx)로 오면 %로 변환해서 보고서에 예쁘게 출력
        # ----------------------------------------------------
        converted = []
        for r in top_ranges:
            center = float(r.get("center", 0.0))
            low = float(r.get("lower", 0.0))
            high = float(r.get("upper", 0.0))

            # 배율이면 %로 변환
            if center <= 2.0:
                center *= 100.0
                low *= 100.0
                high *= 100.0

            prob = float(r.get("probability", 0.0))  # 이미 %라고 가정

            converted.append({
                **r,
                "center": round(center, 2),
                "lower": low,
                "upper": high,
                "range": [low, high],
                "range_display": f"{high:.1f}% ~ {low:.1f}%",
                "probability": round(prob, 2),
            })
        top_ranges = converted

        return {
            "currency": "KRW",
            "point_estimate": award_price,
            "predicted_percent": predicted_percent,
            "confidence": "high",
            "rationale": "TFT v3(pt) median quantile prediction (multiplier)",
            "model_type": "v3_pt",
            "pred_multiplier": pred_multiplier,
            "lower_bound_price": lower_bound_price,
            "top_ranges": top_ranges,
            "statistics": statistics
        }

    except Exception as e:
        return {
            "error": str(e),
            "point_estimate": 0,
            "confidence": "error",
            "rationale": f"V3 Prediction Failed: {str(e)}"
        }



# 어댑터 및 파이프라인 생성
adapter = TFTPredictorAdapter(tft_predictor)
print("🚀 RAG 파이프라인 초기화...")
rag_pipeline = BidRAGPipeline(
    doc_dir="./rag_corpus",
    index_dir="./rag_index",
    # award_predict_fn=adapter.predict
    # award_predict_fn=v2_award_predict
    award_predict_fn=v3_award_predict
)

# ==========================================
# 3. FastAPI 서버 및 PDF 생성 로직
# ==========================================
app = FastAPI(title="Integrated Bid Prediction API with TFT")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RateLimitMiddleware)      # [보안추가] 과도한 요청 차단
app.add_middleware(SecurityLogMiddleware)   # [보안추가] 접근/해킹 시도 로깅

# --- Azure Blob Storage 설정 ---
load_dotenv()
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = "uploads"



if not AZURE_STORAGE_CONNECTION_STRING:
    print("⚠️ Azure 연결 문자열 없음 - 로컬 모드로 실행")


def upload_to_azure(file_path, file_name):
    """Azure Blob Storage에 파일 업로드"""
    if not AZURE_STORAGE_CONNECTION_STRING:
        print("⚠️ Azure 미연결 → 로컬 파일 경로 사용")
        return file_path
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
    """직접 예측 API (V2 모델 적용 버전)"""

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

        # =========================================================
        # [기존모델주석] TFT 모델 예측 로직 (임시 비활성화)
        # =========================================================
        # result = tft_predictor.get_highest_probability_ranges(input_dict, bin_width=0.001, top_k=3)
        #
        # if result and result.get("top_ranges"):
        #     top_ranges = result["top_ranges"]
        #     budget = features[3]
        #
        #     lower_rate = features[1]
        #     pred_sashiritsu = top_ranges[0]["center"]
        #     award_price = round(budget * (1 + pred_sashiritsu) * lower_rate)
        #     award_min = round(budget * (1 + result["statistics"]["q25"]) * lower_rate)
        #     award_max = round(budget * (1 + result["statistics"]["q75"]) * lower_rate)
        #
        #     return {
        #         "predBid": pred_sashiritsu,
        #         "award_price": award_price,
        #         "award_min": award_min,
        #         "award_max": award_max,
        #         "top_ranges": top_ranges,
        #         "statistics": result["statistics"]
        #     }
        # else:
        #     return {"error": "예측 실패", "predBid": 0}

        # =========================================================
        # [모델v2추가] pkl 모델 사용 (투찰율 % 예측)
        # =========================================================
        # if v2_model is None or v2_scaler is None:
        #     return {"error": "V2 model/scaler not loaded", "predBid": 0}
        #
        # x = np.array([[features[0], features[1], features[2], features[3]]], dtype=float)
        # x_scaled = v2_scaler.transform(x)
        # y_pred_transformed = float(v2_model.predict(x_scaled)[0])
        #
        # # 역산 (지시사항)
        # pred_percent = (y_pred_transformed / 100.0) + 100.0
        #
        # return {
        #     "predBid": pred_percent,  # 투찰율 % (101.xx 형태)
        #     "model": "v2",
        #     "y_pred_transformed": y_pred_transformed  # 디버깅용
        # }

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

## 보안 부분 일단은 백업