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
from pyngrok import ngrok
from fpdf import FPDF  # [수정] 오류가 잦은 md2pdf 대신 fpdf2 사용
from azure.storage.blob import BlobServiceClient, ContentSettings

# --- 모듈 임포트 ---
try:
    from model_transformer import TransformerRegressor
    from BidAssitanceModel import BidRAGPipeline
except ImportError as e:
    print(f"❌ 필수 모듈 로딩 실패: {e}")
    exit(1)


# ==========================================
# 0. 유틸리티 함수 및 모델 로드 로직 (기존과 동일)
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


def load_scalers_json(path: str):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transformer_model(model_path: str):
    if not os.path.exists(model_path):
        return None, {"num_features": 4, "d_model": 64}
    state_dict = torch.load(model_path, map_location='cpu')
    config = {"num_features": 4, "d_model": 64, "num_layers": 2, "dim_feedforward": 256, "nhead": 4}
    print(f"🛠 설정된 모델 구조: d_model={config['d_model']}, FFN={config['dim_feedforward']}")
    model = TransformerRegressor(
        num_features=config['num_features'], d_model=config['d_model'],
        num_layers=config['num_layers'], nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'], dropout=0.1
    )
    try:
        model.load_state_dict(state_dict, strict=True)
        print("🎉 Transformer 모델 로드 성공!")
    except RuntimeError as e:
        print(f"❌ 사이즈 에러 발생: {e}")
    model.eval()
    return model, config


# 모델/스케일러 로드
MODEL_PATH = "../results_transformer_4feat/transformer_4feat.pt"
SCALER_PATH = "../results_transformer_4feat/scalers.json"
TF_MODEL, TF_CONFIG = load_transformer_model(MODEL_PATH)
SCALER_DATA = load_scalers_json(SCALER_PATH) or {"x_mean": [0] * 4, "x_std": [1] * 4, "y_mean": 0, "y_std": 1}


# 어댑터 및 파이프라인 생성 (기존과 동일)
class TransformerPredictorAdapter:
    def __init__(self, model, scaler_data):
        self.model = model
        self.x_mean = np.array(scaler_data.get('x_mean', [0.0] * 4))
        self.x_std = np.array(scaler_data.get('x_std', [1.0] * 4))
        self.y_mean = float(scaler_data.get('y_mean', 0.0))
        self.y_std = float(scaler_data.get('y_std', 1.0))
        self.target_log = bool(scaler_data.get('target_log', False))

    def predict(self, requirements: Dict[str, Any], retrieved_context: str = "") -> Dict[str, Any]:
        try:
            estimate = parsenumber(requirements.get('estimate_price')) or 1000000.0
            budget = parsenumber(requirements.get('budget')) or estimate
            pr_range = parsenumber(requirements.get('expected_price_range')) or 0.0
            lower_rate = parsenumber(requirements.get('award_lower_rate')) or 0.0
            features = np.array([budget, estimate, pr_range, lower_rate])
            scaled_features = (features - self.x_mean) / self.x_std
            final_pred = estimate
            if self.model:
                input_tensor = torch.tensor(scaled_features, dtype=torch.float32).reshape(1, -1, 1)
                with torch.no_grad():
                    output = self.model(input_tensor)
                    pred_s = output[0].item() if isinstance(output, (tuple, list)) else output.item()
                pred_log = pred_s * self.y_std + self.y_mean
                final_pred = np.expm1(pred_log) if self.target_log else pred_log
            point_estimate = int(round(final_pred))
            return {
                "currency": "KRW", "point_estimate": point_estimate,
                "predicted_min": int(point_estimate * 0.98), "predicted_max": int(point_estimate * 1.02),
                "confidence": "high", "rationale": "Transformer 분석 완료", "model_type": "Transformer"
            }
        except Exception as e:
            return {"point_estimate": 0, "confidence": "error", "rationale": str(e)}


adapter = TransformerPredictorAdapter(TF_MODEL, SCALER_DATA)
rag_pipeline = BidRAGPipeline(doc_dir="./rag_corpus", index_dir="./rag_index", award_predict_fn=adapter.predict)

# ==========================================
# 3. FastAPI 서버 및 PDF 생성 로직
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --- Azure Blob Storage 설정 ---
load_dotenv()
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = "uploads"

if not AZURE_STORAGE_CONNECTION_STRING:
    raise ValueError("❌환경변수 'AZURE_STORAGE_CONNECTION_STRING'이 설정되지 않았습니다!")

def upload_to_azure(file_path, file_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=file_name)

        # 파일 업로드
        with open(file=file_path, mode="rb") as data:
            blob_client.upload_blob(data, overwrite=True, content_type="application/pdf") # 다운X 브라우저에서 바로 열람
        return blob_client.url
    except Exception as e:
        print(f"Azure 업로드 실패: {e}")
        return e

def generate_pdf(report_text, output_path):
    """fpdf2 OS/2 에러 완벽 해결 버전 (로컬 폰트 사용)"""
    try:
        pdf = FPDF()
        pdf.add_page()

        # 1. 시스템 폰트 대신 프로젝트 폴더 내의 폰트 파일을 직접 지정
        # NanumGothic-Regular.ttf 파일을 RAG_server.py와 같은 위치에 두세요.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = os.path.join(current_dir, "NanumGothic-Regular.ttf")

        # 만약 파일이 없다면 에러를 미리 출력하여 안내
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"폰트 파일이 없습니다: {font_path} (나눔고딕을 다운로드해 폴더에 넣어주세요)")

        # 2. 폰트 등록 및 설정
        pdf.add_font("Nanum", "", font_path)
        pdf.set_font("Nanum", size=11)

        # 3. 텍스트 정제 (불필요한 특수기호 제거)
        clean_text = report_text.replace("#", "").replace("*", "").replace(">", "").replace("- ", "• ").strip()

        # 4. 출력 (OS/2 에러 방지를 위해 latin-1 체크 우회)
        # fpdf2의 multi_cell은 유니코드를 기본적으로 지원합니다.
        pdf.multi_cell(0, 8, txt=clean_text)

        pdf.output(output_path)
    except Exception as e:
        print(f"❌ [Internal generate_pdf Error] : {e}")
        raise e


@app.post("/analyze")
async def analyze(req: Dict[str, Any]):
    try:
        # 1. 분석 수행
        result = rag_pipeline.analyze(req.get("text", ""), thread_id=req.get("thread_id", "default"))
        report_md = result.get("report_markdown", "")

        # 2. PDF 저장 폴더 준비
        output_dir = "./output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pdf_filename = f"report_{uuid.uuid4().hex[:6]}.pdf"
        pdf_path = os.path.join(output_dir, pdf_filename)

        # 3. PDF 생성 시도
        try:
            if not report_md:
                raise ValueError("리포트 생성 실패: 마크다운 내용이 없습니다.")

            generate_pdf(report_md, pdf_path)
            full_pdf_path = os.path.abspath(pdf_path)
            print(f"✅ PDF 생성 성공: {full_pdf_path}")

            final_url = upload_to_azure(full_pdf_path, pdf_filename)
            print(f"Azure 업로드 URL: {final_url}")
        except Exception as e:
            # 실패 시 상세 원인을 JSON 응답에 포함
            print(f"❌ PDF 생성 단계 최종 실패: {e}")
            final_url = f"PDF 생성 실패: {str(e)}"

        return {
            "extracted_requirements": result.get("requirements", {}),
            "prediction": result.get("prediction_result", {}),
            "report": report_md,
            "pdf_link": final_url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=9999)

여기 코드는 됐지 그러면?