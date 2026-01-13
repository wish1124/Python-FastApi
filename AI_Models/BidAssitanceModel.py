"""Bid Assistance RAG Pipeline (LangGraph)

공고문 -> LLM 추출 -> ToolNode(RAG+낙찰가예측+경쟁) -> LLM 리포트

요구 파일/아티팩트
-----------------
- model_1dcnn.py (사용자 업로드 코드)
- best_model.pt (학습 코드에서 저장되는 state_dict)
- scalers.json 또는 scalers.npz (필수 권장: X/y 스케일러 + target_log 설정)

의존성
------
pip install langgraph langchain-core langchain-openai langchain-community langchain-text-splitters pydantic faiss-cpu openai
# PDF 입력을 쓰면(둘 중 하나 권장):
#   pip install pypdf
#   pip install pymupdf
# CNN1D 모델을 쓰면 추가:
pip install torch numpy pandas matplotlib

환경변수
--------
OPENAI_API_KEY 설정(권장) 또는 api_key.txt에 KEY=VALUE 형식으로 저장.

CLI 사용
--------
python BidAssitanceModel_fixed_pdf.py \
  --doc_dir ./rag_corpus \
  --index_dir ./rag_index \
  --input bid_notice.txt \
  --award_model ./model_1dcnn.py \
  --award_weights ./results/best_model.pt \
  --award_scaler ./results/scalers.json

주의
----
- scalers.json(.npz)가 없으면, CNN1D 모델은 올바른 역변환이 불가능하므로
  예측을 수행하지 않고 low-confidence로 반환합니다.
"""

from __future__ import annotations

import inspect
import importlib.util
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode


# ------------------------------
# Utilities
# ------------------------------

def load_api_keys(filepath: str = "api_key.txt") -> None:
    """Load KEY=VALUE lines into os.environ (optional)."""
    if not os.path.exists(filepath):
        return
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def _parse_number(value: Any) -> Optional[float]:
    """Parse numeric values robustly (handles commas, '원', '%', etc.)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None

    s = str(value).strip()
    if not s:
        return None

    # Remove common separators/units
    s = s.replace(",", "")
    s = s.replace("원", "").replace("KRW", "").replace("₩", "")
    s = s.replace("%", "")
    # Keep digits / dot / minus only
    s = re.sub(r"[^0-9\.\-]", "", s)
    if not s or s in ("-", ".", "-."):
        return None
    try:
        return float(s)
    except Exception:
        return None


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _load_module_from_py(py_path: str):
    """Dynamically load a python module from a file path."""
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Python file not found: {py_path}")

    mod_name = f"user_mod_{abs(hash(os.path.abspath(py_path)))}"
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load python module spec: {py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module




def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file.

    - Primary backend: pypdf (pure python)
    - Fallback backend: PyMuPDF (fitz)

    If the PDF is scanned (image-only), extracted text may be empty/short.
    In that case, run OCR upstream and pass the OCR text to this pipeline.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    text = ""
    # Backend 1: pypdf
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(pdf_path)
        parts: List[str] = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        text = "\n\n".join(parts)
    except Exception:
        text = ""

    # Backend 2: PyMuPDF (fallback)
    if len(_clean_whitespace(text)) < 50:
        try:
            import fitz  # type: ignore
            doc = fitz.open(pdf_path)
            parts = []
            for page in doc:
                try:
                    parts.append(page.get_text("text") or "")
                except Exception:
                    parts.append("")
            doc.close()
            text = "\n\n".join(parts)
        except Exception:
            pass

    text = text.replace("\x00", " ")
    if len(_clean_whitespace(text)) < 50:
        try:
            import sys
            sys.stderr.write(
                "[WARN] PDF 텍스트 추출 결과가 매우 짧습니다. 스캔본 PDF일 가능성이 높습니다. "
                "OCR(예: pytesseract+pdf2image) 적용을 고려하세요.\n"
            )
        except Exception:
            pass
    return text


def read_input_text(input_path: str) -> str:
    """Read bid notice text from .txt or .pdf."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(input_path)
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()



def _callable_looks_like_wrapper(fn: Callable[..., Any]) -> bool:
    """Heuristic: wrapper predict function should accept exactly 2 positional args."""
    try:
        sig = inspect.signature(fn)
    except Exception:
        return False

    params = list(sig.parameters.values())
    # count required positional-or-keyword params without defaults
    required = [
        p for p in params
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
    ]
    # We want at least 2 (requirements, retrieved_context), and allow extra optional params
    return len(required) == 2


def _coerce_percent(v: Optional[float]) -> Optional[float]:
    """If v looks like 0~1 ratio, convert to 0~100 percent."""
    if v is None:
        return None
    if 0 < v <= 1.5:
        return v * 100.0
    return v


# ------------------------------
# Award-price predictor adapters
# ------------------------------

class AwardPricePredictor:
    """Interface so ToolNode can call the winning-price predictor safely."""
    def predict(self, requirements: Dict[str, Any], retrieved_context: str) -> Dict[str, Any]:
        raise NotImplementedError


class HeuristicAwardPricePredictor(AwardPricePredictor):
    """Fallback baseline when a user model is not wired yet."""

    def predict(self, requirements: Dict[str, Any], retrieved_context: str) -> Dict[str, Any]:
        budget = _parse_number(requirements.get("budget"))
        estimate = _parse_number(requirements.get("estimate_price"))
        base = estimate if estimate is not None else budget

        if base is None or base <= 0:
            return {
                "currency": "KRW",
                "predicted_min": None,
                "predicted_max": None,
                "point_estimate": None,
                "confidence": "low",
                "rationale": [
                    "공고문에서 예산/추정가격을 확정적으로 파악하지 못했습니다.",
                    "낙찰가 예측 모델을 사용하려면 공고의 기초금액/추정가격 등 핵심 숫자 필드를 보완해야 합니다.",
                ],
                "model": {"type": "heuristic", "name": "baseline_band"},
            }

        strictness = 0
        strictness += 1 if len(requirements.get("qualification_requirements", [])) >= 5 else 0
        strictness += 1 if len(requirements.get("performance_requirements", [])) >= 5 else 0
        strictness += 1 if any("보증" in str(s) for s in requirements.get("risk_flags", [])) else 0

        min_ratio = 0.90 - 0.01 * strictness
        max_ratio = 0.97 - 0.005 * strictness
        min_ratio = max(0.80, min_ratio)
        max_ratio = max(min_ratio + 0.03, max_ratio)

        pred_min = round(base * min_ratio)
        pred_max = round(base * max_ratio)
        point = round((pred_min + pred_max) / 2)

        return {
            "currency": "KRW",
            "predicted_min": pred_min,
            "predicted_max": pred_max,
            "point_estimate": point,
            "confidence": "medium" if strictness <= 1 else "low",
            "rationale": [
                "기초금액/추정가격을 기준(base)으로 낙찰가 밴드(비율) 추정(휴리스틱)입니다.",
                "정확한 예측은 낙찰가 예측 모델로 대체해야 합니다.",
            ],
            "used_base": base,
            "used_band": {"min_ratio": min_ratio, "max_ratio": max_ratio},
            "model": {"type": "heuristic", "name": "baseline_band"},
        }


class CallableAwardPricePredictor(AwardPricePredictor):
    def __init__(self, predict_fn: Callable[[Dict[str, Any], str], Any], model_info: Optional[Dict[str, Any]] = None):
        self.predict_fn = predict_fn
        self.model_info = model_info or {"type": "callable"}

    def predict(self, requirements: Dict[str, Any], retrieved_context: str) -> Dict[str, Any]:
        try:
            out = self.predict_fn(requirements, retrieved_context)
        except Exception as e:
            return {
                "currency": "KRW",
                "predicted_min": None,
                "predicted_max": None,
                "point_estimate": None,
                "confidence": "low",
                "rationale": [
                    "사용자 낙찰가 예측 함수 실행 중 예외가 발생했습니다.",
                    f"예외: {type(e).__name__}: {str(e)}",
                ],
                "model": self.model_info,
            }

        if isinstance(out, (int, float)):
            return {
                "currency": "KRW",
                "point_estimate": float(out),
                "predicted_min": None,
                "predicted_max": None,
                "confidence": "medium",
                "rationale": ["사용자 모델이 단일 낙찰가 예측값(포인트)을 반환했습니다."],
                "model": self.model_info,
            }
        if isinstance(out, dict):
            out.setdefault("model", self.model_info)
            return out

        return {
            "currency": "KRW",
            "predicted_min": None,
            "predicted_max": None,
            "point_estimate": None,
            "confidence": "low",
            "rationale": ["사용자 모델 출력 형식이 예상(dict/숫자)과 달라 해석할 수 없습니다."],
            "model": self.model_info,
        }


class CNN1DAwardPricePredictor(AwardPricePredictor):
    """Auto-adapter for model_1dcnn.py (1D CNN + scalers + log target)."""

    def __init__(
        self,
        module: Any,
        weights_path: str,
        scaler_path: Optional[str] = None,
        device: Optional[str] = None,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        self.module = module
        self.weights_path = weights_path
        self.scaler_path = scaler_path
        self.device = device
        self.hidden = hidden
        self.dropout = dropout

        self._torch = self._import_torch()
        self._np = self._import_numpy()

        if self.device is None:
            self.device = "cuda" if self._torch.cuda.is_available() else "cpu"

        self.model = self.module.CNN1DRegressor(hidden=self.hidden, dropout=self.dropout).to(self.device)
        self._load_weights()

        self.x_scaler = None
        self.y_scaler = None
        self.target_log = True
        self.feature_cols = ["기초금액", "추정가격", "예가범위", "낙찰하한율"]

        self._load_scalers()

    def _import_torch(self):
        try:
            import torch  # type: ignore
            return torch
        except Exception as e:
            raise RuntimeError(
                "PyTorch(torch)가 설치되어 있지 않거나 로딩에 실패했습니다. "
                "CNN1D 낙찰가 예측 모델을 사용하려면 torch가 필요합니다."
            ) from e

    def _import_numpy(self):
        try:
            import numpy as np  # type: ignore
            return np
        except Exception as e:
            raise RuntimeError("numpy가 설치되어 있지 않거나 로딩에 실패했습니다.") from e

    def _load_weights(self) -> None:
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"낙찰가 모델 가중치(.pt) 파일을 찾을 수 없습니다: {self.weights_path}")
        state = self._torch.load(self.weights_path, map_location=self.device)
        if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            self.model.load_state_dict(state)
        elif isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            self.model.load_state_dict(state["state_dict"])
        else:
            raise ValueError("가중치 포맷을 해석할 수 없습니다. state_dict(dict) 형태를 기대합니다.")
        self.model.eval()

    def _load_scalers(self) -> None:
        if not self.scaler_path:
            base_dir = os.path.dirname(self.weights_path) or "."
            c_json = os.path.join(base_dir, "scalers.json")
            c_npz = os.path.join(base_dir, "scalers.npz")
            if os.path.exists(c_json):
                self.scaler_path = c_json
            elif os.path.exists(c_npz):
                self.scaler_path = c_npz

        if not self.scaler_path or not os.path.exists(self.scaler_path):
            return

        x_mean = None
        x_std = None
        y_mean = None
        y_std = None
        feature_cols = None
        target_log = None

        if self.scaler_path.lower().endswith(".json"):
            with open(self.scaler_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            x_mean = cfg.get("x_mean")
            x_std = cfg.get("x_std")
            y_mean = cfg.get("y_mean")
            y_std = cfg.get("y_std")
            feature_cols = cfg.get("feature_cols")
            target_log = cfg.get("target_log")
        elif self.scaler_path.lower().endswith(".npz"):
            arr = self._np.load(self.scaler_path, allow_pickle=True)
            x_mean = arr.get("x_mean")
            x_std = arr.get("x_std")
            y_mean = arr.get("y_mean")
            y_std = arr.get("y_std")
            feature_cols = arr.get("feature_cols")
            target_log = arr.get("target_log")
        else:
            raise ValueError("지원하지 않는 스케일러 파일 확장자입니다. .json 또는 .npz만 지원합니다.")

        if x_mean is None or x_std is None or y_mean is None or y_std is None:
            raise ValueError("scaler 파일에 x_mean/x_std/y_mean/y_std 값이 없습니다.")

        self.x_scaler = self.module.StandardScaler()
        self.x_scaler.mean_ = self._np.asarray(x_mean, dtype=self._np.float32)
        self.x_scaler.std_ = self._np.asarray(x_std, dtype=self._np.float32)

        self.y_scaler = self.module.TargetScaler()
        self.y_scaler.mean_ = float(self._np.asarray(y_mean).reshape(-1)[0])
        self.y_scaler.std_ = float(self._np.asarray(y_std).reshape(-1)[0])

        if feature_cols is not None:
            if isinstance(feature_cols, (list, tuple)):
                self.feature_cols = [str(x) for x in feature_cols]
            else:
                # e.g. numpy array
                try:
                    self.feature_cols = [str(x) for x in list(feature_cols)]
                except Exception:
                    pass

        if target_log is not None:
            self.target_log = bool(target_log)

    def _extract_feature(
        self,
        requirements: Dict[str, Any],
        retrieved_context: str,
        keys: Sequence[str],
        patterns: Sequence[str],
    ) -> Optional[float]:
        for k in keys:
            v = _parse_number(requirements.get(k))
            if v is not None:
                return v

        text = retrieved_context or ""
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                v = _parse_number(m.group(1))
                if v is not None:
                    return v
        return None

    def _build_feature_vector(self, requirements: Dict[str, Any], retrieved_context: str) -> Tuple[Optional[Any], List[str]]:
        missing: List[str] = []

        base_amount = self._extract_feature(
            requirements,
            retrieved_context,
            keys=["budget", "base_amount", "기초금액"],
            patterns=[
                r"기초\s*금액\s*[:：]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
                r"기초금액\s*[:：]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            ],
        )
        if base_amount is None:
            missing.append("기초금액(budget)")

        estimate_price = self._extract_feature(
            requirements,
            retrieved_context,
            keys=["estimate_price", "추정가격", "예정가격"],
            patterns=[
                r"추정\s*가격\s*[:：]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
                r"예정\s*가격\s*[:：]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            ],
        )
        if estimate_price is None:
            missing.append("추정가격(estimate_price)")

        price_range = self._extract_feature(
            requirements,
            retrieved_context,
            keys=["expected_price_range", "예가범위"],
            patterns=[
                r"예가\s*범위\s*[:：]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*%?",
                r"예가범위\s*[:：]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*%?",
            ],
        )
        price_range = _coerce_percent(price_range)
        if price_range is None:
            missing.append("예가범위(expected_price_range)")

        lower_rate = self._extract_feature(
            requirements,
            retrieved_context,
            keys=["award_lower_rate", "낙찰하한율"],
            patterns=[
                r"낙찰\s*하한\s*율\s*[:：]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*%?",
                r"낙찰하한율\s*[:：]?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*%?",
            ],
        )
        lower_rate = _coerce_percent(lower_rate)
        if lower_rate is None:
            missing.append("낙찰하한율(award_lower_rate)")

        if missing:
            return None, missing

        x = self._np.asarray(
            [base_amount, estimate_price, price_range, lower_rate],
            dtype=self._np.float32,
        ).reshape(1, -1)
        return x, []

    def predict(self, requirements: Dict[str, Any], retrieved_context: str) -> Dict[str, Any]:
        x, missing = self._build_feature_vector(requirements, retrieved_context or "")
        if missing:
            return {
                "currency": "KRW",
                "predicted_min": None,
                "predicted_max": None,
                "point_estimate": None,
                "confidence": "low",
                "rationale": [
                    "낙찰가 예측에 필요한 피처를 충분히 확보하지 못해 모델 추론을 수행할 수 없습니다.",
                    "누락 피처: " + ", ".join(missing),
                    "해결: 공고문 추출 필드에 예가범위/낙찰하한율을 포함하거나, RAG 코퍼스에서 해당 수치를 회수할 수 있도록 문서를 보강하세요.",
                ],
                "model": {
                    "type": "cnn1d",
                    "code": getattr(self.module, "__file__", "<module>"),
                    "weights": self.weights_path,
                    "scaler": self.scaler_path,
                    "device": self.device,
                    "hidden": self.hidden,
                    "dropout": self.dropout,
                },
            }

        if self.x_scaler is None or self.y_scaler is None:
            return {
                "currency": "KRW",
                "predicted_min": None,
                "predicted_max": None,
                "point_estimate": None,
                "confidence": "low",
                "rationale": [
                    "CNN1D 모델은 학습 시 X/y 스케일링(및 로그 변환)을 사용했으나, scaler/config 파일이 없어 올바른 역변환이 불가능합니다.",
                    "해결: 학습 시 x_mean/x_std/y_mean/y_std 및 target_log를 scalers.json(.npz)로 저장하세요.",
                ],
                "model": {
                    "type": "cnn1d",
                    "code": getattr(self.module, "__file__", "<module>"),
                    "weights": self.weights_path,
                    "scaler": self.scaler_path,
                    "device": self.device,
                    "hidden": self.hidden,
                    "dropout": self.dropout,
                },
            }

        x_scaled = self.x_scaler.transform(x)  # (1,F)
        x_tensor = self._torch.from_numpy(x_scaled.astype("float32")).reshape(1, -1, 1).to(self.device)

        with self._torch.no_grad():
            y_hat_scaled = self.model(x_tensor).detach().cpu().numpy().reshape(-1)

        y_hat = self.y_scaler.inverse_transform(y_hat_scaled).reshape(-1)
        if self.target_log:
            y_hat = self._np.expm1(y_hat)

        point = float(y_hat[0])
        pred_min = round(point * 0.98)
        pred_max = round(point * 1.02)

        return {
            "currency": "KRW",
            "predicted_min": pred_min,
            "predicted_max": pred_max,
            "point_estimate": round(point),
            "confidence": "medium",
            "rationale": [
                "사용자 1D-CNN 낙찰가 예측 모델 추론 결과입니다.",
                "predicted_min/max는 불확실도 추정치가 아니라, 보고서 표기를 위한 ±2% 휴리스틱 밴드입니다(필요 시 교체).",
                f"피처 사용 순서: {', '.join(self.feature_cols)}",
            ],
            "model": {
                "type": "cnn1d",
                "code": getattr(self.module, "__file__", "<module>"),
                "weights": self.weights_path,
                "scaler": self.scaler_path,
                "device": self.device,
                "hidden": self.hidden,
                "dropout": self.dropout,
            },
        }


# ------------------------------
# Structured schema (extract)
# ------------------------------

class BidRequirements(BaseModel):
    """Extracted fields from bid notice."""

    title: Optional[str] = Field(None, description="공고명")
    agency: Optional[str] = Field(None, description="발주기관/수요기관")
    category: Optional[str] = Field(None, description="용역/물품/공사 등 구분")
    region: Optional[str] = Field(None, description="수행지역/납품지역")
    deadline: Optional[str] = Field(None, description="마감 일시(문자열)")

    budget: Optional[float] = Field(None, description="예산/기초금액 (가능하면 숫자)")
    estimate_price: Optional[float] = Field(None, description="추정가격/예정가격(가능하면 숫자)")
    expected_price_range: Optional[float] = Field(None, description="예가범위(%) 또는 범위값(가능하면 숫자)")
    award_lower_rate: Optional[float] = Field(None, description="낙찰하한율(%) (가능하면 숫자)")

    bid_method: Optional[str] = Field(None, description="낙찰자 결정 방식/평가 방식")

    qualification_requirements: List[str] = Field(default_factory=list, description="참가자격/면허/등급/인증/요건")
    performance_requirements: List[str] = Field(default_factory=list, description="실적요건/경력/유사실적/수행능력")
    document_requirements: List[str] = Field(default_factory=list, description="제출서류/제안서/증빙")
    risk_flags: List[str] = Field(default_factory=list, description="특이사항/리스크(보증금, 제재, 하도급 제한 등)")


# ------------------------------
# LangGraph state
# ------------------------------

class GraphState(TypedDict, total=False):
    messages: List[Any]
    requirements: Dict[str, Any]
    report_markdown: str


# ------------------------------
# RAG index (FAISS)
# ------------------------------

class RagIndex:
    def __init__(self, doc_dir: str, index_dir: str, embedding_model: str):
        self.doc_dir = doc_dir
        self.index_dir = index_dir
        self.embedding_model = embedding_model
        self._vs: Optional[FAISS] = None

    def _load_txt_documents(self) -> List[str]:
        if not os.path.isdir(self.doc_dir):
            return []
        texts: List[str] = []
        for root, _, files in os.walk(self.doc_dir):
            for name in files:
                if not name.lower().endswith(".txt"):
                    continue
                path = os.path.join(root, name)
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
        return texts

    def build_or_load(self, force_rebuild: bool = False) -> FAISS:
        embeddings = OpenAIEmbeddings(model=self.embedding_model)

        if not force_rebuild and os.path.isdir(self.index_dir):
            try:
                self._vs = FAISS.load_local(
                    self.index_dir,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
                return self._vs
            except Exception:
                pass

        texts = self._load_txt_documents()
        if not texts:
            self._vs = FAISS.from_texts([""], embeddings)
            os.makedirs(self.index_dir, exist_ok=True)
            self._vs.save_local(self.index_dir)
            return self._vs

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks: List[str] = []
        for t in texts:
            chunks.extend(splitter.split_text(t))

        self._vs = FAISS.from_texts(chunks, embeddings)
        os.makedirs(self.index_dir, exist_ok=True)
        self._vs.save_local(self.index_dir)
        return self._vs

    @property
    def vs(self) -> FAISS:
        if self._vs is None:
            raise RuntimeError("RAG index is not initialized. Call build_or_load().")
        return self._vs

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        query = _clean_whitespace(query)
        if not query:
            return []
        docs = self.vs.similarity_search(query, k=k)
        return [d.page_content for d in docs if getattr(d, "page_content", None)]


# ------------------------------
# Pipeline (LangGraph)
# ------------------------------

class BidRAGPipeline:
    """End-to-end pipeline: extract -> tools(RAG/award_price/competitor) -> report.

    award_model_path
    ---------------
    1) Wrapper mode (권장): python 파일에 아래 함수 구현
        predict_award_price(requirements: dict, retrieved_context: str) -> dict
        또는 predict(requirements, retrieved_context) -> dict   (단, 시그니처가 2개 인자여야 함)

    2) Auto mode: model_1dcnn.py를 지정하고, 아래 artifacts가 존재하면 자동 연결
        --award_weights ./results/best_model.pt
        --award_scaler  ./results/scalers.json
    """

    def __init__(
        self,
        doc_dir: str = "./rag_corpus",
        index_dir: str = "./rag_index",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large",
        top_k: int = 6,
        award_model_path: Optional[str] = None,
        award_weights_path: Optional[str] = None,
        award_scaler_path: Optional[str] = None,
        award_device: Optional[str] = None,
        award_hidden: int = 64,
        award_dropout: float = 0.1,
        award_predict_fn: Optional[Callable[[Dict[str, Any], str], Any]] = None,
    ):
        load_api_keys("api_key.txt")

        self.doc_dir = doc_dir
        self.index_dir = index_dir
        self.top_k = top_k

        self.llm = ChatOpenAI(model=llm_model, temperature=0)
        self.index = RagIndex(doc_dir=doc_dir, index_dir=index_dir, embedding_model=embedding_model)
        self.index.build_or_load(force_rebuild=False)

        self.award_predictor = self._init_award_predictor(
            award_model_path=award_model_path,
            award_weights_path=award_weights_path,
            award_scaler_path=award_scaler_path,
            award_device=award_device,
            award_hidden=award_hidden,
            award_dropout=award_dropout,
            award_predict_fn=award_predict_fn,
        )

        self.tools = self._build_tools()
        self.tool_node = ToolNode(self.tools)
        self.graph = self._build_graph()

    def _init_award_predictor(
        self,
        award_model_path: Optional[str],
        award_weights_path: Optional[str],
        award_scaler_path: Optional[str],
        award_device: Optional[str],
        award_hidden: int,
        award_dropout: float,
        award_predict_fn: Optional[Callable[[Dict[str, Any], str], Any]],
    ) -> AwardPricePredictor:
        if callable(award_predict_fn):
            return CallableAwardPricePredictor(
                predict_fn=award_predict_fn,
                model_info={"type": "callable", "name": getattr(award_predict_fn, "__name__", "<callable>")},
            )

        if not award_model_path:
            return HeuristicAwardPricePredictor()

        try:
            module = _load_module_from_py(award_model_path)

            # Wrapper function: predict_award_price(requirements, retrieved_context)
            fn = getattr(module, "predict_award_price", None)
            if callable(fn) and _callable_looks_like_wrapper(fn):
                return CallableAwardPricePredictor(
                    predict_fn=fn,
                    model_info={"type": "python_file", "path": award_model_path, "fn": "predict_award_price"},
                )

            # Auto mode: model_1dcnn.py style (important: module also defines predict(model, loader, device))
            if hasattr(module, "CNN1DRegressor") and hasattr(module, "StandardScaler") and hasattr(module, "TargetScaler"):
                weights_path = award_weights_path or "./results/best_model.pt"
                return CNN1DAwardPricePredictor(
                    module=module,
                    weights_path=weights_path,
                    scaler_path=award_scaler_path,
                    device=award_device,
                    hidden=award_hidden,
                    dropout=award_dropout,
                )

            # Generic wrapper: predict(requirements, retrieved_context) with proper signature
            fn = getattr(module, "predict", None)
            if callable(fn) and _callable_looks_like_wrapper(fn):
                return CallableAwardPricePredictor(
                    predict_fn=fn,
                    model_info={"type": "python_file", "path": award_model_path, "fn": "predict"},
                )

            raise AttributeError(
                "Model module must define predict_award_price(requirements, retrieved_context), "
                "or expose CNN1DRegressor+StandardScaler+TargetScaler for auto mode."
            )

        except Exception as e:
            return CallableAwardPricePredictor(
                predict_fn=lambda _r, _c: {
                    "currency": "KRW",
                    "predicted_min": None,
                    "predicted_max": None,
                    "point_estimate": None,
                    "confidence": "low",
                    "rationale": [
                        "낙찰가 예측 모델 로딩에 실패하여 휴리스틱(기본값)으로 대체합니다.",
                        f"로딩 오류: {type(e).__name__}: {str(e)}",
                    ],
                },
                model_info={"type": "error_fallback", "path": award_model_path},
            )

    def _build_tools(self) -> List[Any]:
        index = self.index
        top_k = self.top_k

        @tool
        def rag_retrieve(query: str) -> str:
            """유사 공고/낙찰사례 검색(RAG)."""
            chunks = index.retrieve(query=query, k=top_k)
            if not chunks:
                return "(검색 결과 없음)"
            chunks = [c[:1200] for c in chunks]
            return "\n\n---\n\n".join(chunks)

        @tool
        def predict_award_price(requirements_json: str, retrieved_context: str) -> str:
            """낙찰가 예측 Tool (ToolNode 블록)."""
            try:
                reqs = json.loads(requirements_json)
                if not isinstance(reqs, dict):
                    reqs = {}
            except Exception:
                reqs = {}

            try:
                result = self.award_predictor.predict(reqs, retrieved_context or "")
            except Exception as e:
                result = {
                    "currency": "KRW",
                    "predicted_min": None,
                    "predicted_max": None,
                    "point_estimate": None,
                    "confidence": "low",
                    "rationale": [
                        "낙찰가 예측 중 예외가 발생했습니다. 모델/피처 파이프라인을 점검하세요.",
                        f"예외: {type(e).__name__}: {str(e)}",
                    ],
                    "model": {"type": "runtime_error"},
                }

            if not isinstance(result, dict):
                result = {
                    "currency": "KRW",
                    "predicted_min": None,
                    "predicted_max": None,
                    "point_estimate": None,
                    "confidence": "low",
                    "rationale": ["낙찰가 예측 모델 출력이 dict가 아니어서 처리할 수 없습니다."],
                    "model": {"type": "invalid_output"},
                }

            return json.dumps(result, ensure_ascii=False)

        @tool
        def competitor_analysis(requirements_json: str, retrieved_context: str) -> str:
            """경쟁낙찰/경쟁사 시그널(경량 heuristic)."""
            try:
                reqs = json.loads(requirements_json)
                if not isinstance(reqs, dict):
                    reqs = {}
            except Exception:
                reqs = {}

            text = retrieved_context or ""
            candidates: Dict[str, int] = {}
            for m in re.findall(r"([가-힣A-Za-z0-9&()]{2,40})(?:\s*\(주\)|\s*㈜)", text):
                name = _clean_whitespace(m)
                if 2 <= len(name) <= 40:
                    candidates[name] = candidates.get(name, 0) + 1
            sorted_names = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            top_names = [n for n, _ in sorted_names[:8]]

            barriers = []
            if len(reqs.get("qualification_requirements", [])) >= 5:
                barriers.append("참가자격 요건이 다수로 보이며, 자격 충족이 1차 필터로 작동할 가능성이 큽니다.")
            if len(reqs.get("performance_requirements", [])) >= 5:
                barriers.append("유사실적/실적요건이 강해 신규/중소 사업자 진입이 제한될 수 있습니다.")
            if reqs.get("bid_method"):
                barriers.append(f"평가/낙찰 방식({reqs.get('bid_method')})에 따라 기술/가격 전략이 달라집니다.")

            result = {
                "likely_competitors": top_names or [],
                "market_signals": barriers,
                "recommended_positioning": [
                    "요구사항 매핑표(요구사항-근거-증빙)를 제안서 최상단에 배치해 누락 리스크를 제거합니다.",
                    "유사실적/핵심인력/품질(보안/안전) 체계를 명확히 제시해 기술평가 리스크를 낮춥니다.",
                    "낙찰가 전략은 '예측값 + 유사 낙찰사례 분포 + 내부 원가/마진'으로 최종 결정합니다.",
                ],
                "confidence": "low" if retrieved_context in ("", "(검색 결과 없음)") else "medium",
            }
            return json.dumps(result, ensure_ascii=False)

        return [rag_retrieve, predict_award_price, competitor_analysis]

    def _build_graph(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("extract", self._node_extract)
        workflow.add_node("agent", self._node_agent)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("report", self._node_report)

        workflow.add_edge(START, "extract")
        workflow.add_edge("extract", "agent")

        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "report": "report"},
        )
        workflow.add_edge("tools", "agent")
        workflow.add_edge("report", END)

        return workflow.compile(checkpointer=MemorySaver())

    # ------------------------------
    # Graph nodes
    # ------------------------------

    def _node_extract(self, state: GraphState) -> GraphState:
        messages = state.get("messages", [])
        if not messages:
            raise ValueError("State must start with messages containing HumanMessage.")

        bid_text = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                bid_text = m.content
                break

        sys = SystemMessage(
            content=(
                "너는 조달/입찰(제안/투찰) 분석가다. "
                "사용자가 제공한 공고문 텍스트에서 요구사항을 구조화해 추출하라. "
                "숫자는 가능하면 원 단위 숫자(float/int)로 정규화하고, "
                "확실하지 않으면 null로 둔다. "
                "특히 낙찰가 모델 입력을 위해 예가범위(expected_price_range), 낙찰하한율(award_lower_rate)도 추출을 시도하라."
            )
        )

        try:
            extractor = self.llm.with_structured_output(BidRequirements)
            reqs_obj: BidRequirements = extractor.invoke([sys, HumanMessage(content=bid_text)])
            reqs_dict = reqs_obj.model_dump()
        except Exception:
            fallback = self.llm.invoke(
                [
                    sys,
                    HumanMessage(
                        content=(
                            "다음 공고문을 읽고 아래 키를 갖는 JSON만 출력해라:\n"
                            "title, agency, category, region, deadline, budget, estimate_price, "
                            "expected_price_range, award_lower_rate, bid_method, "
                            "qualification_requirements, performance_requirements, document_requirements, risk_flags\n\n"
                            + bid_text
                        )
                    ),
                ]
            )
            text = fallback.content if isinstance(fallback, AIMessage) else str(fallback)
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            parsed = {}
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except Exception:
                    parsed = {}
            reqs_dict = parsed

        for k in ["qualification_requirements", "performance_requirements", "document_requirements", "risk_flags"]:
            val = reqs_dict.get(k, [])
            if isinstance(val, list):
                reqs_dict[k] = [_clean_whitespace(x) for x in val if str(x).strip()]

        state["requirements"] = reqs_dict
        return state

    def _node_agent(self, state: GraphState) -> GraphState:
        reqs = state.get("requirements", {})
        reqs_json = json.dumps(reqs, ensure_ascii=False)

        sys = SystemMessage(
            content=(
                "너는 제안/투찰 agent다. 다음 순서로 도구를 호출해 근거를 수집하라.\n"
                "1) rag_retrieve(query): 공고 요약 + 핵심 키워드로 유사 공고/낙찰사례 검색\n"
                "2) predict_award_price(requirements_json, retrieved_context): 낙찰가 예측(사용자 모델)\n"
                "3) competitor_analysis(requirements_json, retrieved_context): 경쟁/시장 시그널 산출\n\n"
                "도구 호출이 모두 끝나면, 더 이상 도구를 호출하지 말고 종료하라."
            )
        )

        context_msg = SystemMessage(content="[추출된 요구사항 JSON]\n" + reqs_json)
        messages = state.get("messages", [])
        bound = self.llm.bind_tools(self.tools)
        ai = bound.invoke([sys, context_msg] + messages)
        state["messages"] = messages + [ai]
        return state

    def _should_continue(self, state: GraphState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return "report"
        last = messages[-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "report"

    def _node_report(self, state: GraphState) -> GraphState:
        reqs = state.get("requirements", {})
        reqs_json = json.dumps(reqs, ensure_ascii=False)
        messages = state.get("messages", [])

        sys = SystemMessage(
            content=(
                "너는 조달/입찰(제안/투찰) 컨설턴트다. "
                "아래의 (1) 추출된 요구사항 JSON, (2) 도구 출력들을 근거로 "
                "실무자가 바로 사용할 수 있는 '제안/투찰 분석 리포트'를 한국어 마크다운으로 작성하라.\n\n"
                "필수 섹션(순서 유지):\n"
                "# 1. 공고 요약\n"
                "# 2. 참가자격/실적/제출서류 체크리스트\n"
                "# 3. RAG 근거 요약(유사 공고/낙찰사례 핵심)\n"
                "# 4. 낙찰가 예측(범위/포인트/근거/리스크)\n"
                "# 5. 경쟁낙찰/경쟁사 시그널(가능 경쟁사, 포지셔닝)\n"
                "# 6. 권고 액션(다음 72시간 To-Do)\n\n"
                "제약: 근거가 불충분하면 '가정'으로 명시하고 추가 수집 항목을 제시하라."
            )
        )

        ctx = SystemMessage(content="[추출된 요구사항 JSON]\n" + reqs_json)
        final = self.llm.invoke([sys, ctx] + messages)
        report = final.content if isinstance(final, AIMessage) else str(final)
        state["report_markdown"] = report
        return state

    # ------------------------------
    # Public API
    # ------------------------------

    def analyze(self, bid_notice_text: str, thread_id: str = "default") -> Dict[str, Any]:
        initial: GraphState = {"messages": [HumanMessage(content=bid_notice_text)]}
        final_state: GraphState = self.graph.invoke(initial, config={"configurable": {"thread_id": thread_id}})
        return {
            "requirements": final_state.get("requirements", {}),
            "report_markdown": final_state.get("report_markdown", ""),
            "messages": final_state.get("messages", []),
        }


# ------------------------------
# CLI entry
# ------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Bid Assistance RAG Pipeline")
    parser.add_argument("--doc_dir", default="./rag_corpus")
    parser.add_argument("--index_dir", default="./rag_index")
    parser.add_argument("--input", default=None, help="Bid notice file (.txt or .pdf)")

    parser.add_argument(
        "--award_model",
        default=None,
        help=(
            "낙찰가 예측 모델 python 파일(.py). "
            "Wrapper mode: predict_award_price(requirements, retrieved_context) 또는 predict(requirements, retrieved_context). "
            "Auto mode: model_1dcnn.py를 지정하고 --award_weights/--award_scaler를 함께 지정."
        ),
    )
    parser.add_argument("--award_weights", default=None, help="CNN1D 가중치 파일(.pt). 예: ./results/best_model.pt")
    parser.add_argument("--award_scaler", default=None, help="CNN1D scaler 파일(.json/.npz). 예: ./results/scalers.json")
    parser.add_argument("--award_device", default=None, help="torch device. 예: cpu 또는 cuda")
    parser.add_argument("--award_hidden", type=int, default=64, help="CNN1D hidden size (학습과 동일해야 함)")
    parser.add_argument("--award_dropout", type=float, default=0.1, help="CNN1D dropout (학습과 동일해야 함)")

    args = parser.parse_args()

    text = ""
    if args.input and os.path.exists(args.input):
        text = read_input_text(args.input)
    else:
        print("Paste bid notice text, then end input with EOF (Ctrl-D / Ctrl-Z).")
        try:
            text = sys.stdin.read()
        except Exception:
            text = ""

    pipe = BidRAGPipeline(
        doc_dir=args.doc_dir,
        index_dir=args.index_dir,
        award_model_path=args.award_model,
        award_weights_path=args.award_weights,
        award_scaler_path=args.award_scaler,
        award_device=args.award_device,
        award_hidden=args.award_hidden,
        award_dropout=args.award_dropout,
    )
    out = pipe.analyze(text)
    print(out["report_markdown"])
