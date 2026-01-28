import os
import pandas as pd
from PIL import Image
import pytesseract

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import ImageCaptionLoader, UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict

# from usage_tool import usage_tool : 테스트 하려면 활성화

# test_usage_tool.py : 현재 위치에 faiss_db 폴더를 생성 + usage_tool.py 동작 테스트
# C:\BGPJ\BidAssitance\AI_Models\usage_data\images
# =========================
# 경로 설정
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "usage_data", "images")
API_EXCEL_PATH = os.path.join(BASE_DIR, "usage_data", "api정의서.xlsx")

'''
IMAGE_FAISS_DIR = "faiss_db/image_faiss"     # 웹페이지 스크린샷 FAISS 저장 경로
API_FAISS_DIR = "faiss_db/api_faiss"         # API 정의서 FAISS 저장 경로
# faiss_db 내부에서 image_faiss와 api_faiss 폴더가 각각 생성된다.
os.makedirs("faiss_db", exist_ok=True)        # faiss_db 폴더 생성(이미 있으면 생성하지 않음)
'''

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGE_FAISS_DIR = BASE_DIR / "faiss_db" / "image_faiss"
API_FAISS_DIR= BASE_DIR / "faiss_db" / "api_faiss"

'''
#로컬 테스트용, 경로에 한글이 있으면 C드라이브로 옮겨서 진행할 것
BASE_DIR = Path("C:/faiss_db")
IMAGE_FAISS_DIR = BASE_DIR / "image_faiss"
API_FAISS_DIR= BASE_DIR / "api_faiss"
'''

# =========================
# FAISS 생성 임베딩 모델 설정
# =========================
'''
def load_api_keys(filepath="api_key.txt"): 
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()
        
load_api_keys(os.path.join(BASE_DIR, "usage_api.txt"))   # API 키 로드 및 환경변수 설정
'''
from dotenv import load_dotenv
load_dotenv()


embeddings = OpenAIEmbeddings(model = "text-embedding-3-small") # 임베딩 모델 초기화

# =========================
# 1️⃣ 이미지 → image FAISS 생성 (ImageCaptionLoader, 다른 코드에서 필요시 붙여놓기) 
# =========================
def build_image_faiss():
    print("🔹 image FAISS 생성 중 (ImageCaptionLoader)...")
    if not os.path.exists(IMAGE_DIR):
        raise FileNotFoundError(f"이미지 디렉터리가 없습니다: {IMAGE_DIR}")

    # 1️⃣ 이미지 파일 전체 수집
    image_paths = [
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_paths:
        raise RuntimeError("처리할 이미지 파일이 없습니다.")

    # 2️⃣ ImageCaptionLoader 사용
    loader = ImageCaptionLoader(image_paths)
    documents = loader.load()
    # 3️⃣ metadata 보강
    for doc in documents:
        doc.metadata["source"] = "image"
        doc.metadata["type"] = "screenshot"
    # 4️⃣ FAISS 생성
    faiss = FAISS.from_documents(documents, embeddings)
    faiss.save_local(IMAGE_FAISS_DIR)

# =========================
# 2️⃣ 엑셀 → api FAISS 생성 (UnstructuredExcelLoader, 다른 코드에서 필요시 붙여놓기)
# =========================
def build_api_faiss():
    print("🔹 api FAISS 생성 중 (UnstructuredExcelLoader)...")
    if not os.path.exists(API_EXCEL_PATH):
        raise FileNotFoundError(f"엑셀 파일이 없습니다: {API_EXCEL_PATH}")

    # 1️⃣ 엑셀 로더
    loader = UnstructuredExcelLoader(
        API_EXCEL_PATH,
        mode="elements"   # row / cell 단위 분해
    )

    documents = loader.load()
    if not documents:
        raise RuntimeError("엑셀에서 로드된 문서가 없습니다.")

    # 2️⃣ 메타데이터 보강 (권장)
    for idx, doc in enumerate(documents):
        doc.metadata.update({
            "source": "api_excel",
            "element_id": idx
        })
    # 3️⃣ FAISS 생성
    faiss = FAISS.from_documents(documents, embeddings)
    faiss.save_local(API_FAISS_DIR)

# =========================
# FAISS 값 불러오기 (image / api 분리)
# =========================
def load_image_faiss(image_db_path: str) -> FAISS:
    """웹페이지 스크린샷 기반 vectorDB (OCR / Image Caption 결과가 벡터화되어 있음)"""
    if not os.path.exists(image_db_path):
        raise FileNotFoundError(f"Image FAISS DB not found: {image_db_path}")

    return FAISS.load_local(
        image_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def load_api_faiss(api_db_path: str) -> FAISS:
    """API 정의서 엑셀 기반 vectorDB (API row 단위 벡터화)"""
    if not os.path.exists(api_db_path):
        raise FileNotFoundError(f"API FAISS DB not found: {api_db_path}")

    return FAISS.load_local(
        api_db_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

# =========================
# 벡터 검색 (목적 분리)
# =========================
def search_image_context(image_faiss: FAISS, query: str, k: int = 3) -> List[Document]:
    """UI / 화면 / 사용자 동작 관점 검색"""
    return image_faiss.similarity_search(query, k=k)

def search_api_context(api_faiss: FAISS, query: str, k: int = 3) -> List[Document]:
    """API 기능 / 요청 / 응답 / 필드 관점 검색"""
    return api_faiss.similarity_search(query, k=k)

# =========================
# 컨텍스트 정리 (image / api 분리)
# =========================
def build_context(img_docs: List[Document], api_docs: List[Document]) -> Dict[str, str]:
    """image / api 컨텍스트를 구조적으로 분리하여 반환"""
    image_context = []
    api_context = []

    if img_docs:
        for d in img_docs:
            image_context.append(d.page_content)
    if api_docs:
        for d in api_docs:
            api_context.append(d.page_content)

    return {
        "image": "\n\n".join(image_context),
        "api": "\n\n".join(api_context)
    }

# # =========================
# # 3️⃣ usage_tool 테스트 (usage_tool 정상출력 테스트용, 필요시 생략 가능)
# # =========================
# def test_usage_tool():
#     query = "게시글을 작성하려면 어떻게 해야돼?"

#     result = usage_tool.invoke({
#         "query": query,
#         "img": IMAGE_FAISS_DIR,
#         "api": API_FAISS_DIR
#     })

#     print("\n====================")
#     print("🤖 AI 응답 결과")
#     print("====================\n")
#     print(result)

# # =========================
# # main
# # =========================
# if __name__ == "__main__":      # Python 스크립트가 직접 실행될 때만 작성된 세 개의 함수를 순차적으로 호출.
#     build_image_faiss()
#     build_api_faiss()
#     test_usage_tool()
