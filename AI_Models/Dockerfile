FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 1. 필수 시스템 패키지 설치 (libgomp1 추가 - Torch/FAISS 연산용)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. 라이브러리 설치 (캐시 최적화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. 프로젝트 전체 소스 코드 복사
COPY . .


# 한글 폰트 파일 복사 확인
COPY NanumGothic-Regular.ttf /app/NanumGothic-Regular.ttf

# 5. [추가] PDF 출력을 위한 폴더 생성 및 권한 설정
RUN mkdir -p /app/output && chmod 777 /app/output

# 6. RAG 인덱스 및 코퍼스 폴더 확인
RUN mkdir -p /app/rag_corpus /app/rag_index

# 7. 환경 변수 설정
ENV PYTHONUNBUFFERED=1
# Azure App Service는 기본적으로 PORT 환경변수를 주입하므로 EXPOSE는 참고용입니다
EXPOSE 9999

# 8. 서버 실행
CMD ["python", "RAG_server.py"]