FROM python:3.11-slim

WORKDIR /app

# 1. 필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. 파이썬 라이브러리 설치 (캐시 활용을 위해 소스 복사 전 실행)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. 프로젝트 전체 소스 코드 복사
COPY . .

# 4. [핵심] 모델 폴더 명시적 복사 (보험용)
# .dockerignore에 걸려있더라도 이 명령어가 아래에 있으면 강제로 덮어씌워 복사합니다.
COPY results_transformer/ /app/results_transformer/

# 5. 환경 변수 설정
ENV PYTHONUNBUFFERED=1
EXPOSE 9999

# 6. [매우 중요] CMD는 반드시 파일의 맨 마지막에 딱 한 번만 와야 합니다!
CMD ["python", "RAG_server.py"]