FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 9999

ENV PYTHONUNBUFFERED=1

CMD ["python", "RAG_server.py"]

# 현재 폴더의 모든 내용(모델 폴더 포함)을 컨테이너의 /app으로 복사
COPY . /app/
# 또는 명시적으로 모델 폴더만 복사
COPY results_tft_4feat/ /app/results_tft_4feat/
