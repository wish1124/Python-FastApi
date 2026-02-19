FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*
COPY BE_AI_server/AI_server/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY . .
COPY BE_AI_server/AI_server/NanumGothic-Regular.ttf /app/NanumGothic-Regular.ttf
RUN mkdir -p /app/output && chmod 777 /app/output
RUN mkdir -p /app/rag_corpus /app/rag_index
ENV PYTHONUNBUFFERED=1
EXPOSE 9999
CMD ["python", "BE_AI_server/AI_server/RAG_server.py"]
