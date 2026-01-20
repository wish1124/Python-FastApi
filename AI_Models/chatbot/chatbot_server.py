import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pyngrok import ngrok
from openai import OpenAI

# ------------------
# 1. 환경 설정 (.env 로드)
# ------------------
# .env 파일 명시적 로드 (현재 디렉토리 기준)
load_dotenv(verbose=True)

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
    
    @classmethod
    def check(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("❌ Error: 'OPENAI_API_KEY'가 .env 파일에 없습니다.")
        if not cls.NGROK_AUTH_TOKEN:
            print("⚠️ Warning: 'NGROK_AUTH_TOKEN'이 없어 외부 접속(Public URL)이 불가능합니다.")

# 필수 키 체크 실행
try:
    Config.check()
except ValueError as e:
    print(e)
    exit(1) # 키 없으면 종료

# ------------------
# 2. LLM Client Setup
# ------------------
client = OpenAI(api_key=Config.OPENAI_API_KEY)

def call_llm(prompt: str, system_role: str = "You are a helpful assistant.") -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# ------------------
# 3. Business Logic
# ------------------
def route_intent(query: str) -> str:
    prompt = f"""
    Classify the user request into exactly one of these labels:
    - general_chat
    - report_summary
    - report_qa
    - site_feature_qa

    Do not explain, just return the label.
    User: {query}
    """
    return call_llm(prompt, system_role="You are an intent classifier.")

def summarize_report():
    return "📄 [Mock] 보고서 요약입니다: 이 보고서는 금년도 매출 20% 증가와 주요 리스크 요인을 다루고 있습니다."

def qa_report(query: str):
    return f"📌 [Mock] 보고서 내용 기반 답변: '{query}'에 대한 분석 결과는 3페이지에 있습니다."

def site_feature_answer(query: str):
    return "🧭 [Mock] 사이트 안내: 저희 사이트는 PDF 보고서 요약, AI 질의응답, 실시간 채팅 기능을 제공합니다."

# ------------------
# 4. FastAPI App Setup
# ------------------
app = FastAPI(title="LLM Pipeline PoC")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.get("/status_check")
def root():
    return {"status": "running", "message": "API Active"}

@app.post("/chat")
def chat(req: ChatRequest):
    intent = route_intent(req.query)
    print(f"[Log] Intent: {intent}")

    if "report_summary" in intent:
        answer = summarize_report()
    elif "report_qa" in intent:
        answer = qa_report(req.query)
    elif "site_feature_qa" in intent:
        answer = site_feature_answer(req.query)
    else:
        answer = call_llm(req.query)

    return {"intent": intent, "answer": answer}

# ------------------
# 5. Server Execution
# ------------------
if __name__ == "__main__":
    # ngrok 설정 (토큰이 .env에 있을 때만)
    if Config.NGROK_AUTH_TOKEN:
        ngrok.set_auth_token(Config.NGROK_AUTH_TOKEN)
        public_url = ngrok.connect(8000)
        print(f"\n🌍 Public URL: {public_url.public_url}\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
