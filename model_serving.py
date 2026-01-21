import BidAssitanceModel # 이 모듈은 로컬에 있어야 합니다.
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, List 
from fastapi.middleware.cors import CORSMiddleware
import torch
from model_transformer import TransformerRegressor # 이 모듈은 로컬에 있어야 합니다.
import re
import nest_asyncio
from pyngrok import ngrok

"""
Serve for Bid Price Prediction Model.
"""

app = FastAPI(title="ML API")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET","POST","PUT","DELETE"],
    allow_headers=origins,
)

# 1. 파라미터 파일 로드
pretrainedModelParams = torch.load("../results_transformer/best_model.pt", map_location='cpu')

# 2. 파라미터 분석을 통한 하이퍼파라미터 역추적
def infer_hyperparams(state_dict):
    detected_config = {
        "num_features": 4,   # 기본값
        "d_model": 512,      # 기본값
        "num_layers": 2,     # 기본값
        "nhead": 4           # 기본값
    }

    # (1) 입력 차원 (input_dim) 찾기
    for key, param in state_dict.items():
        if ('input' in key or 'embedding' in key) and 'weight' in key and param.dim() == 2:
            detected_config['num_features'] = param.shape[1]
            detected_config['d_model'] = param.shape[0]
            break

    # (2) 모델 깊이 (num_layers) 찾기
    max_layer_idx = -1
    for key in state_dict.keys():
        match = re.search(r'layers\.(\d+)\.', key)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx > max_layer_idx:
                max_layer_idx = layer_idx
    
    if max_layer_idx != -1:
        detected_config['num_layers'] = max_layer_idx + 1

    # (3) d_model 재확인
    for key, param in state_dict.items():
        if 'norm1.weight' in key or 'self_attn.out_proj.weight' in key:
            detected_config['d_model'] = param.shape[0]
            break

    return detected_config

# 3. 설정값 추출
config = infer_hyperparams(pretrainedModelParams)
print(f"✅ 감지된 설정: {config}")

# 4. 감지된 설정으로 모델 초기화
TFmodel = TransformerRegressor(
    num_features=config['num_features'],
    d_model=config['d_model'],
    num_layers=config['num_layers'],
    nhead=4,              
    dim_feedforward=config['d_model'] * 4, 
    dropout=0.1          
)

# 5. 파라미터 로드
TFmodel.load_state_dict(pretrainedModelParams)
TFmodel.eval()

@app.post("/predictBase", status_code=200)
async def predict_tf(x: List[float]):
    try:
        # [수정됨] detected_input_dim 대신 config['num_features'] 사용
        expected_dim = config['num_features']
        
        if len(x) != expected_dim:
            return {"error": f"입력 데이터 개수가 맞지 않습니다. (기대: {expected_dim}개, 실제: {len(x)}개)"}

        inbound_tensor = torch.tensor([x], dtype=torch.float32)
        
        with torch.no_grad(): 
            predBid = TFmodel(inbound_tensor)
            
        return {"predBid": predBid.item()} 
    except Exception as e:
        print(f"Error: {e}")
        return {"predBid": 9999, "error_msg": str(e)}
    
@app.get("/")
async def root():
    return {"message": "online"}

auth_token = "38H6WIHF5Hn1xV68lPnXu15Tutc_4PDGKRtxpJhbJuVdcUCEp"

ngrok.set_auth_token(auth_token)
ngrokTunnel = ngrok.connect(9999)
print("공용 URL", ngrokTunnel.public_url)

nest_asyncio.apply()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
