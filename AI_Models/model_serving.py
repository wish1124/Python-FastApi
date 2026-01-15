import BidAssitanceModel
import unicorn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch


app = FastAPI(title="ML API")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET","POST","PUT","DELETE"],
    allow_headers=origins,
)

pretrainedModelParams = torch.load("../results_transformer/best_model.pt")

