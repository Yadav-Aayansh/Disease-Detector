# api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import predict_image
from dotenv import load_dotenv
from solution import get_disease_solution_translated

load_dotenv()

app = FastAPI()

# Allow local frontend (Streamlit) to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/{language}")
async def predict(language: str, file: UploadFile = File(...)):
    bytes_data = await file.read()
    message = predict_image(bytes_data)

    if not message['result']:
        return {"result": False, "message": message['message']}

    crop = message['crop']
    disease = message['disease']
    precautions = get_disease_solution_translated(crop, disease, language)

    return {
        "result": True,
        "message": message['message'],
        "crop": crop,
        "disease": disease,
        "precautions": precautions
    }
