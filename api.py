from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import predict_image
from chains.get_precautions import get_precautions_for_disease
from chains.get_translation import translate_precautions
from dotenv import load_dotenv

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
    if (message['result'] == False):
        return {"result": False, "message": message['message']}
    else:
        crop = message['crop']
        disease = message['disease']
        precautions = get_precautions_for_disease(crop, disease)
        precautions = translate_precautions(language, precautions)
        return {
            "result": True,
            "message": message['message'],
            "crop": crop,
            "disease": disease,
            "precautions": precautions
        }