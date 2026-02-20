from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np
import pandas as pd 
import joblib

app = FastAPI()

model = joblib.load(r"C:\Users\joaqu\OneDrive\Desktop\proyecto\mlruns\0\models\m-b76219a15b294d08b5c9ce82c033f346\artifacts\model.pkl")

class PacienteInput(BaseModel):
    pcr: float
    ferritina: float
    vhs: float
    insulina: float
    hemoglobina: float
    glucosa: float

@app.post("/predict")
def predict(paciente: PacienteInput):
    data = pd.DataFrame([paciente.model_dump()])
    probabilities = model.predict_proba(data)[0] # type: ignore
    clases = ["anemia", "diabetes", "infeccion", "normal"]
    
    resultado = {clase: round(float(prob * 100), 1) for clase, prob in zip(clases, probabilities)}
    diagnostico = max(resultado, key=resultado.get) # type: ignore
    
    return {
        "diagnostico": diagnostico,
        "probabilidades": resultado
    }