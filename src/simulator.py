import pandas as pd
import numpy as np
import joblib
import requests

def simulate_patient_flow(n_patients=1000):
    model = joblib.load(r"C:\Users\joaqu\OneDrive\Desktop\proyecto\mlruns\0\models\m-b76219a15b294d08b5c9ce82c033f346\artifacts\model.pkl")
    
    df = pd.read_csv("data/patient.csv")
    sample = df.sample(n=n_patients, random_state=42)
    
    examenes_base = ["pcr", "ferritina", "vhs", "insulina", "hemoglobina", "glucosa"]
    umbral = 0.85
    total_examenes_posibles = 20
    
    resultados = []
    
    for _, paciente in sample.iterrows():
        X = pd.DataFrame([paciente[examenes_base]])
        proba = model.predict_proba(X)[0]
        confianza = max(proba)
        clases = ["anemia", "diabetes", "infeccion", "normal"]
        diagnostico = clases[proba.argmax()]
        
        if confianza >= umbral:
            examenes_usados = len(examenes_base)
            decision = "confirmado"
        else:
            examenes_usados = total_examenes_posibles
            decision = "requirio_mas_examenes"
        
        resultados.append({
            "diagnostico_real": paciente["diagnostico"],
            "diagnostico_predicho": diagnostico,
            "confianza": round(confianza, 3),
            "decision": decision,
            "examenes_usados": examenes_usados,
            "examenes_ahorrados": total_examenes_posibles - examenes_usados
        })
    
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    df_resultados = simulate_patient_flow()
    
    total = len(df_resultados)
    confirmados = len(df_resultados[df_resultados["decision"] == "confirmado"])
    requirieron_mas = len(df_resultados[df_resultados["decision"] == "requirio_mas_examenes"])
    promedio_examenes = df_resultados["examenes_usados"].mean()
    total_ahorrados = df_resultados["examenes_ahorrados"].sum()
    
    print(f"Total pacientes simulados: {total}")
    print(f"Confirmados con 6 exámenes: {confirmados} ({confirmados/total*100:.1f}%)")
    print(f"Requirieron panel completo: {requirieron_mas} ({requirieron_mas/total*100:.1f}%)")
    print(f"Promedio de exámenes usados: {promedio_examenes:.1f} de 20 posibles")
    print(f"Total exámenes ahorrados en 1000 pacientes: {total_ahorrados}")