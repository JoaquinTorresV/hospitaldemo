import numpy as np
import pandas as pd

def generate_patient(diagnostico):
    if diagnostico == "anemia":
        hemoglobina = np.random.normal(loc=9.0, scale=1.0)
    elif diagnostico == "normal":
        hemoglobina = np.random.normal(loc=14.0, scale=1.0)
    else:
        hemoglobina = np.random.normal(loc=13.5, scale=1.0)

    hematocrito = hemoglobina * 3 + np.random.normal(loc=0, scale=0.5)

    if diagnostico == "diabetes":
        glucosa = np.random.normal(loc=180.0, scale=20.0)
    else:
        glucosa = np.random.normal(loc=95.0, scale=15.0)

    hba1c = glucosa * 0.035 + np.random.normal(loc=0, scale=0.2)

    plaquetas = np.random.normal(loc=250000, scale=50000)
    leucocitos = np.random.normal(loc=7000, scale=1500)

        # Anemia
    ferritina = np.random.normal(loc=15.0, scale=5.0) if diagnostico == "anemia" else np.random.normal(loc=80.0, scale=20.0)
    hierro = np.random.normal(loc=40.0, scale=10.0) if diagnostico == "anemia" else np.random.normal(loc=100.0, scale=20.0)

    # Diabetes
    colesterol = np.random.normal(loc=220.0, scale=20.0) if diagnostico == "diabetes" else np.random.normal(loc=170.0, scale=20.0)
    trigliceridos = np.random.normal(loc=250.0, scale=40.0) if diagnostico == "diabetes" else np.random.normal(loc=120.0, scale=30.0)
    insulina = np.random.normal(loc=25.0, scale=5.0) if diagnostico == "diabetes" else np.random.normal(loc=8.0, scale=2.0)

    # Infeccion
    pcr = np.random.normal(loc=50.0, scale=15.0) if diagnostico == "infeccion" else np.random.normal(loc=3.0, scale=1.0)
    vhs = np.random.normal(loc=60.0, scale=15.0) if diagnostico == "infeccion" else np.random.normal(loc=10.0, scale=5.0)
    neutrofilos = np.random.normal(loc=85.0, scale=5.0) if diagnostico == "infeccion" else np.random.normal(loc=55.0, scale=10.0)

    # Generales
    creatinina = np.random.normal(loc=0.9, scale=0.2)
    urea = np.random.normal(loc=30.0, scale=8.0)
    sodio = np.random.normal(loc=140.0, scale=3.0)
    potasio = np.random.normal(loc=4.0, scale=0.4)
    bilirrubina = np.random.normal(loc=0.8, scale=0.2)
    proteinas = np.random.normal(loc=7.0, scale=0.5)

    return {
        "hemoglobina": round(hemoglobina, 2),
        "hematocrito": round(hematocrito, 2),
        "glucosa": round(glucosa, 2),
        "hba1c": round(hba1c, 3),
        "plaquetas": round(plaquetas, 0),
        "leucocitos": round(leucocitos, 0),
        "diagnostico": diagnostico,
        "ferritina": round(ferritina, 2),
        "hierro": round(hierro, 2),
        "colesterol": round(colesterol, 2),
        "trigliceridos": round(trigliceridos, 2),
        "insulina": round(insulina, 2),
        "pcr": round(pcr, 2),
        "vhs": round(vhs, 2),
        "neutrofilos": round(neutrofilos, 2),
        "creatinina": round(creatinina, 3),
        "urea": round(urea, 2),
        "sodio": round(sodio, 2),
        "potasio": round(potasio, 2),
        "bilirrubina": round(bilirrubina, 3),
        "proteinas": round(proteinas, 2)
    }

def generate_dataset(n_patients=10000):
    diagnosticos = ["normal", "anemia", "diabetes", "infeccion"]
    patients = []

    for _ in range(n_patients):
        diagnostico = np.random.choice(diagnosticos)
        patients.append(generate_patient(diagnostico))

    df = pd.DataFrame(patients)
    return df

if __name__ == "__main__":
    df= generate_dataset()
    df.to_csv("data/patient.csv", index=False)
    print(f"Dataset generado: {df.shape[0]} pacientes, {df.shape[1]} exámenes.")