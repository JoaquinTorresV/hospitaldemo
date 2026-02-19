import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator import generate_patient, generate_dataset

def test_generate_patient_keys():
    patient = generate_patient("normal")
    expected_keys = [
        "hemoglobina", "hematocrito", "glucosa", "hba1c", "plaquetas",
        "leucocitos", "ferritina", "hierro", "colesterol", "trigliceridos",
        "insulina", "pcr", "vhs", "neutrofilos", "creatinina", "urea",
        "sodio", "potasio", "bilirrubina", "proteinas", "diagnostico"
    ]
    for key in expected_keys:
        assert key in patient

def test_hemoglobina_anemia():
    resultados = [generate_patient("anemia")["hemoglobina"] for _ in range(100)]
    promedio = sum(resultados) / len(resultados)
    assert promedio < 12.0


def test_generate_dataset_shape():
    df = generate_dataset(100)
    assert df.shape[0] == 100
    assert df.shape[1] == 21