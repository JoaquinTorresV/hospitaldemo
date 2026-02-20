import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss

def load_data():
    df = pd.read_csv("data/patient.csv")
    X = df.drop(columns=["diagnostico"])
    y = df["diagnostico"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le

def train_calibrated_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    calibrated_model = CalibratedClassifierCV(base_model, cv=5, method="sigmoid")
    calibrated_model.fit(X_train, y_train)
    
    probabilities = calibrated_model.predict_proba(X_test)
    
    return calibrated_model, probabilities, X_test, y_test

def show_probabilities(calibrated_model, X_test, le, n_patients=5):
    probabilities = calibrated_model.predict_proba(X_test)
    
    print("\nProbabilidades para los primeros 5 pacientes:")
    for i in range(n_patients):
        print(f"\nPaciente {i+1}:")
        for clase, prob in zip(le.classes_, probabilities[i]):
            print(f"  {clase}: {prob*100:.1f}%")

def evaluate_calibration(probabilities, y_test, le):
    print("\nBrier Score por diagnóstico (menor es mejor):")
    for i, clase in enumerate(le.classes_):
        y_binary = (y_test == i).astype(int)
        score = brier_score_loss(y_binary, probabilities[:, i])
        print(f"  {clase}: {score:.4f}")


if __name__ == "__main__":
    X, y_encoded, le = load_data()
    calibrated_model, probabilities, X_test, y_test = train_calibrated_model(X, y_encoded)
    show_probabilities(calibrated_model, X_test, le)
    evaluate_calibration(probabilities, y_test, le)