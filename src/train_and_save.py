import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import mlflow
import mlflow.sklearn

def train_and_save():
    df = pd.read_csv("data/patient.csv")
    X = df.drop(columns=["diagnostico"])
    y = df["diagnostico"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    top_features = ["pcr", "ferritina", "vhs", "insulina", "hemoglobina", "glucosa"]
    X = X[top_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        model = CalibratedClassifierCV(base_model, cv=5, method="sigmoid")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        recall = recall_score(y_test, y_pred, average="macro")
        
        mlflow.log_param("n_features", len(top_features))
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("recall", recall) # type: ignore
        mlflow.sklearn.log_model(model, "model") # type: ignore
        
        print(f"Recall: {recall:.4f}")
        print("Modelo guardado en MLflow")

if __name__ == "__main__":
    train_and_save()