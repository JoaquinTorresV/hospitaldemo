import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
import shap

def load_data():
    df = pd.read_csv("data/patient.csv")
    X = df.drop(columns=["diagnostico"])
    y = df["diagnostico"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le

def get_feature_importance(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    
    print("Ranking de exámenes por importancia:")
    for exam, score in importances.items():
        print(f"  {exam}: {score:.4f}")
    
    return model, importances, X_test, y_test

def evaluate_feature_subsets(X, y, importances):
    resultados = []
    
    for n in range(4, 21):
        top_features = importances.head(n).index.tolist()
        X_subset = X[top_features]
        
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        recall = recall_score(y_test, y_pred, average="macro")
        resultados.append({"n_examenes": n, "recall": round(recall, 4)}) # type: ignore
        print(f"  {n} exámenes: recall = {recall:.4f}")
    
    return pd.DataFrame(resultados)

if __name__ == "__main__":
    X, y_encoded, le = load_data()
    model, importances, X_test, y_test = get_feature_importance(X, y_encoded)
    print("\nEvaluando subconjuntos de exámenes:")
    resultados = evaluate_feature_subsets(X, y_encoded, importances)

