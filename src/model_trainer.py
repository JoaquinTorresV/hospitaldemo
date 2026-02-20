import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("data/patient.csv")
    X = df.drop(columns=["diagnostico"])
    y = df["diagnostico"]
    return X, y

def train_models(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric="mlogloss")
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X, y_encoded, cv=5, scoring="recall_macro")
        print(f"Recall promedio (CV): {scores.mean():.3f} ± {scores.std():.3f}")

if __name__ == "__main__":
    X, y = load_data()
    train_models(X, y)