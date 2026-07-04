import pickle
import numpy as np

with open("models/xgb_multiclass.pkl", "rb") as f:
    model = pickle.load(f)

# Ver distribución de predicciones
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/processed/gold-features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
X = df.drop(columns=["Date", "target_binary", "target_multiclass"])
split_idx = int(len(X) * 0.8)
X_test = X.iloc[split_idx:]

import pickle
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

X_test_sc = scaler.transform(X_test)
preds = model.predict(X_test_sc)
print("Distribución predicciones XGB multiclase:", np.unique(preds, return_counts=True))