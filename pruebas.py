import pickle
import pandas as pd

with open("models/rf_optimized_binary.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("data/processed/gold-features.csv", parse_dates=["Date"])
features = df.drop(columns=["Date", "target_binary", "target_multiclass"]).columns

importances = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print(importances.tail(20))