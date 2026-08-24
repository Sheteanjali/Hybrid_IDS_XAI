import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = "data/raw/cicids2017_cleaned.csv"
SCALER_PATH = "models/scaler.pkl"

print(f"📂 Fitting StandardScaler on {CSV_PATH}...")
df = pd.read_csv(CSV_PATH, nrows=5000)
df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

numeric_df = df.select_dtypes(include=[np.number])
raw_arr = numeric_df.iloc[:, :52].values

scaler = StandardScaler()
scaler.fit(raw_arr)

os.makedirs("models", exist_ok=True)
joblib.dump(scaler, SCALER_PATH)
print(f"✅ Successfully regenerated {SCALER_PATH}!")
