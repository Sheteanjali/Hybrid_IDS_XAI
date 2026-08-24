import os
import joblib
import numpy as np
import pandas as pd

# Paths matching your directory structure
CSV_PATH = "data/raw/cicids2017_cleaned.csv"
SCALER_PATH = "models/scaler.pkl"
OUTPUT_PATH = "models/background_samples.npy"

def create_background_samples():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Raw CSV file not found at '{CSV_PATH}'")
        return

    if not os.path.exists(SCALER_PATH):
        print(f"❌ Error: StandardScaler file not found at '{SCALER_PATH}'")
        return

    print(f"📂 Loading first 100 rows from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, nrows=100)

    # 1. Clean Inf and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 2. Extract numeric features (Drop non-numeric or target label columns if present)
    numeric_df = df.select_dtypes(include=[np.number])

    # Ensure exactly 52 features to match model input shape
    raw_arr = numeric_df.iloc[:, :52].values
    if raw_arr.shape[1] < 52:
        pad_width = 52 - raw_arr.shape[1]
        raw_arr = np.pad(raw_arr, ((0, 0), (0, pad_width)), mode='constant')
    elif raw_arr.shape[1] > 52:
        raw_arr = raw_arr[:, :52]

    # 3. Apply StandardScaler
    print(f"⚙️ Applying StandardScaler from {SCALER_PATH}...")
    scaler = joblib.load(SCALER_PATH)
    scaled_arr = scaler.transform(raw_arr)

    # 4. Reshape to 3D Tensor: (100, 52, 1) for 1D-CNN + LSTM
    tensor_arr = scaled_arr.reshape((scaled_arr.shape[0], 52, 1))

    # 5. Save to models directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, tensor_arr)
    
    print(f"✅ Success! Saved background samples tensor with shape {tensor_arr.shape} to '{OUTPUT_PATH}'.")

if __name__ == "__main__":
    create_background_samples()