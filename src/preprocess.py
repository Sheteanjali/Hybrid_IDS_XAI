import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data(data_path):
    files = glob.glob(f"{data_path}/*.csv")
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")
    
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        df_list.append(df)
        
    df = pd.concat(df_list, ignore_index=True)
    
    label_col = 'Attack Type'
    if label_col not in df.columns:
        raise KeyError(f"Expected '{label_col}' column in dataset.")
        
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df.drop(columns=[label_col])
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X = X[numeric_cols].iloc[:, :52]
    
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scaler save karein taaki future deployment/SHAP scripts me issue na aaye
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    # 2D (samples, 52) se 3D (samples, 52, 1) me reshape karein
    X_scaled_3d = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
    
    return train_test_split(X_scaled_3d, y, test_size=0.2, random_state=42), le
