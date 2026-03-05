import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def load_and_clean_data(raw_data_path):
    all_files = glob.glob(os.path.join(raw_data_path, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_path}")
    
    print(f"Loading {len(all_files)} files...")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    
    # Cleaning
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Feature Selection & Encoding
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save artifacts for XAI
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    # Reshape for CNN-LSTM: (Samples, Features, 1)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
    
    return train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42), le