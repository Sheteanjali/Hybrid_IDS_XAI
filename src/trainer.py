import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
from preprocess import load_and_clean_data

RAW_DATA_PATH = "data/raw"

def build_hybrid_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_training():
    print("--- Step 1: Loading and Cleaning Data ---")
    (X_train, X_test, y_train, y_test), le = load_and_clean_data(RAW_DATA_PATH)
    
    num_classes = len(np.unique(y_train))
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    print(f"Dataset Loaded | Shape: {X_train.shape} | Classes: {num_classes}")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(le, "models/label_encoder.pkl")
    
    print("\n--- Step 2: Building & Training Hybrid Model ---")
    model = build_hybrid_model(input_shape, num_classes)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=512,
        verbose=1
    )
    
    model.save("models/hybrid_ids_model.h5")
    print("\n✅ Training Complete! Model saved to 'models/hybrid_ids_model.h5'")

if __name__ == "__main__":
    run_training()
