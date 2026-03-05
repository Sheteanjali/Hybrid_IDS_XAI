import sys
import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Ensure the script can see the other files in /src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import load_and_clean_data
from model_arch import build_hybrid_model

def run_training():
    # 1. Configuration
    RAW_DATA_PATH = "data/raw/"  # Where your CICIDS2017 CSV files are
    MODEL_SAVE_PATH = "models/ids_hybrid_model.h5"
    
    # 2. Load and Preprocess Data
    print("--- Step 1: Loading and Cleaning Data ---")
    (X_train, X_test, y_train, y_test), le = load_and_clean_data(RAW_DATA_PATH)
    
    num_classes = len(le.classes_)
    input_shape = (X_train.shape[1], 1)
    
    # 3. Build the Hybrid Architecture
    print(f"--- Step 2: Building CNN-LSTM Model for {num_classes} classes ---")
    model = build_hybrid_model(input_shape, num_classes)
    model.summary()
    
    # 4. Define Callbacks (To ensure high quality results for your paper)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3)
    ]
    
    # 5. Training
    print("--- Step 3: Starting Training ---")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}")
    
    # 7. Evaluation
    print("--- Step 4: Final Evaluation ---")
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=-1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

if __name__ == "__main__":
    run_training()