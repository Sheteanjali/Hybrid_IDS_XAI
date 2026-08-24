import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ---------------------------------------------------------
# 1. GENERATE OR LOAD DATA
# ---------------------------------------------------------
DATA_PATH = "data/processed/cicids2017_sample.csv"

if os.path.exists(DATA_PATH):
    print(f"📁 Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    # Ensure standard cleaning
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X = df.iloc[:, :52].values
    y = df.iloc[:, 52].values if df.shape[1] > 52 else np.random.randint(0, 2, size=len(df))
    feature_names = list(df.columns[:52])
else:
    print("⚠️ Dataset not found. Generating dummy synthetic dataset (4,000 samples, 52 features)...")
    np.random.seed(42)
    X = np.random.randn(4000, 52)
    # Simulate Port Scan features (e.g., Fwd/Bwd Packet Length Min)
    y = (X[:, 0] * 0.4 + X[:, 1] * 0.6 + np.random.randn(4000) * 0.1 > 0).astype(int)
    feature_names = [f"Feature_{i+1}" for i in range(52)]
    
    # Custom names matching paper
    feature_names[0] = "Fwd Packet Length Min"
    feature_names[1] = "Bwd Packet Length Min"
    feature_names[2] = "Flow Duration"
    feature_names[3] = "Total Length of Fwd Packets"

# Save feature names artifact
joblib.dump(feature_names, "models/feature_names.pkl")
print("✅ Saved models/feature_names.pkl")

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# 2. PREPROCESSING & SCALING
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Export Scaler
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Saved models/scaler.pkl")

# Reshape to 3D Tensor for 1D-CNN + LSTM: (samples, 52, 1)
X_train_tensor = X_train_scaled.reshape((X_train_scaled.shape[0], 52, 1))
X_test_tensor = X_test_scaled.reshape((X_test_scaled.shape[0], 52, 1))

# Export background reference set for SHAP (100 samples)
background_samples = X_train_tensor[:100]
np.save("models/background_samples.npy", background_samples)
print("✅ Saved models/background_samples.npy")

# ---------------------------------------------------------
# 3. BUILD HYBRID CNN-LSTM ARCHITECTURE (From Paper)
# ---------------------------------------------------------
model = Sequential([
    # 1D CNN: 64 filters, kernel size 3
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(52, 1), padding='same'),
    MaxPooling1D(pool_size=2),
    
    # LSTM: 64 units
    LSTM(units=64, return_sequences=False),
    
    # Dropout 0.3
    Dropout(0.3),
    
    # Dense Softmax
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # 2 classes: Benign vs Port Scanning
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------------------------------------
# 4. TRAIN MODEL & SAVE
# ---------------------------------------------------------
print("🚀 Training CNN-LSTM model...")
history = model.fit(
    X_train_tensor, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test_tensor, y_test),
    verbose=1
)

# Save Keras Model
model.save("models/cnn_lstm_model.keras")
print("✅ Saved models/cnn_lstm_model.keras")

# ---------------------------------------------------------
# 5. GENERATE EVALUATION PLOTS
# ---------------------------------------------------------
# Plot Training History
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('CNN-LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('static/training_history.png')
plt.close()

# Plot Confusion Matrix
y_pred = np.argmax(model.predict(X_test_tensor), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Benign', 'Port Scan'])
plt.yticks([0, 1], ['Benign', 'Port Scan'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')
plt.tight_layout()
plt.savefig('static/confusion_matrix.png')
plt.close()

# ---------------------------------------------------------
# 6. YOUR SHAP PLOT GENERATOR FUNCTION
# ---------------------------------------------------------
def generate_shap_plots(model, X_sample, feature_names, output_path):
    """Generates and saves the SHAP summary plot for the model predictions."""
    print("⏳ Computing SHAP values (this may take a few seconds)...")
    background = X_sample[:50] 
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_sample[:10])
    
    plt.figure(figsize=(10, 6))
    
    # Extract feature values for Class 1 (Port Scanning)
    if isinstance(shap_values, list):
        target_shap = shap_values[1]
    else:
        target_shap = shap_values

    # Reshape tensor back to 2D for the summary plot
    shap.summary_plot(
        target_shap.reshape(10, -1), 
        X_sample[:10].reshape(10, -1), 
        feature_names=feature_names, 
        show=False
    )
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved SHAP plot to {output_path}")

# Run SHAP plot generator
generate_shap_plots(model, X_test_tensor, feature_names, "static/xai_shap_plot.png")

print("\n🎉 Artifact generation complete! All files ready for main.py.")