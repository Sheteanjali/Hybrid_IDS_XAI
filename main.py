import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
import shap

# --- SETTINGS ---
DATA_PATH = 'data/raw/'
RESULTS_PATH = 'results/figures/'
MODEL_PATH = 'models/'
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 1. LOAD & MERGE DATA
print("[1/5] Reading CSV files from data/raw/...")
all_files = glob.glob(os.path.join(DATA_PATH, "*.csv"))

if not all_files:
    print("❌ Error: No CSV files found in data/raw/!")
    exit()

# Load data subset for efficiency
df_list = [pd.read_csv(f, nrows=200000) for f in all_files]
df = pd.concat(df_list, axis=0, ignore_index=True)

# 2. DATA CLEANING
print("[2/5] Cleaning data and fixing column names...")
df.columns = df.columns.str.strip() 

# Target column identification
if 'Label' in df.columns:
    target_col = 'Label'
elif 'Attack Type' in df.columns:
    target_col = 'Attack Type'
else:
    target_col = df.columns[-1]

print(f"Target column identified as: '{target_col}'")

# Clean Infinity and NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Separate features and label
X = df.drop([target_col], axis=1)
y = df[target_col]

# Keep only numeric features
X = X.select_dtypes(include=[np.number])

# Encoding & Scaling
le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# CNN-LSTM Reshape: (Samples, Features, 1)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_encoded, test_size=0.2, random_state=42)

# 3. BUILD HYBRID CNN-LSTM MODEL
print("[3/5] Initializing Hybrid CNN-LSTM Model...")
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. TRAINING
print("[4/5] Training model...")
history = model.fit(X_train, y_train, epochs=5, batch_size=1024, validation_split=0.1)

# Save Model
model.save(f'{MODEL_PATH}ids_hybrid_model.keras')

# 5. RESULTS & XAI (SHAP)
print("[5/5] Generating Results and XAI Plot...")

# Classification Report
y_pred = np.argmax(model.predict(X_test), axis=1)
print("\n--- PERFORMANCE REPORT ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.savefig(f'{RESULTS_PATH}confusion_matrix.png')

# SHAP Fix: Handling multi-class output and 3D data
background = X_train[:50]
test_instances = X_test[:10]

explainer = shap.GradientExplainer(model, background)
shap_values = explainer.shap_values(test_instances)

# Process SHAP values to fix the Reshape ValueError
# Array size 1040 = 10 samples * 52 features * 2 classes
if isinstance(shap_values, list):
    # For list outputs, extract the first class
    shap_viz = np.array(shap_values[0]).reshape(10, X_train.shape[1])
else:
    # Reshape to (samples, features, classes) and slice the first class index
    temp_shap = shap_values.reshape(10, X_train.shape[1], -1)
    shap_viz = temp_shap[:, :, 0] 

# Reshape test instances for plotting (10, 52, 1) -> (10, 52)
test_instances_viz = test_instances.reshape(10, X_train.shape[1])

# Save SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_viz, test_instances_viz, feature_names=X.columns.tolist(), show=False)
plt.title("XAI Feature Importance (SHAP)")
plt.savefig(f'{RESULTS_PATH}xai_shap_plot.png', bbox_inches='tight')

print("\n" + "="*40)
print("SUCCESS: Full Project Run Complete!")
print(f"Check results in: {RESULTS_PATH}")
print("="*40)
# Accuracy vs Loss Plot code
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.savefig('results/figures/training_history.png')