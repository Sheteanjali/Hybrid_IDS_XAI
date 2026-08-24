import os
import joblib
import pandas as pd

CSV_PATH = "data/raw/cicids2017_cleaned.csv"
OUTPUT_PATH = "models/feature_names.pkl"

def extract_feature_names():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: Raw CSV file not found at '{CSV_PATH}'")
        return

    print(f"📂 Reading columns from {CSV_PATH}...")
    # Read only the header row to save memory and time
    df = pd.read_csv(CSV_PATH, nrows=1)

    # Clean column names (strip leading/trailing whitespace common in CICIDS2017)
    df.columns = df.columns.str.strip()

    # Filter out non-feature or metadata/label columns if they exist in your dataset
    ignore_cols = ['Label', 'Timestamp', 'Flow ID', 'Source IP', 'Source Port', 
                   'Destination IP', 'Destination Port', 'Protocol']
    
    feature_cols = [col for col in df.columns if col not in ignore_cols]

    # Enforce exactly the first 52 features expected by the model architecture
    feature_cols = feature_cols[:52]

    # Save list to pkl file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    joblib.dump(feature_cols, OUTPUT_PATH)

    print(f"✅ Success! Saved {len(feature_cols)} feature names to '{OUTPUT_PATH}':")
    print(feature_cols[:5], "... and", len(feature_cols) - 5, "more.")

if __name__ == "__main__":
    extract_feature_names()