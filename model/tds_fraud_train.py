import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import numpy as np

print("🚀 Starting TDS fraud model training...")

# Define file paths
data_path = os.path.join("models", "tds_fraud_data.csv")
model_path = os.path.join("models", "tds_fraud_model.pkl")

# Define the features (X) and target (y)
required_columns = [
    "deductor_pan", "deductee_pan", "payment_amount", 
    "tds_rate", "tds_deducted", "tds_deposited", 
    "section_code", "nature_of_payment", 
    "date_of_payment", "is_fraud"
]

# Check if data file exists
if not os.path.exists(data_path):
    print(f"❌ Error: Dataset not found at '{data_path}'. Please ensure the file is in the 'models' directory.")
else:
    try:
        df = pd.read_csv(data_path)

        if not all(col in df.columns for col in required_columns):
            print("❌ Error: The CSV file is missing one or more required columns.")
        else:
            # Handle date and categorical data
            df['date_of_payment'] = pd.to_datetime(df['date_of_payment'])
            df['day_of_week'] = df['date_of_payment'].dt.dayofweek
            df['month'] = df['date_of_payment'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

            # Convert categorical columns to dummy variables
            df = pd.get_dummies(df, columns=["section_code", "nature_of_payment"], dtype=int)

            # Drop original non-numeric columns after feature engineering
            X = df.drop(
                ["is_fraud", "deductor_pan", "deductee_pan", "date_of_payment"], 
                axis=1
            )
            y = df["is_fraud"]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Save the trained model with feature names
            with open(model_path, "wb") as f:
                # We save a dictionary containing both the model and the feature names
                pickle.dump({'model': model, 'features': list(X.columns)}, f)

            print(f"✅ TDS fraud model successfully trained and saved to '{model_path}'.")

    except Exception as e:
        print(f"⚠️ Unexpected error during training: {e}")
