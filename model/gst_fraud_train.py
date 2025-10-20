import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

print("Starting the GST fraud model training process with Random Forest...")

# Define file paths
data_path = os.path.join("models", "gst_fraud_data.csv")
model_path = os.path.join("models", "gst_fraud_model.pkl")

# Check if data file exists
if not os.path.exists(data_path):
    print(f"Error: Dataset not found at '{data_path}'. Please ensure the file is in the 'models' directory.")
else:
    try:
        df = pd.read_csv(data_path)
        required_columns = [
            'reported_turnover', 'eway_total_value', 'num_eway_bills', 
            'invoice_count', 'gst_paid', 'gst_rate_applied', 
            'num_suppliers', 'num_customers', 'avg_invoice_value', 
            'gstr1_vs_gstr3b_diff', 'sudden_turnover_jump', 
            'registration_age_months', 'fraud'
        ]
        if not all(col in df.columns for col in required_columns):
            print("Error: The CSV file is missing one or more required columns.")
        else:
            X = df.drop('fraud', axis=1)
            y = df['fraud']
            
            # Use RandomForestClassifier as specified in your abstract
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            
            print(f"✅ GST fraud model successfully trained and saved to '{model_path}'.")

    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
