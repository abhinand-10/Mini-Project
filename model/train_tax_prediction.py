import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# The corrected file path now looks for the CSV in the current directory.
DATA_PATH = "tax_prediction.csv"

def train_model():
    """Trains a tax prediction model and saves it."""
    try:
        # Load the dataset
        df = pd.read_csv(DATA_PATH)
        print("Data loaded successfully.")

        # We will use 'income' as the feature and 'tax_paid' as the target
        # based on the columns available in your CSV file.
        X = df[['income']]
        y = df['tax_paid']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Save the trained model
        model_filename = "tax_prediction_model.pkl"
        joblib.dump(model, model_filename)
        print(f"Model trained and saved as {model_filename}.")

    except FileNotFoundError:
        print(f"❌ Error: The file '{DATA_PATH}' was not found. Please ensure it is in the 'models' directory.")
    except KeyError as e:
        print(f"❌ Error: The column {e} was not found in the CSV file. Please check the column names.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    print("Starting tax prediction model training...")
    train_model()
