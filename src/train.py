import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from src.data_loader import download_stock_data
from src.feature_engineering import add_features
from src.model import create_lstm_model
import os
import joblib

# --- Configuration ---
TICKER = 'RELIANCE.NS'
DATA_PATH = f'data/{TICKER}.csv'
MODEL_SAVE_PATH = 'stock_predictor.h5'
SCALER_SAVE_PATH = 'scaler.joblib'
TIMESTEPS = 60  # How many past days of data to use for a single prediction

def prepare_data(data, timesteps):
    """Prepares the data for the LSTM model."""
    # Create the target variable: 1 if next day's close is higher, 0 otherwise
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    # We need to drop the last row since it has no target
    data = data.iloc[:-1]

    # Define features and target
    features = [col for col in data.columns if col not in ['Target', 'Date']]
    X = data[features]
    y = data['Target']

    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - timesteps):
        X_seq.append(X_scaled[i:i + timesteps])
        y_seq.append(y.iloc[i + timesteps])

    return np.array(X_seq), np.array(y_seq), scaler

def main():
    """Main function to train the model."""
    # --- Load and Process Data ---
    if not os.path.exists(DATA_PATH):
        print(f"Data for {TICKER} not found. Downloading...")
        download_stock_data(TICKER, '2015-01-01', '2023-12-31', DATA_PATH)

    df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
    df_featured = add_features(df)

    X, y, scaler = prepare_data(df_featured, TIMESTEPS)

    # Split data into training and testing sets (80% train, 20% test)
    # Important: shuffle=False for time series data
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- Build and Train Model ---
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)

    print("\nStarting model training...")
    history = model.fit(
        X_train, y_train,
        epochs=10,  # Using a small number of epochs for this example
        batch_size=32,
        validation_split=0.1, # Use part of training data for validation
        verbose=1
    )

    # --- Evaluate Model ---
    print("\nEvaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- Save Model and Scaler ---
    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print(f"Scaler saved to {SCALER_SAVE_PATH}")


if __name__ == '__main__':
    main()
