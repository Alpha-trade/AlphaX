import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from src.feature_engineering import add_features
import datetime as dt

# --- Configuration ---
MODEL_PATH = 'stock_predictor.h5'
SCALER_PATH = 'scaler.joblib'
TIMESTEPS = 60

def get_signal(ticker):
    """
    Generates a trading signal for a given stock ticker.

    Args:
        ticker (str): The stock ticker to predict.

    Returns:
        str: A trading signal ('BUY', 'SELL', or 'HOLD').
    """
    # --- Load Model and Scaler ---
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except IOError as e:
        return f"Error loading model or scaler: {e}. Please train the model first."

    # --- Fetch Latest Data ---
    # Fetch more data than needed to ensure indicators can be calculated
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=150)

    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
        if data.empty:
            return f"No data found for ticker {ticker}."
    except Exception as e:
        return f"Error fetching data for {ticker}: {e}"

    # --- Preprocess Data ---
    # Add features
    data_featured = add_features(data.copy())

    if len(data_featured) < TIMESTEPS:
        return "Not enough data to make a prediction."

    # Get the last 'TIMESTEPS' rows
    last_60_days = data_featured.iloc[-TIMESTEPS:]

    # Define features to be scaled
    features = [col for col in last_60_days.columns if col not in ['Date', 'Target']]

    # Scale the features
    last_60_days_scaled = scaler.transform(last_60_days[features])

    # Reshape for the model
    X_pred = np.reshape(last_60_days_scaled, (1, TIMESTEPS, len(features)))

    # --- Make Prediction ---
    prediction = model.predict(X_pred)[0][0]

    # --- Generate Signal ---
    if prediction > 0.55:
        return "BUY"
    elif prediction < 0.45:
        return "SELL"
    else:
        return "HOLD"

if __name__ == '__main__':
    # Example usage:
    ticker_to_predict = 'RELIANCE.NS'
    print(f"Generating signal for {ticker_to_predict}...")
    signal = get_signal(ticker_to_predict)
    print(f"The predicted signal for {ticker_to_predict} is: {signal}")

    # Example with a different ticker
    ticker_to_predict = 'TCS.NS'
    print(f"\nGenerating signal for {ticker_to_predict}...")
    signal = get_signal(ticker_to_predict)
    print(f"The predicted signal for {ticker_to_predict} is: {signal}")
