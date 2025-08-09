import pandas as pd
import numpy as np

def add_features(df):
    """
    Adds technical analysis features to the stock data DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame with stock data.

    Returns:
        pd.DataFrame: The DataFrame with added features.
    """
    df = df.copy()

    # Simple Moving Average (SMA)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Moving Average Convergence Divergence (MACD)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Drop rows with NaN values created by the indicators
    df.dropna(inplace=True)

    return df

if __name__ == '__main__':
    # Example usage:
    TICKER = 'RELIANCE.NS'
    FILEPATH = f'data/{TICKER}.csv'

    # Load the data
    try:
        data = pd.read_csv(FILEPATH, index_col='Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Data file not found at {FILEPATH}")
        print("Please run the data_loader.py script first.")
    else:
        # Add features
        featured_data = add_features(data)

        # Display the new features
        print(f"Data with features for {TICKER}:")
        print(featured_data.head())
        print("\nColumns added:")
        print([col for col in featured_data.columns if col not in data.columns])
