import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker, start_date, end_date, filepath):
    """
    Downloads historical stock data from Yahoo Finance and saves it to a CSV file.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'RELIANCE.NS').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
        filepath (str): The path to save the CSV file.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Download the data
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    # Collapse the multi-level column headers
    data.columns = data.columns.get_level_values(0)

    if data.empty:
        print(f"No data found for ticker {ticker}. It may be delisted or an invalid ticker.")
        return

    # Reset index to make 'Date' a column, then save without the pandas index
    data.reset_index(inplace=True)
    data.to_csv(filepath, index=False)
    print(f"Data for {ticker} saved to {filepath}")

if __name__ == '__main__':
    # Example usage:
    TICKER = 'RELIANCE.NS'
    START_DATE = '2020-01-01'
    END_DATE = '2023-12-31'
    FILEPATH = f'data/{TICKER}.csv'

    download_stock_data(TICKER, START_DATE, END_DATE, FILEPATH)
