# src/data_prep/data_loader.py

import yfinance as yf
import pandas as pd
import os
from config.settings import START_DATE, END_DATE, TICKER

def load_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE):
    """
    Downloads historical stock data from Yahoo Finance and performs initial validation.
    
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for data download (YYYY-MM-DD).
        end_date (str): End date for data download (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame containing historical price data, or raises an error.
    """
    print(f"[PERCEIVE] Loading raw data for {ticker}...")
    
    # auto_adjust=True handles stock splits and dividends automatically
    # threads=False is used for better stability in single-ticker downloads
    try:
        # Note: Setting auto_adjust=True explicitly to prevent a FutureWarning
        data = yf.download(
            ticker, 
            start=start_date, 
            end=end_date, 
            auto_adjust=True,
            threads=False
        )
    except Exception as e:
        # Handles network issues or yfinance specific errors
        raise ConnectionError(f"Failed to download data for ticker {ticker}. Error: {e}")

    # --- CRITICAL DATA INTEGRITY CHECK ---
    if data.empty:
        # If the download was successful but returned no data (e.g., bad date range, delisted ticker)
        raise ValueError(f"No historical data returned for ticker: {ticker} in the specified date range. Check the ticker and dates in settings.py.")

    # Optional: Ensure the index is a standard DatetimeIndex without timezone info 
    # to prevent potential alignment issues with feature calculations later.
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Clean the DataFrame to keep only necessary columns for feature engineering
    # We use .copy() to ensure we are working on a clean, independent DataFrame
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    return data

# --- You may remove or keep the following helper function if you were using it ---
# def save_data(data, ticker=TICKER):
#     """Saves the raw data to a local file."""
#     # This function is typically used for caching, which is not strictly necessary for this small agent.
#     pass