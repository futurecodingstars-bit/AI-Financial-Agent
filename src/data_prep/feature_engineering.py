# src/data_prep/feature_engineering.py

import pandas as pd
import numpy as np

def calculate_numerical_features(data: pd.DataFrame, window: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    """
    Calculates numerical features (MAs, Volatility) and the target variable.
    
    This function ensures data integrity and alignment by using df.dropna 
    and robust column selection.

    Args:
        data (pd.DataFrame): DataFrame containing historical price data (Close, Open, etc.).
        window (int): The lookback window for rolling calculations (e.g., 20 days).

    Returns:
        tuple[pd.DataFrame, pd.Series]: A tuple containing the aligned features (X) 
                                         and the target variable (y).
    """
    df = data.copy()

    # --- 1. Calculate Numerical Features ---
    print(f"[PERCEIVE] Calculating numerical features (MAs, Volatility with window={window})...")
    
    # Simple Moving Averages (SMA) - Now calculating the missing ones
    df['SMA_20'] = df['Close'].rolling(window=20).mean() # Rename generic MA to SMA_20
    df['SMA_50'] = df['Close'].rolling(window=50).mean() # NEW: Short-term MA
    df['LMA_200'] = df['Close'].rolling(window=200).mean() # NEW: Long-term MA
    
    # Volatility (Standard Deviation of percentage change)
    df['Pct_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Pct_Change'].rolling(window=window).std()
    
    # Relative Price Position (Close price relative to the 20-day average)
    # Uses SMA_20 instead of the old 'MA'
    df['Price_MA_Ratio'] = df['Close'].squeeze() / df['SMA_20'].squeeze() 

    # --- 2. Create Target Variable (Future Prediction) ---
    # Target: 1 if the 'Close' price increases the next day, 0 otherwise.
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # --- 3. CRITICAL ALIGNMENT AND NaN HANDLING ---
    rows_before = len(df)
    # The LMA_200 calculation introduces 200 NaN rows, so we drop them.
    df.dropna(inplace=True) 
    rows_after = len(df)

    print(f"[PERCEIVE] Data integrity checked and features aligned. Dropped {rows_before - rows_after} rows with NaN.")

    # --- 4. Separate Features (X) and Target (y) ---
    features = [
        'Open', 
        'High', 
        'Low', 
        'Close', 
        'Volume', 
        'SMA_20', 
        'SMA_50',       # Include new feature
        'LMA_200',      # Include new feature
        'Volatility', 
        'Price_MA_Ratio'
    ]
    
    X = df[features]
    y = df['Target']
    
    return X, y
