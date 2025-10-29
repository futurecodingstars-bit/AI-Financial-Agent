# config/settings.py

# --- 1. General Agent Settings ----------------------------------------------

# The primary stock ticker the agent will analyze. (Example: Apple)
TICKER = "AAPL"

# --- 2. Data Acquisition (PERCEIVE Phase) -----------------------------------

# Date range for historical data used for training and feature engineering.
# The end date should typically be the date before the agent starts live operation.
START_DATE = "2018-01-01"
END_DATE = "2024-06-30"

# --- 3. Feature Engineering Parameters --------------------------------------

# Periods for the Dual Moving Average (MA) Crossover Strategy (used as features)
# Short-term MA (Fast signal)
MA_SHORT_PERIOD = 50 
# Long-term MA (Slow signal)
MA_LONG_PERIOD = 200

# Period for calculating Volatility (e.g., 20 trading days ≈ 1 month)
VOLATILITY_PERIOD = 20

# --- 4. Machine Learning (DECIDE Phase) -------------------------------------

# Lookahead period (in days) to determine the target variable (future performance).
# Example: We predict the price change 5 days into the future.
TARGET_LOOKAHEAD_DAYS = 5

# Threshold (%) for defining a 'Buy' signal (Target=1). 
# If price increases by this percent or more in TARGET_LOOKAHEAD_DAYS, Target = 1.
TARGET_BUY_THRESHOLD_PERCENT = 1.0  # (i.e., 1.0% gain)

# Features used to train the Random Forest Classifier (must match feature_engineering.py)
ML_FEATURES = [
    'SMA_50', 
    'LMA_200', 
    'Volatility', 
    'Sentiment_Score'  # Added after LLM integration
]

# Random Forest Classifier Hyperparameters
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42
# Percentage of data to reserve for testing the model's performance
TEST_SIZE = 0.2

# --- 5. Model Paths and LLM Settings ----------------------------------------

# Path to save and load the trained ML model (Random Forest Classifier).
MODEL_PATH = "models/RandomForest_Classifier.pkl"

# Placeholder text to simulate aggregated news headlines for LLM sentiment analysis
# In a real app, this would be scraped/fetched live.
LLM_MOCK_NEWS_HEADLINES = (
    "Apple announces record-breaking services revenue and plans "
    "a massive stock buyback, thrilling investors and driving the price up."
)

# --- 6. Data Paths ----------------------------------------------------------

RAW_DATA_PATH = "data/raw_data/stock_data_raw.csv"
PROCESSED_DATA_PATH = "data/processed_data/stock_data_processed.csv"