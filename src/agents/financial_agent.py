# src/agents/financial_agent.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from config import settings
from src.data_prep.data_loader import load_data
from src.data_prep.feature_engineering import calculate_numerical_features
from src.llm_sentiment.sentiment_analyzer import get_current_sentiment_score

# --- DECIDE Phase: Training the AI Brain ---

def train_ai_brain():
    """
    DECIDE: Orchestrates data processing, trains the ML model, and saves it.
    """
    print("\n--- DECIDE PHASE: Training AI Brain ---")
    
    # 1. PERCEIVE: Get Data and Features (Numerical)
    data = load_data()
    # FIX: UNPACK the tuple (X, y) returned by calculate_numerical_features
    X, y = calculate_numerical_features(data)
    
    # 2. Add LLM Sentiment Feature (Mock for all historical rows for simplicity)
    # We add the mock feature directly to the features DataFrame (X).
    # This avoids the 'tuple' object does not support item assignment error.
    X['Sentiment_Score'] = 0.5 
    
    # Final data preparation for training
    # The full list of features now includes the sentiment score
    features = settings.ML_FEATURES + ['Sentiment_Score']
    
    # Ensure X contains only the required feature columns (order matters for the model)
    X = X[features]

    if X.empty or y.empty:
        raise ValueError("Data is empty after feature calculation. Check date range or rolling periods.")
    
    # Split data for training (shuffle=False is standard for time-series data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.TEST_SIZE, shuffle=False
    )
    
    print(f"Training on {len(X_train)} samples.")
    
    # 3. Train the Model
    model = RandomForestClassifier(
        n_estimators=settings.RF_N_ESTIMATORS, 
        random_state=settings.RF_RANDOM_STATE,
        class_weight='balanced' # Helps with imbalanced buy/hold targets
    )
    model.fit(X_train, y_train)
    
    # Evaluate and Save
    accuracy = model.score(X_test, y_test)
    print(f"DECIDE (AI Brain Accuracy): {accuracy:.4f}")
    
    joblib.dump(model, settings.MODEL_PATH)
    print(f"DECIDE: Model saved to {settings.MODEL_PATH}")
    
    return model

# --- ACT Phase: Prediction Logic ---

def load_ai_brain():
    """Loads the trained ML model from the disk."""
    print(f"[INFO] Loading trained model from {settings.MODEL_PATH}...")
    try:
        model = joblib.load(settings.MODEL_PATH)
        return model
    except FileNotFoundError:
        print("ERROR: Model not found. Please run main.py with 'train' argument first.")
        return None

def act_on_signal(model):
    """
    ACT: Gathers latest data, uses the trained model to predict the action, 
    and returns a signal.
    """
    print("\n--- ACT PHASE: Generating Signal ---")
    
    # 1. PERCEIVE: Get the absolute latest data point needed for prediction
    data = load_data() 
    # FIX: UNPACK the tuple (X_full, y_full). We only need X_full for prediction.
    X_full, _ = calculate_numerical_features(data) # Apply features
    
    # 2. PERCEIVE: Get the current LLM Sentiment Score
    current_sentiment = get_current_sentiment_score()
    
    # Prepare the latest data point for the model
    # The full list of features must include Sentiment_Score
    features = settings.ML_FEATURES + ['Sentiment_Score']

    # Get the latest row from the X_full DataFrame
    # This works because X_full is now a DataFrame, not a tuple.
    latest_data_row = X_full.iloc[[-1]].copy()
    
    # Add the live Sentiment_Score to the prediction row
    latest_data_row['Sentiment_Score'] = current_sentiment
    
    # Ensure all required features are present and ordered correctly
    X_latest = latest_data_row[features]

    # 3. DECIDE: Make the prediction
    prediction = model.predict(X_latest)[0]
    
    # 4. ACT: Format the final signal
    if prediction == 1:
        action = "STRONG BUY (ML Confidence)"
    else:
        action = "HOLD / SELL (ML Recommends Caution)"
        
    print(f"ACT: Final LLM Sentiment Score used: {current_sentiment:.4f}")
    print(f"ACT: Final Signal Output: {action}")
    
    return action

if __name__ == '__main__':
    # This block is for testing individual functions only
    print("Run 'main.py train' to train the model, or 'main.py run' to get a signal.")
