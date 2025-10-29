# src/llm_sentiment/sentiment_analyzer.py
from transformers import pipeline
from config import settings

# Initialize the sentiment analysis pipeline using a pre-trained model
# This model will download upon first run.
try:
    SENTIMENT_PIPELINE = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english" # A common, fast model
    )
except Exception as e:
    print(f"Warning: Could not load HuggingFace pipeline. Sentiment will be mocked. Error: {e}")
    SENTIMENT_PIPELINE = None


def get_llm_sentiment(text_input: str) -> float:
    """
    PERCEIVE: Uses a pre-trained LLM/Transformer to determine market sentiment.
    
    Args:
        text_input (str): Aggregated financial news/text.
        
    Returns:
        float: A numerical score (+1 for Positive, -1 for Negative, 0 for Neutral).
    """
    print("[PERCEIVE] Analyzing market sentiment using LLM...")
    
    if SENTIMENT_PIPELINE is None:
        print("Using mock sentiment score due to pipeline failure.")
        # Mocking for robust tutorial/demo run if HF pipeline fails
        return 0.5 

    # Analyze the input text
    result = SENTIMENT_PIPELINE(text_input)[0]
    
    # Convert string label to a numerical score
    if result['label'] == 'POSITIVE':
        # Scale the positive score for a stronger input feature
        score = result['score'] 
    elif result['label'] == 'NEGATIVE':
        # Scale the negative score
        score = -result['score']
    else:
        score = 0.0
        
    return score

def get_current_sentiment_score() -> float:
    """
    Retrieves the current/latest sentiment score, typically by analyzing 
    live news. Here, we use a mock text from settings.
    """
    return get_llm_sentiment(settings.LLM_MOCK_NEWS_HEADLINES)


if __name__ == '__main__':
    # Example usage for testing
    positive_score = get_llm_sentiment("Stock market soared to new highs on strong earnings.")
    negative_score = get_llm_sentiment("Recession fears drove the market down significantly.")
    
    print(f"Positive Example Score: {positive_score:.4f}")
    print(f"Negative Example Score: {negative_score:.4f}")