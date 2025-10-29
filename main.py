# main.py
import sys
from src.agents.financial_agent import train_ai_brain, load_ai_brain, act_on_signal
from config import settings

def main():
    """
    Main entry point for the AI Financial Agent.
    Handles 'train' and 'run' commands.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py [command]")
        print("Commands: train (to train the model), run (to get a live signal)")
        return

    command = sys.argv[1].lower()

    if command == 'train':
        print(f"Starting AI Agent training process for {settings.TICKER}...")
        try:
            train_ai_brain()
            print("\nSUCCESS: Model training complete and saved.")
        except Exception as e:
            print(f"\nFATAL ERROR during training: {e}")
            sys.exit(1)

    elif command == 'run':
        model = load_ai_brain()
        if model:
            try:
                act_on_signal(model)
                print("\nSUCCESS: Signal generated.")
            except Exception as e:
                print(f"\nFATAL ERROR during signal generation: {e}")
                sys.exit(1)
        
    else:
        print(f"Unknown command: {command}")
        
if __name__ == "__main__":
    main()