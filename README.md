# AI Financial Agent (ML & LLM Enhanced)

## 🤖 Overview

The **AI Financial Agent** is a sophisticated Python application designed to generate data-driven, emotionless investment signals. Unlike simple rule-based trading bots, this agent employs a **Machine Learning (ML) Classifier** for its core decision-making and integrates **Large Language Models (LLMs)** for critical market sentiment analysis.

This project is structured around the classic AI Agent paradigm: **Perceive $\rightarrow$ Decide $\rightarrow$ Act**.

## ✨ Key Features

  * **Enhanced Perception:** Gathers both structured price data (`yfinance`, Moving Averages, Volatility) and unstructured market intelligence via LLM-based sentiment analysis (`transformers`).
  * **Intelligent Decision-Making:** Uses a **Random Forest Classifier** (ML model) as the "AI Brain" to learn the optimal combination of features that indicate a future upward trend.
  * **Action Output:** Generates clear, data-backed signals (e.g., `STRONG BUY (ML Confidence)`).
  * **Modular Architecture:** Clean separation of data loading, feature engineering, LLM logic, and agent orchestration.

-----

## 📁 Project Structure

The project is organized to clearly separate the data handling, feature engineering, LLM logic, and core agent orchestration, making it scalable and easy to understand.

### Directory Tree

```
AI_Financial_Agent/
├── config/
│   └── settings.py
├── data/
│   ├── raw_data/
│   └── processed_data/
├── models/
│   └── RandomForest_Classifier.pkl
├── notebooks/
│   └── 01_Exploration_and_Training.ipynb
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── financial_agent.py
│   ├── data_prep/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── feature_engineering.py
│   └── llm_sentiment/
│       ├── __init__.py
│       └── sentiment_analyzer.py
├── main.py
├── requirements.txt
└── README.md
```

### File and Folder Descriptions

| File/Folder | Purpose |
| :--- | :--- |
| **`main.py`** | **Application Entry Point.** Loads the model and runs the full **ACT** phase (live prediction). |
| **`requirements.txt`** | Lists all Python dependencies (excluding PyTorch, which is manually installed). |
| **`config/settings.py`** | Stores crucial constants: stock **ticker**, dates, ML model **hyperparameters**, and MA periods. |
| **`data/raw_data/`** | Placeholder for raw, unprocessed historical data files. |
| **`models/RandomForest_Classifier.pkl`** | The serialized **trained Machine Learning model** (the AI Brain). |
| **`src/agents/financial_agent.py`** | **Agent Orchestrator.** Contains the core training (`train_ai_brain`) and prediction (`act_on_signal`) logic. |
| **`src/data_prep/`** | Handles numerical data loading and feature engineering (**PERCEIVE**). |
| **`src/llm_sentiment/sentiment_analyzer.py`** | Implements the **LLM logic** for the **PERCEIVE** phase: processing text and outputting a numerical **Sentiment Score**. |

-----

## 🚀 Setup and Installation

### A. Environment Setup

1.  **Address Python Compatibility:** Ensure you are using a stable Python version (**3.11 or 3.12**). Python 3.13 often lacks necessary pre-compiled binary packages (wheels).

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/futurecodingstars-bit/AI-Financial-Agent.git
    cd AI_Financial_Agent
    ```

3.  **Create and Activate a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    ```

### B. Installing Dependencies (Crucial Step)

Because the LLM requires a deep learning backend, we install PyTorch first to prevent compilation errors.

1.  **Install PyTorch (CPU Version Recommended):**
    This command installs the CPU-optimized version of PyTorch, which is sufficient for this project and avoids complex GPU setup issues.

    ```bash
    (venv) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

2.  **Install Remaining Dependencies:**

    ```bash
    (venv) pip install -r requirements.txt
    ```

3.  **Configure Settings:**
    Edit `config/settings.py` to adjust the stock ticker, date range, and other parameters as needed.

-----

## 💡 Usage

The project runs in two modes:

### 1\. Training the AI Brain (DECIDE Phase)

This step downloads historical data, calculates features, trains the Random Forest model, and saves the result.

```bash
(venv) python main.py train
```

### 2\. Getting the Live Signal (ACT Phase)

This step loads the saved model, fetches the latest market data and live LLM sentiment (mocked via `settings.py`), and prints the final trading signal.

```bash
(venv) python main.py run
```

## ⚠️ Financial Disclaimer

This project is for **educational and demonstrational purposes only**. It is intended to showcase the potential of AI/ML and LLM technologies in the financial domain. **It does not constitute financial or investment advice.** Trading and investing carry significant risk, and you should consult with a professional financial advisor before making any investment decisions. The author assumes no responsibility for any financial loss incurred using this software.