# math-solver

Simple Streamlit app that solves math problems and answers general questions using a LangChain agent with three tools: Wikipedia search, a calculator for numeric expressions, and a reasoning tool for step‑by‑step logic.

## Features
- Chat-style UI built with Streamlit
- Groq LLM (`gemma2-9b-it`) via `langchain-groq`
- Tools:
  - Wikipedia: factual lookups
  - Calculator: numeric calculations (LLMMathChain)
  - Reasoning Tool: step-by-step logic for non-numeric questions

## Prerequisites
- Python 3.10+
- A Groq API key (`GROQ_API_KEY`)

## Setup
1. Create and activate a virtual environment.
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows (PowerShell): `py -m venv .venv; .\.venv\Scripts\Activate.ps1`
2. Install dependencies: `pip install -r requirements.txt`

## Configure Secrets
Set your Groq API key by either method:

- Streamlit secrets (recommended for deployment):
  - Create `.streamlit/secrets.toml` with:
    ```
    GROQ_API_KEY = "your_api_key_here"
    ```
- Environment variable (works locally):
  - Create a `.env` file with:
    ```
    GROQ_API_KEY=your_api_key_here
    ```
  - Or export in your shell: `export GROQ_API_KEY=your_api_key_here`

## Run
```
streamlit run app.py
```

Open the provided local URL in your browser. If the API key is missing, the app displays a message and stops, avoiding runtime errors.

## Usage Tips
- Ask numeric questions (e.g., "What is 1234^2?") to trigger the Calculator.
- Ask general factual questions to use Wikipedia.
- Ask logic or multi-step reasoning questions when no calculation is required to use the Reasoning Tool.

## Development Notes
- Minimal dependencies are kept in `requirements.txt` to speed up installs.
- The agent is `ZERO_SHOT_REACT_DESCRIPTION` and selects tools based on their descriptions; these descriptions are tuned to reduce mis-selection.
