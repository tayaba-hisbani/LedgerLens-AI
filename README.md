# LedgerLens AI

AI-powered business accounting and cash-flow assistant built with Streamlit, Pandas, Plotly, OpenPyXL and Google's Gemini API.

## Features
- Excel/CSV import
- Transaction ledger
- Deterministic revenue, expense, profit and cash-flow calculations
- Interactive dashboard
- Expense breakdown
- Excel report export
- Gemini-powered transaction categorization
- Gemini finance Q&A and summaries
- Demo data for testing without an API key

## Local run
1. Create a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example` and add your Gemini API key.
4. Run: `streamlit run app.py`

## Deployment
Push the repository to GitHub, connect it to Streamlit Community Cloud, and add `GEMINI_API_KEY` under the app's Secrets settings.

## Important architecture rule
Python calculates financial figures. Gemini explains, classifies and summarizes. Do not use an LLM as the source of truth for accounting arithmetic.
