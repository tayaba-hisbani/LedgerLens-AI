import os
os.environ["GEMINI_API_KEY"] = ""

import pandas as pd
import app


def main():
    df = app.sample_data()
    assert list(df.columns) == app.REQUIRED_COLUMNS
    inc, exp, profit, cash_in, cash_out, cash_net = app.metrics(df)
    assert inc == 470000
    assert exp == 120700
    assert profit == 349300
    assert cash_in == 345000
    assert cash_out == 65300
    assert cash_net == 279700

    normalized = app.normalize_transactions(pd.DataFrame({
        "date": ["2026-09-01"],
        "description": ["Test expense"],
        "category": ["Utilities"],
        "type": ["expense"],
        "payment method": ["bank"],
        "amount": [1234]
    }))
    assert normalized.iloc[0]["Type"] == "Expense"
    assert normalized.iloc[0]["Amount"] == 1234

    xlsx = app.make_excel(df)
    assert xlsx[:2] == b"PK"
    assert len(xlsx) > 1000
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    main()
