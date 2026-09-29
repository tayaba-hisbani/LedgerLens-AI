import io
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

st.set_page_config(
    page_title="LedgerLens AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Theme / UI
# -----------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 10% 0%, rgba(109,93,252,.10), transparent 30%), #f7f8fc; }
.block-container { padding-top: 1.2rem; max-width: 1450px; }
.hero { padding: 28px 30px; border-radius: 24px; background: linear-gradient(135deg,#111827 0%,#202a44 52%,#4638b6 100%); color:white; box-shadow: 0 18px 45px rgba(31,41,55,.16); margin-bottom: 22px; }
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size: 2.35rem; margin:0 0 7px 0; letter-spacing:-1px; }
.hero p { margin:0; color:#dbe4ff; font-size:1.02rem; }
.badge { display:inline-block; padding:6px 11px; border:1px solid rgba(255,255,255,.22); border-radius:999px; font-size:.76rem; margin-bottom:12px; background:rgba(255,255,255,.08); }
.metric-card { background:white; border:1px solid #e8eaf2; border-radius:18px; padding:17px 18px; min-height:108px; box-shadow:0 7px 25px rgba(24,32,56,.05); }
.metric-label { color:#687085; font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em; }
.metric-value { font-family:'Space Grotesk',sans-serif; font-size:1.72rem; font-weight:700; color:#151a2b; margin-top:5px; }
.metric-sub { color:#7b8498; font-size:.78rem; margin-top:2px; }
.section-title { font-family:'Space Grotesk',sans-serif; font-size:1.2rem; font-weight:700; color:#171b2e; margin:20px 0 8px; }
.insight { background:#ffffff; border:1px solid #e8eaf2; border-left:4px solid #6d5dfc; border-radius:15px; padding:16px 18px; margin:8px 0; box-shadow:0 6px 22px rgba(24,32,56,.04); }
.small-note { color:#737d91; font-size:.82rem; }
div[data-testid="stSidebar"] { background:#101625; }
div[data-testid="stSidebar"] * { color:#eef2ff; }
.stButton>button { border-radius:11px; font-weight:600; }
.stDownloadButton>button { border-radius:11px; }
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; }
</style>
""",
    unsafe_allow_html=True,
)

CATEGORIES = [
    "Sales Revenue", "Other Income", "Rent", "Utilities", "Salaries",
    "Transportation", "Office Supplies", "Marketing", "Inventory/Purchases",
    "Bank Charges", "Taxes", "Equipment", "Other Expense", "Uncategorized"
]
TYPES = ["Income", "Expense"]
PAYMENT_METHODS = ["Cash", "Bank", "Card", "Mobile Wallet", "Other"]
REQUIRED_COLUMNS = ["Date", "Description", "Category", "Type", "Payment Method", "Amount"]


def sample_data():
    rows = [
        ["2026-09-01", "Product sales", "Sales Revenue", "Income", "Bank", 125000],
        ["2026-09-02", "Shop rent", "Rent", "Expense", "Bank", 30000],
        ["2026-09-03", "Electricity bill", "Utilities", "Expense", "Bank", 8500],
        ["2026-09-05", "Product sales", "Sales Revenue", "Income", "Cash", 65000],
        ["2026-09-06", "Office stationery", "Office Supplies", "Expense", "Cash", 4200],
        ["2026-09-08", "Staff salaries", "Salaries", "Expense", "Bank", 55000],
        ["2026-09-10", "Fuel and delivery", "Transportation", "Expense", "Cash", 7800],
        ["2026-09-12", "Product sales", "Sales Revenue", "Income", "Bank", 98000],
        ["2026-09-14", "Social media campaign", "Marketing", "Expense", "Card", 12000],
        ["2026-09-18", "Product sales", "Sales Revenue", "Income", "Cash", 72000],
        ["2026-09-21", "Internet bill", "Utilities", "Expense", "Bank", 3200],
        ["2026-09-25", "Product sales", "Sales Revenue", "Income", "Bank", 110000],
    ]
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    return df


def normalize_transactions(df):
    df = df.copy()
    rename = {c: c.strip().title() for c in df.columns}
    df.rename(columns=rename, inplace=True)
    aliases = {
        "Transaction Date": "Date", "Details": "Description", "Category Name": "Category",
        "Transaction Type": "Type", "Payment": "Payment Method", "Value": "Amount"
    }
    df.rename(columns=aliases, inplace=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Description"] = df["Description"].fillna("").astype(str).str.strip()
    df["Category"] = df["Category"].fillna("Uncategorized").astype(str).str.strip()
    df["Type"] = df["Type"].fillna("Expense").astype(str).str.title().str.strip()
    df["Payment Method"] = df["Payment Method"].fillna("Other").astype(str).str.title().str.strip()
    df.loc[~df["Type"].isin(TYPES), "Type"] = "Expense"
    df.loc[df["Category"].eq(""), "Category"] = "Uncategorized"
    df = df.dropna(subset=["Date", "Amount"])
    df["Amount"] = df["Amount"].abs()
    return df.reset_index(drop=True)


def metrics(df):
    income = float(df.loc[df["Type"].eq("Income"), "Amount"].sum())
    expenses = float(df.loc[df["Type"].eq("Expense"), "Amount"].sum())
    profit = income - expenses
    cash_in = float(df.loc[(df["Type"] == "Income") & (df["Payment Method"] == "Cash"), "Amount"].sum())
    cash_out = float(df.loc[(df["Type"] == "Expense") & (df["Payment Method"] == "Cash"), "Amount"].sum())
    cash_net = cash_in - cash_out
    return income, expenses, profit, cash_in, cash_out, cash_net


def money(x):
    return f"PKR {x:,.0f}"


def make_excel(df):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Transactions")
        inc, exp, profit, cash_in, cash_out, cash_net = metrics(df)
        summary = pd.DataFrame({
            "Metric": ["Total Income", "Total Expenses", "Net Profit", "Cash Inflow", "Cash Outflow", "Net Cash Movement"],
            "Amount": [inc, exp, profit, cash_in, cash_out, cash_net]
        })
        summary.to_excel(writer, index=False, sheet_name="Summary")
        by_cat = df[df["Type"] == "Expense"].groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
        by_cat.to_excel(writer, index=False, sheet_name="Expense Breakdown")
    return out.getvalue()


def get_api_key():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


def gemini_text(prompt, max_tokens=900):
    key = get_api_key()
    if not key or genai is None:
        return None, "Gemini API key is not configured. The dashboard still works without AI."
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are LedgerLens, a careful business finance assistant. "
                    "Never invent transactions or financial figures. Use only the data supplied. "
                    "Do not present yourself as a licensed accountant. For accounting/tax decisions, "
                    "clearly say when professional review is appropriate. Keep answers practical and concise."
                ),
                max_output_tokens=max_tokens,
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            return None, "Gemini returned an empty response."
        return text.strip(), None
    except Exception as e:
        return None, f"Gemini request failed: {e}"


def categorize_with_ai(description, amount):
    prompt = f"""
Classify this business transaction.
Description: {description}
Amount: {amount}

Return ONLY valid JSON with these keys:
category, type, confidence, reason

Allowed categories: {', '.join(CATEGORIES)}
Allowed type values: Income, Expense
Confidence must be a number from 0 to 1.
"""
    text, error = gemini_text(prompt, 350)
    if error:
        return None, error
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        if data.get("category") not in CATEGORIES:
            data["category"] = "Uncategorized"
        if data.get("type") not in TYPES:
            data["type"] = "Expense"
        data["confidence"] = max(0, min(1, float(data.get("confidence", 0))))
        return data, None
    except Exception as e:
        return None, f"Could not parse Gemini's classification: {e}"


def financial_context(df):
    inc, exp, profit, cash_in, cash_out, cash_net = metrics(df)
    expense_breakdown = (
        df[df["Type"] == "Expense"].groupby("Category")["Amount"].sum().sort_values(ascending=False).head(10).to_dict()
    )
    income_breakdown = (
        df[df["Type"] == "Income"].groupby("Category")["Amount"].sum().sort_values(ascending=False).head(10).to_dict()
    )
    recent = df.sort_values("Date", ascending=False).head(15).copy()
    recent["Date"] = recent["Date"].dt.strftime("%Y-%m-%d")
    return {
        "total_income": inc,
        "total_expenses": exp,
        "net_profit": profit,
        "cash_inflow": cash_in,
        "cash_outflow": cash_out,
        "net_cash_movement": cash_net,
        "expense_breakdown": expense_breakdown,
        "income_breakdown": income_breakdown,
        "recent_transactions": recent.to_dict(orient="records"),
    }


def ask_finance_ai(question, df):
    context = financial_context(df)
    prompt = f"""
Answer the user's finance question using ONLY this business dataset summary.

DATA:
{json.dumps(context, default=str, indent=2)}

USER QUESTION:
{question}

Rules:
- Never invent missing values.
- If the data cannot answer the question, say so.
- Show PKR amounts where useful.
- Distinguish calculations from interpretation.
- Do not give tax/legal/accounting compliance conclusions as certain facts.
"""
    return gemini_text(prompt, 900)


# -----------------------------
# Session state
# -----------------------------
if "transactions" not in st.session_state:
    st.session_state.transactions = sample_data()
if "ai_history" not in st.session_state:
    st.session_state.ai_history = []

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.markdown("## ◈ LedgerLens")
st.sidebar.caption("AI-powered business finance workspace")
mode = st.sidebar.radio("Workspace", ["Dashboard", "Transactions", "AI Assistant", "Reports"], index=0)
st.sidebar.divider()

with st.sidebar.expander("Data controls", expanded=True):
    uploaded = st.file_uploader("Import Excel or CSV", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                new_df = pd.read_csv(uploaded)
            else:
                new_df = pd.read_excel(uploaded)
            st.session_state.transactions = normalize_transactions(new_df)
            st.success(f"Loaded {len(st.session_state.transactions)} transactions.")
        except Exception as e:
            st.error(str(e))
    if st.button("↺ Restore demo data", use_container_width=True):
        st.session_state.transactions = sample_data()
        st.rerun()

    st.download_button(
        "↓ Download Excel workbook",
        data=make_excel(st.session_state.transactions),
        file_name="LedgerLens_financial_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.sidebar.divider()
key_status = bool(get_api_key())
st.sidebar.markdown(f"**Gemini:** {'🟢 Connected' if key_status else '🟠 Not configured'}")
st.sidebar.caption("Calculations run locally in Python. Gemini is used for AI assistance only.")

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="hero"><div class="badge">FINANCE × AI · MVP</div>'
    '<h1>LedgerLens AI</h1>'
    '<p>Turn messy business transactions into a clear financial picture — without asking an LLM to do the math.</p></div>',
    unsafe_allow_html=True,
)

df = normalize_transactions(st.session_state.transactions)
st.session_state.transactions = df
inc, exp, profit, cash_in, cash_out, cash_net = metrics(df)

# -----------------------------
# Dashboard
# -----------------------------
if mode == "Dashboard":
    st.markdown('<div class="section-title">Financial pulse</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cards = [
        ("Revenue", money(inc), "Recorded income"),
        ("Expenses", money(exp), "Recorded expenses"),
        ("Net profit", money(profit), "Income − expenses"),
        ("Cash movement", money(cash_net), "Cash inflow − cash outflow"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Your money story</div>', unsafe_allow_html=True)
    left, right = st.columns([1.35, 1])
    with left:
        monthly = df.copy()
        monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
        trend = monthly.groupby(["Month", "Type"], as_index=False)["Amount"].sum()
        fig = px.bar(trend, x="Month", y="Amount", color="Type", barmode="group", title="Income vs expenses", template="plotly_white")
        fig.update_layout(height=370, margin=dict(l=10,r=10,t=55,b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        expenses = df[df["Type"] == "Expense"].groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
        if len(expenses):
            fig2 = px.pie(expenses, names="Category", values="Amount", hole=.58, title="Expense mix", template="plotly_white")
            fig2.update_layout(height=370, margin=dict(l=10,r=10,t=55,b=10), showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

    top_exp = df[df["Type"] == "Expense"].groupby("Category")["Amount"].sum().sort_values(ascending=False)
    if len(top_exp):
        biggest = top_exp.index[0]
        share = top_exp.iloc[0] / exp * 100 if exp else 0
        st.markdown(f'<div class="insight"><b>◈ Focus area:</b> {biggest} is the largest expense category at <b>{money(top_exp.iloc[0])}</b> ({share:.1f}% of expenses).</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Latest transactions</div>', unsafe_allow_html=True)
    show = df.sort_values("Date", ascending=False).head(8).copy()
    show["Date"] = show["Date"].dt.strftime("%d %b %Y")
    show["Amount"] = show["Amount"].map(money)
    st.dataframe(show, use_container_width=True, hide_index=True)

# -----------------------------
# Transactions
# -----------------------------
elif mode == "Transactions":
    st.markdown('<div class="section-title">Transaction studio</div>', unsafe_allow_html=True)
    st.caption("Keep the ledger structured. AI can suggest categories, but the final record stays under your control.")

    with st.form("add_transaction", clear_on_submit=True):
        c1, c2, c3 = st.columns([1, 2, 1])
        t_date = c1.date_input("Date", value=date.today())
        description = c2.text_input("Description", placeholder="e.g. Electricity bill")
        amount = c3.number_input("Amount (PKR)", min_value=0.0, step=500.0)
        c4, c5, c6 = st.columns(3)
        category = c4.selectbox("Category", CATEGORIES)
        t_type = c5.selectbox("Type", TYPES)
        payment = c6.selectbox("Payment method", PAYMENT_METHODS)
        submitted = st.form_submit_button("＋ Add transaction", use_container_width=True)
        if submitted:
            if not description.strip() or amount <= 0:
                st.error("Enter a description and an amount greater than zero.")
            else:
                new_row = pd.DataFrame([[pd.Timestamp(t_date), description.strip(), category, t_type, payment, amount]], columns=REQUIRED_COLUMNS)
                st.session_state.transactions = pd.concat([df, new_row], ignore_index=True)
                st.success("Transaction added.")
                st.rerun()

    st.markdown('<div class="section-title">AI category suggestion</div>', unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns([3, 1, 1])
    ai_desc = ac1.text_input("Transaction description", placeholder="e.g. Paid for office internet")
    ai_amount = ac2.number_input("Amount", min_value=0.0, step=100.0, key="ai_amount")
    if ac3.button("✨ Suggest", use_container_width=True):
        if not ai_desc.strip() or ai_amount <= 0:
            st.warning("Enter a description and amount first.")
        else:
            result, err = categorize_with_ai(ai_desc, ai_amount)
            if err:
                st.error(err)
            else:
                st.session_state.last_category = result
    if "last_category" in st.session_state:
        r = st.session_state.last_category
        st.markdown(f'<div class="insight"><b>Suggested category:</b> {r["category"]} · <b>Type:</b> {r["type"]} · <b>Confidence:</b> {r["confidence"]:.0%}<br><span class="small-note">{r.get("reason", "")}</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Ledger</div>', unsafe_allow_html=True)
    edited = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "Amount": st.column_config.NumberColumn("Amount (PKR)", min_value=0, format="%.2f"),
            "Type": st.column_config.SelectboxColumn("Type", options=TYPES),
            "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES),
            "Payment Method": st.column_config.SelectboxColumn("Payment Method", options=PAYMENT_METHODS),
        },
    )
    if st.button("Save ledger edits", type="primary"):
        try:
            st.session_state.transactions = normalize_transactions(edited)
            st.success("Ledger saved for this session.")
        except Exception as e:
            st.error(str(e))

# -----------------------------
# AI Assistant
# -----------------------------
elif mode == "AI Assistant":
    st.markdown('<div class="section-title">Ask your numbers</div>', unsafe_allow_html=True)
    st.caption("Gemini receives a compact financial context derived from your ledger. It does not perform the underlying arithmetic.")

    examples = [
        "Summarize this month's financial position.",
        "What is my biggest expense category?",
        "What should I investigate in these transactions?",
        "Explain my cash movement in simple language.",
    ]
    cols = st.columns(4)
    for i, q in enumerate(examples):
        if cols[i].button(q, use_container_width=True):
            st.session_state.ai_question = q

    question = st.text_area("Your question", value=st.session_state.get("ai_question", ""), height=100, placeholder="Ask something about your business finances...")
    if st.button("Ask LedgerLens ✦", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("LedgerLens is reading the financial context..."):
                answer, error = ask_finance_ai(question, df)
            if error:
                st.error(error)
            else:
                st.session_state.ai_history.append((question, answer))
                st.rerun()

    for q, a in reversed(st.session_state.ai_history[-8:]):
        st.markdown(f"**You:** {q}")
        st.markdown(f'<div class="insight">{a.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

# -----------------------------
# Reports
# -----------------------------
else:
    st.markdown('<div class="section-title">Management reports</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Profit & Loss")
        pnl = pd.DataFrame({"Line item": ["Revenue", "Expenses", "Net Profit"], "PKR": [inc, exp, profit]})
        st.dataframe(pnl, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### Cash movement")
        cash = pd.DataFrame({"Line item": ["Cash inflow", "Cash outflow", "Net cash movement"], "PKR": [cash_in, cash_out, cash_net]})
        st.dataframe(cash, use_container_width=True, hide_index=True)

    st.markdown("### Expense breakdown")
    breakdown = df[df["Type"] == "Expense"].groupby("Category", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
    if len(breakdown):
        breakdown["Share"] = np.where(exp > 0, breakdown["Amount"] / exp * 100, 0).round(1)
        st.dataframe(breakdown, use_container_width=True, hide_index=True, column_config={"Amount": st.column_config.NumberColumn("Amount (PKR)", format="%.2f"), "Share": st.column_config.NumberColumn("Share %", format="%.1f%%")})

    st.markdown("### Download")
    st.download_button("Download complete Excel report", make_excel(df), "LedgerLens_financial_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

st.divider()
st.caption("LedgerLens AI is an MVP for business record management. Verify important accounting, tax, and compliance decisions with a qualified professional.")
