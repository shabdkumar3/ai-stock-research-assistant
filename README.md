# 📈 Stockwise AI — Equity Research Assistant

> An AI-powered stock research dashboard delivering deep fundamental, technical, and sentiment analysis — all in one place.

**🌐 Live Demo → [ai-stock-research-assistant.streamlit.app](https://ai-stock-research-assistant.streamlit.app/)**

---

## ✨ Features

### 🔍 Single Stock Analysis
- Live price, market cap, beta, dividend yield, short float
- Candlestick chart with Bollinger Bands overlay
- Moving averages (MA50 / MA200) with trend detection
- MACD histogram + RSI with overbought/oversold zones
- Volume analysis with 20-day average overlay
- Trading signals — MACD, Bollinger, ADX trend strength

### 📊 Fundamental Intelligence
- Income Statement, Balance Sheet, Cash Flow (annual, USD Millions)
- Revenue & net income growth YoY
- Net profit margin trend chart
- Earnings stability score
- Valuation multiples — P/E vs industry, PEG, P/B, P/S, EV/EBITDA

### 🗞️ Sentiment & News
- Latest news articles with per-article FinBERT sentiment scoring
- Reddit buzz tracking for social sentiment
- AI-generated 2-sentence sentiment verdict (LLaMA 3.1)
- Combined news + Reddit sentiment score

### 🌍 Market Context
- Stock vs S&P 500 normalised performance chart
- VIX fear index reading
- 10-Year Treasury yield with equity impact assessment

### ⚔️ Competitive Intelligence
- Auto-fetched competitor tickers via AI
- Side-by-side comparison table (market cap, P/E, revenue, beta)
- Market cap bar chart vs peers

### ⚠️ Risk & Institutions
- Risk score out of 100 (volatility + RSI + sentiment) with visual gauge
- Analyst ratings breakdown (Strong Buy → Strong Sell)
- Analyst price targets (mean, median, low, high)
- Insider buying vs selling activity
- Top 5 institutional holders with ownership chart
- Options Put/Call ratio (volume + open interest)

### ⚖️ Stock Comparison Mode
- Compare multiple companies side by side
- Normalised price performance chart (base = 100)
- AI verdict ranking best to worst with buy recommendation

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | LLaMA 3.1 8B via Groq API |
| Sentiment NLP | FinBERT (ProsusAI) via HuggingFace |
| Market Data | yahooquery |
| Technical Analysis | `ta` library |
| Charts | Plotly |
| Data | Pandas, NumPy |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/ai-stock-research-assistant.git
cd ai-stock-research-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## 📦 Requirements

```
streamlit
langchain-groq
langchain-core
transformers
torch
requests
ta
pandas
plotly
python-dotenv
yahooquery
```

---

## 📁 Project Structure

```
├── app.py              # Main Streamlit application
├── .env                # API keys (not committed)
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚠️ Disclaimer

This tool is for **informational and educational purposes only**. Nothing on this platform constitutes financial advice. Always do your own research before making investment decisions.

---

## 🙌 Built By

**Shabd Kumar** · B.Tech Energy Engineering · IIT Delhi

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-stock-research-assistant.streamlit.app/)
