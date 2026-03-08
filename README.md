# 📈 Stockwise AI — Equity Research Assistant

> An AI-powered stock research dashboard that gives you deep fundamental, technical, and sentiment analysis — all in one place.

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge)](https://ai-stock-research-assistant.streamlit.app/)
[![Made with Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Powered by LLaMA](https://img.shields.io/badge/LLM-LLaMA%203.1-blueviolet?style=for-the-badge)](https://groq.com)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT-orange?style=for-the-badge)](https://huggingface.co/ProsusAI/finbert)

---

## 🌐 Live Demo

**👉 [https://ai-stock-research-assistant.streamlit.app/](https://ai-stock-research-assistant.streamlit.app/)**

Type any company name (e.g. Apple, Tesla, Reliance) and get a full AI-driven research report instantly.

---

## ✨ Features

### Single Stock Analysis
- **Live price data** with day change, market cap, beta, dividend yield and short float
- **Candlestick chart** with Bollinger Bands overlay
- **Moving averages** (MA50 / MA200) and trend detection
- **MACD histogram** and **RSI** charts with overbought/oversold zones
- **Volume analysis** with 20-day average overlay
- **Trading signals** — MACD, Bollinger, ADX trend strength

### Fundamental Intelligence
- Income Statement, Balance Sheet and Cash Flow (yearly, formatted cleanly)
- Revenue and net income growth YoY
- Net profit margin trend chart
- Earnings stability score
- Valuation multiples — P/E vs industry, PEG, P/B, P/S, EV/EBITDA

### Sentiment & News
- Latest news articles with per-article FinBERT sentiment scoring
- Reddit buzz tracking for social sentiment
- AI-generated 2-sentence sentiment verdict (LLaMA 3.1)
- Combined news + Reddit sentiment score

### Market Context
- Stock vs S&P 500 normalised performance chart
- VIX fear index reading
- 10-Year Treasury yield with equity impact assessment

### Competitive Intelligence
- Auto-fetched competitor tickers via AI
- Side-by-side comparison table (market cap, P/E, revenue, beta)
- Market cap bar chart vs peers

### Risk & Institutions
- Risk score out of 100 (volatility + RSI + sentiment components) with visual gauge
- Analyst ratings breakdown (Strong Buy → Strong Sell bar chart)
- Analyst price targets (mean, median, low, high)
- Insider buying vs selling activity
- Top 5 institutional holders with ownership chart
- Options Put/Call ratio (volume + open interest)

### Stock Comparison Mode
- Enter multiple companies to compare them side by side
- Normalised price performance chart (base = 100)
- AI verdict ranking best to worst with buy recommendation

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | LLaMA 3.1 8B via Groq API |
| Sentiment NLP | FinBERT (ProsusAI) via HuggingFace |
| Market Data | yFinance |
| Technical Analysis | `ta` library |
| Charts | Plotly |
| Data | Pandas, NumPy |

---
## 📦 Requirements

```
streamlit
yfinance
langchain-groq
langchain-core
transformers
torch
ta
pandas
numpy
plotly
requests
python-dotenv
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

This tool is for **informational and educational purposes only**. Nothing on this platform constitutes financial advice. Always do your own research before making any investment decisions.

---

## 🙌 Acknowledgements

- [Groq](https://groq.com) for blazing fast LLaMA inference
- [yFinance](https://github.com/ranaroussi/yfinance) for market data
- [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert) for financial sentiment analysis
- [Streamlit](https://streamlit.io) for the UI framework
