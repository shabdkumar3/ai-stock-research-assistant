import yfinance as yf
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from transformers import pipeline
import requests
import ast
import ta
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="Stockwise AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* Root palette */
  :root {
    --bg:        #0e1117;
    --surface:   #161b26;
    --border:    #1f2937;
    --accent:    #34d399;
    --accent2:   #6ee7b7;
    --muted:     #6b7280;
    --text:      #e5e7eb;
    --text-dim:  #9ca3af;
    --red:       #f87171;
    --amber:     #fbbf24;
    --blue:      #60a5fa;
  }

  /* Global resets */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    background-color: var(--bg);
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2rem 3rem; max-width: 1400px; }

  /* Masthead */
  .masthead {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.25rem;
  }
  .masthead h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: var(--accent);
    margin: 0;
    letter-spacing: -0.5px;
  }
  .masthead span {
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Section headers */
  .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 2rem 0 0.6rem;
  }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
  }
  div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    font-weight: 500;
  }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.5rem !important;
    font-weight: 600;
  }

  /* Info / success / error boxes */
  div[data-testid="stAlert"] {
    border-radius: 8px;
    border-left-width: 3px;
    font-size: 0.88rem;
  }

  /* Divider */
  hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

  /* Dataframe */
  [data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    overflow: hidden;
  }

  /* Expander */
  [data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px;
    background: var(--surface);
  }

  /* Tab strip */
  [data-testid="stTabs"] [role="tab"] {
    font-size: 0.82rem;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.04em;
  }
  [data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
  }

  /* Input fields */
  [data-testid="stTextInput"] input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-size: 0.9rem;
  }
  [data-testid="stTextInput"] input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(52,211,153,0.15) !important;
  }

  /* Spinner text */
  [data-testid="stSpinner"] p { color: var(--muted) !important; font-size: 0.82rem; }

  /* Tag pill */
  .pill {
    display: inline-block;
    background: rgba(52,211,153,0.12);
    color: var(--accent2);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 20px;
    padding: 0.15rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 0 0.2rem;
  }

  /* Signal badge */
  .badge-bull { background:#052e16; color:#34d399; border:1px solid #166534; border-radius:6px; padding:0.2rem 0.65rem; font-size:0.78rem; font-weight:600; }
  .badge-bear { background:#2d0d0d; color:#f87171; border:1px solid #7f1d1d; border-radius:6px; padding:0.2rem 0.65rem; font-size:0.78rem; font-weight:600; }
  .badge-neut { background:#1c1a0a; color:#fbbf24; border:1px solid #78350f; border-radius:6px; padding:0.2rem 0.65rem; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# Plotly theme helper
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#9ca3af", size=12),
    xaxis=dict(gridcolor="#1f2937", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1f2937", showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1f2937"),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode="x unified",
)
ACCENT = "#34d399"
RED    = "#f87171"
AMBER  = "#fbbf24"
BLUE   = "#60a5fa"

# Model + NLP
API_KEY = os.environ.get("GROQ_API_KEY")
model   = ChatGroq(api_key=API_KEY, model_name="llama-3.1-8b-instant")

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

sentiment_model = load_sentiment_model()

# Helpers
def get_ticker(company: str) -> str:
    res = model.invoke([HumanMessage(
        content=f"Return ONLY the stock ticker symbol for '{company}'. Just the ticker, nothing else."
    )])
    return res.content.strip().upper()


def get_competitors(ticker: str) -> list:
    res = model.invoke([HumanMessage(
        content=f"Give 4 main competitor ticker symbols of {ticker}. Return ONLY a Python list like ['MSFT', 'GOOGL']. Nothing else."
    )])
    try:
        return ast.literal_eval(res.content.strip())
    except Exception:
        return []


def get_tickers_from_text(text: str) -> list:
    res = model.invoke([HumanMessage(
        content=(
            f"Extract all company names from this text and return ONLY a Python list of their stock ticker symbols. "
            f"No markdown, no explanation, just the list. Text: '{text}'. Example output: ['AAPL', 'TSLA', 'META']"
        )
    )])
    cleaned = res.content.strip().replace("```python", "").replace("```", "").strip()
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        return []


def get_reddit_titles(ticker: str) -> list[str]:
    url = f"https://www.reddit.com/search.json?q={ticker}+stock&limit=8"
    titles = []
    try:
        res  = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        data = res.json()
        for post in data["data"]["children"]:
            t = post["data"]["title"]
            if t:
                titles.append(t)
    except Exception:
        pass
    return titles


def score_sentiment(text: str) -> float:
    try:
        result = sentiment_model(text[:512])[0]
        return result["score"] if result["label"] == "positive" else -result["score"]
    except Exception:
        return 0.0


def badge(signal: str) -> str:
    s = signal.lower()
    if any(x in s for x in ["bull", "buy", "up", "strong up", "overs"]):
        return f'<span class="badge-bull">{signal}</span>'
    if any(x in s for x in ["bear", "sell", "down", "strong down"]):
        return f'<span class="badge-bear">{signal}</span>'
    return f'<span class="badge-neut">{signal}</span>'


def fmt_large(val):
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    if val >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"


# Masthead
st.markdown("""
<div class="masthead">
  <h1>Stockwise AI</h1>
  <span>Equity Research · Powered by LLaMA 3.1 · FinBERT · yFinance</span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input row
col_a, col_b = st.columns([1, 1], gap="large")
with col_a:
    company = st.text_input("", placeholder="🔍  Single stock — e.g. Apple, Tesla, Reliance", label_visibility="collapsed")
with col_b:
    compare_input = st.text_input("", placeholder="⚖️  Compare stocks — e.g. Apple, Tesla, Google", label_visibility="collapsed")

# COMPARISON MODE
if compare_input:
    st.markdown("<p class='section-label'>Comparison Dashboard</p>", unsafe_allow_html=True)

    with st.spinner("Resolving tickers…"):
        tickers = get_tickers_from_text(compare_input)

    if not tickers:
        st.error("Could not extract tickers from your input. Please try again.")
    else:
        rows = []
        fig  = go.Figure()
        colors = [ACCENT, BLUE, AMBER, RED, "#a78bfa", "#fb923c"]

        for i, sym in enumerate(tickers):
            try:
                stock = yf.Ticker(sym)
                info  = stock.info
                hist  = stock.history(period="1y")["Close"]
                if hist.empty:
                    continue

                norm = hist * 100 / hist.iloc[0]
                fig.add_trace(go.Scatter(
                    x=norm.index, y=norm.values,
                    name=info.get("shortName", sym),
                    mode="lines",
                    line=dict(color=colors[i % len(colors)], width=2),
                ))

                sentiment_score = 0.0
                for article in stock.news[:5]:
                    content = article.get("content", {})
                    text    = content.get("summary", "") or content.get("title", "")
                    if text:
                        sentiment_score += score_sentiment(text)

                ytd = round((hist.iloc[-1] - hist.iloc[0]) * 100 / hist.iloc[0], 2)
                rows.append({
                    "Ticker":        sym,
                    "Company":       info.get("shortName", sym),
                    "Price":         info.get("currentPrice"),
                    "Mkt Cap":       fmt_large(info.get("marketCap")),
                    "P/E":           round(info.get("trailingPE") or 0, 1) or "—",
                    "52W High":      info.get("fiftyTwoWeekHigh"),
                    "Revenue":       fmt_large(info.get("totalRevenue")),
                    "YTD (%)":       ytd,
                    "Volatility (%)": round(hist.pct_change().std() * 100, 2),
                    "RSI":           round(ta.momentum.RSIIndicator(hist).rsi().iloc[-1], 1),
                    "Sentiment":     round(sentiment_score, 2),
                })
            except Exception as e:
                st.warning(f"Could not fetch data for {sym}: {e}")

        if rows:
            df = pd.DataFrame(rows)

            # Style the dataframe
            def color_ytd(val):
                try:
                    v = float(val)
                    return "color: #34d399" if v > 0 else "color: #f87171"
                except Exception:
                    return ""

            styled = df.style.applymap(color_ytd, subset=["YTD (%)"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            fig.update_layout(
                title="Normalised Price Performance (Base = 100)",
                **PLOTLY_LAYOUT
            )
            st.plotly_chart(fig, use_container_width=True)

            # AI verdict — concise
            st.markdown("<p class='section-label'>AI Verdict</p>", unsafe_allow_html=True)
            with st.spinner("Analysing…"):
                r = model.invoke([
                    SystemMessage(content="You are a concise financial analyst. Be direct, avoid fluff."),
                    HumanMessage(content=f"""
Stocks data (Price, PE, YTD Return, Volatility, RSI, Sentiment):
{df[['Ticker','Company','Price','P/E','YTD (%)','Volatility (%)','RSI','Sentiment']].to_string(index=False)}

In 6 bullet points max:
- Rank stocks best to worst
- One-line reason per stock
- Which to BUY and why
- Key risk to watch

No fluff. End with: ⚠️ Not financial advice.
""")
                ])
            st.info(r.content)

    st.divider()

# SINGLE STOCK MODE
if company:

    with st.spinner("Looking up ticker…"):
        ticker = get_ticker(company)

    stock = yf.Ticker(ticker)
    info  = stock.info

    if not info or "longName" not in info:
        st.error("Could not find stock data. Please check the company name.")
        st.stop()

    # Header
    name   = info.get("longName", ticker)
    sector = info.get("sector", "N/A")
    indust = info.get("industry", "N/A")
    exch   = info.get("exchange", "")

    st.markdown(f"""
    <h2 style="font-family:'DM Serif Display',serif; color:var(--text); margin-bottom:0.1rem;">{name}</h2>
    <p style="color:var(--muted); font-size:0.85rem; margin-top:0;">
      <span class="pill">{ticker}</span>
      <span class="pill">{exch}</span>
      <span class="pill">{sector}</span>
      <span class="pill">{indust}</span>
    </p>
    """, unsafe_allow_html=True)

    # Key metrics row
    mc  = info.get("marketCap", 0) or 0
    cur = info.get("currentPrice") or info.get("regularMarketPrice") or "N/A"
    prev_close = info.get("previousClose") or cur
    try:
        day_chg = round((float(cur) - float(prev_close)) / float(prev_close) * 100, 2)
        delta_str = f"{day_chg:+.2f}%"
    except Exception:
        delta_str = None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Price",       f"${cur}",                  delta_str)
    c2.metric("Market Cap",  fmt_large(mc))
    c3.metric("P/E Ratio",   round(info.get("trailingPE") or 0, 2) or "N/A")
    c4.metric("52W High",    f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
    c5.metric("52W Low",     f"${info.get('fiftyTwoWeekLow', 'N/A')}")
    c6.metric("Beta",        round(info.get("beta") or 0, 2) or "N/A")

    # Dividend + short interest sub-row
    d1, d2, d3, d4 = st.columns(4)
    div_yield = info.get("dividendYield")
    div_rate  = info.get("dividendRate")
    short_pct = info.get("shortPercentOfFloat")
    fwd_pe    = info.get("forwardPE")
    d1.metric("Div Yield",   f"{div_yield*100:.2f}%" if div_yield else "N/A")
    d2.metric("Div Rate",    f"${div_rate:.2f}" if div_rate else "N/A")
    d3.metric("Short Float", f"{short_pct*100:.1f}%" if short_pct else "N/A")
    d4.metric("Fwd P/E",     round(fwd_pe, 2) if fwd_pe else "N/A")

    st.divider()

    # Historical data
    hist = stock.history(period="1y")
    if hist.empty:
        st.warning("No historical price data available.")
        st.stop()

    hist["MA50"]        = hist["Close"].rolling(50).mean()
    hist["MA200"]       = hist["Close"].rolling(200).mean()
    hist["RSI"]         = ta.momentum.RSIIndicator(hist["Close"]).rsi()
    hist["MACD"]        = ta.trend.MACD(hist["Close"]).macd()
    hist["MACD_Signal"] = ta.trend.MACD(hist["Close"]).macd_signal()
    hist["hband"]       = ta.volatility.BollingerBands(hist["Close"]).bollinger_hband()
    hist["lband"]       = ta.volatility.BollingerBands(hist["Close"]).bollinger_lband()
    hist["mband"]       = ta.volatility.BollingerBands(hist["Close"]).bollinger_mavg()
    hist["ADX"]         = ta.trend.ADXIndicator(hist["High"], hist["Low"], hist["Close"]).adx()

    rsi_val     = round(hist["RSI"].iloc[-1], 1)
    macd_val    = hist["MACD"].iloc[-1]
    macd_sig_val = hist["MACD_Signal"].iloc[-1]
    adx_val     = round(hist["ADX"].iloc[-1], 1)
    current_px  = hist["Close"].iloc[-1]
    ma50_val    = hist["MA50"].iloc[-1]
    ma200_val   = hist["MA200"].iloc[-1]
    volatility  = round(hist["Close"].pct_change().std() * 100, 3)

    macd_signal = "Bullish" if macd_val > macd_sig_val else "Bearish"
    if current_px > hist["hband"].iloc[-1]:
        bb_signal = "Overbought"
    elif current_px < hist["lband"].iloc[-1]:
        bb_signal = "Oversold"
    else:
        bb_signal = "Neutral"

    if current_px > ma50_val > ma200_val:
        trend_signal = "Strong Uptrend"
    elif current_px < ma50_val < ma200_val:
        trend_signal = "Strong Downtrend"
    else:
        trend_signal = "Mixed Trend"

    # Price chart tabs
    st.markdown("<p class='section-label'>Price & Technicals</p>", unsafe_allow_html=True)
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Candlestick + Bollinger", "Moving Averages", "MACD / RSI"])

    with chart_tab1:
        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            name="Price",
            increasing_line_color=ACCENT, decreasing_line_color=RED,
        ))
        fig_candle.add_trace(go.Scatter(x=hist.index, y=hist["hband"], name="Upper BB", line=dict(color=AMBER, width=1, dash="dot")))
        fig_candle.add_trace(go.Scatter(x=hist.index, y=hist["mband"], name="Mid BB",   line=dict(color="#9ca3af", width=1)))
        fig_candle.add_trace(go.Scatter(x=hist.index, y=hist["lband"], name="Lower BB", line=dict(color=AMBER, width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(251,191,36,0.04)"))
        fig_candle.update_layout(xaxis_rangeslider_visible=False, title="Candlestick + Bollinger Bands", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_candle, use_container_width=True)

    with chart_tab2:
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Close", line=dict(color=ACCENT, width=2)))
        fig_ma.add_trace(go.Scatter(x=hist.index, y=hist["MA50"],  name="MA50",  line=dict(color=BLUE,  width=1.5, dash="dash")))
        fig_ma.add_trace(go.Scatter(x=hist.index, y=hist["MA200"], name="MA200", line=dict(color=AMBER, width=1.5, dash="dash")))
        fig_ma.update_layout(title="Close Price with Moving Averages", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_ma, use_container_width=True)

    with chart_tab3:
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=hist.index, y=hist["MACD"],        name="MACD",   line=dict(color=ACCENT, width=2)))
        fig_macd.add_trace(go.Scatter(x=hist.index, y=hist["MACD_Signal"], name="Signal", line=dict(color=AMBER,  width=1.5, dash="dash")))
        macd_hist_vals = hist["MACD"] - hist["MACD_Signal"]
        fig_macd.add_trace(go.Bar(
            x=hist.index, y=macd_hist_vals,
            name="Histogram",
            marker_color=[ACCENT if v >= 0 else RED for v in macd_hist_vals],
            opacity=0.6,
        ))
        fig_macd.update_layout(title="MACD", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_macd, use_container_width=True)

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=hist.index, y=hist["RSI"], name="RSI", line=dict(color=BLUE, width=2)))
        fig_rsi.add_hline(y=70, line_dash="dot", line_color=RED,   annotation_text="Overbought 70")
        fig_rsi.add_hline(y=30, line_dash="dot", line_color=ACCENT, annotation_text="Oversold 30")
        fig_rsi.update_layout(title="RSI", yaxis_range=[0, 100], **PLOTLY_LAYOUT)
        st.plotly_chart(fig_rsi, use_container_width=True)

    # Trading Signals
    st.markdown("<p class='section-label'>Trading Signals</p>", unsafe_allow_html=True)
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("RSI",              rsi_val)
    s2.metric("MACD Signal",      macd_signal)
    s3.metric("Bollinger Signal", bb_signal)
    s4.metric("ADX (Trend Str.)", adx_val)
    s5.metric("Trend",            trend_signal)

    st.markdown(
        f"RSI status: {badge('Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral')} &nbsp; "
        f"MACD: {badge(macd_signal)} &nbsp; Trend: {badge(trend_signal)}",
        unsafe_allow_html=True
    )

    st.divider()

    # Volume Analysis
    st.markdown("<p class='section-label'>Volume Analysis</p>", unsafe_allow_html=True)
    hist["Vol_MA20"] = hist["Volume"].rolling(20).mean()
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(
        x=hist.index, y=hist["Volume"],
        name="Volume",
        marker_color=[ACCENT if c >= o else RED for c, o in zip(hist["Close"], hist["Open"])],
        opacity=0.7,
    ))
    fig_vol.add_trace(go.Scatter(x=hist.index, y=hist["Vol_MA20"], name="20-day Avg", line=dict(color=AMBER, width=2)))
    fig_vol.update_layout(title="Daily Volume with 20-Day Average", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_vol, use_container_width=True)

    latest_vol    = hist["Volume"].iloc[-1]
    avg_vol       = hist["Vol_MA20"].iloc[-1]
    vol_pct       = round((latest_vol - avg_vol) / avg_vol * 100, 1) if avg_vol else 0
    vol1, vol2, vol3 = st.columns(3)
    vol1.metric("Latest Volume",    f"{int(latest_vol):,}")
    vol2.metric("20-Day Avg Volume", f"{int(avg_vol):,}")
    vol3.metric("vs Avg",           f"{vol_pct:+.1f}%")

    st.divider()

    # News & Sentiment
    st.markdown("<p class='section-label'>News & Sentiment</p>", unsafe_allow_html=True)

    headlines       = []
    sentiment_score = 0.0
    news_articles   = []

    for article in stock.news[:6]:
        content  = article.get("content", {})
        title    = content.get("title", "")
        summary  = content.get("summary", "") or content.get("description", "")
        pub_date = content.get("pubDate", "")
        url      = content.get("canonicalUrl", {}).get("url", "")
        if title:
            headlines.append(title)
            s = score_sentiment(summary if summary else title)
            sentiment_score += s
            news_articles.append({
                "title": title, "summary": summary,
                "pub_date": pub_date, "url": url, "score": round(s, 3)
            })

    with st.spinner("Fetching Reddit sentiment…"):
        reddit_titles = get_reddit_titles(ticker)
        reddit_score  = score_sentiment(" ".join(reddit_titles[:5]))

    # Combined sentiment gauge
    avg_news_sent = sentiment_score / max(len(news_articles), 1)
    overall_sent  = (avg_news_sent + reddit_score) / 2

    sent1, sent2, sent3 = st.columns(3)
    sent1.metric("News Sentiment",   f"{round(avg_news_sent, 3):+.3f}")
    sent2.metric("Reddit Sentiment", f"{round(reddit_score, 3):+.3f}")
    sent3.metric("Combined Score",   f"{round(overall_sent, 3):+.3f}")

    if overall_sent > 0.15:
        st.success("Overall Sentiment: **Positive** — Markets appear optimistic about this stock.")
    elif overall_sent < -0.15:
        st.error("Overall Sentiment: **Negative** — Caution advised; negative tone in coverage.")
    else:
        st.info("Overall Sentiment: **Neutral** — Mixed signals in news and social media.")

    with st.expander("📰  Latest News Articles"):
        for a in news_articles:
            sent_label = "🟢" if a["score"] > 0.05 else ("🔴" if a["score"] < -0.05 else "🟡")
            st.markdown(f"**{a['title']}** {sent_label}")
            st.caption(a["pub_date"])
            if a["summary"]:
                st.write(a["summary"])
            if a["url"]:
                st.markdown(f"[Read full article →]({a['url']})")
            st.divider()

    if reddit_titles:
        with st.expander("💬  Reddit Buzz"):
            for t in reddit_titles:
                st.markdown(f"— {t}")

    if headlines:
        with st.spinner("Generating sentiment verdict…"):
            sent_res = model.invoke([
                SystemMessage(content="You are a financial analyst. Be concise and direct."),
                HumanMessage(content=(
                    f"Headlines for {ticker}:\n" + "\n".join(f"- {h}" for h in headlines) +
                    (f"\n\nReddit posts:\n" + "\n".join(f"- {t}" for t in reddit_titles[:5]) if reddit_titles else "") +
                    "\n\nIn 2 sentences: Is this stock Bullish, Bearish or Neutral based on sentiment? State key reason."
                ))
            ])
        st.info(f"🤖 AI Sentiment Verdict: {sent_res.content}")

    st.divider()

    # Financials
    st.markdown("<p class='section-label'>Financial Statements</p>", unsafe_allow_html=True)
    def fmt_fin(df):
        """Rename columns to just the year (e.g. 2024-09-30 00:00:00 → 2024)."""
        if df is None or df.empty:
            return df
        df = df.copy()
        df.columns = [str(c)[:4] if hasattr(c, '__str__') else c for c in df.columns]
        return df

    fin_tab1, fin_tab2, fin_tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
    with fin_tab1:
        st.dataframe(fmt_fin(stock.financials),    use_container_width=True)
    with fin_tab2:
        st.dataframe(fmt_fin(stock.balance_sheet), use_container_width=True)
    with fin_tab3:
        st.dataframe(fmt_fin(stock.cash_flow),     use_container_width=True)

    # Growth metrics
    st.markdown("<p class='section-label'>Growth & Margins</p>", unsafe_allow_html=True)

    try:
        rev_series  = stock.financials.loc["Total Revenue"]
        inc_series  = stock.financials.loc["Net Income"]
        margin_ser  = (inc_series / rev_series * 100).sort_index()

        gm1, gm2, gm3 = st.columns(3)
        if len(rev_series) >= 2:
            rev_growth = (rev_series.iloc[0] - rev_series.iloc[1]) / abs(rev_series.iloc[1]) * 100
            gm1.metric("Revenue Growth YoY", f"{rev_growth:.1f}%", delta="YoY")
        if len(inc_series) >= 2 and inc_series.iloc[1] != 0:
            profit_growth = (inc_series.iloc[0] - inc_series.iloc[1]) / abs(inc_series.iloc[1]) * 100
            gm2.metric("Net Income Growth YoY", f"{profit_growth:.1f}%", delta="YoY")
        gm3.metric("Latest Net Margin", f"{margin_ser.iloc[-1]:.2f}%")

        fig_margin = go.Figure()
        fig_margin.add_trace(go.Scatter(
            x=margin_ser.index, y=margin_ser.values,
            fill="tozeroy", name="Net Margin %",
            line=dict(color=ACCENT, width=2),
            fillcolor="rgba(52,211,153,0.1)",
        ))
        fig_margin.update_layout(title="Net Profit Margin Trend", yaxis_ticksuffix="%", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_margin, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render growth metrics: {e}")

    # Financial AI analysis — short
    ana_str = ""
    try:
        ana_str = stock.financials.loc[
            ["Total Revenue", "Net Income", "Operating Income", "Gross Profit"]
        ].to_string()
        with st.spinner("Analysing financials…"):
            fin_res = model.invoke([
                SystemMessage(content="You are a financial analyst. Be concise — 3 bullet points max."),
                HumanMessage(content=f"Analyse financial trend for {ticker}:\n{ana_str}\nHighlight: revenue trend, margin direction, key risk.")
            ])
        st.info(f"📊 Financial Summary: {fin_res.content}")
    except Exception as e:
        st.warning(f"Financial analysis unavailable: {e}")

    st.divider()

    # Competitor Comparison
    st.markdown("<p class='section-label'>Competitor Comparison</p>", unsafe_allow_html=True)
    with st.spinner("Fetching competitors…"):
        competitors = get_competitors(ticker)

    if competitors:
        main_mc  = (info.get("marketCap")    or 0) / 1e9
        main_pe  =  info.get("trailingPE")   or 0
        main_rev = (info.get("totalRevenue") or 0) / 1e9

        comp_rows = [{"Ticker": ticker, "Company": info.get("shortName", ticker),
                      "Mkt Cap (B)": round(main_mc, 2), "P/E": round(main_pe, 2),
                      "Revenue (B)": round(main_rev, 2), "52W High": info.get("fiftyTwoWeekHigh"),
                      "Beta": round(info.get("beta") or 0, 2)}]

        for t in competitors:
            try:
                ci  = yf.Ticker(t).info
                c_mc  = round((ci.get("marketCap")    or 0) / 1e9, 2)
                c_pe  = round( ci.get("trailingPE")   or 0, 2)
                c_rev = round((ci.get("totalRevenue") or 0) / 1e9, 2)
                comp_rows.append({
                    "Ticker":     t,
                    "Company":    ci.get("shortName", t),
                    "Mkt Cap (B)": c_mc,
                    "P/E":        c_pe,
                    "Revenue (B)": c_rev,
                    "52W High":   ci.get("fiftyTwoWeekHigh", "N/A"),
                    "Beta":       round(ci.get("beta") or 0, 2),
                })
            except Exception:
                pass

        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Competitor bar chart
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            x=comp_df["Company"], y=comp_df["Mkt Cap (B)"],
            name="Market Cap (B)", marker_color=ACCENT, opacity=0.8,
        ))
        fig_comp.update_layout(title="Market Cap Comparison", yaxis_title="$B", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    # vs S&P 500
    st.markdown("<p class='section-label'>Macro Context</p>", unsafe_allow_html=True)

    try:
        sp500 = yf.Ticker("^GSPC").history(period="1y")["Close"]
        norm_stock = hist["Close"] * 100 / hist["Close"].iloc[0]
        norm_sp    = sp500 * 100 / sp500.iloc[0]
        fig_vs = go.Figure()
        fig_vs.add_trace(go.Scatter(x=norm_stock.index, y=norm_stock.values, name=ticker, line=dict(color=ACCENT, width=2)))
        fig_vs.add_trace(go.Scatter(x=norm_sp.index,    y=norm_sp.values,    name="S&P 500", line=dict(color="#9ca3af", width=1.5, dash="dash")))
        fig_vs.update_layout(title=f"{ticker} vs S&P 500 (Normalised, Base = 100)", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_vs, use_container_width=True)
    except Exception:
        st.warning("S&P 500 comparison unavailable.")

    mac1, mac2 = st.columns(2)
    with mac1:
        try:
            vix_val = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
            if vix_val < 20:
                st.success(f"VIX {vix_val:.1f} — Market Calm")
            elif vix_val < 30:
                st.warning(f"VIX {vix_val:.1f} — Elevated Uncertainty")
            else:
                st.error(f"VIX {vix_val:.1f} — High Fear / Volatility")
        except Exception:
            st.warning("VIX data unavailable.")
    with mac2:
        try:
            tnx_val = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
            st.metric("10Y Treasury Yield", f"{tnx_val:.2f}%")
            if tnx_val > 4.5:
                st.error("High Rates — Headwind for equities")
            elif tnx_val >= 3:
                st.warning("Moderate Rates — Neutral")
            else:
                st.success("Low Rates — Tailwind for equities")
        except Exception:
            st.warning("Treasury yield data unavailable.")

    st.divider()

    # Risk Assessment
    st.markdown("<p class='section-label'>Risk Assessment</p>", unsafe_allow_html=True)

    rsi_risk        = max(0, min(40, rsi_val - 50)) if rsi_val > 50 else 0
    sentiment_risk  = max(0, min(30, -sentiment_score * 2))
    volatility_risk = min(30, volatility * 5)
    risk_score      = min(100, volatility_risk + rsi_risk + sentiment_risk)

    ri1, ri2, ri3, ri4 = st.columns(4)
    ri1.metric("Overall Risk", f"{risk_score:.1f} / 100")
    ri2.metric("Volatility Risk (30)", f"{volatility_risk:.1f}")
    ri3.metric("RSI Risk (40)",        f"{rsi_risk:.1f}")
    ri4.metric("Sentiment Risk (30)",  f"{sentiment_risk:.1f}")

    # Risk gauge bar
    risk_color = RED if risk_score > 70 else (AMBER if risk_score > 40 else ACCENT)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"suffix": "/100", "font": {"color": risk_color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b5563"},
            "bar":  {"color": risk_color},
            "steps": [
                {"range": [0,  40], "color": "#052e16"},
                {"range": [40, 70], "color": "#1c1a0a"},
                {"range": [70, 100],"color": "#2d0d0d"},
            ],
            "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.75, "value": risk_score},
            "bgcolor": "rgba(0,0,0,0)",
        },
        title={"text": "Risk Score", "font": {"color": "#9ca3af"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"), height=260, margin=dict(t=40,b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if risk_score > 70:
        st.error("High Risk — Proceed with significant caution.")
    elif risk_score > 40:
        st.warning("Medium Risk — Monitor positions closely.")
    else:
        st.success("Low Risk — Relatively stable profile.")

    st.divider()

    # Analyst Ratings
    st.markdown("<p class='section-label'>Analyst Ratings</p>", unsafe_allow_html=True)
    try:
        recs = stock.recommendations_summary
        if recs is not None and not recs.empty:
            w    = recs.iloc[0]
            buy  = 2 * w["buy"]   + w["strongBuy"]
            sell =     w["sell"]  + 2 * w["strongSell"]
            an1, an2 = st.columns([1, 2])
            with an1:
                signal_analyst = "Bullish" if buy > sell else "Bearish"
                st.markdown(f"**Consensus:** {badge(signal_analyst)}", unsafe_allow_html=True)
                st.markdown(f"**Strong Buy:** {int(w['strongBuy'])}  |  **Buy:** {int(w['buy'])}  |  **Hold:** {int(w['hold'])}  |  **Sell:** {int(w['sell'])}  |  **Strong Sell:** {int(w['strongSell'])}")
            with an2:
                fig_bar = go.Figure(data=[go.Bar(
                    x=["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
                    y=[w["strongBuy"], w["buy"], w["hold"], w["sell"], w["strongSell"]],
                    marker_color=["#14532d", ACCENT, AMBER, RED, "#7f1d1d"],
                )])
                fig_bar.update_layout(title="", showlegend=False, height=220, **PLOTLY_LAYOUT)
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No analyst rating data available.")
    except Exception as e:
        st.warning(f"Could not load analyst ratings: {e}")

    st.divider()

    # Price Target
    st.markdown("<p class='section-label'>Analyst Price Targets</p>", unsafe_allow_html=True)
    try:
        target_mean = info.get("targetMeanPrice")
        target_low  = info.get("targetLowPrice")
        target_high = info.get("targetHighPrice")
        target_med  = info.get("targetMedianPrice")

        if target_mean:
            updown = round((target_mean - float(cur)) / float(cur) * 100, 1)
            pt1, pt2, pt3, pt4 = st.columns(4)
            pt1.metric("Mean Target",   f"${target_mean}",  f"{updown:+.1f}% vs current")
            pt2.metric("Median Target", f"${target_med}"  if target_med  else "N/A")
            pt3.metric("Low Target",    f"${target_low}"   if target_low  else "N/A")
            pt4.metric("High Target",   f"${target_high}"  if target_high else "N/A")
        else:
            st.warning("No analyst price target data found.")
    except Exception as e:
        st.warning(f"Price target unavailable: {e}")

    st.divider()

    # Insider + Institutional
    st.markdown("<p class='section-label'>Insider & Institutional Activity</p>", unsafe_allow_html=True)
    Total_Bought = 0
    Total_Sold   = 0

    insider_df = stock.insider_transactions
    if insider_df is not None and not insider_df.empty and "Text" in insider_df.columns:
        Total_Bought = insider_df[insider_df["Text"].str.contains("Buy|Purchase", case=False, na=False)]["Shares"].sum()
        Total_Sold   = insider_df[insider_df["Text"].str.contains("Sale", case=False, na=False)]["Shares"].sum()

    ins1, ins2 = st.columns(2)
    with ins1:
        st.metric("Insider Bought", f"{int(Total_Bought):,}")
        st.metric("Insider Sold",   f"{int(Total_Sold):,}")
        if Total_Bought > Total_Sold:
            st.success("Insiders net buyers — bullish signal.")
        elif Total_Sold > Total_Bought:
            st.warning("Insiders net sellers — watch closely.")
        else:
            st.info("Neutral insider activity.")

    with ins2:
        inst_df = stock.institutional_holders
        if inst_df is not None and not inst_df.empty:
            top_holder      = inst_df.iloc[0]["Holder"]
            total_ownership = round(inst_df["pctHeld"].sum() * 100, 2)
            st.metric("Top Holder",     top_holder)
            st.metric("Institutional Ownership", f"{total_ownership}%")

            # Mini bar chart for top 5 holders
            top5 = inst_df.head(5)
            fig_inst = go.Figure(go.Bar(
                x=top5["Holder"], y=(top5["pctHeld"] * 100).round(2),
                marker_color=BLUE, opacity=0.85,
            ))
            fig_inst.update_layout(title="Top 5 Institutional Holders", yaxis_ticksuffix="%", height=240, **PLOTLY_LAYOUT)
            st.plotly_chart(fig_inst, use_container_width=True)
        else:
            st.warning("No institutional holder data available.")

    st.divider()

    # Options Market
    st.markdown("<p class='section-label'>Options Market Intelligence</p>", unsafe_allow_html=True)
    pcr = None
    try:
        if stock.options:
            opt          = stock.option_chain(stock.options[0])
            call_volume  = opt.calls["volume"].sum()
            put_volume   = opt.puts["volume"].sum()
            call_oi      = opt.calls["openInterest"].sum()
            put_oi       = opt.puts["openInterest"].sum()
            if call_volume > 0:
                pcr     = round(put_volume / call_volume, 2)
            pcr_oi  = round(put_oi / call_oi, 2) if call_oi > 0 else None

            op1, op2, op3, op4 = st.columns(4)
            op1.metric("Put/Call Ratio (Vol)", pcr if pcr else "N/A")
            op2.metric("Put/Call Ratio (OI)",  pcr_oi if pcr_oi else "N/A")
            op3.metric("Total Call Volume",    f"{int(call_volume):,}")
            op4.metric("Total Put Volume",     f"{int(put_volume):,}")

            if pcr is not None:
                if pcr < 0.7:
                    st.success("Options flow: Bullish (more calls than puts).")
                elif pcr > 1.3:
                    st.error("Options flow: Bearish (heavy put buying).")
                else:
                    st.info("Options flow: Neutral.")
    except Exception:
        st.warning("Options data not available for this ticker.")

    st.divider()

    # Valuation Engine
    st.markdown("<p class='section-label'>Valuation</p>", unsafe_allow_html=True)
    stock_pe = info.get("trailingPE")
    pe_list  = []

    for comp in competitors:
        try:
            comp_pe = yf.Ticker(comp).info.get("trailingPE")
            if comp_pe and comp_pe > 0:
                pe_list.append(comp_pe)
        except Exception:
            pass

    if stock_pe and stock_pe > 0 and pe_list:
        industry_pe = sum(pe_list) / len(pe_list)
        if   stock_pe < industry_pe * 0.8:
            val_signal = "Undervalued"
        elif stock_pe > industry_pe * 1.2:
            val_signal = "Overvalued"
        else:
            val_signal = "Fairly Valued"

        v1, v2, v3 = st.columns(3)
        v1.metric("Stock P/E",        f"{stock_pe:.2f}")
        v2.metric("Industry Avg P/E", f"{industry_pe:.2f}")
        v3.metric("Valuation Signal", val_signal)

        # PEG Ratio
        peg = info.get("pegRatio")
        pb  = info.get("priceToBook")
        ps  = info.get("priceToSalesTrailing12Months")
        ev_ebitda = info.get("enterpriseToEbitda")

        vv1, vv2, vv3, vv4 = st.columns(4)
        vv1.metric("PEG Ratio",       round(peg, 2) if peg else "N/A")
        vv2.metric("Price/Book",      round(pb, 2)  if pb  else "N/A")
        vv3.metric("Price/Sales",     round(ps, 2)  if ps  else "N/A")
        vv4.metric("EV/EBITDA",       round(ev_ebitda, 2) if ev_ebitda else "N/A")
    else:
        st.warning("Not enough P/E data to run valuation comparison.")

    st.divider()

    # Earnings Stability
    st.markdown("<p class='section-label'>Earnings Stability</p>", unsafe_allow_html=True)
    stability_score  = 0
    stability_signal = "N/A"
    try:
        income_series   = stock.financials.loc["Net Income"]
        cv              = income_series.std() / income_series.mean() if income_series.mean() != 0 else float("inf")
        stability_score = round(max(0, 100 - cv * 100), 1)
        stability_signal = ("Very Stable" if stability_score > 80 else
                            "Stable"      if stability_score > 60 else
                            "Moderate"    if stability_score > 40 else "Unstable")
    except Exception:
        pass

    es1, es2 = st.columns(2)
    es1.metric("Stability Score",  f"{stability_score} / 100")
    es2.metric("Stability Signal",  stability_signal)

    st.divider()

    # Final AI Recommendation
    st.markdown("<p class='section-label'>Final AI Recommendation</p>", unsafe_allow_html=True)
    with st.spinner("Synthesising all data…"):
        final_res = model.invoke([
            SystemMessage(content="You are a professional buy-side analyst. Be concise and structured."),
            HumanMessage(content=f"""
Stock: {ticker} | Price: {cur}
52W High: {info.get('fiftyTwoWeekHigh')} | 52W Low: {info.get('fiftyTwoWeekLow')}
RSI: {rsi_val} | MACD: {macd_signal} | Bollinger: {bb_signal} | ADX: {adx_val}
Trend: {trend_signal} | Volatility: {volatility:.2f}%
Sentiment Score: {round(sentiment_score, 2)} | Risk Score: {risk_score:.1f}/100
Stability: {stability_signal} | Valuation vs peers: {val_signal if stock_pe and pe_list else 'N/A'}
Insider Net: {'Buying' if Total_Bought > Total_Sold else 'Selling' if Total_Sold > Total_Bought else 'Neutral'}
Financials: {ana_str[:500] if ana_str else 'N/A'}

Provide in this exact format:
VERDICT: [BUY / HOLD / SELL]
RATIONALE: [2-3 sentences]
KEY RISK: [1 sentence]
TIMEFRAME: [short/medium/long term]

End with: ⚠️ Not financial advice.
""")
        ])

    # Parse and style recommendation
    resp_text = final_res.content
    if "BUY" in resp_text.upper()[:50]:
        rec_color = "success"
    elif "SELL" in resp_text.upper()[:50]:
        rec_color = "error"
    else:
        rec_color = "info"

    if rec_color == "success":
        st.success(resp_text)
    elif rec_color == "error":
        st.error(resp_text)
    else:
        st.info(resp_text)