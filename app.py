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
import os
from dotenv import load_dotenv
load_dotenv()


API_KEY = os.environ.get("GROQ_API_KEY")
model = ChatGroq(api_key=API_KEY, model_name="llama-3.1-8b-instant")

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

sentiment_model = load_sentiment_model()


def get_ticker(company: str) -> str:
    res = model.invoke([HumanMessage(
        content=f"Return ONLY the stock ticker symbol for '{company}'. Example: 'Apple' -> 'AAPL'. Just the ticker, nothing else."
    )])
    return res.content.strip()


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


def get_reddit_topics(ticker: str) -> str:
    url = f"https://www.reddit.com/search.json?q={ticker}+stock&limit=10"
    texts = []
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = res.json()
        for post in data["data"]["children"]:
            title = post["data"]["title"]
            texts.append(title)
            permalink = post["data"]["permalink"]
            comment_res = requests.get(
                f"https://www.reddit.com{permalink}.json",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            comment_json = comment_res.json()
            if len(comment_json) > 1:
                for c in comment_json[1]["data"]["children"][:3]:
                    if "body" in c["data"]:
                        texts.append(c["data"]["body"])
    except Exception:
        pass
    return " ".join(texts)


def summarize_text(text: str) -> str:
    if not text:
        return ""
    try:
        res = model.invoke([HumanMessage(content=f"Summarize this in 2-3 sentences:\n{text[:3000]}")])
        return res.content
    except Exception:
        return ""


def score_sentiment(text: str) -> float:
    try:
        result = sentiment_model(text[:512])[0]
        return result["score"] if result["label"] == "positive" else -result["score"]
    except Exception:
        return 0.0


st.set_page_config(page_title="AI Stock Research", layout="wide")

st.title("AI Stock Research Assistant")
st.caption("Powered by LLaMA 3.1 · FinBERT · yFinance")
st.divider()

col_a, col_b = st.columns(2)
with col_a:
    company = st.text_input("Enter Company Name", placeholder="e.g. Apple, Tesla, Reliance")
with col_b:
    compare_input = st.text_input("Compare Multiple Stocks", placeholder="e.g. Apple, Tesla, Google")

if compare_input:
    st.divider()
    st.header("Stock Comparison")

    with st.spinner("Finding tickers..."):
        tickers = get_tickers_from_text(compare_input)

    if not tickers:
        st.error("Could not extract tickers. Please try again.")
    else:
        rows = []
        fig = go.Figure()

        for compan in tickers:
            try:
                stock = yf.Ticker(compan)
                info = stock.info
                hist = stock.history(period="1y")["Close"]
                if hist.empty:
                    continue

                norm = hist * 100 / hist.iloc[0]
                fig.add_trace(go.Scatter(
                    x=norm.index, y=norm.values,
                    name=info.get("longName", compan),
                    mode="lines"
                ))

                sentiment_score = 0.0
                summ = ""
                for article in stock.news[:5]:
                    content = article.get("content", {})
                    title = content.get("title", "")
                    summary = content.get("summary", "") or content.get("description", "")
                    if title:
                        summ += summarize_text(summary)
                        sentiment_score += score_sentiment(summary if summary else title)

                rows.append({
                    "Company": info.get("longName", compan),
                    "Price": info.get("currentPrice"),
                    "Market Cap": info.get("marketCap"),
                    "Trailing PE": round(info.get("trailingPE") or 0, 2),
                    "52W High": info.get("fiftyTwoWeekHigh"),
                    "Total Revenue": info.get("totalRevenue"),
                    "YTD Return (%)": round((hist.iloc[-1] - hist.iloc[0]) * 100 / hist.iloc[0], 2),
                    "Volatility (%)": round(hist.pct_change().std() * 100, 2),
                    "RSI": round(ta.momentum.RSIIndicator(hist).rsi().iloc[-1], 2),
                    "Sentiment Score": round(sentiment_score, 2),
                    "News Summary": summ,
                })
            except Exception as e:
                st.warning(f"Could not fetch data for {compan}: {e}")

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.drop(columns=["Sentiment Score", "News Summary"], errors="ignore"),
                use_container_width=True
            )

            fig.update_layout(
                title="Normalised Price Performance (Base = 100)",
                xaxis_title="Date",
                yaxis_title="Normalised Price",
                legend_title="Company",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("AI Verdict")
            with st.spinner("Analysing all stocks..."):
                r = model.invoke([
                    SystemMessage(content="You are an expert financial analyst."),
                    HumanMessage(content=f"""
You are given data of multiple stocks including Price, PE Ratio, YTD Return, Volatility, RSI,
Sentiment Score, and News Summaries.

Analyse each stock and provide:
1. A short analysis of each stock
2. Ranking from best to worst investment
3. Which one to BUY and why
4. Expected return potential for the top pick
5. Risk warning if any

End with: This is not financial advice.

Data:
{df.to_string()}
""")
                ])
            st.write(r.content)

if company:
    st.divider()

    with st.spinner("Finding ticker..."):
        ticker = get_ticker(company)

    stock = yf.Ticker(ticker)
    info = stock.info

    if not info or "longName" not in info:
        st.error("Could not find stock data. Please check the company name.")
        st.stop()

    st.header(info.get("longName", ticker))
    st.caption(f"`{ticker}` · {info.get('sector', 'N/A')} · {info.get('industry', 'N/A')}")
    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    mc = info.get("marketCap", 0) or 0
    c1.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
    c2.metric("Market Cap",    f"${round(mc/1e9, 2)}B" if mc else "N/A")
    c3.metric("PE Ratio",      round(info.get("trailingPE") or 0, 2) or "N/A")
    c4.metric("52W High",      f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
    c5.metric("52W Low",       f"${info.get('fiftyTwoWeekLow', 'N/A')}")
    st.divider()

    hist = stock.history(period="1y")
    if hist.empty:
        st.warning("No historical data available.")
        st.stop()

    hist["MA50"]  = hist["Close"].rolling(50).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()
    hist["RSI"]   = ta.momentum.RSIIndicator(hist["Close"]).rsi()

    st.subheader("Price Chart with Moving Averages")
    st.line_chart(hist[["Close", "MA50", "MA200"]])

    rsi_val = round(hist["RSI"].iloc[-1], 2)
    current = hist["Close"].iloc[-1]
    ma50    = hist["MA50"].iloc[-1]
    ma200   = hist["MA200"].iloc[-1]

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("RSI", rsi_val)
    with rc2:
        if rsi_val > 70:
            st.error("Overbought (RSI > 70)")
        elif rsi_val < 30:
            st.success("Oversold (RSI < 30)")
        else:
            st.info("Neutral RSI")
    with rc3:
        if current > ma50 > ma200:
            st.success("Strong Uptrend")
        elif current < ma50 < ma200:
            st.error("Strong Downtrend")
        else:
            st.info("Mixed Trend")

    st.divider()

    st.subheader("Latest News and Sentiment")
    headlines       = []
    sentiment_score = 0.0
    news_articles   = []

    for article in stock.news[:5]:
        content  = article.get("content", {})
        title    = content.get("title", "")
        summary  = content.get("summary", "") or content.get("description", "")
        pub_date = content.get("pubDate", "")
        url      = content.get("canonicalUrl", {}).get("url", "")
        if title:
            headlines.append(title)
            ai_summary = summarize_text(summary)
            sentiment_score += score_sentiment(summary if summary else title)
            news_articles.append({
                "title": title, "summary": summary,
                "pub_date": pub_date, "url": url, "ai_summary": ai_summary
            })

    with st.expander("Click to expand news articles"):
        for a in news_articles:
            st.markdown(f"**{a['title']}**")
            st.caption(a["pub_date"])
            if a["summary"]:
                st.write(a["summary"])
            if a["ai_summary"]:
                st.info(f"AI Summary: {a['ai_summary']}")
            if a["url"]:
                st.markdown(f"[Read More]({a['url']})")
            st.divider()

    with st.spinner("Fetching Reddit sentiment..."):
        reddit_text  = get_reddit_topics(ticker)
        reddit_score = score_sentiment(reddit_text)

    if headlines:
        with st.spinner("Analysing sentiment..."):
            sentiment_res = model.invoke([
                SystemMessage(content="You are a financial analyst."),
                HumanMessage(content=(
                    f"Based on these headlines:\n{chr(10).join(headlines)}\n\n"
                    f"And Reddit content:\n{reddit_text[:1000]}\n\n"
                    f"Is {ticker} Bullish, Bearish or Neutral? Give reason."
                ))
            ])
        st.info(f"News Sentiment: {sentiment_res.content}")

    st.divider()

    st.subheader("Financial Statements")
    t1, t2, t3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
    with t1:
        st.dataframe(stock.financials, use_container_width=True)
    with t2:
        st.dataframe(stock.balance_sheet, use_container_width=True)
    with t3:
        st.dataframe(stock.cash_flow, use_container_width=True)

    ana = ""
    try:
        ana = stock.financials.loc[
            ["Total Revenue", "Net Income", "Operating Income", "Gross Profit"]
        ].to_string()
        with st.spinner("Analysing financials..."):
            fin_res = model.invoke([
                SystemMessage(content="You are a financial analyst."),
                HumanMessage(content=(
                    f"Analyse this company's financial trend:\n{ana}\n"
                    f"Is revenue growing? Are margins improving?"
                ))
            ])
        st.subheader("Financial Analysis")
        st.write(fin_res.content)
    except Exception as e:
        st.warning(f"Could not analyse financials: {e}")

    st.divider()

    st.subheader("Earnings and Fundamental Intelligence")
    try:
        rev = stock.financials.loc["Total Revenue"]
        inc = stock.financials.loc["Net Income"]

        ef1, ef2 = st.columns(2)
        with ef1:
            if len(rev) >= 2:
                rev_growth = (rev.iloc[0] - rev.iloc[1]) / abs(rev.iloc[1]) * 100
                st.metric("Revenue Growth YoY", f"{rev_growth:.1f}%", delta="YoY")
                if rev_growth > 10:
                    st.success("Strong Revenue Growth")
                elif rev_growth > 0:
                    st.info("Moderate Growth")
                else:
                    st.error("Revenue Declining")
            else:
                st.write("Not enough data")

        with ef2:
            if len(inc) >= 2:
                prev_profit = inc.iloc[1]
                if prev_profit != 0:
                    profit_growth = (inc.iloc[0] - inc.iloc[1]) / abs(prev_profit) * 100
                    st.metric("Profit Growth YoY", f"{profit_growth:.1f}%", delta="YoY")
                    if profit_growth > 10:
                        st.success("Strong Profit Growth")
                    elif profit_growth > 0:
                        st.info("Moderate Growth")
                    else:
                        st.error("Profit Declining")
                else:
                    st.info("Previous profit was zero — unable to calculate growth rate")
            else:
                st.write("Not enough data")
    except Exception as e:
        st.warning(f"Could not compute growth metrics: {e}")

    st.subheader("Profit Margin Trend")
    try:
        revenue_series = stock.financials.loc["Total Revenue"]
        profit_series  = stock.financials.loc["Net Income"]
        margin         = (profit_series / revenue_series * 100).sort_index()
        st.line_chart(margin)
        st.metric("Latest Profit Margin", f"{margin.iloc[-1]:.2f}%")
    except Exception:
        st.warning("Profit margin data unavailable.")

    st.divider()

    st.subheader("Competitor Comparison")
    with st.spinner("Fetching competitors..."):
        competitors = get_competitors(ticker)

    if competitors:
        main_mc  = (info.get("marketCap")    or 0) / 1e9
        main_pe  =  info.get("trailingPE")   or 0
        main_rev = (info.get("totalRevenue") or 0) / 1e9

        comp_rows = []
        for t in competitors:
            try:
                ci = yf.Ticker(t).info
                c_mc  = round((ci.get("marketCap")    or 0) / 1e9, 2)
                c_pe  = round( ci.get("trailingPE")   or 0, 2)
                c_rev = round((ci.get("totalRevenue") or 0) / 1e9, 2)
                comp_rows.append({
                    "Ticker":         t,
                    "Company":        ci.get("longName", t),
                    "Market Cap (B)": f"{c_mc} {'up' if c_mc  > main_mc  else 'down'}",
                    "PE Ratio":       f"{c_pe} {'up' if c_pe  > main_pe  else 'down'}",
                    "Revenue (B)":    f"{c_rev} {'up' if c_rev > main_rev else 'down'}",
                    "52W High":       ci.get("fiftyTwoWeekHigh", "N/A"),
                })
            except Exception:
                pass
        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)
    else:
        st.info("No competitor data found.")

    st.divider()

    st.subheader("Risk Assessment")
    volatility      = hist["Close"].pct_change().std() * 100
    rsi_risk        = max(0, min(40, rsi_val - 50)) if rsi_val > 50 else 0
    sentiment_risk  = max(0, min(30, -sentiment_score * 2))
    volatility_risk = min(30, volatility * 5)
    risk_score      = min(100, volatility_risk + rsi_risk + sentiment_risk)

    ri1, ri2, ri3, ri4 = st.columns(4)
    ri1.metric("Risk Score",      f"{risk_score:.1f}/100")
    ri2.metric("Volatility Risk", f"{volatility_risk:.1f}/30")
    ri3.metric("RSI Risk",        f"{rsi_risk:.1f}/40")
    ri4.metric("Sentiment Risk",  f"{sentiment_risk:.1f}/30")

    if risk_score > 70:
        st.error("High Risk — Invest with caution")
    elif risk_score > 40:
        st.warning("Medium Risk — Monitor closely")
    else:
        st.success("Low Risk — Relatively stable")

    st.divider()

    st.subheader("Analyst Ratings")
    try:
        recs = stock.recommendations_summary
        if recs is not None and not recs.empty:
            w    = recs.iloc[0]
            buy  = 2 * w["buy"] + w["strongBuy"]
            sell = w["sell"] + 2 * w["strongSell"]
            signal_analyst = "Bullish" if buy > sell else "Bearish"
            st.info(f"Overall Signal: {signal_analyst}")

            fig_bar = go.Figure(data=[go.Bar(
                x=["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"],
                y=[w["strongBuy"], w["buy"], w["hold"], w["sell"], w["strongSell"]],
                marker_color=["#26a641", "#2ea043", "#d29922", "#f85149", "#b91c1c"]
            )])
            fig_bar.update_layout(title="Analyst Recommendations", yaxis_title="Count")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No analyst rating data available.")
    except Exception as e:
        st.warning(f"Could not load analyst ratings: {e}")

    st.divider()

    st.subheader("Insider and Institutional Activity")
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
            st.success("Insiders buying more than selling")
        elif Total_Sold > Total_Bought:
            st.warning("Insiders selling more than buying")
        else:
            st.info("Neutral insider activity")

    with ins2:
        inst_df = stock.institutional_holders
        if inst_df is not None and not inst_df.empty:
            top_holder      = inst_df.iloc[0]["Holder"]
            total_ownership = round(inst_df["pctHeld"].sum() * 100, 2)
            inst_str        = inst_df[["Holder", "Shares", "pctHeld"]].to_string()

            with st.spinner("Analysing institutional holders..."):
                inst_res = model.invoke([
                    SystemMessage(content="You are a financial analyst."),
                    HumanMessage(content=f"""
Institutional Holders data for {ticker}:
{inst_str}

Top Holder: {top_holder}
Total Institutional Ownership: {total_ownership}%

Give a 2 line analysis:
- Are institutions confident in this stock?
- Is this bullish or bearish?
""")
                ])
            st.metric("Top Holder", top_holder)
            st.metric("Total Institutional Ownership", f"{total_ownership}%")
            st.info(inst_res.content)
        else:
            st.warning("No institutional data available.")

    st.divider()

    st.subheader("Smart Trading Signals")

    hist["MACD"]        = ta.trend.MACD(hist["Close"]).macd()
    hist["MACD_Signal"] = ta.trend.MACD(hist["Close"]).macd_signal()
    hist["hband"]       = ta.volatility.BollingerBands(hist["Close"]).bollinger_hband()
    hist["lband"]       = ta.volatility.BollingerBands(hist["Close"]).bollinger_lband()
    hist["ADX"]         = ta.trend.ADXIndicator(hist["High"], hist["Low"], hist["Close"]).adx()

    macd_signal = "Bullish" if hist["MACD"].iloc[-1] > hist["MACD_Signal"].iloc[-1] else "Bearish"

    if   hist["Close"].iloc[-1] > hist["hband"].iloc[-1]:
        bb_signal = "Overbought"
    elif hist["Close"].iloc[-1] < hist["lband"].iloc[-1]:
        bb_signal = "Oversold"
    else:
        bb_signal = "Neutral"

    adx_val = round(hist["ADX"].iloc[-1], 2)

    sig1, sig2, sig3 = st.columns(3)
    sig1.metric("MACD Signal",     macd_signal)
    sig2.metric("Bollinger Signal", bb_signal)
    sig3.metric("ADX Value",        adx_val)

    st.divider()

    st.subheader("Sector and Macro Analysis")

    stock2 = yf.Ticker("^GSPC")
    info2  = stock2.info
    hist2  = stock2.history(period="1y")

    fig_compare = go.Figure()
    norm_stock  = hist["Close"]  * 100 / hist["Close"].iloc[0]
    norm_sp500  = hist2["Close"] * 100 / hist2["Close"].iloc[0]
    fig_compare.add_trace(go.Scatter(x=norm_stock.index, y=norm_stock.values, name=info.get("longName", ticker), mode="lines"))
    fig_compare.add_trace(go.Scatter(x=norm_sp500.index, y=norm_sp500.values, name="S&P 500", mode="lines", line=dict(dash="dash")))
    fig_compare.update_layout(title=f"{ticker} vs S&P 500 (Normalised)", xaxis_title="Date", yaxis_title="Normalised Price", hovermode="x unified")
    st.plotly_chart(fig_compare, use_container_width=True)

    mac1, mac2 = st.columns(2)

    with mac1:
        try:
            vix_val = yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1]
            if vix_val < 20:
                st.success(f"VIX: {vix_val:.1f} — Market Calm")
            elif vix_val < 30:
                st.warning(f"VIX: {vix_val:.1f} — Market Uncertain")
            else:
                st.error(f"VIX: {vix_val:.1f} — High Fear")
        except Exception:
            st.warning("VIX data unavailable.")

    with mac2:
        try:
            rate_val = yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1]
            st.metric("10Y Treasury Yield", f"{rate_val:.2f}%")
            if rate_val > 4.5:
                st.error("High Rates — Bearish for stocks")
            elif rate_val >= 3:
                st.warning("Moderate Rates — Neutral impact")
            else:
                st.success("Low Rates — Bullish for stocks")
        except Exception:
            st.warning("Treasury yield data unavailable.")

    st.divider()

    st.subheader("Options Market Intelligence")
    pcr = None
    try:
        if stock.options:
            opt         = stock.option_chain(stock.options[0])
            call_volume = opt.calls["volume"].sum()
            put_volume  = opt.puts["volume"].sum()
            if call_volume > 0:
                pcr = round(put_volume / call_volume, 2)
    except Exception:
        pass

    if pcr is not None:
        st.metric("Put/Call Ratio", pcr)
        if pcr < 0.7:
            st.success("Options Sentiment: Bullish")
        elif pcr > 1.3:
            st.error("Options Sentiment: Bearish")
        else:
            st.info("Options Sentiment: Neutral")
    else:
        st.warning("Options data not available for this ticker.")

    st.divider()

    st.subheader("Valuation Engine")
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
        v1.metric("Stock PE",        f"{stock_pe:.2f}")
        v2.metric("Industry Avg PE", f"{industry_pe:.2f}")
        v3.metric("Valuation Signal", val_signal)
    else:
        st.warning("Not enough PE data to evaluate valuation.")

    st.divider()

    st.subheader("Earnings Stability Score")
    stability_score  = 0
    stability_signal = "N/A"
    try:
        income_series   = stock.financials.loc["Net Income"]
        std             = income_series.std()
        mean            = income_series.mean()
        cv              = std / mean if mean != 0 else float("inf")
        stability_score = round(max(0, 100 - cv * 100), 1)

        if   stability_score > 80:
            stability_signal = "Very Stable"
        elif stability_score > 60:
            stability_signal = "Stable"
        elif stability_score > 40:
            stability_signal = "Moderate"
        else:
            stability_signal = "Unstable"
    except Exception:
        pass

    es1, es2 = st.columns(2)
    es1.metric("Stability Score",  f"{stability_score}/100")
    es2.metric("Stability Signal",  stability_signal)

    st.divider()

    st.subheader("Final AI Recommendation")
    with st.spinner("Generating recommendation..."):
        final_res = model.invoke([
            SystemMessage(content="You are a professional financial analyst. Give a clear investment recommendation."),
            HumanMessage(content=f"""
Stock: {ticker}
Current Price:    {info.get('currentPrice')}
52W High:         {info.get('fiftyTwoWeekHigh')}
RSI:              {rsi_val}
Sentiment Score:  {round(sentiment_score, 2)}
Risk Score:       {risk_score:.1f}/100
MACD Signal:      {macd_signal}
Bollinger Signal: {bb_signal}
ADX Value:        {adx_val}
Stability Signal: {stability_signal}
Insider Buying:   {int(Total_Bought):,}
Insider Selling:  {int(Total_Sold):,}
Financial Summary: {ana if ana else 'N/A'}

Give a BUY / HOLD / SELL recommendation with 2-3 lines of reasoning.
End with: This is not financial advice.
""")
        ])
    st.success(final_res.content)