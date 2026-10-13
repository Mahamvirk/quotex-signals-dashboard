import os
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forex Signal Lab", page_icon="📈", layout="wide")
st.title("📈 Forex Signal Lab — Quotex-style Signals")
st.caption("Market-analysis dashboard only. It does not connect to or place trades on Quotex.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Twelve Data API key",
        value=os.getenv("TWELVE_DATA_API_KEY", ""),
        type="password",
        help="Get an API key from Twelve Data. Do not share it publicly."
    )
    symbol = st.selectbox("Forex pair", ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "USD/CAD"])
    interval = st.selectbox("Candle timeframe", ["1min", "5min", "15min", "30min", "1h"], index=1)
    outputsize = st.slider("Historical candles", 100, 5000, 500, step=100)
    min_score = st.slider("Minimum signal score", 2, 4, 3)
    payout = st.slider("Illustrative payout on a winning trade", 0.50, 0.95, 0.80, 0.05)
    refresh = st.button("Fetch latest data", type="primary")

st.info(
    "Live data depends on your data provider, plan, symbol coverage, and API limits. "
    "Provider candles may differ from Quotex's displayed prices. Test in demo mode first."
)

@st.cache_data(ttl=30, show_spinner=False)
def fetch_candles(pair: str, tf: str, count: int, key: str):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": pair,
        "interval": tf,
        "outputsize": count,
        "apikey": key,
        "format": "JSON",
        "order": "ASC",
    }
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if "values" not in payload:
        raise ValueError(payload.get("message", str(payload)))
    df = pd.DataFrame(payload["values"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    return df.sort_values("datetime").reset_index(drop=True)

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["ema9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()

    delta = x["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    x["rsi"] = 100 - (100 / (1 + rs))
    x["rsi"] = x["rsi"].fillna(50)

    ema12 = x["close"].ewm(span=12, adjust=False).mean()
    ema26 = x["close"].ewm(span=26, adjust=False).mean()
    x["macd"] = ema12 - ema26
    x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()

    x["bull"] = (
        (x["ema9"] > x["ema21"]).astype(int)
        + (x["close"] > x["ema50"]).astype(int)
        + (x["macd"] > x["macd_signal"]).astype(int)
        + ((x["rsi"] >= 52) & (x["rsi"] <= 70)).astype(int)
    )
    x["bear"] = (
        (x["ema9"] < x["ema21"]).astype(int)
        + (x["close"] < x["ema50"]).astype(int)
        + (x["macd"] < x["macd_signal"]).astype(int)
        + ((x["rsi"] <= 48) & (x["rsi"] >= 30)).astype(int)
    )
    x["signal"] = "NO TRADE"
    x.loc[(x["bull"] >= min_score) & (x["bull"] > x["bear"]), "signal"] = "CALL"
    x.loc[(x["bear"] >= min_score) & (x["bear"] > x["bull"]), "signal"] = "PUT"
    x["score"] = x[["bull", "bear"]].max(axis=1)
    return x

def backtest(x: pd.DataFrame, illustrative_payout: float):
    # Signal is taken from candle i; outcome uses the next candle's direction.
    # This is a rough directional proxy, not an exact Quotex settlement simulator.
    d = x.copy()
    d["next_open"] = d["open"].shift(-1)
    d["next_close"] = d["close"].shift(-1)
    d = d[d["signal"].isin(["CALL", "PUT"]) & d["next_open"].notna()].copy()
    if d.empty:
        return d, {"trades": 0, "wins": 0, "win_rate": None, "net_units": None}
    d["win"] = (
        ((d["signal"] == "CALL") & (d["next_close"] > d["next_open"]))
        | ((d["signal"] == "PUT") & (d["next_close"] < d["next_open"]))
    )
    d["net_units"] = d["win"].map(lambda won: illustrative_payout if won else -1.0)
    # Flat candles are treated as losses in this simplified proxy.
    return d, {
        "trades": int(len(d)),
        "wins": int(d["win"].sum()),
        "win_rate": float(d["win"].mean() * 100),
        "net_units": float(d["net_units"].sum()),
    }

if not api_key:
    st.warning("Sidebar mein Twelve Data API key enter karein, phir 'Fetch latest data' dabayein.")
    st.markdown("### Setup")
    st.markdown(
        "1. Twelve Data par account bana kar API key lein.\n"
        "2. Neeche diye gaye `requirements.txt` se packages install karein.\n"
        "3. `streamlit run app.py` command chalayein.\n"
        "4. API key sidebar mein enter karein aur data fetch karein."
    )
    st.stop()

try:
    with st.spinner("Candles fetch ho rahi hain..."):
        raw = fetch_candles(symbol, interval, outputsize, api_key)
    data = add_indicators(raw)
except Exception as exc:
    st.error(f"Data load nahi ho saka: {exc}")
    st.caption("API key, provider limits, symbol format, internet connection aur plan check karein.")
    st.stop()

if len(data) < 60:
    st.warning("Indicators ke liye data kam hai. Zyada historical candles load karein.")

latest = data.iloc[-1]
# Provider may include a candle still in progress; treat signal as provisional.
col1, col2, col3, col4 = st.columns(4)
col1.metric("Pair", symbol)
col2.metric("Timeframe", interval)
col3.metric("Latest close", f"{latest['close']:.5f}")
col4.metric("Latest candle time", str(latest["datetime"]))

sig_col, score_col, rsi_col = st.columns(3)
sig_col.subheader("Latest signal")
signal = latest["signal"]
if signal == "CALL":
    sig_col.success("CALL")
elif signal == "PUT":
    sig_col.error("PUT")
else:
    sig_col.warning("NO TRADE")
score_col.metric("Indicator agreement", f"{int(latest['score'])}/4")
rsi_col.metric("RSI (14)", f"{latest['rsi']:.1f}")
st.caption("Signal current provider candle par based hai aur candle close hone se pehle change ho sakta hai. Entry se pehle candle close confirm karein.")

st.subheader("Price and trend")
chart_data = data.set_index("datetime")[["close", "ema9", "ema21", "ema50"]].tail(250)
st.line_chart(chart_data)

st.subheader("Backtest (historical directional proxy)")
bt, stats = backtest(data, payout)
if stats["trades"] == 0:
    st.warning("Is data aur filter par backtest trades nahi mile.")
else:
    a, b, c, d = st.columns(4)
    a.metric("Signals tested", stats["trades"])
    b.metric("Wins (proxy)", stats["wins"])
    c.metric("Directional win rate", f"{stats['win_rate']:.1f}%")
    d.metric("Illustrative net units", f"{stats['net_units']:.2f}")
    st.caption(
        f"Net units assume winning stake returns +{payout:.0%} of stake and losing stake is -1 unit. "
        "Real payout, tie rules, entry price, spread, latency and expiry settlement can differ."
    )
    st.dataframe(
        bt[["datetime", "close", "signal", "score", "rsi", "next_open", "next_close", "win", "net_units"]]
        .tail(100).sort_values("datetime", ascending=False),
        use_container_width=True,
        hide_index=True
    )

st.subheader("Latest candles and indicators")
st.dataframe(
    data[["datetime", "open", "high", "low", "close", "ema9", "ema21", "ema50", "rsi", "macd", "macd_signal", "signal", "score"]]
    .tail(30).sort_values("datetime", ascending=False),
    use_container_width=True,
    hide_index=True
)

st.warning(
    "Important: yeh starter strategy hai, verified profitable system nahi. "
    "Historical win rate future accuracy ki guarantee nahi. Out-of-sample testing, demo forward-test, "
    "data quality checks aur transaction/expiry rules ko verify kiye baghair real money risk na karein."
)
