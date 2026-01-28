import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="RADAR IA PRO", layout="wide")
st.title("🚀 RADAR IA PRO — TEMPO REAL")

symbol = st.selectbox("Escolha o ativo", ["btcusdt", "ethusdt"])

# ---------------- WEBSOCKET BINANCE ----------------
async def get_realtime_data():
    url = f"wss://stream.binance.com:9443/ws/{symbol}@trade"
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            price = float(data['p'])
            return price

price = asyncio.run(get_realtime_data())

st.metric("Preço ao vivo", f"${price}")

# ---------------- HISTÓRICO SIMULADO ----------------
if "prices" not in st.session_state:
    st.session_state.prices = []

st.session_state.prices.append(price)

if len(st.session_state.prices) > 200:
    df = pd.DataFrame(st.session_state.prices, columns=["Close"])
    df["ema"] = df["Close"].ewm(span=20).mean()
    df["retorno"] = df["Close"].pct_change()
    df["alvo"] = np.where(df["retorno"].shift(-1) > 0, 1, 0)

    df.dropna(inplace=True)

    X = df[["ema", "retorno"]]
    y = df["alvo"]

    model = RandomForestClassifier()
    model.fit(X, y)

    prob = model.predict_proba(X.iloc[-1:])[0][1]

    st.subheader("📡 SINAL DA IA")

    if prob > 0.6:
        st.success(f"📈 ALTA provável ({prob*100:.1f}%)")
    else:
        st.error(f"📉 QUEDA provável ({(1-prob)*100:.1f}%)")

    st.line_chart(df[["Close", "ema"]].tail(100))
else:
    st.warning("Coletando dados ao vivo...")
