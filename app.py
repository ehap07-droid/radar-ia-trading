import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="RADAR IA PRO", layout="wide")
st.title("🚀 RADAR IA PRO — TEMPO REAL (VERSÃO ESTÁVEL)")

symbol = st.selectbox("Escolha o ativo", ["bitcoin", "ethereum"])

# ---------- PEGAR PREÇO (COINGECKO) ----------
@st.cache_data(ttl=5)
def get_price(symbol):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        return float(data[symbol]["usd"])
    except:
        return None

price = get_price(symbol)

if price is None:
    st.error("⚠️ Falha ao obter preço. Aguarde...")
    time.sleep(3)
    st.rerun()

st.metric("Preço atual", f"${price}")

# ---------- HISTÓRICO ----------
if "prices" not in st.session_state:
    st.session_state.prices = []

st.session_state.prices.append(price)

if len(st.session_state.prices) > 120:

    df = pd.DataFrame(st.session_state.prices, columns=["Close"])
    df["ema"] = df["Close"].ewm(span=20).mean()
    df["retorno"] = df["Close"].pct_change()
    df["alvo"] = np.where(df["retorno"].shift(-1) > 0, 1, 0)

    df.dropna(inplace=True)

    if len(df) > 50:
        X = df[["ema", "retorno"]]
        y = df["alvo"]

        model = RandomForestClassifier(n_estimators=80)
        model.fit(X, y)

        prob = model.predict_proba(X.iloc[-1:])[0][1]

        st.subheader("📡 SINAL DA IA")

        if prob > 0.6:
            st.success(f"📈 ALTA provável ({prob*100:.1f}%)")
        else:
            st.error(f"📉 QUEDA provável ({(1-prob)*100:.1f}%)")

        st.line_chart(df[["Close", "ema"]].tail(100))

else:
    st.warning("Coletando dados para IA... Aguarde 1–2 minutos.")

# ---------- AUTO REFRESH ----------
time.sleep(3)
st.rerun()
