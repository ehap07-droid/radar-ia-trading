import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="RADAR IA PRO", layout="wide")
st.title("🚀 RADAR IA PRO — QUASE TEMPO REAL")

symbol = st.selectbox("Escolha o ativo", ["BTCUSDT", "ETHUSDT"])

# ---------- PEGAR PREÇO COM PROTEÇÃO ----------
@st.cache_data(ttl=2)
def get_price(symbol):
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        if "price" in data:
            return float(data["price"])
        else:
            return None
    except:
        return None

price = get_price(symbol)

if price is None:
    st.error("⚠️ Falha ao obter preço da Binance. Aguarde 2 segundos...")
    time.sleep(2)
    st.rerun()

st.metric("Preço atual", f"${price}")

# ---------- HISTÓRICO ----------
if "prices" not in st.session_state:
    st.session_state.prices = []

st.session_state.prices.append(price)

# Só treina se tiver dados suficientes
if len(st.session_state.prices) > 120:

    df = pd.DataFrame(st.session_state.prices, columns=["Close"])
    df["ema"] = df["Close"].ewm(span=20).mean()
    df["retorno"] = df["Close"].pct_change()
    df["alvo"] = np.where(df["retorno"].shift(-1) > 0, 1, 0)

    df.dropna(inplace=True)

    # Segurança extra
    if len(df) < 50:
        st.warning("Coletando dados para IA...")
    else:
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
    st.warning("Coletando dados para treinar a IA... Aguarde 1–2 minutos.")

# ---------- AUTO REFRESH ----------
time.sleep(2)
st.rerun()
