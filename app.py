import streamlit as st
import pandas as pd
import ta
import requests
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Radar IA M1", layout="centered")
st.title("📊 RADAR DE OPERAÇÃO M1 COM IA")

def get_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=300"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        'time','open','high','low','close','volume',
        'c1','c2','c3','c4','c5','c6'
    ])
    df = df[['open','high','low','close','volume']].astype(float)
    return df

df = get_data()

# Indicadores
df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
df['ema'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
df['macd'] = ta.trend.MACD(df['close']).macd()

# Alvo
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Remove qualquer linha com valor vazio
df = df.dropna()

X = df[['rsi','ema','macd']]
y = df['target']

# Treinar modelo apenas se houver dados suficientes
if len(df) > 50:
    model = RandomForestClassifier()
    model.fit(X, y)

    last = X.iloc[-1:]
    prediction = model.predict(last)[0]

    st.subheader("📡 SINAL DA IA")

    if prediction == 1:
        st.success("✅ PROBABILIDADE DE ALTA — POSSÍVEL COMPRA")
    else:
        st.error("🔻 PROBABILIDADE DE QUEDA — POSSÍVEL VENDA")
else:
    st.warning("Carregando dados do mercado... Aguarde alguns segundos.")

st.caption("Modelo educacional — não é recomendação financeira.")
