import streamlit as st
import pandas as pd
import ta
import requests
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Radar IA M1", layout="centered")
st.title("📊 RADAR DE OPERAÇÃO M1 COM IA")

def get_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=200"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=[
        'time','open','high','low','close','volume',
        'c1','c2','c3','c4','c5','c6'
    ])
    df = df[['open','high','low','close','volume']].astype(float)
    return df

df = get_data()

df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
df['ema'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
df['macd'] = ta.trend.MACD(df['close']).macd()
df = df.dropna()

df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

X = df[['rsi','ema','macd']]
y = df['target']

model = RandomForestClassifier()
model.fit(X, y)

last = X.iloc[-1:]
prediction = model.predict(last)[0]

st.subheader("📡 SINAL DA IA")

if prediction == 1:
    st.success("✅ PROBABILIDADE DE ALTA — POSSÍVEL COMPRA")
else:
    st.error("🔻 PROBABILIDADE DE QUEDA — POSSÍVEL VENDA")

st.caption("Modelo educacional — não é recomendação financeira.")
