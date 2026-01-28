import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Radar IA M1", layout="centered")

st.title("📊 RADAR DE OPERAÇÃO M1 COM IA")
st.caption("Modelo educacional — não é recomendação financeira.")

st.info("Carregando dados do mercado... Aguarde alguns segundos.")

# =========================
# COLETA DE DADOS
# =========================
ticker = "EURUSD=X"  # pode trocar depois
df = yf.download(ticker, interval="1m", period="5d")

if df.empty:
    st.error("Não foi possível carregar dados do mercado.")
    st.stop()

# =========================
# INDICADORES
# =========================
df['ema'] = df['Close'].ewm(span=10).mean()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

exp1 = df['Close'].ewm(span=12).mean()
exp2 = df['Close'].ewm(span=26).mean()
df['macd'] = exp1 - exp2

# Alvo (se preço subiu na próxima vela)
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# =========================
# LIMPEZA DE DADOS (CORREÇÃO DO ERRO)
# =========================
df = df.replace([np.inf, -np.inf], np.nan)
df.dropna(inplace=True)

if len(df) < 50:
    st.warning("Coletando dados suficientes do mercado... Aguarde.")
    st.stop()

# =========================
# IA
# =========================
features = ['rsi', 'ema', 'macd']
X = df[features]
y = df['target']

if X.isnull().values.any():
    st.warning("Ainda há dados inválidos. Aguardando mercado gerar mais histórico.")
    st.stop()

model = RandomForestClassifier(n_estimators=80)
model.fit(X, y)

last = X.iloc[-1:]
prediction = model.predict(last)[0]
prob = model.predict_proba(last)[0]

# =========================
# EXIBIÇÃO DO SINAL
# =========================
st.subheader("📡 SINAL DA IA")

if prediction == 1:
    st.success(f"✅ PROBABILIDADE DE ALTA — COMPRA ({prob[1]*100:.1f}%)")
else:
    st.error(f"🔻 PROBABILIDADE DE QUEDA — VENDA ({prob[0]*100:.1f}%)")

# =========================
# GRÁFICO
# =========================
st.subheader("📈 Últimos dados do mercado")
st.line_chart(df[['Close', 'ema']].tail(100))
