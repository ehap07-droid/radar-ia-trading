import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("📊 RADAR DE OPERAÇÃO M1 COM IA")

st.write("Carregando dados do mercado... Aguarde alguns segundos.")

# ==============================
# BAIXAR DADOS
# ==============================
df = yf.download("EURUSD=X", period="2d", interval="1m")

if df.empty:
    st.error("Não foi possível carregar dados.")
    st.stop()

# Corrige colunas multinível (ERRO DO GRÁFICO)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ==============================
# INDICADORES
# ==============================
df['ema'] = df['Close'].ewm(span=14).mean()
df['ret'] = df['Close'].pct_change()
df['vol'] = df['ret'].rolling(10).std()

df.dropna(inplace=True)

# ==============================
# TREINAMENTO IA
# ==============================
X = df[['ema', 'vol']]
y = np.where(df['ret'].shift(-1) > 0, 1, 0)

# Garante alinhamento
X = X[:-1]
y = y[:-1]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# ==============================
# PREVISÃO
# ==============================
ultima = X.iloc[-1:].values
prob = model.predict_proba(ultima)[0]
direcao = np.argmax(prob)
conf = prob[direcao] * 100

st.subheader("📡 SINAL DA IA")

if direcao == 1:
    st.success(f"🔺 PROBABILIDADE DE ALTA — COMPRA ({conf:.1f}%)")
else:
    st.error(f"🔻 PROBABILIDADE DE QUEDA — VENDA ({conf:.1f}%)")

# ==============================
# GRÁFICO (CORRIGIDO)
# ==============================
st.subheader("📈 Últimos dados do mercado")

grafico = df[['Close', 'ema']].tail(120)
st.line_chart(grafico)

st.caption("Modelo educacional — não é recomendação financeira.")
