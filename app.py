import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
st.title("📊 RADAR DE OPERAÇÃO M1 COM IA")

# ==============================
# ESCOLHA DO ATIVO
# ==============================
ativo = st.selectbox(
    "Selecione o ativo:",
    ["EURUSD=X", "GBPUSD=X", "BTC-USD", "ETH-USD", "AAPL", "TSLA"]
)

st.write(f"📡 Analisando agora: **{ativo}** | Timeframe: **1 minuto (M1)**")

st.write("Carregando dados do mercado...")

# ==============================
# BAIXAR DADOS
# ==============================
df = yf.download(ativo, period="2d", interval="1m")

if df.empty:
    st.error("Não foi possível carregar dados.")
    st.stop()

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
# IA
# ==============================
X = df[['ema', 'vol']]
y = np.where(df['ret'].shift(-1) > 0, 1, 0)

X = X[:-1]
y = y[:-1]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

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
# GRÁFICO
# ==============================
st.subheader("📈 Últimos dados do mercado")
st.line_chart(df[['Close', 'ema']].tail(120))

st.caption("Modelo educacional — não é recomendação financeira.")
