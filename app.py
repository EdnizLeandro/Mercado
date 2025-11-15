import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import plotly.graph_objects as go
import numpy as np
import os

st.set_page_config(page_title="Plataforma Jovem Futuro", layout="wide")

# ======================================================
# 1) Carregar arquivos principais com tratamento de erro
# ======================================================
PARQUET_FILE = "dados.parquet"
CBO_FILE = "cbo.xlsx"

def load_data():
    # Verificar parquet
    if not os.path.exists(PARQUET_FILE):
        st.error(f"❌ Arquivo não encontrado: **{PARQUET_FILE}**")
        st.stop()

    # Verificar cbo
    if not os.path.exists(CBO_FILE):
        st.error(f"❌ Arquivo não encontrado: **{CBO_FILE}**")
        st.stop()

    df = pd.read_parquet(PARQUET_FILE)

    df_cbo = pd.read_excel(CBO_FILE)
    df_cbo.columns = ["codigo", "descricao"]

    return df, df_cbo


df, df_cbo = load_data()

st.success("✅ Dados carregados com sucesso!")
st.write("### Colunas detectadas no dataset:")
st.json(list(df.columns))


# ======================================================
# Normalização das colunas esperadas
# ======================================================

COLUMN_MAP = {
    "cbo": "cbo2002ocupacao",
    "date": "competenciadec",
    "salary": "salario",
    "saldo": "saldomovimentacao"
}

for alias, realname in COLUMN_MAP.items():
    if realname not in df.columns:
        st.error(f"❌ Coluna obrigatória não encontrada: **{realname}**")
        st.stop()

df["competenciadec"] = pd.to_datetime(df["competenciadec"], errors="coerce")


# ======================================================
# Busca por profissão (CBO)
# ======================================================
st.header("🔎 Buscar profissão (por nome ou código CBO)")

query = st.text_input("Digite nome ou código da profissão:")

if query:
    # Filtro seguro
    mask = (
        df_cbo["descricao"].str.contains(query, case=False, na=False)
        | df_cbo["codigo"].astype(str).str.contains(query, na=False)
    )

    resultados = df_cbo[mask]

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        st.write("### Resultados encontrados:")
        st.dataframe(resultados)

        # Selecionar profissão
        selected_code = st.selectbox(
            "Selecione um código CBO para análise:",
            resultados["codigo"].astype(str).unique()
        )

        if selected_code:
            st.info(f"📌 Mostrando análise para CBO **{selected_code}**")

            df_job = df[df["cbo2002ocupacao"].astype(str) == selected_code]

            if df_job.empty:
                st.warning("Não existem registros para este CBO nos dados.")
            else:
                st.write("### 📊 Distribuição salarial")
                fig = px.box(df_job, x="cbo2002ocupacao", y="salario")
                st.plotly_chart(fig, use_container_width=True)

                st.write("### 📈 Evolução do saldo de contratações")
                fig2 = px.line(df_job, x="competenciadec", y="saldomovimentacao")
                st.plotly_chart(fig2, use_container_width=True)

                # ======================================
                # 3) PREVISÃO (ML) — Prophet
                # ======================================
                st.subheader("🤖 Previsão de demanda futura (Prophet)")

                df_prophet = df_job[["competenciadec", "saldomovimentacao"]].rename(
                    columns={"competenciadec": "ds", "saldomovimentacao": "y"}
                ).dropna()

                if len(df_prophet) >= 12:
                    model = Prophet()
                    model.fit(df_prophet)

                    future = model.make_future_dataframe(periods=12, freq="M")
                    forecast = model.predict(future)

                    fig3 = model.plot(forecast)
                    st.pyplot(fig3)

                    st.write("### 🔮 Previsão numérica dos próximos 12 meses")
                    st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12))

                else:
                    st.warning("Dados insuficientes para previsão.")
