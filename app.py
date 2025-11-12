import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.linear_model import LinearRegression

# ==============================================================
# CONFIGURAÇÕES DO APP
# ==============================================================

st.set_page_config(
    page_title="Jobin – Analytics & Mercado",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Jobin – Analytics & Mercado")
st.markdown("""
**Iniciativa que transforma a vida de jovens em Recife por meio de dados e inteligência de mercado.**  
Conectamos talentos a oportunidades reais de trabalho, educação e renda, promovendo inclusão e impacto social.
""")

# ==============================================================
# FUNÇÃO DE CARREGAMENTO DE DADOS
# ==============================================================

@st.cache_data
def carregar_dados():
    try:
        base_path = os.path.dirname(__file__)
        dados_path = os.path.join(base_path, "dados.parquet")

        if not os.path.exists(dados_path):
            raise FileNotFoundError("Arquivo 'dados.parquet' não encontrado no diretório do app.")

        df = pd.read_parquet(dados_path)

        if df.empty:
            raise ValueError("O arquivo 'dados.parquet' está vazio.")

        st.success("✅ Dados carregados com sucesso!")
        return df
    except Exception as e:
        st.error(f"❌ Erro ao carregar os dados: {e}")
        return None


# ==============================================================
# CARREGAMENTO DOS DADOS
# ==============================================================

df = carregar_dados()

if df is None:
    st.stop()

# Mostra preview
st.subheader("📋 Visualização Inicial dos Dados")
st.dataframe(df.head())

# ==============================================================
# IDENTIFICAÇÃO AUTOMÁTICA DE COLUNAS
# ==============================================================

coluna_data = next((c for c in df.columns if "competencia" in c.lower()), None)
coluna_salario = next((c for c in df.columns if "salario" in c.lower()), None)
coluna_saldo = next((c for c in df.columns if "saldo" in c.lower()), None)

if not any([coluna_data, coluna_salario, coluna_saldo]):
    st.warning("⚠️ Nenhuma coluna padrão (competência, salário, saldo) foi encontrada.")
else:
    st.markdown("### 🔍 Colunas identificadas automaticamente:")
    st.write(f"- Data: **{coluna_data or 'não encontrada'}**")
    st.write(f"- Salário: **{coluna_salario or 'não encontrada'}**")
    st.write(f"- Saldo: **{coluna_saldo or 'não encontrada'}**")

# ==============================================================
# GRÁFICOS BÁSICOS
# ==============================================================

if coluna_salario:
    st.markdown("### 💰 Distribuição Salarial")
    fig_sal = px.histogram(df, x=coluna_salario, nbins=40, title="Distribuição dos Salários")
    st.plotly_chart(fig_sal, use_container_width=True)

if coluna_saldo:
    st.markdown("### 📊 Distribuição do Saldo de Movimentação")
    fig_saldo = px.histogram(df, x=coluna_saldo, nbins=40, title="Distribuição do Saldo")
    st.plotly_chart(fig_saldo, use_container_width=True)

# ==============================================================
# PREVISÃO SALARIAL (OPCIONAL)
# ==============================================================

if coluna_data and coluna_salario:
    st.markdown("### 📈 Previsão Simples de Salário")

    df[coluna_data] = pd.to_datetime(df[coluna_data], errors="coerce")
    df = df.dropna(subset=[coluna_data, coluna_salario])
    df["tempo_meses"] = ((df[coluna_data].dt.year - 2020) * 12 + df[coluna_data].dt.month)

    df_mensal = df.groupby("tempo_meses")[coluna_salario].mean().reset_index()

    if len(df_mensal) > 2:
        X = df_mensal[["tempo_meses"]]
        y = df_mensal[coluna_salario]
        model = LinearRegression().fit(X, y)

        ult_mes = df_mensal["tempo_meses"].max()
        anos_futuros = [5, 10, 15]
        previsoes = []

        for anos in anos_futuros:
            mes_futuro = ult_mes + anos * 12
            pred = model.predict(np.array([[mes_futuro]]))[0]
            previsoes.append((anos, pred))

        df_prev = pd.DataFrame(previsoes, columns=["Anos", "Salário Previsto"])
        st.dataframe(df_prev.style.format({"Salário Previsto": "R$ {:,.2f}"}))

        fig_prev = px.line(df_prev, x="Anos", y="Salário Previsto", markers=True, title="Projeção Salarial Futura")
        st.plotly_chart(fig_prev, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar previsão salarial.")
