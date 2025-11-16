# app.py
import pandas as pd
import streamlit as st
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import numpy as np

# ---------------------------------------
# Funções auxiliares
# ---------------------------------------
@st.cache_data
def carregar_historico():
    df = pd.read_parquet("dados.parquet")
    df.columns = [unidecode(c.lower()).replace(" ", "") for c in df.columns]
    df["cbo2002ocupacao"] = df["cbo2002ocupacao"].astype(str).str.strip()
    df["salario"] = pd.to_numeric(df["salario"], errors='coerce')
    return df

@st.cache_data
def carregar_cbo():
    df_cbo = pd.read_excel("cbo.xlsx")
    df_cbo.columns = [unidecode(c.lower()).replace(" ", "") for c in df_cbo.columns]
    df_cbo["descricao"] = df_cbo["descricao"].astype(str)
    df_cbo["codigo"] = df_cbo["codigo"].astype(str)
    return df_cbo

def busca_profissao(df_cbo, termo):
    termo_norm = unidecode(termo.lower())
    df_cbo["descricao_norm"] = df_cbo["descricao"].apply(lambda x: unidecode(str(x).lower()))
    df_filtrada = df_cbo[df_cbo["descricao_norm"].str.contains(termo_norm)]
    return df_filtrada

def treinar_modelo_salario(df_prof):
    X = df_prof[["idade", "horascontratuais"]].fillna(0)
    y = df_prof["salario"].fillna(0)
    if len(X) < 10:
        return None  # dados insuficientes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

def prever_salario(model, anos_futuro, idade_atual, horas):
    X_new = pd.DataFrame({"idade": [idade_atual + anos_futuro], "horascontratuais": [horas]})
    return model.predict(X_new)[0]

def treinar_modelo_vagas(df_prof):
    df_prof["saldomovimentacao"] = pd.to_numeric(df_prof["saldomovimentacao"], errors='coerce').fillna(0)
    X = df_prof[["idade", "horascontratuais"]].fillna(0)
    y = df_prof["saldomovimentacao"]
    if len(X) < 10:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model

def prever_vagas(model, anos_futuro, idade_atual, horas):
    X_new = pd.DataFrame({"idade": [idade_atual + anos_futuro], "horascontratuais": [horas]})
    return model.predict(X_new)[0]

def rank_profissoes(df_hist):
    resumo = df_hist.groupby("cbo2002ocupacao")["salario"].mean().sort_values(ascending=False)
    return resumo.head(10)

# ---------------------------------------
# Aplicativo Streamlit
# ---------------------------------------
st.set_page_config(page_title="Mercado de Trabalho", layout="wide")

st.title("📊 Previsão Salarial e Tendência de Mercado")

df_hist = carregar_historico()
df_cbo = carregar_cbo()

# Input do usuário
termo = st.text_input("Digite o nome ou código da profissão:")

if termo:
    df_opcoes = busca_profissao(df_cbo, termo)
    if df_opcoes.empty:
        st.warning("Profissão não encontrada. Digite outro nome ou código.")
    else:
        primeira_opcao = df_opcoes.iloc[0]
        cbo_selecionado = st.selectbox("Selecione o CBO:", df_opcoes["descricao"], index=0)
        codigo_cbo = df_opcoes[df_opcoes["descricao"]==cbo_selecionado]["codigo"].values[0]

        st.subheader(f"Profissão: {cbo_selecionado}")
        df_prof = df_hist[df_hist["cbo2002ocupacao"] == str(codigo_cbo)]

        if not df_prof.empty:
            salario_medio = df_prof["salario"].mean()
            st.write(f"Salário médio atual: R$ {salario_medio:,.2f}")

            # Treinar modelo XGBoost para salário
            modelo_salario = treinar_modelo_salario(df_prof)
            if modelo_salario:
                st.write("Previsão salarial futura do melhor modelo:")
                idade_atual = int(df_prof["idade"].median())
                horas = int(df_prof["horascontratuais"].median())
                for anos in [5, 10, 15, 20]:
                    previsao = prever_salario(modelo_salario, anos, idade_atual, horas)
                    st.write(f"  {anos} anos → R$ {previsao:,.2f}")
            else:
                st.info("Sem dados suficientes para previsão salarial.")

            # Treinar modelo XGBoost para tendência de vagas
            modelo_vagas = treinar_modelo_vagas(df_prof)
            if modelo_vagas:
                st.write("\nTendência de vagas futuras (XGBoost):")
                for anos in [5, 10, 15, 20]:
                    vagas = prever_vagas(modelo_vagas, anos, idade_atual, horas)
                    st.write(f"  {anos} anos: {vagas:.0f}")
            else:
                st.info("Sem dados suficientes para previsão de vagas.")

            # Ranking profissões mais promissoras
            st.write("\n🔥 Ranking de profissões mais promissoras (maior salário médio):")
            rank = rank_profissoes(df_hist)
            st.table(rank.reset_index().rename(columns={"cbo2002ocupacao": "Código CBO", "salario": "Salário Médio"}))
        else:
            st.info("Sem dados históricos suficientes para esta profissão.")
