import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

# ---------------------------------------------------------
#                CARREGAR E PREPARAR DADOS
# ---------------------------------------------------------

@st.cache_data
def carregar_dados():
    df = pd.read_parquet("dados.parquet")

    # Renomear colunas CAGED
    df = df.rename(columns={
        "competênciamov": "competenciamov",
        "saldomovimentação": "saldomovimentacao",
        "cbo2002ocupação": "cbo2002ocupacao",
        "salário": "salario"
    })

    df["cbo2002ocupacao"] = df["cbo2002ocupacao"].astype(str).str.strip()

    # Conversão de salário
    df["salario"] = (
        df["salario"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["salario"] = pd.to_numeric(df["salario"], errors="coerce")

    mediana_sal = df["salario"].median()
    df["salario"] = df["salario"].fillna(mediana_sal)

    return df


@st.cache_data
def carregar_cbo():
    df = pd.read_excel("cbo.xlsx")
    df.columns = ["cbo_codigo", "cbo_descricao"]
    df["cbo_codigo"] = df["cbo_codigo"].astype(str).str.strip()
    df["cbo_descricao"] = df["cbo_descricao"].astype(str).str.strip()
    return df


df = carregar_dados()
df_cbo = carregar_cbo()


# ---------------------------------------------------------
#                  FORMATAR MOEDA
# ---------------------------------------------------------

def formatar_moeda(v):
    try:
        return f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return v


# ---------------------------------------------------------
#             FUNÇÃO PARA BUSCAR PROFISSÃO
# ---------------------------------------------------------

def buscar_profissao(entrada: str):
    entrada = entrada.strip().lower()

    if entrada.isdigit():
        resultado = df_cbo[df_cbo["cbo_codigo"] == entrada]
        return resultado

    # busca textual simples (sem unidecode)
    resultado = df_cbo[
        df_cbo["cbo_descricao"].str.lower().str.contains(entrada, na=False)
    ]
    return resultado


# ---------------------------------------------------------
#            RELATÓRIO COMPLETO DA PROFISSÃO
# ---------------------------------------------------------

def relatorio(cbo_codigo):

    st.subheader("📌 Relatório Completo da Profissão")

    info = df_cbo[df_cbo["cbo_codigo"] == cbo_codigo]

    if info.empty:
        st.error("Profissão não encontrada na tabela CBO.")
        return

    nome = info.iloc[0]["cbo_descricao"]
    st.markdown(f"## **{nome}**")

    # Filtrar dados do CBO escolhido
    dfx = df[df["cbo2002ocupacao"] == cbo_codigo].copy()

    if dfx.empty:
        st.warning("Sem dados suficientes no CAGED para gerar previsões.")
        return

    # -------------------------------
    # SALÁRIO ATUAL
    # -------------------------------
    salario_atual = dfx["salario"].mean()
    st.write(f"**Salário médio atual:** R$ {formatar_moeda(salario_atual)}")

    # -------------------------------
    # PREPARAR BASE TEMPORAL
    # -------------------------------
    dfx["competenciamov"] = pd.to_datetime(dfx["competenciamov"], errors="coerce")
    dfx = dfx.dropna(subset=["competenciamov"])

    dfx["mes"] = (dfx["competenciamov"].dt.year - 2020) * 12 + dfx["competenciamov"].dt.month

    df_mensal = dfx.groupby("mes")["salario"].mean().reset_index()

    # -------------------------------
    # PREVISÃO DE SALÁRIO
    # -------------------------------

    if len(df_mensal) < 2:
        st.warning("Sem dados suficientes para previsão salarial.")
    else:
        st.subheader("📈 Previsão Salarial")

        X = df_mensal[["mes"]]
        y = df_mensal["salario"]

        modelo = LinearRegression().fit(X, y)

        ultimo_mes = df_mensal["mes"].max()

        anos = [5, 10, 15, 20]

        st.markdown("### **Previsão salarial futura:**")
        for a in anos:
            futuro = ultimo_mes + a * 12
            pred = modelo.predict([[futuro]])[0]
            st.write(f"**{a} anos → R$ {formatar_moeda(pred)}**")

    # -------------------------------
    # PREVISÃO DE VAGAS
    # -------------------------------

    st.subheader("📊 Tendência de Mercado")

    if "saldomovimentacao" not in dfx.columns:
        st.warning("Sem dados de vagas.")
        return

    saldo_medio = dfx["saldomovimentacao"].mean()

    if saldo_medio > 10:
        tendencia = "CRESCIMENTO ACELERADO"
    elif saldo_medio > 0:
        tendencia = "CRESCIMENTO LEVE"
    elif saldo_medio < -10:
        tendencia = "QUEDA ACELERADA"
    elif saldo_medio < 0:
        tendencia = "QUEDA LEVE"
    else:
        tendencia = "ESTÁVEL"

    st.write(f"**Situação recente:** {tendencia}")

    st.markdown("### **Projeção de vagas (admissões − desligamentos):**")

    for a in [5, 10, 15, 20]:
        st.write(f"{a} anos: {int(saldo_medio)} vagas/mês")


# ---------------------------------------------------------
#                      INTERFACE STREAMLIT
# ---------------------------------------------------------

st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho (CAGED / CBO)")

busca = st.text_input("Digite nome ou código da profissão:")

if busca:
    resultados = buscar_profissao(busca)

    if resultados.empty:
        st.error("❌ Profissão não encontrada. Tente outro nome ou código.")
    else:
        lista = resultados["cbo_codigo"] + " - " + resultados["cbo_descricao"]

        primeira_opcao = lista.iloc[0]

        st.success("Profissão encontrada! Veja as opções relacionadas abaixo:")

        escolha = st.selectbox(
            "Selecione o CBO:",
            lista,
            index=0  # primeira opção já marcada
        )

        cbo = escolha.split(" - ")[0]

        if st.button("Gerar Relatório Completo"):
            relatorio(cbo)
