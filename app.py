import streamlit as st
import pandas as pd
import numpy as np
import unicodedata

# =======================================================
# Função para normalizar strings (sem acentos)
# =======================================================
def normalizar(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

# =======================================================
# Carregar tabela CBO
# =======================================================
@st.cache_data
def carregar_dados_cbo(cbo_path="cbo.xlsx"):
    df = pd.read_excel(cbo_path)
    df.columns = ["Código", "Descrição"]
    df["Código"] = df["Código"].astype(str).str.strip()
    df["Descrição"] = df["Descrição"].astype(str).str.strip()
    df["Descrição_norm"] = df["Descrição"].apply(normalizar)
    return df

# =======================================================
# Carregar parquet com detecção automática de colunas
# =======================================================
@st.cache_data
def carregar_historico(path="dados.parquet"):
    df = pd.read_parquet(path)

    # Normalizar todos nomes de colunas
    cols_norm = {}
    for col in df.columns:
        col_norm = "".join(
            c for c in unicodedata.normalize("NFD", col.lower())
            if unicodedata.category(c) != "Mn"
        )
        cols_norm[col] = col_norm

    df.columns = cols_norm.values()

    # Detectar coluna CBO
    col_cbo = None
    for col in df.columns:
        if "cbo2002" in col or "ocupacao" in col or col.startswith("cbo"):
            col_cbo = col
            break

    if col_cbo is None:
        st.error("❌ ERRO: Nenhuma coluna de CBO encontrada no dados.parquet.")
        st.stop()

    # Detectar coluna salario
    col_sal = None
    for col in df.columns:
        if "sal" in col:
            col_sal = col
            break

    if col_sal is None:
        st.error("❌ ERRO: Nenhuma coluna salarial encontrada no dados.parquet.")
        st.stop()

    df[col_cbo] = df[col_cbo].astype(str).str.strip()
    df[col_sal] = pd.to_numeric(df[col_sal], errors="coerce").fillna(0)

    return df, col_cbo, col_sal

# =======================================================
# Buscar profissão
# =======================================================
def buscar_profissao(df_cbo, entrada):
    entrada_norm = normalizar(entrada)

    if entrada.isdigit():
        return df_cbo[df_cbo["Código"] == entrada]

    return df_cbo[df_cbo["Descrição_norm"].str.contains(entrada_norm)]

# =======================================================
# Prever salário (modelo simples)
# =======================================================
def prever_salario(salario_atual):
    anos = [5, 10, 15, 20]
    taxa = 0.02  # Crescimento anual
    return {ano: salario_atual * ((1 + taxa) ** ano) for ano in anos}

# =======================================================
# Tendência de mercado
# =======================================================
def tendencia_mercado(df, col_cbo, cbo_codigo):
    df_cbo = df[df[col_cbo] == cbo_codigo]

    if df_cbo.empty:
        return "Sem dados suficientes", {5:0, 10:0, 15:0, 20:0}

    saldo_medio = df_cbo["saldomovimentacao"].mean()

    if saldo_medio > 10:
        status = "CRESCIMENTO ACELERADO"
    elif saldo_medio > 0:
        status = "CRESCIMENTO LEVE"
    elif saldo_medio < -10:
        status = "QUEDA ACELERADA"
    elif saldo_medio < 0:
        status = "QUEDA LEVE"
    else:
        status = "ESTÁVEL"

    previsao = {ano: int(saldo_medio) for ano in [5,10,15,20]}
    return status, previsao

# =======================================================
# Aplicativo Streamlit
# =======================================================
st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("📊 Previsão Salarial e Tendência do Mercado (CAGED / CBO)")

df_cbo = carregar_dados_cbo()
df_hist, COL_CBO, COL_SALARIO = carregar_historico()

entrada = st.text_input("Digite nome ou código da profissão:")

if entrada:
    resultados = buscar_profissao(df_cbo, entrada)

    if resultados.empty:
        st.error("❌ Profissão não encontrada. Tente novamente.")
        st.stop()

    # múltiplas opções → usuário escolhe
    if len(resultados) > 1:
        st.warning("Foram encontradas várias profissões. Selecione uma:")
        opcoes = resultados["Descrição"] + " (" + resultados["Código"] + ")"
        escolha = st.selectbox("Selecione:", opcoes)
        cbo_codigo = escolha.split("(")[-1].replace(")", "").strip()
    else:
        cbo_codigo = resultados.iloc[0]["Código"]

    descricao = resultados[resultados["Código"] == cbo_codigo]["Descrição"].values[0]

    st.header(f"👷 Profissão: {descricao}")

    # Filtrar histórico
    df_cbo_hist = df_hist[df_hist[COL_CBO] == cbo_codigo]

    if df_cbo_hist.empty:
        st.error("Sem dados históricos suficientes para gerar previsões.")
        st.stop()

    # Salário atual
    salario_atual = df_cbo_hist[COL_SALARIO].mean()

    st.subheader("💰 Salário Atual")
    st.write(f"Salário médio atual: **R$ {salario_atual:,.2f}**")

    # Previsão salarial
    st.subheader("📈 Previsão Salarial Futura")
    previsoes = prever_salario(salario_atual)

    for ano, valor in previsoes.items():
        st.write(f"**{ano} anos → R$ {valor:,.2f}**")

    st.write("*Tendência de crescimento do salário no longo prazo.*")

    # Tendência de mercado
    st.markdown("---")
    st.subheader("📊 Tendência de Mercado")

    status, vagas = tendencia_mercado(df_hist, COL_CBO, cbo_codigo)

    st.write(f"Situação histórica recente: **{status}**")

    for ano, val in vagas.items():
        seta = "↑" if val > 0 else "↓" if val < 0 else "→"
        st.write(f"**{ano} anos: {val} ({seta})**")
