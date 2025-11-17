import streamlit as st
import pandas as pd
import numpy as np
import unicodedata

from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------
# NORMALIZAÇÃO DE TEXTO
# ----------------------------------------------------------
def normalizar(texto):
    if not isinstance(texto, str):
        return ""
    texto = texto.lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )

# ----------------------------------------------------------
# CARREGA CBO
# ----------------------------------------------------------
@st.cache_data
def carregar_dados_cbo():
    df = pd.read_excel("cbo.xlsx")
    df.columns = ["Código", "Descrição"]

    df["Código"] = df["Código"].astype(str).strip()
    df["Descrição"] = df["Descrição"].astype(str).strip()
    df["Descrição_norm"] = df["Descrição"].apply(normalizar)

    return df

# ----------------------------------------------------------
# CARREGA HISTÓRICO CAGED
# ----------------------------------------------------------
@st.cache_data
def carregar_historico():
    df = pd.read_parquet("dados.parquet")

    # normalizar nomes das colunas
    def norm(c):
        return "".join(
            x for x in unicodedata.normalize("NFD", c.lower())
            if unicodedata.category(x) != "Mn"
        )

    old_cols = list(df.columns)
    new_cols = [norm(c) for c in old_cols]
    df.columns = new_cols

    col_cbo = next((c for c in df.columns if "cbo" in c), None)
    col_sal = next((c for c in df.columns if "sal" in c), None)

    df[col_cbo] = df[col_cbo].astype(str).str.strip()
    df[col_sal] = pd.to_numeric(df[col_sal], errors="coerce").fillna(0)

    return df, col_cbo, col_sal

# ----------------------------------------------------------
# BUSCA PROFISSÕES
# ----------------------------------------------------------
def buscar_profissoes(df_cbo, texto):
    tnorm = normalizar(texto)
    if texto.isdigit():
        return df_cbo[df_cbo["Código"] == texto]
    return df_cbo[df_cbo["Descrição_norm"].str.contains(tnorm, na=False)]

# ----------------------------------------------------------
# CRIA DATAS SEGURAS (2020–2025)
# ----------------------------------------------------------
def criar_datas_seguras(df):
    df = df.copy()
    df["y"] = df.iloc[:, 0]

    # Caso tenha ano/mes
    col_ano = next((c for c in df.columns if "ano" in c), None)
    col_mes = next((c for c in df.columns if "mes" in c), None)

    if col_ano and col_mes:
        df["data"] = pd.to_datetime(df[col_ano].astype(str) + "-" +
                                    df[col_mes].astype(str) + "-01", errors="coerce")
        df = df[df["data"].between("2020-01-01", "2025-12-01")]
        return df[["data", "y"]]

    # Caso tenha competenciamov (202001)
    if "competenciamov" in df.columns:
        df["data"] = pd.to_datetime(df["competenciamov"].astype(str),
                                    format="%Y%m", errors="coerce")
        df = df[df["data"].between("2020-01-01", "2025-12-01")]
        return df[["data", "y"]]

    # Caso não tenha data, gerar manual
    datas = pd.date_range("2020-01-01", "2025-12-01", freq="M")
    datas = datas[: len(df)]
    df["data"] = datas
    return df[["data", "y"]]

# ----------------------------------------------------------
# TREINA TODOS MODELOS E ESCOLHE O MELHOR
# ----------------------------------------------------------
def treinar_e_escolher_melhor_modelo(df):
    df = df.dropna().sort_values("data")

    if len(df) < 24:
        return None

    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    valid = df.iloc[split:]

    resultados = {}

    # PROPHET
    try:
        ptrain = train.rename(columns={"data": "ds", "y": "y"})
        model_p = Prophet()
        model_p.fit(ptrain)

        pfuture = valid.rename(columns={"data": "ds"})
        pred = model_p.predict(pfuture)["yhat"].values

        rmse = np.sqrt(mean_squared_error(valid["y"], pred))
        resultados["prophet"] = (rmse, model_p)
    except Exception as e:
        print("Erro Prophet:", e)

    # XGBOOST
    try:
        df_ml = df.copy()
        df_ml["mes"] = df_ml["data"].dt.month
        df_ml["ano"] = df_ml["data"].dt.year

        tml = df_ml.iloc[:split]
        vml = df_ml.iloc[split:]

        xgb = XGBRegressor(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=4
        )
        xgb.fit(tml[["mes", "ano"]], tml["y"])

        pred = xgb.predict(vml[["mes", "ano"]])
        rmse = np.sqrt(mean_squared_error(vml["y"], pred))
        resultados["xgboost"] = (rmse, xgb)
    except Exception as e:
        print("Erro XGBoost:", e)

    if not resultados:
        return None

    melhor = min(resultados, key=lambda k: resultados[k][0])
    rmse, modelo = resultados[melhor]

    return {"modelo_nome": melhor, "melhor_modelo": modelo, "rmse": rmse}

# ----------------------------------------------------------
# PREVISÃO
# ----------------------------------------------------------
def prever(modelo, modelo_nome, df):
    limite = pd.Timestamp("2025-12-01")

    if modelo_nome == "prophet":
        future = modelo.make_future_dataframe(periods=36, freq="M")
        forecast = modelo.predict(future)
        forecast = forecast[["ds", "yhat"]].rename(columns={"ds": "data", "yhat": "y"})
        forecast = forecast[forecast["data"] <= limite]
        return forecast

    if modelo_nome == "xgboost":
        datas = pd.date_range(df["data"].max(), limite, freq="M")
        temp = pd.DataFrame({"data": datas})
        temp["mes"] = temp["data"].dt.month
        temp["ano"] = temp["data"].dt.year
        temp["y"] = modelo.predict(temp[["mes", "ano"]])
        return temp

# ----------------------------------------------------------
# INTERFACE STREAMLIT
# ----------------------------------------------------------
st.set_page_config(page_title="Mercado de Trabalho - IA", layout="wide")
st.title("Previsão Inteligente do Mercado de Trabalho (CAGED + IA)")

df_cbo = carregar_dados_cbo()
df_hist, COL_CBO, COL_SALARIO = carregar_historico()

entrada = st.text_input("Digite nome ou código da profissão:")

if entrada:
    res = buscar_profissoes(df_cbo, entrada)
    if res.empty:
        st.warning("Nenhuma profissão encontrada.")
        st.stop()
    opcoes = (res["Descrição"] + " (" + res["Código"] + ")").tolist()
else:
    opcoes = []

escolha = st.selectbox("Selecione a profissão:", [""] + opcoes)

if escolha:
    codigo = escolha.split("(")[-1].replace(")", "").strip()
    descricao = escolha.split("(")[0].strip()

    st.header(f"Profissão: {descricao}")

    dados = df_hist[df_hist[COL_CBO] == codigo]

    if dados.empty:
        st.error("Sem dados para esta profissão.")
        st.stop()

    df_sal = criar_datas_seguras(dados[[COL_SALARIO]])

    st.subheader("Treinando modelos...")
    modelo = treinar_e_escolher_melhor_modelo(df_sal)

    if modelo is None:
        st.error("Dados insuficientes para prever.")
        st.stop()

    st.success(f"Modelo escolhido: **{modelo['modelo_nome']}** (RMSE: {modelo['rmse']:.2f})")

    previsao = prever(modelo["melhor_modelo"], modelo["modelo_nome"], df_sal)

    st.subheader("📈 Previsão até 2025")
    st.line_chart(previsao.set_index("data")["y"])

    st.info("""
### Como interpretar o gráfico:
- A linha mostra a evolução estimada do salário médio da profissão escolhida.
- A parte final representa as **previsões até dezembro de 2025**.
- O algoritmo selecionado automaticamente é o de **melhor RMSE**, ou seja, menor erro.
""")
