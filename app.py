import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import streamlit as st

# ------------------------------
# CLASSE DE PREDIÇÃO DE MERCADO
# ------------------------------
class MercadoTrabalhoPredictor:
    def __init__(self, parquet_file: str, codigos_filepath: str):
        self.parquet_file = parquet_file
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False

    def formatar_moeda(self, valor):
        try:
            return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(valor)

    def carregar_dados(self):
        self.df = pd.read_parquet(self.parquet_file)
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ["cbo_codigo", "cbo_descricao"]
        self.df_codigos["cbo_codigo"] = self.df_codigos["cbo_codigo"].astype(str)

        if "salario" in self.df.columns:
            self.df["salario"] = pd.to_numeric(self.df["salario"].astype(str).str.replace(",", "."), errors="coerce")
            mediana = self.df["salario"].median()
            self.df["salario"] = self.df["salario"].fillna(mediana)

        self.cleaned = True

    def buscar_profissao(self, entrada: str):
        if not self.cleaned:
            return pd.DataFrame()
        entrada = entrada.strip()
        if entrada.isdigit():
            return self.df_codigos[self.df_codigos["cbo_codigo"] == entrada]
        mask = self.df_codigos["cbo_descricao"].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5,10,15,20]):
        df = self.df
        col_cbo = "cbo2002ocupacao"
        col_data = "competenciamov"
        col_salario = "salario"
        col_saldo = "saldomovimentacao"

        prof_info = self.df_codigos[self.df_codigos["cbo_codigo"] == cbo_codigo]
        titulo = prof_info.iloc[0]["cbo_descricao"] if not prof_info.empty else f"CBO {cbo_codigo}"

        # Filtra dados da profissão
        df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()
        if df_cbo.empty:
            st.warning("Nenhum dado disponível para esta profissão.")
            return

        # Converte datas
        df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors="coerce")
        df_cbo = df_cbo.dropna(subset=[col_data])
        df_cbo["tempo_meses"] = (df_cbo[col_data].dt.year - 2020) * 12 + df_cbo[col_data].dt.month

        salario_atual = df_cbo[col_salario].mean()

        # Agrupa mensal
        df_mensal = df_cbo.groupby("tempo_meses")[col_salario].mean().reset_index()
        if len(df_mensal) < 2:
            st.info("Sem dados suficientes para fazer previsões.")
            return

        X = df_mensal[["tempo_meses"]]
        y = df_mensal[col_salario]

        # Treina modelos
        modelos = {
            "LinearRegression": LinearRegression(),
            "XGBoost": XGBRegressor(n_estimators=100, objective="reg:squarederror")
        }

        resultados = {}
        for nome, model in modelos.items():
            model.fit(X, y)
            pred = model.predict(X)
            r2 = r2_score(y, pred)
            mae = mean_absolute_error(y, pred)
            resultados[nome] = {"model": model, "r2": r2, "mae": mae}

        melhor_nome = max(resultados, key=lambda k: resultados[k]["r2"])
        melhor_modelo = resultados[melhor_nome]["model"]

        # ---------- PREVISÃO SALARIAL FUTURA ----------
        ult_mes = df_mensal["tempo_meses"].max()
        previsoes = []
        for anos in anos_futuros:
            futuro = ult_mes + anos*12
            pred = melhor_modelo.predict([[futuro]])[0]
            previsoes.append([anos, pred])

        # Tendência de salário
        if previsoes[-1][1] > salario_atual:
            tendencia_salario = "TENDÊNCIA DE CRESCIMENTO"
        elif previsoes[-1][1] < salario_atual:
            tendencia_salario = "TENDÊNCIA DE QUEDA"
        else:
            tendencia_salario = "TENDÊNCIA ESTÁVEL"

        # ---------- SALDO DE VAGAS ----------
        if col_saldo not in df_cbo.columns:
            df_cbo[col_saldo] = 0  # Caso não exista

        df_saldo = df_cbo.groupby("tempo_meses")[col_saldo].sum().reset_index()
        mod_saldo = LinearRegression().fit(df_saldo[["tempo_meses"]], df_saldo[col_saldo])
        ult_mes_s = df_saldo["tempo_meses"].max()

        # Situação histórica recente
        saldo_recente = df_saldo[col_saldo].iloc[-1]
        if saldo_recente > 100: situacao_hist = "ALTA DEMANDA"
        elif saldo_recente > 50: situacao_hist = "CRESCIMENTO MODERADO"
        elif saldo_recente > 0: situacao_hist = "CRESCIMENTO LEVE"
        elif saldo_recente > -50: situacao_hist = "RETRAÇÃO LEVE"
        else: situacao_hist = "RETRAÇÃO"

        # Projeção de saldo de vagas
        proj_saldo = []
        for anos in anos_futuros:
            futuro = ult_mes_s + anos*12
            pred = mod_saldo.predict([[futuro]])[0]
            proj_saldo.append([anos, round(pred), "→"])

        # ---------- IMPRESSÃO FORMATADA ----------
        console_output = []

        console_output.append(f"Previsão salarial futura do melhor modelo:")
        for anos, valor in previsoes:
            console_output.append(f"  {anos} anos → R$ {self.formatar_moeda(valor)}")
        console_output.append(f"* Tendência de crescimento do salário no longo prazo: {tendencia_salario}")
        console_output.append("\n" + "="*70)
        console_output.append("TENDÊNCIA DE MERCADO (Projeção de demanda para a profissão):")
        console_output.append("="*70)
        console_output.append(f"Situação histórica recente: {situacao_hist}")
        console_output.append("\nProjeção de saldo de vagas (admissões - desligamentos):")
        for anos, valor, seta in proj_saldo:
            console_output.append(f"  {anos} anos: {valor} ({seta})")
        console_output.append("Digite o nome ou código da profissão (ou 'sair' para encerrar):")

        st.text("\n".join(console_output))


# ------------------------------
# APLICATIVO STREAMLIT
# ------------------------------
st.set_page_config(page_title="Previsão Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho (CAGED / CBO)")

PARQUET_FILE = "dados.parquet"
CBO_FILE = "cbo.xlsx"

with st.spinner("Carregando dados..."):
    app = MercadoTrabalhoPredictor(PARQUET_FILE, CBO_FILE)
    app.carregar_dados()

busca = st.text_input("Digite nome ou código da profissão:")

if busca:
    resultados = app.buscar_profissao(busca)

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        lista = resultados["cbo_codigo"] + " - " + resultados["cbo_descricao"]
        escolha = st.selectbox("Selecione o CBO:", lista)
        cbo_codigo = escolha.split(" - ")[0]

        if st.button("Gerar Relatório Completo"):
            app.relatorio_previsao(cbo_codigo)
