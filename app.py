import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st

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
        except Exception:
            return str(valor)

    def carregar_dados(self):
        # Apenas o PARQUET agora
        self.df = pd.read_parquet(self.parquet_file)

        # Padroniza nomes
        self.df.columns = [c.lower() for c in self.df.columns]

        # Carrega o CBO
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ["cbo_codigo", "cbo_descricao"]
        self.df_codigos["cbo_codigo"] = self.df_codigos["cbo_codigo"].astype(str)

        self.cleaned = True

    def buscar_profissao(self, entrada: str):
        if not self.cleaned:
            return pd.DataFrame()

        if entrada.isdigit():
            return self.df_codigos[self.df_codigos["cbo_codigo"] == entrada]

        mask = self.df_codigos["cbo_descricao"].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5,10,15,20]):
        df = self.df
        col_cbo = "cbo2002ocupacao"
        col_data = "competenciamov"
        col_salario = "salario"
        saldo_col = "saldomovimentacao"

        # Info da profissão
        info = self.df_codigos[self.df_codigos["cbo_codigo"] == cbo_codigo]
        nome = info.iloc[0]["cbo_descricao"] if not info.empty else cbo_codigo
        st.subheader(f"Profissão: {nome}")

        df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()

        if df_cbo.empty:
            st.warning("Nenhum registro encontrado para essa profissão.")
            return

        # Dados gerais
        st.write(f"Registros encontrados: **{len(df_cbo):,}**")

        # =====================
        # PERFIL DEMOGRÁFICO
        # =====================
        with st.expander("Perfil Demográfico"):
            if "idade" in df_cbo:
                st.write(f"Idade média: {pd.to_numeric(df_cbo['idade'], errors='coerce').mean():.1f}")

            if "sexo" in df_cbo:
                sexo_dist = df_cbo["sexo"].value_counts()
                mapa = {"1": "Masculino", "3": "Feminino"}
                lista = [f"{mapa.get(str(k), k)}: {(v/len(df_cbo))*100:.1f}%" for k, v in sexo_dist.items()]
                st.write("Distribuição por sexo:", ", ".join(lista))

        # =====================
        # SALDO DO MERCADO
        # =====================
        if saldo_col in df_cbo:
            saldo_total = pd.to_numeric(df_cbo[saldo_col], errors="coerce").sum()
            if saldo_total > 0: status = "EXPANSÃO"
            elif saldo_total < 0: status = "RETRAÇÃO"
            else: status = "ESTÁVEL"
            st.subheader("Situação do Mercado de Trabalho")
            st.write(f"Saldo total: {saldo_total:+,.0f} → **{status}**")

        # =====================
        # PREVISÃO SALARIAL
        # =====================
        st.subheader("Previsão Salarial")

        df_cbo[col_salario] = pd.to_numeric(
            df_cbo[col_salario].astype(str).str.replace(",", ".").str.replace(" ", ""),
            errors="coerce"
        )
        df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors="coerce")

        df_cbo = df_cbo.dropna(subset=[col_salario, col_data])

        if df_cbo.empty:
            st.warning("Sem dados temporais válidos.")
            return

        df_cbo["tempo_meses"] = (df_cbo[col_data].dt.year - 2020) * 12 + df_cbo[col_data].dt.month
        df_mensal = df_cbo.groupby("tempo_meses")[col_salario].mean().reset_index()
        salario_atual = df_mensal[col_salario].iloc[-1]

        st.write(f"Salário médio atual: **R$ {self.formatar_moeda(salario_atual)}**")

        model = LinearRegression().fit(df_mensal[["tempo_meses"]], df_mensal[col_salario])
        ult_mes = df_mensal["tempo_meses"].max()

        previsoes = []
        for anos in anos_futuros:
            futuro = ult_mes + anos * 12
            pred = model.predict([[futuro]])[0]
            var_pct = ((pred - salario_atual) / salario_atual) * 100
            previsoes.append([anos, self.formatar_moeda(pred), f"{var_pct:+.1f}%"])

        st.table(pd.DataFrame(previsoes, columns=["Anos", "Salário Previsto", "Variação (%)"]))

        # =====================
        # PREVISÃO DE VAGAS
        # =====================
        st.subheader("Tendência de Vagas")

        if saldo_col in df_cbo:
            df_saldo = df_cbo.groupby("tempo_meses")[saldo_col].sum().reset_index()

            model2 = LinearRegression().fit(df_saldo[["tempo_meses"]], df_saldo[saldo_col])

            tendencias = []
            for anos in anos_futuros:
                futuro = ult_mes + anos * 12
                pred = model2.predict([[futuro]])[0]

                if pred > 100: status = "ALTA DEMANDA"
                elif pred > 50: status = "CRESCIMENTO MODERADO"
                elif pred > 0: status = "CRESCIMENTO LEVE"
                elif pred > -50: status = "RETRAÇÃO LEVE"
                elif pred > -100: status = "RETRAÇÃO MODERADA"
                else: status = "RETRAÇÃO FORTE"

                tendencias.append([anos, f"{pred:+,.0f}", status])

            st.table(pd.DataFrame(tendencias, columns=["Anos", "Vagas Previstas/mês", "Tendência"]))


# ===============================
# STREAMLIT APP
# ===============================
st.set_page_config(page_title="Mercado de Trabalho", layout="wide")
st.title("📊 Previsão do Mercado de Trabalho (CAGED / CBO)")

app = MercadoTrabalhoPredictor("dados.parquet", "cbo.xlsx")

with st.spinner("Carregando dados..."):
    app.carregar_dados()

st.success("Dados carregados!")

busca = st.text_input("Digite nome ou código da profissão:")

if busca:
    resultados = app.buscar_profissao(busca)

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        cbo_opcao = st.selectbox(
            "Selecione:",
            resultados["cbo_codigo"] + " - " + resultados["cbo_descricao"]
        )

        cbo_codigo = cbo_opcao.split(" - ")[0]

        if st.button("Gerar análise"):
            app.relatorio_previsao(cbo_codigo)
