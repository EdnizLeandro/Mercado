import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ==============================
# CLASSE ORIGINAL (com prints -> st.write)
# ==============================
class MercadoTrabalhoPredictor:
    def __init__(self, filepath: str, codigos_filepath: str):
        self.filepath = filepath
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False
        self.coluna_cbo = None
        self.coluna_data = None
        self.coluna_salario = None

    def formatar_moeda(self, valor):
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    def carregar_dados(self):
        st.info("📂 Carregando dataset...")
        self.df = pd.read_parquet(self.filepath)
        st.success(f"Dataset carregado com {self.df.shape[0]} linhas e {self.df.shape[1]} colunas.")

        st.info("📋 Carregando lista de códigos CBO...")
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ['cbo_codigo', 'cbo_descricao']
        self.df_codigos['cbo_codigo'] = self.df_codigos['cbo_codigo'].astype(str)
        st.success(f"{len(self.df_codigos)} profissões carregadas.")

    def limpar_dados(self):
        st.info("🧹 Limpando dataset...")
        obj_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in obj_cols:
            self.df[col] = self.df[col].astype(str)

        for col in self.df.select_dtypes(include=['number']).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())

        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.cleaned = True
        st.success(f"✅ Limpeza finalizada! ({self.df.shape[0]} linhas)")

        self._identificar_colunas()

    def _identificar_colunas(self):
        for col in self.df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if 'cbo' in col_lower and 'ocupa' in col_lower:
                self.coluna_cbo = col
            if 'competencia' in col_lower and 'mov' in col_lower:
                self.coluna_data = col
            if 'salario' in col_lower and 'fixo' in col_lower:
                self.coluna_salario = col

        st.write("### 🔍 Colunas identificadas automaticamente")
        st.write(f"- CBO: **{self.coluna_cbo}**")
        st.write(f"- DATA: **{self.coluna_data}**")
        st.write(f"- SALÁRIO: **{self.coluna_salario}**")

    def buscar_profissao(self, entrada: str):
        if not self.cleaned:
            st.warning("⚠️ Limpe o dataset antes de buscar profissões.")
            return pd.DataFrame()

        if entrada.isdigit():
            resultados = self.df_codigos[self.df_codigos['cbo_codigo'] == entrada]
        else:
            mask = self.df_codigos['cbo_descricao'].str.contains(entrada, case=False, na=False)
            resultados = self.df_codigos[mask]

        if resultados.empty:
            st.warning("Nenhuma profissão encontrada com esse nome ou código.")
        else:
            st.success(f"{len(resultados)} resultado(s) encontrado(s).")
            st.dataframe(resultados)
        return resultados

    def prever_mercado(self, cbo_codigo: str, anos_futuros=[5, 10, 15, 20]):
        if not self.cleaned:
            st.warning("Dataset não limpo.")
            return

        if not all([self.coluna_cbo, self.coluna_data, self.coluna_salario]):
            st.warning("Colunas necessárias não identificadas.")
            return

        prof_info = self.df_codigos[self.df_codigos['cbo_codigo'] == cbo_codigo]
        if not prof_info.empty:
            st.header(f"📊 Análise de Mercado: {prof_info.iloc[0]['cbo_descricao']} ({cbo_codigo})")

        df_cbo = self.df[self.df[self.coluna_cbo].astype(str) == cbo_codigo].copy()
        if df_cbo.empty:
            st.warning("Nenhum registro encontrado para este CBO.")
            return

        st.write(f"**{len(df_cbo):,} registros encontrados.**")

        # SALÁRIO MÉDIO E TENDÊNCIA
        df_cbo[self.coluna_data] = pd.to_datetime(df_cbo[self.coluna_data], errors='coerce')
        df_cbo = df_cbo.dropna(subset=[self.coluna_data])
        if df_cbo.empty:
            st.warning("Sem dados temporais válidos para previsão.")
            return

        df_cbo['tempo_meses'] = ((df_cbo[self.coluna_data].dt.year - 2020) * 12 +
                                  df_cbo[self.coluna_data].dt.month)
        df_mensal = df_cbo.groupby('tempo_meses')[self.coluna_salario].mean().reset_index()

        salario_atual = df_cbo[self.coluna_salario].mean()
        st.metric("💰 Salário médio atual", f"R$ {self.formatar_moeda(salario_atual)}")

        if len(df_mensal) >= 2:
            X = df_mensal[['tempo_meses']]
            y = df_mensal[self.coluna_salario]
            model = LinearRegression()
            model.fit(X, y)

            ult_mes = df_mensal['tempo_meses'].max()
            previsoes = []
            for anos in anos_futuros:
                mes_futuro = ult_mes + anos * 12
                pred = model.predict(np.array([[mes_futuro]]))[0]
                variacao = ((pred - salario_atual) / salario_atual) * 100
                previsoes.append((anos, pred, variacao))

            df_prev = pd.DataFrame(previsoes, columns=['Anos Futuro', 'Salário Previsto', 'Variação %'])
            df_prev['Salário Previsto'] = df_prev['Salário Previsto'].apply(lambda x: f"R$ {self.formatar_moeda(x)}")
            df_prev['Variação %'] = df_prev['Variação %'].apply(lambda x: f"{x:+.1f}%")
            st.subheader("📈 Projeção Salarial")
            st.dataframe(df_prev)
        else:
            st.info("Previsão baseada apenas na média atual (dados insuficientes).")


# ==============================
# INTERFACE STREAMLIT
# ==============================
def main():
    st.title("💼 Previsor de Mercado de Trabalho - CBO")
    st.write("Faça upload dos arquivos necessários e explore as previsões por profissão.")

    dados_file = st.file_uploader("Selecione o arquivo de dados (.parquet)", type=["parquet"])
    codigos_file = st.file_uploader("Selecione o arquivo de códigos CBO (.xlsx)", type=["xlsx"])

    if dados_file and codigos_file:
        app = MercadoTrabalhoPredictor(dados_file, codigos_file)
        app.carregar_dados()
        app.limpar_dados()

        entrada = st.text_input("🔍 Digite o nome ou código da profissão:")
        if entrada:
            resultados = app.buscar_profissao(entrada)
            if not resultados.empty:
                cbo_opcoes = resultados['cbo_codigo'].tolist()
                cbo = st.selectbox("Selecione o código CBO para previsão:", cbo_opcoes)
                if st.button("Gerar previsão"):
                    app.prever_mercado(cbo)
    else:
        st.info("Envie os arquivos de dados para começar.")

if __name__ == "__main__":
    main()
