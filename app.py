import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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
            self.df["salario"] = (
                self.df["salario"]
                .astype(str)
                .str.replace(",", ".")
            )
            self.df["salario"] = pd.to_numeric(self.df["salario"], errors="coerce")
            self.df["salario"] = self.df["salario"].fillna(self.df["salario"].median())

        self.cleaned = True

    def buscar_profissao(self, entrada: str):
        if not self.cleaned:
            return pd.DataFrame()

        entrada = entrada.strip()

        if entrada.isdigit():
            return self.df_codigos[self.df_codigos["cbo_codigo"] == entrada]

        mask = self.df_codigos["cbo_descricao"].str.contains(entrada, case=False, na=False)
        return self.df_codigos[mask]

    def relatorio_previsao(self, cbo_codigo, anos_futuros=[5, 10, 15, 20]):
        df = self.df

        col_cbo = "cbo2002ocupacao"
        col_data = "competenciamov"
        col_salario = "salario"
        col_saldo = "saldomovimentacao"

        prof_info = self.df_codigos[self.df_codigos["cbo_codigo"] == cbo_codigo]
        nome = prof_info.iloc[0]["cbo_descricao"] if not prof_info.empty else f"CBO {cbo_codigo}"

        print(f"\nProfissão: {nome}\n")

        df_cbo = df[df[col_cbo].astype(str) == cbo_codigo].copy()

        if df_cbo.empty:
            print("Nenhum dado disponível para esta profissão.")
            return

        # ======= SALÁRIOS =========
        df_cbo[col_data] = pd.to_datetime(df_cbo[col_data], errors="coerce")
        df_cbo = df_cbo.dropna(subset=[col_data])

        df_cbo["tempo_meses"] = (df_cbo[col_data].dt.year - 2020) * 12 + df_cbo[col_data].dt.month
        df_mensal = df_cbo.groupby("tempo_meses")[col_salario].mean().reset_index()

        salario_atual = df_cbo[col_salario].mean()
        print(f"Salário médio atual: R$ {self.formatar_moeda(salario_atual)}\n")

        modelo = LinearRegression().fit(df_mensal[["tempo_meses"]], df_mensal[col_salario])

        print("Modelo usado: LinearRegression (simples)\n")

        # ====== PREVISÕES ======
        print("Previsão salarial futura:")
        ult_mes = df_mensal["tempo_meses"].max()

        for anos in anos_futuros:
            futuro = ult_mes + anos * 12
            pred = modelo.predict([[futuro]])[0]
            print(f"  {anos} anos → R$ {self.formatar_moeda(pred)}")

        print("\n==============================================================")
        print("TENDÊNCIA DE MERCADO (Projeção de demanda para a profissão):")
        print("==============================================================")

        # ===== Pré-processamento vagas ======
        if col_saldo not in df_cbo.columns:
            print("Sem dados de movimentação (vagas).")
            return

        df_saldo = df_cbo.groupby("tempo_meses")[col_saldo].sum().reset_index()

        if len(df_saldo) < 2:
            print("Dados insuficientes para prever vagas.")
            return

        mod_saldo = LinearRegression().fit(df_saldo[["tempo_meses"]], df_saldo[col_saldo])

        ult_mes_s = df_saldo["tempo_meses"].max()

        saldo_atual = df_saldo[col_saldo].iloc[-1]
        if saldo_atual > 0:
            status_hist = "CRESCIMENTO LEVE"
        elif saldo_atual == 0:
            status_hist = "ESTABILIDADE"
        else:
            status_hist = "RETRAÇÃO"

        print(f"Situação histórica recente: {status_hist}\n")
        print("Projeção de saldo de vagas (admissões - desligamentos):")

        for anos in anos_futuros:
            futuro = ult_mes_s + anos * 12
            pred = mod_saldo.predict([[futuro]])[0]
            seta = "→" if abs(pred) < 5 else ("↑" if pred > 0 else "↓")
            print(f"  {anos} anos: {int(pred)} ({seta})")

        print("\nDigite o nome ou código da profissão (ou 'sair' para encerrar): ")


# ===========================
#         EXECUÇÃO
# ===========================
if __name__ == "__main__":
    PARQUET_FILE = "dados.parquet"
    CBO_FILE = "cbo.xlsx"

    app = MercadoTrabalhoPredictor(PARQUET_FILE, CBO_FILE)
    app.carregar_dados()

    while True:
        entrada = input("Digite o nome ou código da profissão (ou 'sair' para encerrar): ").strip()
        if entrada.lower() == "sair":
            break

        print(f"\nBuscando '{entrada}'...")
        resultados = app.buscar_profissao(entrada)

        if resultados.empty:
            print("Nenhuma profissão encontrada.\n")
            continue

        print(f"{len(resultados)} profissão(ões) encontrada(s):")
        for idx, row in resultados.iterrows():
            print(f"  [{row['cbo_codigo']}] {row['cbo_descricao']}")

        cbo = input("\nDigite o código CBO desejado: ").strip()
        print(f"\nCódigo CBO selecionado: {cbo}\n")

        app.relatorio_previsao(cbo)
