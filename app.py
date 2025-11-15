import os
import sys
import io
import warnings
import logging

# ---------------------------
# Supressão global de logs
# ---------------------------
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CMDSTAN_LOG_LEVEL'] = 'ERROR'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('cmdstan').setLevel(logging.ERROR)

try:
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm
from xgboost import XGBRegressor
from contextlib import redirect_stderr
with redirect_stderr(io.StringIO()):
    from prophet import Prophet

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

try:
    from pmdarima import auto_arima
    PMDARIMA_OK = True
except Exception:
    PMDARIMA_OK = False

# ==========================================================
# Classe principal — Agora usando CSV como fonte principal
# ==========================================================
class MercadoTrabalhoPredictor:
    def __init__(self, csv_files, codigos_filepath):
        self.csv_files = csv_files
        self.codigos_filepath = codigos_filepath
        self.df = None
        self.df_codigos = None
        self.cleaned = False
        self.coluna_cbo = None
        self.coluna_data = None
        self.coluna_salario = None
        self.lstm_model = None
        self.lstm_lag = 12

    def formatar_moeda(self, valor):
        try:
            return f"{float(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(valor)

    def converter_data_robusta(self, df_cbo):
        try:
            df_cbo = df_cbo.copy()
            df_cbo[self.coluna_data] = df_cbo[self.coluna_data].astype(str).str.strip().str.replace(".", "").str.replace(",", "")
            df_cbo = df_cbo[df_cbo[self.coluna_data].str.match(r'^\d{6}$', na=False)]
            if df_cbo.empty:
                return pd.DataFrame()
            df_cbo['ano'] = df_cbo[self.coluna_data].str[:4].astype(int)
            df_cbo['mes'] = df_cbo[self.coluna_data].str[4:].astype(int)
            df_cbo = df_cbo[(df_cbo['ano'] >= 2000) & (df_cbo['ano'] <= 2100)]
            df_cbo = df_cbo[(df_cbo['mes'] >= 1) & (df_cbo['mes'] <= 12)]
            df_cbo['data_convertida'] = pd.to_datetime({
                'year': df_cbo['ano'],
                'month': df_cbo['mes'],
                'day': 1
            })
            return df_cbo.sort_values('data_convertida')
        except Exception:
            return pd.DataFrame()

    def carregar_dados(self):
        # Carrega todos os CSV da pasta local, pulando linhas problemáticas
        print(" Carregando datasets dos arquivos .csv...")
        dfs = [pd.read_csv(path, encoding='utf-8', on_bad_lines='skip') for path in self.csv_files]
        self.df = pd.concat(dfs, ignore_index=True)
        print(f" Dataset carregado: {self.df.shape[0]:,} linhas e {self.df.shape[1]} colunas.")
        # códigos CBO
        self.df_codigos = pd.read_excel(self.codigos_filepath)
        self.df_codigos.columns = ['cbo_codigo', 'cbo_descricao']
        self.df_codigos['cbo_codigo'] = self.df_codigos['cbo_codigo'].astype(str)
        print(f" {len(self.df_codigos)} profissões carregadas.\n")

    def limpar_dados(self):
        print(" Limpando e otimizando dataset...")
        obj_cols = [col for col in self.df.columns if self.df[col].dtype == 'object']
        for col in obj_cols:
            self.df[col] = self.df[col].astype(str)
        for col in self.df.select_dtypes(include=['number']).columns:
            self.df[col] = self.df[col].fillna(self.df[col].median())
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.cleaned = True
        print(f" Limpeza concluída! {self.df.shape[0]:,} linhas e {self.df.shape[1]} colunas.")
        print(" Identificando colunas...")
        self._identificar_colunas()

    def _identificar_colunas(self):
        for col in self.df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            if 'cbo' in col_lower and 'ocupa' in col_lower:
                self.coluna_cbo = col
            if 'competencia' in col_lower and 'mov' in col_lower:
                self.coluna_data = col
            if 'salario' in col_lower:
                self.coluna_salario = col
        print(f"  ✓ Coluna CBO: {self.coluna_cbo}")
        print(f"  ✓ Coluna DATA: {self.coluna_data}")
        print(f"  ✓ Coluna SALÁRIO: {self.coluna_salario}\n")

    # ... [o restante dos métodos permanece igual ao seu código original] ...

# ==========================================================
# main — ajuste para sequência de arquivos CSV
# ==========================================================
def main():
    # Defina a lista dos CSV presentes na pasta local
    csv_files = [
        "2020_PE1.csv",
        "2021_PE1.csv",
        "2022_PE1.csv",
        "2023_PE1.csv",
        "2024_PE1.csv",
        "2025_PE1.csv"
    ]
    codigos_filepath = "cbo.xlsx"
    app = MercadoTrabalhoPredictor(csv_files=csv_files, codigos_filepath=codigos_filepath)
    app.carregar_dados()
    app.limpar_dados()
    while True:
        entrada = input("Digite o nome ou código da profissão (ou 'sair' para encerrar): ").strip()
        if entrada.lower() == 'sair':
            print(" Encerrando aplicação.")
            break
        resultados = app.buscar_profissao(entrada)
        if resultados.empty:
            continue
        if len(resultados) == 1:
            cbo = resultados['cbo_codigo'].iloc[0]
            print(f" Código CBO selecionado: {cbo}\n")
        else:
            cbo = input("\nDigite o código CBO desejado: ").strip()
        app.prever_mercado(cbo)

if __name__ == "__main__":
    main()
