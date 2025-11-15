# app.py — Plataforma Jovem Futuro — Mostra somente o MELHOR modelo (RMSE)
import os
import math
import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Suppress warnings in UI
warnings.filterwarnings("ignore")
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('CMDSTAN_LOG_LEVEL', 'ERROR')

# Optional imports — safe
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    HAS_TF = True
except Exception:
    HAS_TF = False

st.set_page_config(page_title="Plataforma Jovem Futuro — Melhor Modelo (RMSE)", layout="wide")
st.title("🔎 Previsões do Mercado de Trabalho — Jovem Futuro")

# Files expected
PARQUET_FILE = "dados.parquet"
CBO_FILE = "cbo.xlsx"

# -------------------------
# Helpers
# -------------------------
def format_brl(x):
    try:
        s = f"{float(x):,.2f}"
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"
    except:
        return str(x)

def safe_rmse(y_true, y_pred):
    try:
        if len(y_true) == 0 or len(y_pred) == 0:
            return float('inf')
        # align lengths
        n = min(len(y_true), len(y_pred))
        return math.sqrt(mean_squared_error(y_true[:n], y_pred[:n]))
    except:
        return float('inf')

def find_col(df, keywords):
    for c in df.columns:
        low = c.lower().replace(" ", "").replace("_","")
        for k in keywords:
            if k in low:
                return c
    return None

def parse_competencia(col):
    s = col.astype(str).str.strip().str.replace(r'\D','', regex=True)
    def _p(v):
        if v is None or v=='' or v.lower()=='nan':
            return pd.NaT
        if len(v)==6:
            return pd.to_datetime(v, format='%Y%m', errors='coerce')
        if len(v)==8:
            return pd.to_datetime(v, format='%Y%m%d', errors='coerce')
        try:
            return pd.to_datetime(v, errors='coerce')
        except:
            return pd.NaT
    return s.apply(_p)

def fix_salary_col(s):
    s = s.astype(str).str.strip()
    s = s.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
    s = pd.to_numeric(s, errors='coerce')
    s.loc[s < 0] = np.nan
    s.loc[s > 1_000_000] = np.nan
    med = s.median()
    if np.isnan(med):
        med = 0.0
    s = s.fillna(med)
    return s

def create_lag_df(series, lags=12):
    df_l = pd.DataFrame({'y':series})
    for i in range(1,lags+1):
        df_l[f'lag_{i}'] = df_l['y'].shift(i)
    return df_l.dropna()

# -------------------------
# Load data (cached)
# -------------------------
@st.cache_resource
def load_data():
    if not os.path.exists(PARQUET_FILE):
        return None, "PARQUET_MISSING"
    if not os.path.exists(CBO_FILE):
        return None, "CBO_MISSING"
    try:
        df = pd.read_parquet(PARQUET_FILE)
    except Exception:
        return None, "PARQUET_INVALID"
    try:
        cbo = pd.read_excel(CBO_FILE)
    except Exception:
        return None, "CBO_INVALID"
    return (df, cbo), None

loaded, err = load_data()
if err:
    if err == "PARQUET_MISSING":
        st.error("Arquivo necessário ausente: dados.parquet")
    elif err == "CBO_MISSING":
        st.error("Arquivo necessário ausente: cbo.xlsx")
    else:
        st.error("Erro lendo arquivos. Verifique os arquivos.")
    st.stop()

df, df_cbo = loaded

# Don't show raw data or column lists anywhere (requirement)
# --------------------------------
# Detect columns robustly
col_cbo = find_col(df, ['cbo','ocupacao','ocupação'])
col_date = find_col(df, ['competencia','competenciamov','competenciadec','data'])
col_salary = find_col(df, ['salario','valorsalario','remuneracao'])
col_saldo = find_col(df, ['saldomovimentacao','saldomovimentação','saldo'])

if not all([col_cbo, col_date, col_salary, col_saldo]):
    st.error("Não foi possível identificar automaticamente todas as colunas necessárias no dataset. Certifique-se que o arquivo contém colunas de: CBO, competência (data), salário, saldomovimentacao.")
    st.stop()

# Parse dates robustly
df[col_date] = parse_competencia(df[col_date])
df = df.dropna(subset=[col_date]).copy()

# Fix salary
df[col_salary] = fix_salary_col(df[col_salary])

# Prepare CBO sheet: detect code & description columns
df_cbo.columns = [str(c).strip() for c in df_cbo.columns]
cbo_code_col = next((c for c in df_cbo.columns if 'cod' in c.lower()), df_cbo.columns[0])
cbo_desc_col = next((c for c in df_cbo.columns if 'descr' in c.lower() or 'nome' in c.lower() or 'titulo' in c.lower()), df_cbo.columns[1] if len(df_cbo.columns)>1 else df_cbo.columns[0])
df_cbo = df_cbo.rename(columns={cbo_code_col:'codigo', cbo_desc_col:'descricao'})
df_cbo['codigo'] = df_cbo['codigo'].astype(str)

# -------------------------
# UI: minimal inputs
# -------------------------
query = st.text_input("Digite nome ou código da profissão:", "")
if not query:
    st.info("Digite o nome (ex: pintor) ou código CBO para ver previsão.")
    st.stop()

# search CBO
c_mask = df_cbo['descricao'].astype(str).str.contains(query, case=False, na=False) | df_cbo['codigo'].astype(str).str.contains(query, na=False)
candidates = df_cbo[c_mask]
if candidates.empty:
    st.warning("Nenhuma profissão encontrada.")
    st.stop()

# if many candidates, show selectbox of codes (no table)
chosen_code = st.selectbox("Selecione o CBO", options=candidates['codigo'].astype(str).unique())
if not chosen_code:
    st.stop()

# subset job
df_job = df[df[col_cbo].astype(str)==str(chosen_code)].copy()
if df_job.empty:
    st.warning("Não há registros para o CBO selecionado.")
    st.stop()

# aggregate monthly by mean of saldomovimentacao
ts = df_job.set_index(col_date).resample('M')[col_saldo].mean().ffill().reset_index().rename(columns={col_date:'ds', col_saldo:'y'})

if ts['ds'].isnull().all() or len(ts) < 6:
    st.warning("Série temporal insuficiente para modelagem. Precisa de ao menos alguns meses de dados.")
    st.stop()

# Show simple historical plot (no tables)
fig_hist = px.line(ts, x='ds', y='y', title='Histórico de saldo médio mensal')
st.plotly_chart(fig_hist, use_container_width=True)

# Forecast horizon selection
horizon = st.selectbox("Horizonte (meses) para previsão:", options=[6,12,24], index=1)

# train/test split heuristic: last min(6, horizon//6) months as test
test_months = min(6, max(1, horizon//6))
train = ts[:-test_months] if len(ts) > test_months else ts
test = ts[-test_months:] if len(ts) >= test_months else ts.copy()

y_train = train['y'].values
y_test = test['y'].values

# container for results
results = {}

# 1) Linear reg on time index
try:
    X_tr = np.arange(len(train)).reshape(-1,1)
    X_te = np.arange(len(train), len(train)+len(test)).reshape(-1,1)
    lr = LinearRegression().fit(X_tr, y_train)
    pred_test = lr.predict(X_te) if len(X_te)>0 else np.array([])
    pred_full = lr.predict(np.arange(len(ts), len(ts)+horizon).reshape(-1,1))
    results['Linear'] = {
        'pred_test': np.array(pred_test),
        'pred_full': np.array(pred_full),
        'rmse': safe_rmse(y_test, np.array(pred_test)),
        'mae': mean_absolute_error(y_test, np.array(pred_test)) if len(y_test)>0 else float('inf'),
        'r2': r2_score(y_test, np.array(pred_test)) if len(y_test)>0 else float('-inf')
    }
except Exception:
    results['Linear'] = None

# 2) Prophet
if HAS_PROPHET:
    try:
        dfp = ts.rename(columns={'ds':'ds','y':'y'})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(dfp.iloc[:len(train)])
        fut_test = m.make_future_dataframe(periods=len(test), freq='M')
        fc_test = m.predict(fut_test)
        pred_test = fc_test['yhat'].iloc[-len(test):].values if len(test)>0 else np.array([])
        fut_full = m.make_future_dataframe(periods=horizon, freq='M')
        fc_full = m.predict(fut_full)
        pred_full = fc_full['yhat'].iloc[-horizon:].values
        results['Prophet'] = {
            'pred_test': np.array(pred_test),
            'pred_full': np.array(pred_full),
            'rmse': safe_rmse(y_test, np.array(pred_test)),
            'mae': mean_absolute_error(y_test, np.array(pred_test)) if len(y_test)>0 else float('inf'),
            'r2': r2_score(y_test, np.array(pred_test)) if len(y_test)>0 else float('-inf'),
            'model': m,
            'forecast_full': fc_full
        }
    except Exception:
        results['Prophet'] = None
else:
    results['Prophet'] = None

# 3) ARIMA / SARIMA
if HAS_STATSMODELS:
    try:
        arima = ARIMA(y_train, order=(1,1,1)).fit()
        pred_test = arima.forecast(steps=len(test)) if len(test)>0 else np.array([])
        pred_full = arima.forecast(steps=horizon)
        results['ARIMA'] = {
            'pred_test': np.array(pred_test),
            'pred_full': np.array(pred_full),
            'rmse': safe_rmse(y_test, np.array(pred_test)),
            'mae': mean_absolute_error(y_test, np.array(pred_test)) if len(y_test)>0 else float('inf'),
            'r2': r2_score(y_test, np.array(pred_test)) if len(y_test)>0 else float('-inf'),
            'model': arima
        }
    except Exception:
        results['ARIMA'] = None
else:
    results['ARIMA'] = None

# 4) XGBoost with lags
if HAS_XGBOOST:
    try:
        lags = min(12, max(3, len(ts)//6))
        df_lag = create_lag_df(ts['y'].values, lags=lags)
        X = df_lag.drop(columns='y').values; y = df_lag['y'].values
        if len(X) > 0:
            split = int(0.8*len(X))
            X_tr, X_val = X[:split], X[split:]
            y_tr, y_val = y[:split], y[split:]
            xgb = XGBRegressor(n_estimators=200, verbosity=0)
            xgb.fit(X_tr, y_tr)
            # test preds: rolling
            preds_test = []
            last = ts['y'].values[-lags:].tolist()
            for _ in range(len(test)):
                arr = np.array(last[-lags:]).reshape(1,-1)
                p = float(xgb.predict(arr)[0])
                preds_test.append(p); last.append(p)
            # full future
            last2 = ts['y'].values[-lags:].tolist()
            preds_full = []
            for _ in range(horizon):
                arr = np.array(last2[-lags:]).reshape(1,-1)
                p = float(xgb.predict(arr)[0])
                preds_full.append(p); last2.append(p)
            results['XGBoost'] = {
                'pred_test': np.array(preds_test),
                'pred_full': np.array(preds_full),
                'rmse': safe_rmse(y_test, np.array(preds_test)),
                'mae': mean_absolute_error(y_test, np.array(preds_test)) if len(y_test)>0 else float('inf'),
                'r2': r2_score(y_test, np.array(preds_test)) if len(y_test)>0 else float('-inf'),
                'model': xgb
            }
        else:
            results['XGBoost'] = None
    except Exception:
        results['XGBoost'] = None
else:
    results['XGBoost'] = None

# 5) LSTM (small) if TF available
if HAS_TF:
    try:
        window = min(6, max(3, len(ts)//12))
        arr = ts['y'].values
        Xs, ys = [], []
        for i in range(window, len(arr)):
            Xs.append(arr[i-window:i]); ys.append(arr[i])
        Xs = np.array(Xs); ys = np.array(ys)
        if len(Xs) > 10:
            minv, maxv = Xs.min(), Xs.max()
            scale = maxv-minv if maxv!=minv else 1.0
            Xs_s = (Xs-minv)/scale; ys_s = (ys-minv)/scale
            Xs_s = Xs_s.reshape((Xs_s.shape[0], Xs_s.shape[1], 1))
            tf.keras.backend.clear_session()
            model_l = Sequential()
            model_l.add(LSTM(32, input_shape=(Xs_s.shape[1],1)))
            model_l.add(Dropout(0.2))
            model_l.add(Dense(1))
            model_l.compile(optimizer='adam', loss='mse')
            model_l.fit(Xs_s, ys_s, epochs=15, batch_size=8, verbose=0)
            # test preds
            preds_test = []
            last_window = arr[-(window+len(test)):-len(test)] if len(test)>0 else arr[-window:]
            last = arr[-window:].tolist()
            for _ in range(len(test)):
                arr_in = (np.array(last[-window:]) - minv)/scale
                p = model_l.predict(arr_in.reshape(1,window,1), verbose=0)[0][0]
                p = p*scale + minv; preds_test.append(p); last.append(p)
            # full future
            last2 = arr[-window:].tolist(); preds_full = []
            for _ in range(horizon):
                arr_in = (np.array(last2[-window:]) - minv)/scale
                p = model_l.predict(arr_in.reshape(1,window,1), verbose=0)[0][0]
                p = p*scale + minv; preds_full.append(p); last2.append(p)
            results['LSTM'] = {
                'pred_test': np.array(preds_test),
                'pred_full': np.array(preds_full),
                'rmse': safe_rmse(y_test, np.array(preds_test)),
                'mae': mean_absolute_error(y_test, np.array(preds_test)) if len(y_test)>0 else float('inf'),
                'r2': r2_score(y_test, np.array(preds_test)) if len(y_test)>0 else float('-inf'),
                'model': model_l
            }
        else:
            results['LSTM'] = None
    except Exception:
        results['LSTM'] = None
else:
    results['LSTM'] = None

# Filter executed models
executed = {k:v for k,v in results.items() if v is not None}
if len(executed)==0:
    st.error("Nenhum modelo foi executado com sucesso. Verifique dependências.")
    st.stop()

# Build metrics table (internal only) and pick by RMSE
metrics = []
for name, r in executed.items():
    metrics.append({'model':name, 'rmse': float(r['rmse']), 'mae': float(r['mae']), 'r2': float(r['r2'])})
metrics_df = pd.DataFrame(metrics).sort_values('rmse').reset_index(drop=True)
best_name = metrics_df.loc[0,'model']
best = executed[best_name]

# Show only minimal outputs
st.markdown(f"### 🏆 Melhor modelo (critério = RMSE): **{best_name}**")
st.markdown(f"- RMSE: **{metrics_df.loc[0,'rmse']:.2f}**  •  MAE: **{metrics_df.loc[0,'mae']:.2f}**  •  R²: **{metrics_df.loc[0,'r2']:.3f}**")

# Plot predicted test vs real and future forecast for best model
fig = go.Figure()
fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Histórico (treino)'))
if len(test)>0:
    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='markers+lines', name='Real (teste)'))
if 'pred_test' in best and len(best['pred_test'])>0:
    # align x for pred_test -> use ds_test
    fig.add_trace(go.Scatter(x=test['ds'], y=best['pred_test'][:len(test)], mode='lines+markers', name=f'Previsto (teste) — {best_name}'))
# future dates
last_date = ts['ds'].max()
future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq='M')
fig.add_trace(go.Scatter(x=future_dates, y=best['pred_full'] if 'pred_full' in best else best['pred_full_future'], mode='lines+markers', name=f'Forecast {horizon}m — {best_name}', line=dict(dash='dash')))
fig.update_layout(title=f"Melhor modelo: {best_name} — Previsão ({horizon} meses)", xaxis_title='Data', yaxis_title='Saldo médio')
st.plotly_chart(fig, use_container_width=True)

# Prepare numeric forecast for download (formatted)
forecast_vals = best.get('pred_full') if 'pred_full' in best else best.get('pred_full_future')
if forecast_vals is None:
    # try other keys
    forecast_vals = best.get('pred_full_future', np.zeros(horizon))
forecast_vals = np.array(forecast_vals).astype(float)
df_download = pd.DataFrame({'date': future_dates, 'forecast': forecast_vals})
# no large displays; provide download
csv = df_download.to_csv(index=False).encode('utf-8')
st.download_button("📥 Baixar forecast (CSV)", data=csv, file_name=f"forecast_{chosen_code}_{best_name}.csv", mime="text/csv")

st.success("Previsão gerada com sucesso.")
