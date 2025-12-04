import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# ---------------------------------------------------------------------
# Estilo moderno Jobin
# ---------------------------------------------------------------------
def css():
    st.markdown("""
    <style>
        .main > div {
            background: linear-gradient(135deg, #EC008C 0%, #673AB7 100%);
            padding: 25px;
            border-radius: 14px;
        }
        h1 {
            color: white !important;
            font-size: 34px;
            font-weight: 900;
        }
        .indicador-box {
            background: rgba(255,255,255,0.25);
            padding: 18px;
            border-radius: 12px;
            text-align: center;
            font-weight: 600;
            color: white;
            font-size: 1.05rem;
        }
        .tendencia {
            padding: 14px;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1.2rem;
            text-align: center;
            margin-top: 15px;
        }
        .footer {
            text-align: center;
            color: #eee;
            margin-top: 30px;
        }
    </style>
    """, unsafe_allow_html=True)

css()

st.set_page_config(page_title="Jobin - Salários & Mercado", layout="centered")

st.title("🔎 Jobin Inteligente - Salários & Tendências do Mercado")
st.write("Pesquise uma profissão e visualize o futuro dela no Brasil.")

@st.cache_data
def load():
    return pd.read_csv("cache_Jobin1.csv")

df = load()

termo = st.text_input("Digite parte do nome da profissão:", placeholder="Ex: Analista")

if termo:
    resultados = df[df["descricao"].str.contains(termo, case=False, na=False)]

    if resultados.empty:
        st.warning("Nenhuma profissão encontrada.")
    else:
        opcao = st.selectbox("Selecione o CBO:", resultados.apply(lambda x: f"{x['codigo']} - {x['descricao']}", axis=1))

        codigo = int(opcao.split(" - ")[0])
        info = resultados[resultados["codigo"] == codigo].iloc[0]

        st.subheader(f"{info['descricao']} • CBO {codigo}")

        # Indicadores
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div class='indicador-box'>💰<br>R$ {info['salario_medio_atual']:.2f}<br><small>Salário Médio</small></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='indicador-box'>🧠<br>{info['modelo_vencedor']}<br><small>Modelo</small></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='indicador-box'>📊<br>{info['score']:.3f}<br><small>Score</small></div>", unsafe_allow_html=True)

        # Tendência do mercado (nova lógica)
        tendencia_raw = str(info.get("tendencia_mercado", "")).lower()
        if "alta" in tendencia_raw:
            icon, cor = "🔥", "#00e676"
        elif "baixa" in tendencia_raw:
            icon, cor = "⚠️", "#ff5252"
        elif "estável" in tendencia_raw:
            icon, cor = "ℹ️", "#2196F3"
        else:
            icon, cor = "📌", "#ffffff"

        col4.markdown(
            f"<div class='indicador-box' style='background:{cor}cc;'>{icon}<br>{info['tendencia_mercado']}<br><small>Mercado</small></div>",
            unsafe_allow_html=True
        )

        # Projeções salariais
        anos = ["+5 anos", "+10 anos", "+15 anos", "+20 anos"]
        valores = [info["previsao_5"], info["previsao_10"], info["previsao_15"], info["previsao_20"]]

        crescimento = ((valores[-1] - valores[0]) / valores[0]) * 100

        if crescimento > 15:
            mensagem = f"🚀 Crescimento Acelerado ({crescimento:.1f}%)"
            cor_t = "#00e676"
        elif crescimento > 2:
            mensagem = f"📈 Crescimento Moderado ({crescimento:.1f}%)"
            cor_t = "#ffeb3b"
        elif crescimento > -2:
            mensagem = f"⚖️ Estável ({crescimento:.1f}%)"
            cor_t = "#ffffff"
        else:
            mensagem = f"📉 Queda Salarial ({crescimento:.1f}%)"
            cor_t = "#ff5252"

        fig = go.Figure(go.Scatter(x=anos, y=valores, mode="lines+markers", line=dict(width=4)))
        fig.update_layout(title="Evolução Salarial", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"<div class='tendencia' style='background:{cor_t};'>{mensagem}</div>",
            unsafe_allow_html=True
        )

st.markdown("<div class='footer'>© 2025 Jobin Analytics</div>", unsafe_allow_html=True)
