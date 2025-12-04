import streamlit as st
import pandas as pd
import plotly.graph_objs as go

# ========== CONFIGURAÇÃO DA PÁGINA ==========
st.set_page_config(
    page_title="Dashboard Jobin | Mercado de Trabalho",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 ESTILOS PERSONALIZADOS
custom_css = """
<style>
    .main {
        background-color: #f7f9fc;
    }

    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #bbb;
    }

    h1 {
        font-weight: 900;
        background: -webkit-linear-gradient(#7b2ff7, #f107a3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Cards das métricas */
    .metric-container {
        background: linear-gradient(135deg, #7b2ff7cc, #f107a3cc);
        color: white !important;
        padding: 25px;
        border-radius: 20px;
        min-height: 130px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }

    .footer {
        font-size: 14px;
        opacity: 0.6;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ========== CABEÇALHO ==========
st.title("🔎 Jobin Inteligente — Salários & Tendências do Mercado")
st.markdown("### O futuro da sua carreira, em um clique! 🚀")
st.write(
    "Busque profissões **pelo nome completo ou parcial** "
    "(ex: *desenvolvedor*, *enfermeiro*, *motorista*) e veja projeções e tendências de mercado com base no Novo CAGED 📊"
)

# ========== CARREGAMENTO DOS DADOS ==========
@st.cache_data
def carregar_dados():
    try:
        return pd.read_csv("cache_Jobin1.csv")
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

df = carregar_dados()

# ========== BUSCA ==========
if df is not None:
    
    termo = st.text_input(
        "🔍 Pesquisar profissão:",
        placeholder="Ex: Analista"
    )

    resultado_filtro = pd.DataFrame()
    cbo_selecionado = None
    
    if termo:
        resultado_filtro = df[df['descricao'].str.contains(termo, case=False, na=False)]
        
        if resultado_filtro.empty:
            st.warning("Nenhuma profissão encontrada. Tente outro termo 👀")
        
        else:
            st.success(f"{resultado_filtro.shape[0]} profissões encontradas!")

            opcao = st.selectbox(
                "Escolha a profissão desejada:",
                [
                    f"{row['codigo']} - {row['descricao']}" 
                    for _, row in resultado_filtro.iterrows()
                ]
            )
            cbo_selecionado = int(opcao.split(" - ")[0])

    if cbo_selecionado:
        info = resultado_filtro[resultado_filtro['codigo'] == cbo_selecionado].iloc[0]

        st.subheader(f"👔 {info['descricao']} — CBO {info['codigo']}")

        # ========== CARDS DE MÉTRICAS ==========
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"<div class='metric-container'><h4>Salário Médio<br>R$ {info['salario_medio_atual']:.2f}</h4></div>",
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"<div class='metric-container'><h4>Modelo<br>{info['modelo_vencedor']}</h4></div>",
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"<div class='metric-container'><h4>Score<br>{info['score']:.3f}</h4></div>",
                unsafe_allow_html=True
            )

        # ===== Tendência Salarial Inteligente =====
        sal_atual = float(info['salario_medio_atual'])
        projecoes = [
            float(info['previsao_5']),
            float(info['previsao_10']),
            float(info['previsao_15']),
            float(info['previsao_20'])
        ]

        variacao_total = ((projecoes[-1] - sal_atual) / sal_atual) * 100

        if variacao_total >= 8:
            tendencia_label = "Crescimento Acelerado"
            tendencia_icon = "🚀"
        elif 0 < variacao_total < 8:
            tendencia_label = "Crescimento"
            tendencia_icon = "📈"
        elif -3 <= variacao_total <= 3:
            tendencia_label = "Estabilidade"
            tendencia_icon = "➖"
        elif -8 < variacao_total < -3:
            tendencia_label = "Leve Queda"
            tendencia_icon = "📉"
        else:
            tendencia_label = "Queda Acentuada"
            tendencia_icon = "⚠️"

        tendencia_html = f"""
        <div class='metric-container'>
            <div style='text-align:center;'>
                <span style='font-size:32px'>{tendencia_icon}</span><br>
                <span style='font-size:14px;font-weight:600;'>{tendencia_label}</span><br>
                <span style='font-size:11px;opacity:0.8;'>({variacao_total:.1f}% em 20 anos)</span>
            </div>
        </div>
        """

        with col4:
            st.markdown(tendencia_html, unsafe_allow_html=True)

        # ========== GRÁFICO DE PROJEÇÃO ==========
        anos = ["+5 anos", "+10 anos", "+15 anos", "+20 anos"]

        fig = go.Figure(go.Scatter(
            x=anos, y=projecoes,
            mode="lines+markers",
            marker={"size": 12},
            line={"width": 3}
        ))
        
        fig.update_layout(
            title=f"📈 Projeção Salarial para {info['descricao']}",
            xaxis_title="Horizonte de Tempo",
            yaxis_title="Salário (R$)",
            template="plotly_white",
            title_font_size=20
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"📊 **Tendência do Mercado**: {info['tendencia_mercado']}"
        )

else:
    st.error("Não foi possível carregar os dados. Verifique o arquivo CSV.")

# ========== RODAPÉ ==========
st.markdown(
    "<div class='footer' style='text-align:center;margin-top:40px;'>"
    "© 2025 Jobin Analytics — Powered by Streamlit 👨‍💻✨"
    "</div>",
    unsafe_allow_html=True
)
