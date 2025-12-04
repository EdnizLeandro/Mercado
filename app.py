import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import math

# ===================== CONFIGURAÇÃO DA PÁGINA =====================
st.set_page_config(
    page_title="Jobin — Salários & Tendências",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== CSS & VISUAL =====================
st.markdown(
    """
    <style>
    /* Body & header */
    .reportview-container .main {
        background: #f5f7fb;
        padding-top: 12px;
        padding-bottom: 30px;
    }
    .title-banner {
        background: linear-gradient(90deg,#7b2ff7 0%, #f107a3 100%);
        padding: 18px 22px;
        border-radius: 12px;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 6px 18px rgba(23,0,102,0.12);
        margin-bottom: 18px;
    }
    .title-banner h1 {
        margin: 0;
        font-size: 20px;
        font-weight: 900;
        color: white;
    }
    .subtitle {
        margin: 0;
        color: #f1e7ff;
        opacity: 0.95;
        font-size: 13px;
    }

    /* Cards */
    .card {
        background: rgba(255,255,255,0.7);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 6px 18px rgba(15,15,20,0.04);
        min-height: 110px;
    }
    .card .icon {
        font-size: 26px;
        margin-bottom: 6px;
    }
    .card .value {
        font-size: 18px;
        font-weight: 800;
        color: #111827;
    }
    .card .label {
        display:block;
        font-size: 12px;
        color: #6b7280;
        margin-top: 6px;
        font-weight:600;
    }

    /* Tendência badge */
    .trend-badge {
        padding: 10px 12px;
        border-radius: 10px;
        color: white;
        font-weight: 700;
        display: inline-block;
        font-size: 14px;
    }

    /* Small note */
    .muted {
        font-size: 12px;
        color: #6b7280;
    }

    /* Footer */
    .footer {
        text-align:center;
        color:#9aa0b4;
        font-size:13px;
        margin-top:30px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ===================== BANNER / TÍTULO =====================
st.markdown(
    """
    <div class="title-banner">
        <div>
            <h1>🔎 Jobin Inteligente — Salários & Tendências do Mercado</h1>
            <div class="subtitle">Pesquise profissões, veja projeções salariais e tendência de mercado — informações claras para decisões de carreira.</div>
        </div>
        <div style="text-align:right;">
            <div style="font-size:13px;color:#fff;opacity:0.9;">© 2025 Jobin Analytics</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("**Busque por profissão (nome parcial ou completo) e selecione a opção desejada.**")

# ===================== CARREGAMENTO DE DADOS =====================
@st.cache_data
def carregar_dados(path="cache_Jobin1.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None

df = carregar_dados()

# ===================== INTERAÇÃO: BUSCA E SELEÇÃO =====================
if df is None:
    st.error("Base de dados não carregada. Verifique o arquivo 'cache_Jobin1.csv'.")
    st.stop()

col_search, col_help = st.columns([3,1])
with col_search:
    termo = st.text_input("🔎 Digite parte do nome da profissão:", placeholder="Ex.: analista, enfermeiro, pintor")
with col_help:
    st.markdown("<div class='muted'>Dica: Use palavras-chave — ex: 'analista' ou 'auxiliar'</div>", unsafe_allow_html=True)

if termo:
    resultados = df[df["descricao"].str.contains(termo, case=False, na=False)].copy()
    if resultados.empty:
        st.warning("Nenhuma profissão encontrada. Tente outro termo.")
    else:
        st.success(f"{resultados.shape[0]} resultados encontrados")
        escolha = st.selectbox(
            "Selecione a profissão (CBO - descrição):",
            results := resultados.apply(lambda x: f"{int(x['codigo'])} - {x['descricao']}", axis=1).tolist()
        )

        # Extrair CBO e linha
        cbo_selecionado = int(escolha.split(" - ")[0])
        info = resultados[resultados["codigo"] == cbo_selecionado].iloc[0]

        # ===================== CABEÇALHO DA PROFISSÃO =====================
        st.markdown(f"### {info['descricao']}  •  CBO {int(info['codigo'])}")
        st.markdown(f"<div class='muted'>Salário atual e projeções automáticas — dados de base Jobin + Novo CAGED</div>", unsafe_allow_html=True)
        st.write("")  # espaço

        # ===================== CARDS DE INDICADORES =====================
        c1, c2, c3, c4 = st.columns(4, gap="large")

        # ícones consistentes
        icon_salary = "💰"
        icon_model = "🧠"
        icon_score = "📊"
        icon_market = "📈"

        # Formatação dos valores seguros
        try:
            salario_atual = float(info.get("salario_medio_atual", 0.0))
        except:
            salario_atual = 0.0
        try:
            score_val = float(info.get("score", 0.0))
        except:
            score_val = 0.0
        modelo_vencedor = str(info.get("modelo_vencedor", "—"))

        # Card Salário
        c1.markdown(
            f"""
            <div class="card">
                <div class="icon">{icon_salary}</div>
                <div class="value">R$ {salario_atual:,.2f}</div>
                <span class="label">Salário Médio</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Card Modelo
        c2.markdown(
            f"""
            <div class="card">
                <div class="icon">{icon_model}</div>
                <div class="value">{modelo_vencedor}</div>
                <span class="label">Modelo de Previsão</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Card Score
        c3.markdown(
            f"""
            <div class="card">
                <div class="icon">{icon_score}</div>
                <div class="value">{score_val:.3f}</div>
                <span class="label">Score do Modelo</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Tendência do Mercado (campo vindo do CSV)
        tendencia_raw = str(info.get("tendencia_mercado", "")).strip()
        tendencia_lower = tendencia_raw.lower()
        # Mapear para ícone e cor
        if "alta" in tendencia_lower or "aumento" in tendencia_lower or "cres" in tendencia_lower:
            market_icon = "📈"
            market_color = "#16a34a"  # verde
        elif "baixa" in tendencia_lower or "queda" in tendencia_lower or "redu" in tendencia_lower:
            market_icon = "📉"
            market_color = "#ef4444"  # vermelho
        elif "est" in tendencia_lower or "estável" in tendencia_lower or "estabilidade" in tendencia_lower:
            market_icon = "⚖️"
            market_color = "#0ea5e9"  # azul
        else:
            market_icon = "📌"
            market_color = "#8b5cf6"  # roxo neutro

        c4.markdown(
            f"""
            <div class="card">
                <div class="icon">{market_icon}</div>
                <div class="value">{tendencia_raw if tendencia_raw else 'Informação N/A'}</div>
                <span class="label">Tendência do Mercado</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")  # espaço

        # ===================== PROJEÇÃO SALARIAL e CÁLCULOS DE TENDÊNCIA =====================
        st.markdown("#### 📊 Projeção Salarial — Horizontes: +5 / +10 / +15 / +20 anos")

        # Ler projeções (fallback para 0.0 se missing)
        def to_float_safe(x):
            try:
                return float(x)
            except:
                return 0.0

        p5  = to_float_safe(info.get("previsao_5", 0.0))
        p10 = to_float_safe(info.get("previsao_10", 0.0))
        p15 = to_float_safe(info.get("previsao_15", 0.0))
        p20 = to_float_safe(info.get("previsao_20", 0.0))

        anos = ["+5 anos", "+10 anos", "+15 anos", "+20 anos"]
        proj = [p5, p10, p15, p20]

        # Se valores inválidos (zeros), evitar divisão por zero
        valid_base = p5 if p5 > 0 else (salario_atual if salario_atual > 0 else 1)

        # Percentual entre p5 e p20 (coerência com gráfico)
        pct_5_to_20 = ((p20 - p5) / p5 * 100) if p5 > 0 else 0.0

        # Percentual entre salário atual e p20 (20 anos)
        pct_now_to_20 = ((p20 - salario_atual) / salario_atual * 100) if salario_atual > 0 else 0.0

        # CAGR anual estimado (entre +5 e +20 => n=15 anos)
        try:
            cagr_5_20 = (p20 / p5) ** (1 / 15) - 1 if (p5 > 0 and p20 > 0) else 0.0
        except:
            cagr_5_20 = 0.0

        # CAGR anual entre agora e +20 (n = 20)
        try:
            cagr_now_20 = (p20 / salario_atual) ** (1 / 20) - 1 if (salario_atual > 0 and p20 > 0) else 0.0
        except:
            cagr_now_20 = 0.0

        # Definir rótulo de tendência dependente de pct_5_to_20 (coerente com gráfico)
        # thresholds ajustáveis
        if pct_5_to_20 >= 15:
            trend_label = "Crescimento Acelerado"
            trend_icon = "🚀"
            trend_color = "#16a34a"
        elif pct_5_to_20 >= 2:
            trend_label = "Tendência de Crescimento Positiva"
            trend_icon = "📈"
            trend_color = "#22c55e"
        elif -2 <= pct_5_to_20 < 2:
            trend_label = "Estabilidade Projetada"
            trend_icon = "⚖️"
            trend_color = "#64748b"
        elif pct_5_to_20 < -2:
            trend_label = "Tendência de Queda"
            trend_icon = "📉"
            trend_color = "#ef4444"
        else:
            trend_label = "Neutro"
            trend_icon = "📌"
            trend_color = "#8b5cf6"

        # Montar texto profissional coerente
        # Ex.: "Tendência de Crescimento Positiva (+12,2% entre +5 e +20 anos — CAGR +0,76% a.a.)"
        pct_display = f"{pct_5_to_20:+.1f}%"
        cagr_display_percent = cagr_5_20 * 100
        cagr_display = f"{cagr_display_percent:+.2f}% a.a."

        trend_text_line = f"{trend_icon} {trend_label} ({pct_display} entre +5 e +20 anos — CAGR {cagr_display})"
        # Também um subtítulo explicativo curto
        trend_subtext = "Projeção baseada na curva salarial prevista; CAGR = taxa média anual composta."

        # ===================== GRÁFICO (Plotly) =====================
        # Escolher cor da linha conforme sinal
        line_color = trend_color

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=anos,
                y=proj,
                mode="lines+markers",
                marker=dict(size=10),
                line=dict(width=3, color=line_color),
                hovertemplate="%{x}<br>Salário: R$ %{y:,.2f}<extra></extra>"
            )
        )
        fig.update_layout(
            margin=dict(t=30, r=20, l=40, b=20),
            xaxis_title="Horizonte",
            yaxis_title="Salário (R$)",
            template="plotly_white",
            height=420
        )

        st.plotly_chart(fig, use_container_width=True)

        # ===================== EXIBIÇÃO DA TENDÊNCIA (badge profissional) =====================
        st.markdown(
            f"""<div style="margin-top:8px;">
                    <span class="trend-badge" style="background:{trend_color};">{trend_text_line}</span>
                </div>
                <div style="margin-top:6px;"><span class="muted">{trend_subtext}</span></div>
            """,
            unsafe_allow_html=True
        )

        st.write("")  # espaçamento

        # ===================== DETALHES ADICIONAIS (opcional) =====================
        # Mostra resumo numérico coerente
        col_a, col_b, col_c = st.columns([1,1,2])
        with col_a:
            st.markdown(f"**Variação (+5 → +20):**<br><span class='muted'>{pct_display}</span>", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"**CAGR (+5 → +20):**<br><span class='muted'>{cagr_display}</span>", unsafe_allow_html=True)
        with col_c:
            st.markdown(
                f"**Variação (Agora → +20 anos):**<br><span class='muted'>{pct_now_to_20:+.1f}% (estimada)</span>",
                unsafe_allow_html=True
            )

        # ===================== TENDÊNCIA DO MERCADO (mais detalhada) =====================
        market_note = ""
        if tendencia_raw:
            market_note = f"A nota de mercado registrada: {tendencia_raw}."
        else:
            market_note = "Sem descrição detalhada de demanda no registro."

        st.markdown(f"<div class='muted' style='margin-top:10px;'>{market_note}</div>", unsafe_allow_html=True)

# ===================== RODAPÉ =====================
st.markdown("<div class='footer'>Jobin Analytics © 2025 — Insights para decisões de carreira</div>", unsafe_allow_html=True)
