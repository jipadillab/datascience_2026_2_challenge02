"""
Dashboard Inteligente — Agro Colombia + Analista IA (Groq · Llama 3.3 70B)
----------------------------------------------------------------------------
EDA completo (cuantitativo, cualitativo, gráfico) de fincas agrícolas,
con interpretaciones generadas por un LLM (vía Groq) ancladas a las cifras
reales del dataset filtrado, más un chat abierto con el "analista IA".

Ejecutar con:
    streamlit run app.py
"""

import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq, APIError, AuthenticationError

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(page_title="Agro IA | Dashboard Inteligente", page_icon="🌾", layout="wide",
                    initial_sidebar_state="expanded")

NUMERIC_COLS = ["Area_Hectareas", "Produccion_Anual_Ton", "Precio_Venta_Por_Ton_COP"]
CATEGORICAL_COLS = ["Tipo_Cultivo", "Sistema_Riego_Tecnificado", "Nivel_Tecnificacion", "Tipo_Suelo", "Departamento"]
DATE_COL = "Fecha_Ultima_Auditoria"
ID_COL = "ID_FincaDepartamento"

MODELOS_DISPONIBLES = {
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "GPT-OSS 120B (alternativa Groq)": "openai/gpt-oss-120b",
    "Qwen 3.6 27B (alternativa Groq)": "qwen/qwen3.6-27b",
}

# ---------------------------------------------------------
# PALETA
# ---------------------------------------------------------
BG_DEEP, BG_PANEL, BG_PANEL_LIGHT = "#0D1F17", "#16291F", "#1E3A2B"
TEXT_MAIN, TEXT_MUTED = "#F5EFE0", "#B9C4B4"
GOLD, CLAY, TEAL, SPROUT = "#E8B84B", "#C1663D", "#4FA6A8", "#7FB069"
BORDER = "rgba(245,239,224,0.12)"
COLORWAY = [GOLD, CLAY, TEAL, SPROUT, "#8C6E4A", "#E8955C", "#6FA98C", "#C97B84"]
CORR_COLORSCALE = [[0, CLAY], [0.5, "#3A4A3E"], [1, SPROUT]]

px.defaults.color_discrete_sequence = COLORWAY
px.defaults.template = "plotly_dark"


def aplicar_tema(fig, altura=None, titulo=None):
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=TEXT_MAIN, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", colorway=COLORWAY,
        title=dict(text=titulo if titulo else fig.layout.title.text,
                    font=dict(family="Fraunces, serif", size=20, color=GOLD), x=0.01, xanchor="left"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED)),
        hoverlabel=dict(bgcolor=BG_PANEL_LIGHT, font_family="Inter, sans-serif", font_color=TEXT_MAIN, bordercolor=GOLD),
        margin=dict(t=70, l=10, r=10, b=10),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, color=TEXT_MUTED)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, color=TEXT_MUTED)
    if altura:
        fig.update_layout(height=altura)
    return fig


def inyectar_css():
    st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,700;9..144,900&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{ background: radial-gradient(circle at 15% 0%, {BG_PANEL_LIGHT} 0%, {BG_DEEP} 45%) fixed; color: {TEXT_MAIN}; }}
        h1, h2, h3 {{ font-family: 'Fraunces', serif !important; color: {TEXT_MAIN}; }}
        section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, {BG_PANEL} 0%, {BG_DEEP} 100%); border-right: 1px solid {BORDER}; }}
        section[data-testid="stSidebar"] * {{ color: {TEXT_MAIN} !important; }}
        .eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:0.75rem; letter-spacing:0.18em; text-transform:uppercase; color:{GOLD}; }}
        .chapter-title {{ font-size:1.9rem; font-weight:700; margin-top:0; }}
        .chapter-sub {{ color:{TEXT_MUTED}; font-size:1rem; max-width:760px; margin-bottom:0.5rem; }}
        .harvest-divider {{ height:6px; border-radius:6px; margin:1.4rem 0 1.6rem 0;
            background:linear-gradient(90deg,{CLAY} 0%,{SPROUT} 50%,{GOLD} 100%); opacity:0.85; }}
        .hero-wrap {{ padding:2.2rem 2rem; border-radius:18px; margin-bottom:1.4rem;
            background:linear-gradient(135deg,{BG_PANEL_LIGHT} 0%,{BG_DEEP} 100%); border:1px solid {BORDER}; position:relative; overflow:hidden; }}
        .hero-wrap::after {{ content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px;
            background:radial-gradient(circle,{GOLD}33 0%,transparent 70%); }}
        .hero-eyebrow {{ font-family:'IBM Plex Mono',monospace; color:{TEAL}; letter-spacing:0.2em; text-transform:uppercase; font-size:0.8rem; }}
        .hero-title {{ font-family:'Fraunces',serif; font-weight:900; font-size:2.5rem; line-height:1.05; margin:0.4rem 0 0.6rem 0; }}
        .hero-title span {{ color:{GOLD}; }}
        .hero-sub {{ color:{TEXT_MUTED}; font-size:1.02rem; max-width:720px; }}
        .kpi-card {{ background:{BG_PANEL}; border:1px solid {BORDER}; border-left:4px solid var(--accent,{GOLD});
            border-radius:12px; padding:1rem 1.1rem; transition:transform .18s,box-shadow .18s; height:100%; }}
        .kpi-card:hover {{ transform:translateY(-4px); box-shadow:0 10px 24px rgba(0,0,0,.35); }}
        .kpi-label {{ color:{TEXT_MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; margin-top:.3rem; }}
        .kpi-value {{ font-family:'IBM Plex Mono',monospace; font-size:1.5rem; font-weight:600; }}
        .insight-box {{ background:linear-gradient(90deg,{BG_PANEL_LIGHT} 0%,{BG_PANEL} 100%); border-left:4px solid {TEAL};
            border-radius:10px; padding:.9rem 1.2rem; margin:.8rem 0 1rem 0; font-size:.96rem; }}
        .insight-box b {{ color:{GOLD}; }}
        .insight-label {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:{TEAL}; letter-spacing:.14em;
            text-transform:uppercase; display:block; margin-bottom:.25rem; }}
        .ia-box {{ background:linear-gradient(90deg,#2A2416 0%,{BG_PANEL} 100%); border-left:4px solid {GOLD};
            border-radius:10px; padding:1rem 1.2rem; margin:.6rem 0 1.2rem 0; font-size:.97rem; }}
        .ia-label {{ font-family:'IBM Plex Mono',monospace; font-size:.72rem; color:{GOLD}; letter-spacing:.14em;
            text-transform:uppercase; display:block; margin-bottom:.35rem; }}
        .stTabs [data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid {BORDER}; }}
        .stTabs [data-baseweb="tab"] {{ background-color:{BG_PANEL}; border-radius:10px 10px 0 0; color:{TEXT_MUTED};
            font-weight:600; padding:10px 16px; border:1px solid {BORDER}; border-bottom:none; }}
        .stTabs [aria-selected="true"] {{ background-color:{GOLD} !important; color:{BG_DEEP} !important; }}
        .stButton>button {{ background:linear-gradient(90deg,{GOLD},{CLAY}); color:{BG_DEEP}; border:none;
            font-weight:700; border-radius:10px; padding:.5rem 1.1rem; }}
        .stButton>button:hover {{ filter:brightness(1.08); color:{BG_DEEP}; }}
        footer {{visibility:hidden;}}
        .footer-brand {{ text-align:center; color:{TEXT_MUTED}; font-family:'IBM Plex Mono',monospace; font-size:.78rem; padding:1.3rem 0 .4rem 0; }}
    </style>
    """, unsafe_allow_html=True)


def divisor():
    st.markdown('<div class="harvest-divider"></div>', unsafe_allow_html=True)


def encabezado_capitulo(numero, titulo, subtitulo):
    st.markdown(f'<div class="eyebrow">Capítulo {numero}</div><div class="chapter-title">{titulo}</div>'
                f'<div class="chapter-sub">{subtitulo}</div>', unsafe_allow_html=True)


def insight(texto, etiqueta="Insight automático"):
    st.markdown(f'<div class="insight-box"><span class="insight-label">💡 {etiqueta}</span>{texto}</div>', unsafe_allow_html=True)


def kpi_card(col, icono, label, valor, accent):
    col.markdown(f'<div class="kpi-card" style="--accent:{accent}"><div style="font-size:1.4rem">{icono}</div>'
                 f'<div class="kpi-value">{valor}</div><div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)


# =========================================================
# CARGA Y LIMPIEZA DE DATOS
# =========================================================
@st.cache_data
def cargar_datos(archivo, sep=",", encoding="utf-8"):
    archivo.seek(0)
    df = pd.read_csv(archivo, sep=sep, encoding=encoding)
    df.columns = [c.strip() for c in df.columns]

    if ID_COL not in df.columns:
        if "ID_Finca" in df.columns and "Departamento" in df.columns:
            df[ID_COL] = df["ID_Finca"].astype(str) + " - " + df["Departamento"].astype(str)
        elif "ID_Finca" in df.columns:
            df[ID_COL] = df["ID_Finca"].astype(str)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if DATE_COL in df.columns:
        fechas_iso = pd.to_datetime(df[DATE_COL], format="%Y-%m-%d", errors="coerce")
        if fechas_iso.isna().mean() > 0.3:
            fechas_iso = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)
        df[DATE_COL] = fechas_iso

    if "Sistema_Riego_Tecnificado" in df.columns:
        col = df["Sistema_Riego_Tecnificado"]
        if col.dtype == bool:
            df["Sistema_Riego_Tecnificado"] = col.map({True: "Sí", False: "No"})
        else:
            texto = col.astype(str).str.strip().str.lower()
            mapeo = {"1": "Sí", "0": "No", "true": "Sí", "false": "No", "si": "Sí", "sí": "Sí", "yes": "Sí", "no": "No"}
            df["Sistema_Riego_Tecnificado"] = texto.map(mapeo).fillna(col.astype(str))
    return df


def generar_datos_demo():
    rng = np.random.default_rng(42)
    n = 300
    cultivos = ["Café", "Cacao", "Aguacate", "Plátano", "Caña de azúcar", "Maíz"]
    suelos = ["Franco", "Arcilloso", "Arenoso", "Limoso"]
    niveles = ["Bajo", "Medio", "Alto"]
    depas = ["Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Tolima", "Huila"]
    return pd.DataFrame({
        ID_COL: [f"{rng.choice(depas)}-{i:04d}" for i in range(n)],
        "Departamento": rng.choice(depas, n), "Tipo_Cultivo": rng.choice(cultivos, n),
        "Area_Hectareas": np.round(rng.gamma(4, 5, n), 2), "Produccion_Anual_Ton": np.round(rng.gamma(6, 8, n), 2),
        "Sistema_Riego_Tecnificado": rng.choice(["Sí", "No"], n, p=[0.4, 0.6]),
        "Nivel_Tecnificacion": rng.choice(niveles, n, p=[0.3, 0.45, 0.25]),
        "Precio_Venta_Por_Ton_COP": np.round(rng.normal(1_800_000, 400_000, n), -3),
        "Tipo_Suelo": rng.choice(suelos, n),
        DATE_COL: pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
    })


REQUIRED_COLS_BASE = ["Tipo_Cultivo", "Area_Hectareas", "Produccion_Anual_Ton", "Sistema_Riego_Tecnificado",
                       "Nivel_Tecnificacion", "Precio_Venta_Por_Ton_COP", "Tipo_Suelo", DATE_COL]


def validar_columnas(columnas_df):
    faltantes = [c for c in REQUIRED_COLS_BASE if c not in columnas_df]
    if not ((ID_COL in columnas_df) or ("ID_Finca" in columnas_df)):
        faltantes = [f"{ID_COL} (o ID_Finca)"] + faltantes
    return faltantes


def calcular_kpis(df):
    total = len(df)
    area = df["Area_Hectareas"].sum() if "Area_Hectareas" in df else np.nan
    prod = df["Produccion_Anual_Ton"].sum() if "Produccion_Anual_Ton" in df else np.nan
    precio = df["Precio_Venta_Por_Ton_COP"].mean() if "Precio_Venta_Por_Ton_COP" in df else np.nan
    rend = ((df["Produccion_Anual_Ton"] / df["Area_Hectareas"]).replace([np.inf, -np.inf], np.nan).mean()
            if "Produccion_Anual_Ton" in df and "Area_Hectareas" in df else np.nan)
    return total, area, prod, precio, rend


# =========================================================
# RESUMEN DE DATOS PARA ANCLAR AL LLM (grounding)
# =========================================================
def hash_df(df):
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:10]


def generar_resumen_datos(df):
    """Construye un resumen textual denso y verídico del dataset filtrado,
    para que el LLM interprete SOLO a partir de cifras reales (evita alucinaciones)."""
    total, area_t, prod_t, precio_p, rend_p = calcular_kpis(df)
    partes = [f"### Ficha general\n- Fincas en la vista actual: {total}\n"
              f"- Área total: {area_t:,.1f} Ha | Producción total: {prod_t:,.1f} Ton\n"
              f"- Precio promedio: ${precio_p:,.0f} COP/Ton | Rendimiento promedio: {rend_p:,.2f} Ton/Ha"]

    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        partes.append(f"- Rango de fechas de auditoría: {df[DATE_COL].min().date()} a {df[DATE_COL].max().date()}")

    cols_num = [c for c in NUMERIC_COLS if c in df.columns]
    if cols_num:
        desc = df[cols_num].describe().T
        desc["CV%"] = (desc["std"] / desc["mean"] * 100).round(1)
        partes.append("\n### Estadística descriptiva (numéricas)\n" + desc[["mean", "median" if "median" in desc else "50%", "std", "min", "max", "CV%"]].round(2).to_string())

    if len(cols_num) >= 2:
        corr = df[cols_num].corr(numeric_only=True)
        pares = []
        for i, a in enumerate(cols_num):
            for b in cols_num[i + 1:]:
                pares.append(f"{a} vs {b}: r={corr.loc[a, b]:.2f}")
        partes.append("\n### Correlaciones\n" + "; ".join(pares))

    for c in [c for c in CATEGORICAL_COLS if c in df.columns]:
        top = df[c].value_counts(normalize=True).head(3) * 100
        resumen_top = ", ".join(f"{idx} ({val:.1f}%)" for idx, val in top.items())
        partes.append(f"\n### Top categorías de {c}\n{resumen_top}")

    if "Tipo_Cultivo" in df.columns and "Produccion_Anual_Ton" in df.columns:
        top_cultivo = df.groupby("Tipo_Cultivo")["Produccion_Anual_Ton"].sum().sort_values(ascending=False).head(5)
        partes.append("\n### Producción total (Ton) por cultivo (top 5)\n" + top_cultivo.round(1).to_string())

    if "Departamento" in df.columns and "Produccion_Anual_Ton" in df.columns:
        top_depto = df.groupby("Departamento")["Produccion_Anual_Ton"].sum().sort_values(ascending=False).head(5)
        partes.append("\n### Producción total (Ton) por departamento (top 5)\n" + top_depto.round(1).to_string())

    if "Sistema_Riego_Tecnificado" in df.columns and "Precio_Venta_Por_Ton_COP" in df.columns:
        precio_riego = df.groupby("Sistema_Riego_Tecnificado")["Precio_Venta_Por_Ton_COP"].mean()
        partes.append("\n### Precio promedio (COP/Ton) según riego tecnificado\n" + precio_riego.round(0).to_string())

    if "Nivel_Tecnificacion" in df.columns and "Produccion_Anual_Ton" in df.columns and "Area_Hectareas" in df.columns:
        rend_tec = (df.groupby("Nivel_Tecnificacion").apply(
            lambda g: (g["Produccion_Anual_Ton"] / g["Area_Hectareas"]).replace([np.inf, -np.inf], np.nan).mean()))
        partes.append("\n### Rendimiento promedio (Ton/Ha) según nivel de tecnificación\n" + rend_tec.round(2).to_string())

    for c in cols_num:
        q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
        n_out = df[(df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)].shape[0]
        partes.append(f"\n### Outliers IQR en {c}: {n_out} ({n_out / len(df) * 100:.1f}%)")

    dup = df.duplicated().sum()
    pct_nulos = df.isna().mean().mean() * 100
    partes.append(f"\n### Calidad de datos\n- Filas duplicadas: {dup}\n- % nulos promedio por columna: {pct_nulos:.1f}%")

    return "\n".join(partes)


SYSTEM_PROMPT_BASE = """Eres un analista de datos agrícolas experto, que interpreta resultados de un dashboard EDA
de fincas colombianas para un usuario no técnico. Reglas estrictas:
- Basa TODAS tus afirmaciones numéricas ÚNICAMENTE en el "Resumen de datos" que se te entrega abajo. NUNCA inventes
  cifras, porcentajes o registros que no aparezcan ahí.
- Si te preguntan algo que el resumen no permite responder con precisión, dilo honestamente y sugiere qué filtro
  o vista del dashboard revisar para obtenerlo.
- Responde en español, en tono cercano y claro, como un analista explicándole a un gerente de finca — evita jerga
  estadística innecesaria, pero sé preciso con los números que sí tengas.
- Estructura respuestas medianamente largas con viñetas o párrafos cortos. Sé conciso: prioriza 2-4 hallazgos clave
  sobre listar todo.
- Cuando interpretes correlaciones, aclara siempre que correlación no implica causalidad.

Resumen de datos (dataset actualmente filtrado por el usuario):
{resumen}
"""


def llamar_groq(client, modelo, resumen_datos, mensajes_historial, instruccion_extra="",
                 placeholder=None, temperature=0.4, max_tokens=800):
    system_prompt = SYSTEM_PROMPT_BASE.format(resumen=resumen_datos)
    if instruccion_extra:
        mensajes = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruccion_extra}]
    else:
        mensajes = [{"role": "system", "content": system_prompt}] + mensajes_historial
    try:
        stream = client.chat.completions.create(model=modelo, messages=mensajes, temperature=temperature,
                                                  max_tokens=max_tokens, stream=True)
        texto = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            texto += delta
            if placeholder is not None:
                placeholder.markdown(f'<div class="ia-box"><span class="ia-label">🤖 Analista IA</span>{texto}▌</div>', unsafe_allow_html=True)
        if placeholder is not None:
            placeholder.markdown(f'<div class="ia-box"><span class="ia-label">🤖 Analista IA</span>{texto}</div>', unsafe_allow_html=True)
        return texto, None
    except AuthenticationError:
        return None, "API Key inválida o sin permisos. Verifica tu GROQ API Key en la barra lateral."
    except APIError as e:
        return None, f"Error de la API de Groq: {e}. Si el modelo fue dado de baja, prueba otro en la barra lateral."
    except Exception as e:
        return None, f"Error inesperado: {e}"


def boton_interpretar(clave, instruccion, client, modelo, resumen_datos):
    """Botón reutilizable que llama al LLM y cachea el resultado en session_state."""
    col_btn, _ = st.columns([1, 3])
    disparar = col_btn.button("🤖 Interpretar con IA", key=f"btn_{clave}")
    cache_key = f"ia_resp_{clave}"
    placeholder = st.empty()

    if disparar:
        if not client:
            st.warning("👈 Ingresa tu GROQ API Key en la barra lateral para usar la interpretación IA.")
        else:
            with st.spinner("El analista IA está revisando los datos..."):
                texto, error = llamar_groq(client, modelo, resumen_datos, [], instruccion_extra=instruccion, placeholder=placeholder)
            if error:
                st.error(f"❌ {error}")
            else:
                st.session_state[cache_key] = texto
    elif cache_key in st.session_state:
        placeholder.markdown(f'<div class="ia-box"><span class="ia-label">🤖 Analista IA</span>{st.session_state[cache_key]}</div>', unsafe_allow_html=True)


# =========================================================
# CSS
# =========================================================
inyectar_css()

# =========================================================
# SIDEBAR: IA + CARGA + FILTROS
# =========================================================
st.sidebar.markdown("## 🌾🤖 Agro IA")
st.sidebar.caption("Dashboard inteligente con interpretación por LLM (Groq).")

st.sidebar.subheader("🔑 Analista IA (Groq)")
api_key = st.sidebar.text_input("GROQ API Key", type="password", placeholder="gsk_...",
                                 help="Se usa solo para llamar a la API de Groq desde tu sesión.")
modelo_label = st.sidebar.selectbox("Modelo", list(MODELOS_DISPONIBLES.keys()), index=0)
modelo_id = MODELOS_DISPONIBLES[modelo_label]

with st.sidebar.expander("⚠️ Nota sobre disponibilidad del modelo"):
    st.write("Groq anunció el 17-jun-2026 la baja de `llama-3.3-70b-versatile` para cuentas no-enterprise, "
             "recomendando `openai/gpt-oss-120b` o `qwen/qwen3.6-27b`. Si falla, cambia de modelo arriba.")

client = None
if api_key:
    try:
        client = Groq(api_key=api_key)
    except Exception as e:
        st.sidebar.error(f"No se pudo inicializar Groq: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("📤 Datos")
archivo = st.sidebar.file_uploader("CSV de fincas", type=["csv"])
sep = st.sidebar.selectbox("Separador", [",", ";", "\t", "|"], index=0,
                            format_func=lambda s: {",": "Coma (,)", ";": "Punto y coma (;)", "\t": "Tabulación", "|": "Barra (|)"}[s])
encoding = st.sidebar.selectbox("Codificación", ["utf-8", "latin-1", "utf-8-sig"], index=0)
usar_demo = st.sidebar.checkbox("Usar datos de ejemplo (demo)", value=False) if archivo is None else False

if "df_confirmado" not in st.session_state:
    st.session_state.df_confirmado = None
    st.session_state.archivo_nombre = None

REQUIRED_INFO = REQUIRED_COLS_BASE + [ID_COL, "ID_Finca", "Departamento"]

if archivo is None and not usar_demo:
    inyectar_css()
    st.markdown(f"""<div class="hero-wrap"><div class="hero-eyebrow">EDA + Analista IA · Fincas agrícolas de Colombia</div>
    <div class="hero-title">Datos que se <span>explican solos</span>.</div>
    <div class="hero-sub">Sube tu dataset y explóralo con gráficos interactivos — luego pídele al Analista IA
    (Llama 3.3 70B vía Groq) que interprete cada resultado, siempre anclado a las cifras reales del dashboard.</div></div>""",
                unsafe_allow_html=True)
    st.markdown("### Carga tu archivo CSV")
    archivo_central = st.file_uploader("Arrastra y suelta o busca tu archivo", type=["csv"], key="uploader_central")
    with st.expander("ℹ️ Columnas requeridas"):
        st.code(", ".join([f"{ID_COL} (o ID_Finca [+ Departamento])"] + REQUIRED_COLS_BASE))
    if archivo_central is not None:
        archivo = archivo_central
    else:
        st.info("También puedes activar **'Usar datos de ejemplo (demo)'** en la barra lateral.")
        st.stop()

if archivo is not None:
    try:
        df_raw = cargar_datos(archivo, sep=sep, encoding=encoding)
    except Exception as e:
        st.error(f"❌ No se pudo leer el archivo CSV. Detalle: {e}")
        st.stop()

    faltantes = validar_columnas(df_raw.columns)
    extra = [c for c in df_raw.columns if c not in REQUIRED_INFO]

    st.markdown("### 🔍 Vista previa del archivo cargado")
    st.write(f"**Archivo:** `{archivo.name}` — **Filas:** {df_raw.shape[0]:,} — **Columnas:** {df_raw.shape[1]}")
    st.dataframe(df_raw.head(10), use_container_width=True)

    if faltantes:
        st.error("❌ Faltan columnas requeridas: " + ", ".join(f"`{c}`" for c in faltantes))
        st.stop()
    st.success("✅ El archivo contiene todas las columnas requeridas.")
    if extra:
        st.info("ℹ️ Columnas adicionales detectadas: " + ", ".join(extra))

    continuar = st.button("🚀 Continuar al Dashboard", type="primary")
    if not continuar and st.session_state.archivo_nombre != archivo.name:
        st.stop()
    st.session_state.df_confirmado = df_raw
    st.session_state.archivo_nombre = archivo.name
elif usar_demo:
    st.session_state.df_confirmado = generar_datos_demo()
    st.session_state.archivo_nombre = "demo"

df = st.session_state.df_confirmado.copy()

# ---------------------------------------------------------
# FILTROS
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Filtros")
if "Departamento" in df.columns:
    sel = st.sidebar.multiselect("Departamento", sorted(df["Departamento"].dropna().unique()))
    if sel: df = df[df["Departamento"].isin(sel)]
if "Tipo_Cultivo" in df.columns:
    sel = st.sidebar.multiselect("Tipo de Cultivo", sorted(df["Tipo_Cultivo"].dropna().unique()))
    if sel: df = df[df["Tipo_Cultivo"].isin(sel)]
if "Tipo_Suelo" in df.columns:
    sel = st.sidebar.multiselect("Tipo de Suelo", sorted(df["Tipo_Suelo"].dropna().unique()))
    if sel: df = df[df["Tipo_Suelo"].isin(sel)]
if "Nivel_Tecnificacion" in df.columns:
    sel = st.sidebar.multiselect("Nivel de Tecnificación", sorted(df["Nivel_Tecnificacion"].dropna().unique()))
    if sel: df = df[df["Nivel_Tecnificacion"].isin(sel)]
if "Area_Hectareas" in df.columns and df["Area_Hectareas"].notna().any():
    mn, mx = float(df["Area_Hectareas"].min()), float(df["Area_Hectareas"].max())
    rango = st.sidebar.slider("Área (Hectáreas)", mn, mx, (mn, mx))
    df = df[df["Area_Hectareas"].between(*rango)]
if DATE_COL in df.columns and df[DATE_COL].notna().any():
    fmin, fmax = df[DATE_COL].min(), df[DATE_COL].max()
    rf = st.sidebar.date_input("Fecha última auditoría", (fmin.date(), fmax.date()))
    if isinstance(rf, tuple) and len(rf) == 2:
        df = df[(df[DATE_COL] >= pd.to_datetime(rf[0])) & (df[DATE_COL] <= pd.to_datetime(rf[1]))]

if df.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

resumen_datos = generar_resumen_datos(df)

# =========================================================
# HERO + KPIs
# =========================================================
total, area_t, prod_t, precio_p, rend_p = calcular_kpis(df)
cultivo_top = df.groupby("Tipo_Cultivo")["Produccion_Anual_Ton"].sum().idxmax() if "Tipo_Cultivo" in df.columns else "—"

st.markdown(f"""<div class="hero-wrap"><div class="hero-eyebrow">Dashboard Inteligente · {total:,} fincas</div>
<div class="hero-title">Datos que se <span>explican solos</span>.</div>
<div class="hero-sub">{total:,} fincas, {area_t:,.0f} Ha, cultivo protagonista: <b>{cultivo_top}</b>.
Explora los capítulos y pídele al Analista IA que interprete cada resultado.</div></div>""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
kpi_card(k1, "🏡", "Fincas", f"{total:,}", GOLD)
kpi_card(k2, "🌍", "Área total (Ha)", f"{area_t:,.0f}", SPROUT)
kpi_card(k3, "📦", "Producción (Ton)", f"{prod_t:,.0f}", TEAL)
kpi_card(k4, "💰", "Precio prom. (COP/Ton)", f"${precio_p:,.0f}", CLAY)
kpi_card(k5, "📈", "Rendimiento (Ton/Ha)", f"{rend_p:,.2f}", GOLD)

st.markdown("<br>", unsafe_allow_html=True)
boton_interpretar("resumen_ejecutivo",
                   "Da un resumen ejecutivo de 3-4 hallazgos clave del estado general de estas fincas (producción, "
                   "rentabilidad y tecnificación), en un tono útil para un gerente agrícola.",
                   client, modelo_id, resumen_datos)

divisor()

# =========================================================
# TABS
# =========================================================
tab_resumen, tab_cuanti, tab_cuali, tab_grafico, tab_chat, tab_datos = st.tabs(
    ["📋 01·Panorama", "🔢 02·Cuantitativo", "🔤 03·Cualitativo", "📊 04·Relaciones", "💬 Chat IA", "🗂️ Datos"]
)

# ---------------------------------------------------------
# 01 PANORAMA
# ---------------------------------------------------------
with tab_resumen:
    encabezado_capitulo("01", "El punto de partida", "Estructura, calidad y vacíos del dataset filtrado.")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(df.head(10), use_container_width=True)
    with c2:
        info_df = pd.DataFrame({"Columna": df.columns, "Tipo": df.dtypes.astype(str).values,
                                 "Nulos": df.isna().sum().values, "% Nulos": (df.isna().mean() * 100).round(2).values,
                                 "Únicos": [df[c].nunique() for c in df.columns]})
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    dup = df.duplicated().sum(); pct_nulos = df.isna().mean().mean() * 100
    insight(f"El dataset filtrado tiene <b>{dup}</b> filas duplicadas y {pct_nulos:.1f}% de nulos en promedio.")
    boton_interpretar("panorama", "Interpreta la calidad general de los datos (nulos, duplicados) y qué implica "
                       "para la confiabilidad del análisis.", client, modelo_id, resumen_datos)

    if df.isna().sum().sum() > 0:
        fig_na = px.imshow(df.isna().T, aspect="auto", color_continuous_scale=[[0, BG_PANEL_LIGHT], [1, CLAY]])
        st.plotly_chart(aplicar_tema(fig_na, altura=280, titulo="Faltantes por columna"), use_container_width=True)

# ---------------------------------------------------------
# 02 CUANTITATIVO
# ---------------------------------------------------------
with tab_cuanti:
    encabezado_capitulo("02", "Los números detrás del cultivo", "Distribución, dispersión y correlación de las variables numéricas.")
    cols_num = [c for c in NUMERIC_COLS if c in df.columns]
    if cols_num:
        desc = df[cols_num].describe().T
        desc["mediana"] = df[cols_num].median(); desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
        st.dataframe(desc.round(2), use_container_width=True)

        col_sel = st.selectbox("Variable numérica", cols_num)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(aplicar_tema(px.histogram(df, x=col_sel, nbins=30, marginal="box", color_discrete_sequence=[GOLD]),
                                          titulo=f"Distribución de {col_sel}"), use_container_width=True)
        with c2:
            st.plotly_chart(aplicar_tema(px.box(df, y=col_sel, points="outliers", color_discrete_sequence=[CLAY]),
                                          titulo=f"Boxplot de {col_sel}"), use_container_width=True)

        corr = df[cols_num].corr(numeric_only=True)
        st.plotly_chart(aplicar_tema(px.imshow(corr, text_auto=".2f", color_continuous_scale=CORR_COLORSCALE, zmin=-1, zmax=1),
                                      titulo="Correlación entre variables"), use_container_width=True)

        boton_interpretar("cuantitativo", "Interpreta la distribución, dispersión (CV%) y correlaciones de las "
                           "variables numéricas. Explica qué significan en términos prácticos para el negocio agrícola, "
                           "y recuerda que correlación no implica causalidad.", client, modelo_id, resumen_datos)

# ---------------------------------------------------------
# 03 CUALITATIVO
# ---------------------------------------------------------
with tab_cuali:
    encabezado_capitulo("03", "El carácter del campo", "Cultivos, suelos, tecnificación y territorio.")
    cols_cat = [c for c in CATEGORICAL_COLS if c in df.columns]
    if cols_cat:
        col_cat = st.selectbox("Variable categórica", cols_cat)
        freq = df[col_cat].value_counts(dropna=False).reset_index()
        freq.columns = [col_cat, "Frecuencia"]; freq["Porcentaje (%)"] = (freq["Frecuencia"] / freq["Frecuencia"].sum() * 100).round(2)
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(freq, use_container_width=True, hide_index=True)
        with c2:
            fig_bar = px.bar(freq, x=col_cat, y="Frecuencia", text="Frecuencia", color=col_cat)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(aplicar_tema(fig_bar, titulo=f"Frecuencia de {col_cat}"), use_container_width=True)

        boton_interpretar("cualitativo", "Interpreta la composición categórica del dataset (cultivos, suelos, "
                           "tecnificación, departamentos): qué patrones o concentraciones destacan y qué riesgos "
                           "u oportunidades sugieren.", client, modelo_id, resumen_datos)

# ---------------------------------------------------------
# 04 RELACIONES
# ---------------------------------------------------------
with tab_grafico:
    encabezado_capitulo("04", "Donde todo se conecta", "Cruces entre rendimiento, precio, tecnificación y territorio.")
    cols_num = [c for c in NUMERIC_COLS if c in df.columns]
    if len(cols_num) >= 2:
        c1, c2, c3 = st.columns(3)
        with c1: eje_x = st.selectbox("Eje X", cols_num, index=0)
        with c2: eje_y = st.selectbox("Eje Y", cols_num, index=min(1, len(cols_num) - 1))
        with c3: color_por = st.selectbox("Colorear por", ["Ninguno"] + [c for c in CATEGORICAL_COLS if c in df.columns])
        fig_sc = px.scatter(df, x=eje_x, y=eje_y, color=None if color_por == "Ninguno" else color_por,
                             size="Area_Hectareas" if "Area_Hectareas" in df.columns else None,
                             trendline="ols" if df[eje_x].notna().sum() > 2 else None)
        st.plotly_chart(aplicar_tema(fig_sc, titulo=f"{eje_y} vs {eje_x}"), use_container_width=True)

    if "Nivel_Tecnificacion" in df.columns and "Precio_Venta_Por_Ton_COP" in df.columns:
        fig_violin = px.violin(df, x="Nivel_Tecnificacion", y="Precio_Venta_Por_Ton_COP", box=True, points="all",
                                color="Nivel_Tecnificacion")
        fig_violin.update_layout(showlegend=False)
        st.plotly_chart(aplicar_tema(fig_violin, titulo="Precio según nivel de tecnificación"), use_container_width=True)

    if "Departamento" in df.columns and "Produccion_Anual_Ton" in df.columns:
        prod_dep = df.groupby("Departamento", as_index=False)["Produccion_Anual_Ton"].sum().sort_values("Produccion_Anual_Ton", ascending=False)
        fig_dep = px.bar(prod_dep, x="Departamento", y="Produccion_Anual_Ton", color="Departamento")
        fig_dep.update_layout(showlegend=False)
        st.plotly_chart(aplicar_tema(fig_dep, titulo="Producción por Departamento"), use_container_width=True)

    boton_interpretar("relaciones", "Interpreta las relaciones clave entre tecnificación, riego, precio y "
                       "rendimiento, y el panorama territorial. Da 2-3 recomendaciones accionables para mejorar "
                       "la rentabilidad de las fincas con peor desempeño.", client, modelo_id, resumen_datos)

# ---------------------------------------------------------
# CHAT IA
# ---------------------------------------------------------
with tab_chat:
    encabezado_capitulo("💬", "Habla con el Analista IA",
                         "Pregunta lo que quieras sobre los datos filtrados actualmente — el modelo responde anclado a las cifras reales del dashboard.")

    if "chat_ia" not in st.session_state:
        st.session_state.chat_ia = []

    with st.expander("🔍 Ver el resumen de datos que recibe el modelo (contexto/grounding)"):
        st.text(resumen_datos)

    for m in st.session_state.chat_ia:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    pregunta = st.chat_input("Pregúntale al analista sobre estos datos...")
    if pregunta:
        if not client:
            st.warning("👈 Ingresa tu GROQ API Key en la barra lateral para chatear con el Analista IA.")
        else:
            st.session_state.chat_ia.append({"role": "user", "content": pregunta})
            with st.chat_message("user"):
                st.markdown(pregunta)
            with st.chat_message("assistant"):
                placeholder = st.empty()
                texto, error = None, None
                try:
                    mensajes_hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_ia]
                    system_prompt = SYSTEM_PROMPT_BASE.format(resumen=resumen_datos)
                    stream = client.chat.completions.create(
                        model=modelo_id, messages=[{"role": "system", "content": system_prompt}] + mensajes_hist,
                        temperature=0.4, max_tokens=800, stream=True)
                    texto = ""
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        texto += delta
                        placeholder.markdown(texto + "▌")
                    placeholder.markdown(texto)
                except AuthenticationError:
                    placeholder.error("❌ API Key inválida o sin permisos.")
                except APIError as e:
                    placeholder.error(f"❌ Error de la API de Groq: {e}. Prueba otro modelo en la barra lateral.")
                except Exception as e:
                    placeholder.error(f"❌ Error inesperado: {e}")
            if texto:
                st.session_state.chat_ia.append({"role": "assistant", "content": texto})

    if st.button("🗑️ Limpiar chat"):
        st.session_state.chat_ia = []
        st.rerun()

# ---------------------------------------------------------
# DATOS
# ---------------------------------------------------------
with tab_datos:
    encabezado_capitulo("Anexo", "Los datos, sin filtro narrativo", "Explora y descarga la tabla filtrada.")
    st.dataframe(df, use_container_width=True)
    st.download_button("⬇️ Descargar datos filtrados (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                        file_name="fincas_filtrado.csv", mime="text/csv")

divisor()
st.markdown('<div class="footer-brand">🌾🤖 AGRO IA — Streamlit + Plotly + Groq (Llama 3.3 70B)</div>', unsafe_allow_html=True)
