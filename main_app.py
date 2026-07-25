"""
Super Dashboard EDA — Tierra, Cultivo y Cosecha
-------------------------------------------------
Storytelling de datos agrícolas colombianos: análisis cuantitativo,
cualitativo y gráfico con identidad visual propia y alta interactividad.

Columnas esperadas en el CSV:
- ID_FincaDepartamento (o ID_Finca [+ Departamento])
- Tipo_Cultivo, Area_Hectareas, Produccion_Anual_Ton
- Sistema_Riego_Tecnificado, Nivel_Tecnificacion
- Precio_Venta_Por_Ton_COP, Tipo_Suelo, Fecha_Ultima_Auditoria

Ejecutar con:
    streamlit run main_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="Tierra, Cultivo y Cosecha | EDA",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERIC_COLS = ["Area_Hectareas", "Produccion_Anual_Ton", "Precio_Venta_Por_Ton_COP"]
CATEGORICAL_COLS = ["Tipo_Cultivo", "Sistema_Riego_Tecnificado", "Nivel_Tecnificacion", "Tipo_Suelo", "Departamento"]
DATE_COL = "Fecha_Ultima_Auditoria"
ID_COL = "ID_FincaDepartamento"

# ---------------------------------------------------------
# PALETA E IDENTIDAD VISUAL
# ---------------------------------------------------------
BG_DEEP = "#0D1F17"
BG_PANEL = "#16291F"
BG_PANEL_LIGHT = "#1E3A2B"
TEXT_MAIN = "#F5EFE0"
TEXT_MUTED = "#B9C4B4"
GOLD = "#E8B84B"
CLAY = "#C1663D"
TEAL = "#4FA6A8"
SPROUT = "#7FB069"
BORDER = "rgba(245,239,224,0.12)"

COLORWAY = [GOLD, CLAY, TEAL, SPROUT, "#8C6E4A", "#E8955C", "#6FA98C", "#C97B84"]
CORR_COLORSCALE = [[0, CLAY], [0.5, "#3A4A3E"], [1, SPROUT]]
SEQ_COLORSCALE = [[0, BG_PANEL_LIGHT], [0.5, TEAL], [1, GOLD]]

px.defaults.color_discrete_sequence = COLORWAY
px.defaults.template = "plotly_dark"


def aplicar_tema(fig, altura=None, titulo=None):
    """Aplica el tema visual de la marca a cualquier figura Plotly."""
    fig.update_layout(
        font=dict(family="Inter, sans-serif", color=TEXT_MAIN, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        title=dict(
            text=titulo if titulo else fig.layout.title.text,
            font=dict(family="Fraunces, serif", size=20, color=GOLD),
            x=0.01, xanchor="left",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_MUTED)),
        hoverlabel=dict(bgcolor=BG_PANEL_LIGHT, font_family="Inter, sans-serif",
                         font_color=TEXT_MAIN, bordercolor=GOLD),
        margin=dict(t=70, l=10, r=10, b=10),
        hovermode="closest",
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, color=TEXT_MUTED)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, color=TEXT_MUTED)
    if altura:
        fig.update_layout(height=altura)
    return fig


# =========================================================
# CSS / TIPOGRAFÍA / MARCA
# =========================================================
def inyectar_css():
    st.markdown(f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700;9..144,900&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
        .stApp {{
            background: radial-gradient(circle at 15% 0%, {BG_PANEL_LIGHT} 0%, {BG_DEEP} 45%) fixed;
            color: {TEXT_MAIN};
        }}
        h1, h2, h3 {{ font-family: 'Fraunces', serif !important; color: {TEXT_MAIN}; letter-spacing: -0.01em; }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {BG_PANEL} 0%, {BG_DEEP} 100%);
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{ color: {TEXT_MAIN} !important; }}

        /* --- Eyebrow / etiquetas de capítulo --- */
        .eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.75rem; letter-spacing: 0.18em; text-transform: uppercase;
            color: {GOLD}; margin-bottom: 0.2rem;
        }}
        .chapter-title {{ font-size: 2rem; font-weight: 700; margin-top: 0; }}
        .chapter-sub {{ color: {TEXT_MUTED}; font-size: 1rem; max-width: 760px; margin-bottom: 0.5rem; }}

        /* --- Divisor firma: tierra -> cultivo -> cosecha --- */
        .harvest-divider {{
            height: 6px; border-radius: 6px; margin: 1.6rem 0 1.8rem 0;
            background: linear-gradient(90deg, {CLAY} 0%, {SPROUT} 50%, {GOLD} 100%);
            opacity: 0.85;
        }}

        /* --- Hero --- */
        .hero-wrap {{
            padding: 2.4rem 2rem; border-radius: 18px; margin-bottom: 1.6rem;
            background: linear-gradient(135deg, {BG_PANEL_LIGHT} 0%, {BG_DEEP} 100%);
            border: 1px solid {BORDER};
            position: relative; overflow: hidden;
        }}
        .hero-wrap::after {{
            content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px;
            background: radial-gradient(circle, {GOLD}33 0%, transparent 70%);
        }}
        .hero-eyebrow {{ font-family: 'IBM Plex Mono', monospace; color: {TEAL}; letter-spacing: 0.2em;
            text-transform: uppercase; font-size: 0.8rem; }}
        .hero-title {{ font-family: 'Fraunces', serif; font-weight: 900; font-size: 2.8rem; line-height: 1.05;
            margin: 0.4rem 0 0.7rem 0; color: {TEXT_MAIN}; }}
        .hero-title span {{ color: {GOLD}; }}
        .hero-sub {{ color: {TEXT_MUTED}; font-size: 1.05rem; max-width: 700px; }}
        .hero-stat {{ font-family: 'IBM Plex Mono', monospace; font-size: 2.6rem; font-weight: 600; color: {GOLD}; }}
        .hero-stat-label {{ color: {TEXT_MUTED}; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em;}}

        /* --- KPI cards --- */
        .kpi-card {{
            background: {BG_PANEL}; border: 1px solid {BORDER}; border-left: 4px solid var(--accent, {GOLD});
            border-radius: 12px; padding: 1rem 1.1rem; transition: transform 0.18s ease, box-shadow 0.18s ease;
            height: 100%;
        }}
        .kpi-card:hover {{ transform: translateY(-4px); box-shadow: 0 10px 24px rgba(0,0,0,0.35); }}
        .kpi-icon {{ font-size: 1.4rem; }}
        .kpi-label {{ color: {TEXT_MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.3rem;}}
        .kpi-value {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.5rem; font-weight: 600; color: {TEXT_MAIN}; }}

        /* --- Insight callouts --- */
        .insight-box {{
            background: linear-gradient(90deg, {BG_PANEL_LIGHT} 0%, {BG_PANEL} 100%);
            border-left: 4px solid {TEAL}; border-radius: 10px; padding: 0.9rem 1.2rem; margin: 0.8rem 0 1.4rem 0;
            font-size: 0.96rem; color: {TEXT_MAIN};
        }}
        .insight-box b {{ color: {GOLD}; }}
        .insight-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: {TEAL};
            letter-spacing: 0.14em; text-transform: uppercase; display:block; margin-bottom: 0.25rem;}}

        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {BORDER}; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: {BG_PANEL}; border-radius: 10px 10px 0 0; color: {TEXT_MUTED};
            font-family: 'Inter', sans-serif; font-weight: 600; padding: 10px 18px; border: 1px solid {BORDER}; border-bottom: none;
        }}
        .stTabs [aria-selected="true"] {{ background-color: {GOLD} !important; color: {BG_DEEP} !important; }}

        /* --- Botones / inputs --- */
        .stButton>button {{
            background: linear-gradient(90deg, {GOLD}, {CLAY}); color: {BG_DEEP}; border: none;
            font-weight: 700; border-radius: 10px; padding: 0.5rem 1.2rem;
        }}
        .stButton>button:hover {{ filter: brightness(1.08); color: {BG_DEEP}; }}
        div[data-baseweb="select"] > div {{ background-color: {BG_PANEL}; border-color: {BORDER}; }}

        /* --- Dataframes --- */
        .stDataFrame {{ border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}

        footer {{visibility: hidden;}}
        .footer-brand {{ text-align: center; color: {TEXT_MUTED}; font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem; padding: 1.5rem 0 0.5rem 0; letter-spacing: 0.05em;}}
    </style>
    """, unsafe_allow_html=True)


def divisor():
    st.markdown('<div class="harvest-divider"></div>', unsafe_allow_html=True)


def encabezado_capitulo(numero, titulo, subtitulo):
    st.markdown(f"""
    <div class="eyebrow">Capítulo {numero}</div>
    <div class="chapter-title">{titulo}</div>
    <div class="chapter-sub">{subtitulo}</div>
    """, unsafe_allow_html=True)


def insight(texto, etiqueta="Insight automático"):
    st.markdown(f"""
    <div class="insight-box">
        <span class="insight-label">💡 {etiqueta}</span>{texto}
    </div>
    """, unsafe_allow_html=True)


def kpi_card(col, icono, label, valor, accent):
    col.markdown(f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-icon">{icono}</div>
        <div class="kpi-value">{valor}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FUNCIONES DE CARGA Y LIMPIEZA
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
            mapeo_texto = {"1": "Sí", "0": "No", "true": "Sí", "false": "No",
                           "si": "Sí", "sí": "Sí", "yes": "Sí", "no": "No"}
            df["Sistema_Riego_Tecnificado"] = texto.map(mapeo_texto).fillna(col.astype(str))

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
        "Departamento": rng.choice(depas, n),
        "Tipo_Cultivo": rng.choice(cultivos, n),
        "Area_Hectareas": np.round(rng.gamma(4, 5, n), 2),
        "Produccion_Anual_Ton": np.round(rng.gamma(6, 8, n), 2),
        "Sistema_Riego_Tecnificado": rng.choice(["Sí", "No"], n, p=[0.4, 0.6]),
        "Nivel_Tecnificacion": rng.choice(niveles, n, p=[0.3, 0.45, 0.25]),
        "Precio_Venta_Por_Ton_COP": np.round(rng.normal(1_800_000, 400_000, n), -3),
        "Tipo_Suelo": rng.choice(suelos, n),
        DATE_COL: pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
    })


REQUIRED_COLS_BASE = ["Tipo_Cultivo", "Area_Hectareas", "Produccion_Anual_Ton",
                       "Sistema_Riego_Tecnificado", "Nivel_Tecnificacion",
                       "Precio_Venta_Por_Ton_COP", "Tipo_Suelo", DATE_COL]


def validar_columnas(columnas_df):
    faltantes = [c for c in REQUIRED_COLS_BASE if c not in columnas_df]
    tiene_id = (ID_COL in columnas_df) or ("ID_Finca" in columnas_df)
    if not tiene_id:
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
# CSS
# =========================================================
inyectar_css()

# =========================================================
# SIDEBAR: CARGA DE DATOS
# =========================================================
st.sidebar.markdown("## 🌾 Tierra · Cultivo · Cosecha")
st.sidebar.caption("Sube tu archivo CSV para comenzar el análisis.")

archivo = st.sidebar.file_uploader("Selecciona el archivo CSV", type=["csv"], help="Arrastra o busca tu archivo (máx. 200MB).")
sep = st.sidebar.selectbox("Separador del CSV", [",", ";", "\t", "|"], index=0,
                            format_func=lambda s: {",": "Coma (,)", ";": "Punto y coma (;)", "\t": "Tabulación", "|": "Barra vertical (|)"}[s])
encoding = st.sidebar.selectbox("Codificación", ["utf-8", "latin-1", "utf-8-sig"], index=0)

usar_demo = False
if archivo is None:
    usar_demo = st.sidebar.checkbox("Usar datos de ejemplo (demo)", value=False)

if "df_confirmado" not in st.session_state:
    st.session_state.df_confirmado = None
    st.session_state.archivo_nombre = None

if archivo is None and not usar_demo:
    inyectar_css()
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">EDA · Fincas agrícolas de Colombia</div>
        <div class="hero-title">De la <span>tierra</span> al dato:<br>una historia de cosechas.</div>
        <div class="hero-sub">Sube tu dataset y convierte cientos de registros de fincas en una narrativa
        clara de producción, rentabilidad y tecnificación — con analítica cuantitativa, cualitativa y gráfica.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📤 Carga tu archivo CSV")
    archivo_central = st.file_uploader("Arrastra y suelta tu archivo aquí, o haz clic para buscarlo",
                                        type=["csv"], key="uploader_central")

    with st.expander("ℹ️ Columnas que debe contener el archivo"):
        st.code(", ".join([f"{ID_COL} (o ID_Finca [+ Departamento])"] + REQUIRED_COLS_BASE))

    if archivo_central is not None:
        archivo = archivo_central
    else:
        st.info("También puedes activar **'Usar datos de ejemplo (demo)'** en la barra lateral para explorar el dashboard sin subir un archivo.")
        st.stop()

if archivo is not None:
    try:
        df_raw = cargar_datos(archivo, sep=sep, encoding=encoding)
    except Exception as e:
        st.error(f"❌ No se pudo leer el archivo CSV. Detalle: {e}")
        st.stop()

    columnas_faltantes = validar_columnas(df_raw.columns)
    columnas_esperadas_ok = REQUIRED_COLS_BASE + [ID_COL, "ID_Finca", "Departamento"]
    columnas_extra = [c for c in df_raw.columns if c not in columnas_esperadas_ok]

    st.markdown("### 🔍 Vista previa del archivo cargado")
    st.write(f"**Archivo:** `{archivo.name}` — **Filas:** {df_raw.shape[0]:,} — **Columnas:** {df_raw.shape[1]}")
    st.dataframe(df_raw.head(10), use_container_width=True)

    if columnas_faltantes:
        st.error("❌ Faltan columnas requeridas: " + ", ".join(f"`{c}`" for c in columnas_faltantes))
        st.warning("Corrige el archivo y vuelve a cargarlo para continuar.")
        st.stop()
    else:
        st.success("✅ El archivo contiene todas las columnas requeridas.")
        if columnas_extra:
            st.info("ℹ️ Columnas adicionales detectadas: " + ", ".join(columnas_extra))

    continuar = st.button("🚀 Continuar al Dashboard", type="primary")
    if not continuar and st.session_state.archivo_nombre != archivo.name:
        st.stop()

    st.session_state.df_confirmado = df_raw
    st.session_state.archivo_nombre = archivo.name

elif usar_demo:
    st.session_state.df_confirmado = generar_datos_demo()
    st.session_state.archivo_nombre = "demo"

df = st.session_state.df_confirmado.copy()

# =========================================================
# SIDEBAR: FILTROS
# =========================================================
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

# =========================================================
# HERO + KPIs
# =========================================================
total, area_t, prod_t, precio_p, rend_p = calcular_kpis(df)
cultivo_top = df.groupby("Tipo_Cultivo")["Produccion_Anual_Ton"].sum().idxmax() if "Tipo_Cultivo" in df.columns else "—"

st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-eyebrow">Dashboard EDA · {total:,} fincas analizadas</div>
    <div class="hero-title">De la <span>tierra</span> al dato:<br>una historia de cosechas.</div>
    <div class="hero-sub">Este panorama recorre {total:,} fincas colombianas, sumando
    <b>{area_t:,.0f} hectáreas</b> productivas. El cultivo protagonista de esta cosecha es
    <b>{cultivo_top}</b>. Recorre los capítulos abajo para explorar la producción, la rentabilidad
    y el nivel de tecnificación del campo.</div>
    <br>
    <span class="hero-stat">{prod_t:,.0f}</span>
    <div class="hero-stat-label">Toneladas producidas en total</div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
kpi_card(k1, "🏡", "Fincas analizadas", f"{total:,}", GOLD)
kpi_card(k2, "🌍", "Área total (Ha)", f"{area_t:,.0f}", SPROUT)
kpi_card(k3, "📦", "Producción (Ton)", f"{prod_t:,.0f}", TEAL)
kpi_card(k4, "💰", "Precio prom. (COP/Ton)", f"${precio_p:,.0f}", CLAY)
kpi_card(k5, "📈", "Rendimiento (Ton/Ha)", f"{rend_p:,.2f}", GOLD)

divisor()

# =========================================================
# TABS — CAPÍTULOS DE LA HISTORIA
# =========================================================
tab_resumen, tab_cuanti, tab_cuali, tab_grafico, tab_datos = st.tabs(
    ["📋 01 · Panorama", "🔢 02 · Cuantitativo", "🔤 03 · Cualitativo", "📊 04 · Relaciones", "🗂️ Datos"]
)

# ---------------------------------------------------------
# CAPÍTULO 1: PANORAMA GENERAL
# ---------------------------------------------------------
with tab_resumen:
    encabezado_capitulo("01", "El punto de partida",
                         "Antes de sembrar conclusiones, entendemos la forma cruda del dataset: su estructura, calidad y vacíos.")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Primeras filas del dataset filtrado**")
        st.dataframe(df.head(10), use_container_width=True)
    with c2:
        info_df = pd.DataFrame({
            "Columna": df.columns, "Tipo": df.dtypes.astype(str).values,
            "Nulos": df.isna().sum().values, "% Nulos": (df.isna().mean() * 100).round(2).values,
            "Únicos": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    dup = df.duplicated().sum()
    pct_nulos = df.isna().mean().mean() * 100
    insight(f"El dataset filtrado tiene <b>{dup}</b> filas duplicadas y un promedio de "
            f"<b>{pct_nulos:.1f}%</b> de valores nulos por columna. "
            + ("La calidad de los datos es sólida para el análisis." if pct_nulos < 5 and dup == 0
               else "Vale la pena revisar la fuente antes de sacar conclusiones fuertes."))

    if df.isna().sum().sum() > 0:
        st.write("**Mapa de valores faltantes**")
        fig_na = px.imshow(df.isna().T, aspect="auto", color_continuous_scale=[[0, BG_PANEL_LIGHT], [1, CLAY]],
                            labels=dict(color="Faltante"))
        st.plotly_chart(aplicar_tema(fig_na, altura=300, titulo="Faltantes por columna"), use_container_width=True)
    else:
        st.success("✅ No se detectaron valores nulos en el dataset filtrado.")

# ---------------------------------------------------------
# CAPÍTULO 2: CUANTITATIVO
# ---------------------------------------------------------
with tab_cuanti:
    encabezado_capitulo("02", "Los números detrás del cultivo",
                         "Distribución, dispersión y correlación de las variables numéricas: área, producción y precio.")
    cols_num = [c for c in NUMERIC_COLS if c in df.columns]

    if cols_num:
        desc = df[cols_num].describe().T
        desc["mediana"] = df[cols_num].median()
        desc["asimetría"] = df[cols_num].skew()
        desc["curtosis"] = df[cols_num].kurt()
        desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
        st.dataframe(desc.round(2), use_container_width=True)

        var_mas_variable = desc["CV (%)"].idxmax()
        insight(f"<b>{var_mas_variable}</b> es la variable con mayor dispersión relativa "
                f"(CV = {desc.loc[var_mas_variable, 'CV (%)']:.1f}%), lo que sugiere fincas muy heterogéneas en ese frente.",
                "Dispersión")

        st.markdown("#### Explorador de distribución")
        col_sel = st.selectbox("Selecciona una variable numérica", cols_num)
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(df, x=col_sel, nbins=30, marginal="box", color_discrete_sequence=[GOLD])
            st.plotly_chart(aplicar_tema(fig_hist, titulo=f"Distribución de {col_sel}"), use_container_width=True)
        with c2:
            fig_box = px.box(df, y=col_sel, points="outliers", color_discrete_sequence=[CLAY])
            st.plotly_chart(aplicar_tema(fig_box, titulo=f"Boxplot de {col_sel}"), use_container_width=True)

        st.markdown("#### Matriz de correlación")
        corr = df[cols_num].corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale=CORR_COLORSCALE, zmin=-1, zmax=1)
        st.plotly_chart(aplicar_tema(fig_corr, titulo="Correlación entre variables numéricas"), use_container_width=True)

        if len(cols_num) >= 2:
            corr_abs = corr.where(~np.eye(len(corr), dtype=bool)).abs()
            par = corr_abs.stack().idxmax()
            valor_corr = corr.loc[par]
            insight(f"La relación más fuerte es entre <b>{par[0]}</b> y <b>{par[1]}</b> "
                    f"(r = {valor_corr:.2f}), {'una correlación positiva' if valor_corr > 0 else 'una correlación negativa'} "
                    f"{'notable' if abs(valor_corr) > 0.5 else 'moderada'}.", "Correlación")

        st.markdown("#### Valores atípicos (método IQR)")
        rows = []
        for c in cols_num:
            q1, q3 = df[c].quantile([0.25, 0.75]); iqr = q3 - q1
            li, ls = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = df[(df[c] < li) | (df[c] > ls)].shape[0]
            rows.append({"Variable": c, "Límite inf.": round(li, 2), "Límite sup.": round(ls, 2),
                         "Nº Outliers": n_out, "% Outliers": round(n_out / len(df) * 100, 2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No se encontraron columnas numéricas esperadas.")

# ---------------------------------------------------------
# CAPÍTULO 3: CUALITATIVO
# ---------------------------------------------------------
with tab_cuali:
    encabezado_capitulo("03", "El carácter del campo",
                         "Cultivos, suelos, tecnificación y territorio: cómo se distribuyen las categorías que definen cada finca.")
    cols_cat = [c for c in CATEGORICAL_COLS if c in df.columns]

    if cols_cat:
        col_cat = st.selectbox("Selecciona una variable categórica", cols_cat)
        freq = df[col_cat].value_counts(dropna=False).reset_index()
        freq.columns = [col_cat, "Frecuencia"]
        freq["Porcentaje (%)"] = (freq["Frecuencia"] / freq["Frecuencia"].sum() * 100).round(2)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(freq, use_container_width=True, hide_index=True)
        with c2:
            fig_bar = px.bar(freq, x=col_cat, y="Frecuencia", text="Frecuencia", color=col_cat)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(aplicar_tema(fig_bar, titulo=f"Frecuencia de {col_cat}"), use_container_width=True)

        categoria_top = freq.iloc[0]
        insight(f"<b>{categoria_top[col_cat]}</b> domina la variable <b>{col_cat}</b> con "
                f"{categoria_top['Porcentaje (%)']:.1f}% de las fincas filtradas.", "Categoría dominante")

        st.markdown("#### Tabla cruzada (crosstab)")
        c3, c4 = st.columns(2)
        with c3:
            var_a = st.selectbox("Variable A", cols_cat, index=0, key="va")
        with c4:
            opciones_b = [c for c in cols_cat if c != var_a] or cols_cat
            var_b = st.selectbox("Variable B", opciones_b, index=0, key="vb")

        if var_a != var_b:
            st.dataframe(pd.crosstab(df[var_a], df[var_b]), use_container_width=True)
            fig_stack = px.bar(df, x=var_a, color=var_b, barmode="stack")
            st.plotly_chart(aplicar_tema(fig_stack, titulo=f"{var_a} vs {var_b}"), use_container_width=True)
        else:
            st.info("Selecciona dos variables distintas para ver la tabla cruzada.")
    else:
        st.warning("No se encontraron columnas categóricas esperadas.")

# ---------------------------------------------------------
# CAPÍTULO 4: RELACIONES Y TERRITORIO
# ---------------------------------------------------------
with tab_grafico:
    encabezado_capitulo("04", "Donde todo se conecta",
                         "Cruces dinámicos entre rendimiento, precio, tecnificación y territorio — con controles interactivos.")
    cols_num = [c for c in NUMERIC_COLS if c in df.columns]

    if len(cols_num) >= 2:
        c1, c2, c3 = st.columns(3)
        with c1: eje_x = st.selectbox("Eje X", cols_num, index=0)
        with c2: eje_y = st.selectbox("Eje Y", cols_num, index=min(1, len(cols_num) - 1))
        with c3:
            color_por = st.selectbox("Colorear por", ["Ninguno"] + [c for c in CATEGORICAL_COLS if c in df.columns])
        fig_sc = px.scatter(df, x=eje_x, y=eje_y, color=None if color_por == "Ninguno" else color_por,
                             size="Area_Hectareas" if "Area_Hectareas" in df.columns else None,
                             hover_data=[ID_COL] if ID_COL in df.columns else None,
                             trendline="ols" if df[eje_x].notna().sum() > 2 else None)
        st.plotly_chart(aplicar_tema(fig_sc, titulo=f"{eje_y} vs {eje_x}"), use_container_width=True)

    divisor()

    if "Produccion_Anual_Ton" in df.columns and "Tipo_Cultivo" in df.columns:
        st.markdown("#### 🌱 Ranking dinámico por cultivo")
        metrica = st.radio("Métrica a comparar", ["Producción total (Ton)", "Área total (Ha)", "Precio promedio (COP/Ton)"],
                            horizontal=True)
        map_metrica = {"Producción total (Ton)": ("Produccion_Anual_Ton", "sum"),
                       "Área total (Ha)": ("Area_Hectareas", "sum"),
                       "Precio promedio (COP/Ton)": ("Precio_Venta_Por_Ton_COP", "mean")}
        col_m, agg_m = map_metrica[metrica]
        rank = df.groupby("Tipo_Cultivo", as_index=False)[col_m].agg(agg_m).sort_values(col_m, ascending=False)
        fig_rank = px.bar(rank, x=col_m, y="Tipo_Cultivo", orientation="h", color="Tipo_Cultivo", text_auto=".2s")
        fig_rank.update_layout(showlegend=False, yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(aplicar_tema(fig_rank, titulo=f"{metrica} por Tipo de Cultivo"), use_container_width=True)

    if "Departamento" in df.columns and "Produccion_Anual_Ton" in df.columns:
        prod_dep = df.groupby("Departamento", as_index=False)["Produccion_Anual_Ton"].sum().sort_values("Produccion_Anual_Ton", ascending=False)
        fig_dep = px.bar(prod_dep, x="Departamento", y="Produccion_Anual_Ton", color="Departamento")
        fig_dep.update_layout(showlegend=False)
        st.plotly_chart(aplicar_tema(fig_dep, titulo="Producción Anual (Ton) por Departamento"), use_container_width=True)
        top_dep = prod_dep.iloc[0]
        insight(f"<b>{top_dep['Departamento']}</b> lidera la producción territorial con "
                f"{top_dep['Produccion_Anual_Ton']:,.0f} toneladas.", "Territorio")

    if "Nivel_Tecnificacion" in df.columns and "Precio_Venta_Por_Ton_COP" in df.columns:
        st.markdown("#### 💧 Tecnificación, riego y rentabilidad")
        fig_violin = px.violin(df, x="Nivel_Tecnificacion", y="Precio_Venta_Por_Ton_COP", box=True, points="all",
                                color="Nivel_Tecnificacion")
        fig_violin.update_layout(showlegend=False)
        st.plotly_chart(aplicar_tema(fig_violin, titulo="Precio de venta según nivel de tecnificación"), use_container_width=True)

        if "Sistema_Riego_Tecnificado" in df.columns:
            precio_riego = df.groupby("Sistema_Riego_Tecnificado")["Precio_Venta_Por_Ton_COP"].mean()
            if len(precio_riego) == 2 and "Sí" in precio_riego.index and "No" in precio_riego.index:
                dif_pct = (precio_riego["Sí"] - precio_riego["No"]) / precio_riego["No"] * 100
                insight(f"Las fincas con <b>riego tecnificado</b> venden en promedio a "
                        f"${precio_riego['Sí']:,.0f} COP/Ton, un {abs(dif_pct):.1f}% "
                        f"{'más alto' if dif_pct > 0 else 'más bajo'} que las que no lo tienen.", "Riego y precio")

    if "Area_Hectareas" in df.columns and "Produccion_Anual_Ton" in df.columns and "Nivel_Tecnificacion" in df.columns:
        st.markdown("#### 🎬 Burbujas animadas: área vs producción por nivel de tecnificación")
        orden_niveles = [n for n in ["Bajo", "Medio", "Alto", "Muy Alto"] if n in df["Nivel_Tecnificacion"].unique()]
        fig_bubble = px.scatter(
            df, x="Area_Hectareas", y="Produccion_Anual_Ton",
            size="Precio_Venta_Por_Ton_COP", color="Tipo_Cultivo" if "Tipo_Cultivo" in df.columns else None,
            animation_frame="Nivel_Tecnificacion", category_orders={"Nivel_Tecnificacion": orden_niveles} if orden_niveles else None,
            size_max=40, range_x=[0, df["Area_Hectareas"].max() * 1.1], range_y=[0, df["Produccion_Anual_Ton"].max() * 1.1],
        )
        st.plotly_chart(aplicar_tema(fig_bubble, titulo="Evolución por nivel de tecnificación"), use_container_width=True)

    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        st.markdown("#### 🗓️ Línea de tiempo de auditorías")
        serie = df.set_index(DATE_COL).resample("M").size().reset_index(name="Nº Auditorías")
        fig_serie = px.line(serie, x=DATE_COL, y="Nº Auditorías", markers=True, color_discrete_sequence=[TEAL])
        fig_serie.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(aplicar_tema(fig_serie, titulo="Auditorías realizadas por mes"), use_container_width=True)

    if "Sistema_Riego_Tecnificado" in df.columns:
        rc = df["Sistema_Riego_Tecnificado"].value_counts().reset_index()
        rc.columns = ["Sistema_Riego_Tecnificado", "Conteo"]
        fig_pie = px.pie(rc, names="Sistema_Riego_Tecnificado", values="Conteo", hole=0.5,
                          color="Sistema_Riego_Tecnificado", color_discrete_map={"Sí": TEAL, "No": CLAY})
        st.plotly_chart(aplicar_tema(fig_pie, titulo="Fincas con riego tecnificado"), use_container_width=True)

# ---------------------------------------------------------
# TAB: DATOS
# ---------------------------------------------------------
with tab_datos:
    encabezado_capitulo("Anexo", "Los datos, sin filtro narrativo",
                         "Explora y descarga la tabla completa tal como quedó tras aplicar tus filtros.")
    st.dataframe(df, use_container_width=True)
    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar datos filtrados (CSV)", data=csv_export,
                        file_name="fincas_filtrado.csv", mime="text/csv")

divisor()
st.markdown('<div class="footer-brand">🌾 TIERRA · CULTIVO · COSECHA — Dashboard EDA generado con Streamlit &amp; Plotly</div>', unsafe_allow_html=True)
