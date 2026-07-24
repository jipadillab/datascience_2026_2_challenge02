"""
Dashboard EDA - Fincas y Cultivos
----------------------------------
Análisis Exploratorio de Datos (EDA) cuantitativo, cualitativo y gráfico
para un dataset de fincas agrícolas.

Columnas esperadas en el CSV:
- ID_FincaDepartamento
- Tipo_Cultivo
- Area_Hectareas
- Produccion_Anual_Ton
- Sistema_Riego_Tecnificado
- Nivel_Tecnificacion
- Precio_Venta_Por_Ton_COP
- Tipo_Suelo
- Fecha_Ultima_Auditoria

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
    page_title="EDA Fincas y Cultivos",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERIC_COLS = [
    "Area_Hectareas",
    "Produccion_Anual_Ton",
    "Precio_Venta_Por_Ton_COP",
]

CATEGORICAL_COLS = [
    "Tipo_Cultivo",
    "Sistema_Riego_Tecnificado",
    "Nivel_Tecnificacion",
    "Tipo_Suelo",
]

DATE_COL = "Fecha_Ultima_Auditoria"
ID_COL = "ID_FincaDepartamento"


# =========================================================
# FUNCIONES DE CARGA Y LIMPIEZA
# =========================================================
@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo)
    df.columns = [c.strip() for c in df.columns]

    # Conversión de tipos
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", dayfirst=True)

    # Normalizar riego tecnificado a categoría legible si viene como 0/1 o bool
    if "Sistema_Riego_Tecnificado" in df.columns:
        mapeo = {
            1: "Sí", 0: "No", True: "Sí", False: "No",
            "1": "Sí", "0": "No",
            "si": "Sí", "Si": "Sí", "SI": "Sí", "sí": "Sí",
            "no": "No", "No": "No", "NO": "No",
        }
        df["Sistema_Riego_Tecnificado"] = df["Sistema_Riego_Tecnificado"].replace(mapeo)

    return df


def calcular_kpis(df):
    total_fincas = len(df)
    area_total = df["Area_Hectareas"].sum() if "Area_Hectareas" in df else np.nan
    produccion_total = df["Produccion_Anual_Ton"].sum() if "Produccion_Anual_Ton" in df else np.nan
    precio_promedio = df["Precio_Venta_Por_Ton_COP"].mean() if "Precio_Venta_Por_Ton_COP" in df else np.nan
    rendimiento_prom = (
        (df["Produccion_Anual_Ton"] / df["Area_Hectareas"]).replace([np.inf, -np.inf], np.nan).mean()
        if "Produccion_Anual_Ton" in df and "Area_Hectareas" in df else np.nan
    )
    return total_fincas, area_total, produccion_total, precio_promedio, rendimiento_prom


# =========================================================
# SIDEBAR: CARGA DE DATOS
# =========================================================
st.sidebar.title("🌾 EDA Fincas y Cultivos")
st.sidebar.markdown("Sube tu archivo CSV para comenzar el análisis.")

archivo = st.sidebar.file_uploader("Selecciona el archivo CSV", type=["csv"])

usar_demo = False
if archivo is None:
    usar_demo = st.sidebar.checkbox("Usar datos de ejemplo (demo)", value=True)

if archivo is not None:
    df_raw = cargar_datos(archivo)
elif usar_demo:
    rng = np.random.default_rng(42)
    n = 300
    cultivos = ["Café", "Cacao", "Aguacate", "Plátano", "Caña de azúcar", "Maíz"]
    suelos = ["Franco", "Arcilloso", "Arenoso", "Limoso"]
    niveles = ["Bajo", "Medio", "Alto"]
    depas = ["Antioquia", "Cundinamarca", "Valle del Cauca", "Santander", "Tolima", "Huila"]

    df_demo = pd.DataFrame({
        ID_COL: [f"{rng.choice(depas)}-{i:04d}" for i in range(n)],
        "Tipo_Cultivo": rng.choice(cultivos, n),
        "Area_Hectareas": np.round(rng.gamma(4, 5, n), 2),
        "Produccion_Anual_Ton": np.round(rng.gamma(6, 8, n), 2),
        "Sistema_Riego_Tecnificado": rng.choice(["Sí", "No"], n, p=[0.4, 0.6]),
        "Nivel_Tecnificacion": rng.choice(niveles, n, p=[0.3, 0.45, 0.25]),
        "Precio_Venta_Por_Ton_COP": np.round(rng.normal(1_800_000, 400_000, n), -3),
        "Tipo_Suelo": rng.choice(suelos, n),
        DATE_COL: pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
    })
    df_raw = df_demo
else:
    st.title("🌾 Dashboard EDA — Fincas y Cultivos")
    st.info("👈 Sube un archivo CSV en la barra lateral, o activa los datos de ejemplo, para comenzar.")
    st.stop()

df = df_raw.copy()

# =========================================================
# SIDEBAR: FILTROS
# =========================================================
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros")

if "Tipo_Cultivo" in df.columns:
    cultivos_sel = st.sidebar.multiselect(
        "Tipo de Cultivo",
        options=sorted(df["Tipo_Cultivo"].dropna().unique()),
        default=None,
    )
    if cultivos_sel:
        df = df[df["Tipo_Cultivo"].isin(cultivos_sel)]

if "Tipo_Suelo" in df.columns:
    suelos_sel = st.sidebar.multiselect(
        "Tipo de Suelo",
        options=sorted(df["Tipo_Suelo"].dropna().unique()),
        default=None,
    )
    if suelos_sel:
        df = df[df["Tipo_Suelo"].isin(suelos_sel)]

if "Nivel_Tecnificacion" in df.columns:
    nivel_sel = st.sidebar.multiselect(
        "Nivel de Tecnificación",
        options=sorted(df["Nivel_Tecnificacion"].dropna().unique()),
        default=None,
    )
    if nivel_sel:
        df = df[df["Nivel_Tecnificacion"].isin(nivel_sel)]

if "Area_Hectareas" in df.columns and df["Area_Hectareas"].notna().any():
    min_a, max_a = float(df["Area_Hectareas"].min()), float(df["Area_Hectareas"].max())
    rango_area = st.sidebar.slider("Área (Hectáreas)", min_a, max_a, (min_a, max_a))
    df = df[df["Area_Hectareas"].between(*rango_area)]

if DATE_COL in df.columns and df[DATE_COL].notna().any():
    fmin, fmax = df[DATE_COL].min(), df[DATE_COL].max()
    rango_fecha = st.sidebar.date_input("Fecha última auditoría", (fmin.date(), fmax.date()))
    if isinstance(rango_fecha, tuple) and len(rango_fecha) == 2:
        df = df[
            (df[DATE_COL] >= pd.to_datetime(rango_fecha[0]))
            & (df[DATE_COL] <= pd.to_datetime(rango_fecha[1]))
        ]

if df.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# =========================================================
# TÍTULO Y KPIs
# =========================================================
st.title("🌾 Dashboard EDA — Fincas y Cultivos")
st.caption("Análisis exploratorio cuantitativo, cualitativo y gráfico del dataset agrícola.")

total_fincas, area_total, produccion_total, precio_prom, rend_prom = calcular_kpis(df)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Nº Fincas", f"{total_fincas:,}")
k2.metric("Área Total (Ha)", f"{area_total:,.1f}")
k3.metric("Producción Total (Ton)", f"{produccion_total:,.1f}")
k4.metric("Precio Promedio (COP/Ton)", f"${precio_prom:,.0f}")
k5.metric("Rendimiento Prom. (Ton/Ha)", f"{rend_prom:,.2f}")

st.markdown("---")

# =========================================================
# TABS PRINCIPALES
# =========================================================
tab_resumen, tab_cuanti, tab_cuali, tab_grafico, tab_datos = st.tabs(
    ["📋 Resumen General", "🔢 Análisis Cuantitativo", "🔤 Análisis Cualitativo",
     "📊 Análisis Gráfico", "🗂️ Datos"]
)

# ---------------------------------------------------------
# TAB 1: RESUMEN GENERAL
# ---------------------------------------------------------
with tab_resumen:
    st.subheader("Vista general del dataset")
    c1, c2 = st.columns([2, 1])

    with c1:
        st.write("**Primeras filas:**")
        st.dataframe(df.head(10), use_container_width=True)

    with c2:
        st.write("**Estructura del dataset**")
        info_df = pd.DataFrame({
            "Columna": df.columns,
            "Tipo de dato": df.dtypes.astype(str).values,
            "Nulos": df.isna().sum().values,
            "% Nulos": (df.isna().mean() * 100).round(2).values,
            "Únicos": [df[c].nunique() for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.markdown("**Filas duplicadas:** " + str(df.duplicated().sum()))

    if df.isna().sum().sum() > 0:
        st.write("**Mapa de valores faltantes**")
        fig_na = px.imshow(
            df.isna().T,
            aspect="auto",
            color_continuous_scale=["#2ca02c", "#d62728"],
            labels=dict(color="Faltante"),
        )
        fig_na.update_layout(height=300)
        st.plotly_chart(fig_na, use_container_width=True)
    else:
        st.success("No se detectaron valores nulos en el dataset filtrado.")

# ---------------------------------------------------------
# TAB 2: ANÁLISIS CUANTITATIVO
# ---------------------------------------------------------
with tab_cuanti:
    st.subheader("Estadística descriptiva de variables numéricas")
    cols_num_presentes = [c for c in NUMERIC_COLS if c in df.columns]

    if cols_num_presentes:
        desc = df[cols_num_presentes].describe().T
        desc["mediana"] = df[cols_num_presentes].median()
        desc["asimetria"] = df[cols_num_presentes].skew()
        desc["curtosis"] = df[cols_num_presentes].kurt()
        desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(2)
        st.dataframe(desc.round(2), use_container_width=True)

        st.markdown("---")
        st.write("**Distribución individual**")
        col_sel = st.selectbox("Selecciona una variable numérica", cols_num_presentes)

        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                df, x=col_sel, nbins=30, marginal="box",
                title=f"Distribución de {col_sel}",
                color_discrete_sequence=["#2E8B57"],
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_box = px.box(
                df, y=col_sel, points="outliers",
                title=f"Boxplot de {col_sel}",
                color_discrete_sequence=["#8B4513"],
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")
        st.write("**Matriz de correlación**")
        corr = df[cols_num_presentes].corr(numeric_only=True)
        fig_corr = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Correlación entre variables numéricas",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Detección simple de outliers (IQR)
        st.markdown("---")
        st.write("**Detección de valores atípicos (método IQR)**")
        outlier_rows = []
        for c in cols_num_presentes:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = df[(df[c] < lim_inf) | (df[c] > lim_sup)].shape[0]
            outlier_rows.append({"Variable": c, "Límite inferior": round(lim_inf, 2),
                                  "Límite superior": round(lim_sup, 2), "Nº Outliers": n_out,
                                  "% Outliers": round(n_out / len(df) * 100, 2)})
        st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("No se encontraron columnas numéricas esperadas en el dataset.")

# ---------------------------------------------------------
# TAB 3: ANÁLISIS CUALITATIVO
# ---------------------------------------------------------
with tab_cuali:
    st.subheader("Análisis de variables categóricas")
    cols_cat_presentes = [c for c in CATEGORICAL_COLS if c in df.columns]

    if cols_cat_presentes:
        col_cat_sel = st.selectbox("Selecciona una variable categórica", cols_cat_presentes)

        freq = df[col_cat_sel].value_counts(dropna=False).reset_index()
        freq.columns = [col_cat_sel, "Frecuencia"]
        freq["Porcentaje (%)"] = (freq["Frecuencia"] / freq["Frecuencia"].sum() * 100).round(2)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("**Tabla de frecuencias**")
            st.dataframe(freq, use_container_width=True, hide_index=True)
        with c2:
            fig_bar = px.bar(
                freq, x=col_cat_sel, y="Frecuencia", text="Frecuencia",
                title=f"Frecuencia de {col_cat_sel}",
                color=col_cat_sel,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.write("**Tabla cruzada (crosstab) entre dos variables categóricas**")
        c3, c4 = st.columns(2)
        with c3:
            var_a = st.selectbox("Variable A", cols_cat_presentes, index=0, key="var_a")
        with c4:
            opciones_b = [c for c in cols_cat_presentes if c != var_a] or cols_cat_presentes
            var_b = st.selectbox("Variable B", opciones_b, index=0, key="var_b")

        if var_a != var_b:
            tabla_cruzada = pd.crosstab(df[var_a], df[var_b])
            st.dataframe(tabla_cruzada, use_container_width=True)

            fig_stack = px.bar(
                df, x=var_a, color=var_b, barmode="stack",
                title=f"{var_a} vs {var_b} (conteo apilado)",
            )
            st.plotly_chart(fig_stack, use_container_width=True)
        else:
            st.info("Selecciona dos variables distintas para ver la tabla cruzada.")
    else:
        st.warning("No se encontraron columnas categóricas esperadas en el dataset.")

# ---------------------------------------------------------
# TAB 4: ANÁLISIS GRÁFICO (relaciones y series de tiempo)
# ---------------------------------------------------------
with tab_grafico:
    st.subheader("Relaciones entre variables")
    cols_num_presentes = [c for c in NUMERIC_COLS if c in df.columns]

    if len(cols_num_presentes) >= 2:
        c1, c2, c3 = st.columns(3)
        with c1:
            eje_x = st.selectbox("Eje X", cols_num_presentes, index=0)
        with c2:
            eje_y = st.selectbox("Eje Y", cols_num_presentes, index=min(1, len(cols_num_presentes) - 1))
        with c3:
            color_por = st.selectbox(
                "Colorear por",
                ["Ninguno"] + [c for c in CATEGORICAL_COLS if c in df.columns],
            )

        fig_scatter = px.scatter(
            df, x=eje_x, y=eje_y,
            color=None if color_por == "Ninguno" else color_por,
            size="Area_Hectareas" if "Area_Hectareas" in df.columns else None,
            hover_data=[ID_COL] if ID_COL in df.columns else None,
            title=f"{eje_y} vs {eje_x}",
            trendline="ols" if df[eje_x].notna().sum() > 2 else None,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    if "Produccion_Anual_Ton" in df.columns and "Tipo_Cultivo" in df.columns:
        st.write("**Producción total por tipo de cultivo**")
        prod_cultivo = df.groupby("Tipo_Cultivo", as_index=False)["Produccion_Anual_Ton"].sum()
        fig_prod = px.bar(
            prod_cultivo.sort_values("Produccion_Anual_Ton", ascending=False),
            x="Tipo_Cultivo", y="Produccion_Anual_Ton", color="Tipo_Cultivo",
            title="Producción Anual (Ton) por Tipo de Cultivo",
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    if "Nivel_Tecnificacion" in df.columns and "Precio_Venta_Por_Ton_COP" in df.columns:
        st.write("**Precio de venta según nivel de tecnificación**")
        fig_violin = px.violin(
            df, x="Nivel_Tecnificacion", y="Precio_Venta_Por_Ton_COP",
            box=True, points="all", color="Nivel_Tecnificacion",
            title="Distribución del precio por nivel de tecnificación",
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    if DATE_COL in df.columns and df[DATE_COL].notna().any():
        st.markdown("---")
        st.write("**Evolución temporal de auditorías**")
        serie = df.set_index(DATE_COL).resample("M").size().reset_index(name="Nº Auditorías")
        fig_serie = px.line(
            serie, x=DATE_COL, y="Nº Auditorías", markers=True,
            title="Auditorías realizadas por mes",
        )
        st.plotly_chart(fig_serie, use_container_width=True)

    if "Sistema_Riego_Tecnificado" in df.columns:
        st.markdown("---")
        st.write("**Proporción de fincas con riego tecnificado**")
        riego_counts = df["Sistema_Riego_Tecnificado"].value_counts().reset_index()
        riego_counts.columns = ["Sistema_Riego_Tecnificado", "Conteo"]
        fig_pie = px.pie(
            riego_counts, names="Sistema_Riego_Tecnificado", values="Conteo",
            title="Fincas con riego tecnificado",
            hole=0.4,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------
# TAB 5: DATOS (tabla completa + descarga)
# ---------------------------------------------------------
with tab_datos:
    st.subheader("Datos filtrados")
    st.dataframe(df, use_container_width=True)
    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)",
        data=csv_export,
        file_name="fincas_filtrado.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption("Dashboard EDA generado con Streamlit · Datos de fincas y cultivos agrícolas.")
