import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cargar datos ---
@st.cache_data
def load_data():
    df = pd.read_csv("metricas_curva.csv")
    if "error" in df.columns:
        df = df[df["error"].isna()]
    columnas_requeridas = ["ticker", "precio_teorico", "precio_mercado", "spread", "duracion_modificada", "convexidad"]
    columnas_presentes = [col for col in columnas_requeridas if col in df.columns]

    if len(columnas_presentes) < len(columnas_requeridas):
        st.warning("⚠️ El archivo metricas_curva.csv no contiene todas las columnas necesarias para el screener.")
        return pd.DataFrame()

    df = df.dropna(subset=columnas_requeridas)
    df["score"] = (
        0.4 * df["spread"] +
        0.3 * df["convexidad"] +
        0.3 * (1 / (1 + abs(df["duracion_modificada"] - 5)))
    )
    return df

df = load_data()
if df.empty:
    st.stop()

estilo =  """
<style>
.stApp {
    background-color: #131722;
    color: #d1d4dc;
    font-family: 'Inter', 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
}
div[data-testid="stSidebar"] {
    background-color: #1e222d;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: white !important;
}
.stButton>button {
    background-color: #2962ff;
    color: white;
    border-radius: 4px;
    padding: 0.4em 1em;
    border: none;
}
.stSlider>div, .stSelectbox>div {
    color: #f5f6fa;
}
.stDataFrame, .stTable {
    background-color: #131722;
    color: #f5f6fa;
}
.css-1v3fvcr, .css-1kyxreq, .css-1d391kg {
    color: #d1d4dc !important;
}
.block-container {
    padding: 2rem 2rem;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1e222d;
    color: #d1d4dc;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #009dff20;
}
</style>
"""

st.markdown(estilo, unsafe_allow_html=True)

st.sidebar.markdown("## BondScreener AR")
st.sidebar.markdown("**Análisis integral de bonos soberanos argentinos**")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Menú principal", ["Screener", "Cartera recomendada avanzada", "Análisis comparativo", "Datos históricos", "Cotizaciones históricas"])

if menu == "Datos históricos":
    st.markdown("## Flujos Futuros de Bonos")
    st.markdown("---")
    selected_ticker = st.selectbox("Seleccioná un bono para ver sus datos históricos", df["ticker"].unique())

    st.subheader("Flujos de Fondos")
    try:
        df_flujo = pd.read_csv(f"flujos_completo_{selected_ticker}.csv")
        df_flujo = df_flujo.dropna(subset=["fecha", "monto"])
        df_flujo["fecha"] = pd.to_datetime(df_flujo["fecha"], errors="coerce")
        df_flujo = df_flujo[df_flujo["fecha"] > pd.Timestamp.today()].sort_values("fecha")
        st.dataframe(df_flujo[["fecha", "monto"]].round(2))

        fig, ax = plt.subplots()
        ax.bar(df_flujo["fecha"], df_flujo["monto"], width=10, color="dodgerblue")
        ax.set_ylabel("Monto (USD)")
        ax.set_xlabel("Fecha")
        ax.set_title(f"Flujos futuros de {selected_ticker}")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo de flujos para {selected_ticker}. Detalle: {e}")

elif menu == "Cotizaciones históricas":
    st.markdown("## 📉 Cotizaciones Históricas de Bonos")
    st.markdown("---")
    selected_ticker = st.selectbox("Seleccioná un bono para ver su cotización histórica", df["ticker"].unique())
    try:
        df_hist = pd.read_csv(f"cotizaciones_completo_{selected_ticker}.csv")
        df_hist["fecha"] = pd.to_datetime(df_hist["fecha"], errors="coerce")
        df_hist = df_hist.sort_values("fecha")

        col_hist = st.selectbox("Elegí la variable a graficar", ["cierre", "apertura", "max", "min"])
        fig, ax = plt.subplots()
        ax.plot(df_hist["fecha"], df_hist[col_hist], label=col_hist.capitalize())
        ax.set_title(f"{col_hist.capitalize()} de {selected_ticker} en el tiempo")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio (USD)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo de cotizaciones históricas para {selected_ticker}. Detalle: {e}")

    # --- Cotizaciones históricas ---
    st.subheader("Cotizaciones Históricas")
    try:
        df_hist = pd.read_csv(f"cotizaciones_completo_{selected_ticker}.csv")
        df_hist["fecha"] = pd.to_datetime(df_hist["fecha"], errors="coerce")
        df_hist = df_hist.sort_values("fecha")

        col_hist = st.selectbox("Elegí la variable a graficar", ["cierre", "apertura", "max", "min"])
        fig, ax = plt.subplots()
        ax.plot(df_hist["fecha"], df_hist[col_hist], label=col_hist.capitalize())
        ax.set_title(f"{col_hist.capitalize()} de {selected_ticker} en el tiempo")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio (USD)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo de cotizaciones históricas para {selected_ticker}. Detalle: {e}")

elif menu == "Screener":
    st.sidebar.title("Filtros del Screener")
    tir_min = st.sidebar.slider("TIR mínima (%)", 0.0, 30.0, 0.0, 0.5)
    tir_max = st.sidebar.slider("TIR máxima (%)", 0.0, 30.0, 20.0, 0.5)
    spread_min = st.sidebar.slider("Spread mínimo ($)", -10.0, 10.0, -5.0, 0.5)
    spread_max = st.sidebar.slider("Spread máximo ($)", -10.0, 10.0, 5.0, 0.5)
    duracion_max = st.sidebar.slider("Duración máxima", 0.0, 20.0, 10.0, 0.5)

    df_filtrado = df[
        (df["precio_teorico"].notna()) &
        (df["precio_mercado"].notna()) &
        (df["spread"] >= spread_min) & (df["spread"] <= spread_max) &
        (df["duracion_modificada"] <= duracion_max)
    ]

    st.markdown("## Screener de Bonos Argentinos")
    st.markdown("---")
    st.dataframe(df_filtrado.sort_values("score", ascending=False), use_container_width=True)

    st.subheader("TIR vs Duración Modificada")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df_filtrado, x="duracion_modificada", y="precio_teorico", hue="spread", palette="coolwarm", ax=ax1)
    ax1.set_xlabel("Duración Modificada")
    ax1.set_ylabel("Precio Teórico")
    st.pyplot(fig1)
   
elif menu == "Cartera recomendada avanzada":
    st.markdown("## Constructor de Cartera Avanzada")
    st.markdown("---")
    perfil = st.selectbox("Seleccioná el perfil de inversión", ["spread", "conservador", "convexidad", "mixto"])
    n_bonos = st.slider("Cantidad de bonos a incluir", 3, min(20, len(df)), 6)

    if perfil == "spread":
        df_sel = df[df["spread"] > 0].sort_values("spread", ascending=False).head(n_bonos)
    elif perfil == "conservador":
        df_sel = df[(df["spread"] > 0) & (df["duracion_modificada"] < 6)]
        df_sel = df_sel.sort_values(["duracion_modificada", "spread"], ascending=[True, False]).head(n_bonos)
    elif perfil == "convexidad":
        df_sel = df[df["spread"] > 0].sort_values("convexidad", ascending=False).head(n_bonos)
    else:
        df_sel = df.sort_values("score", ascending=False).head(n_bonos)

    df_sel["peso"] = df_sel["score"] / df_sel["score"].sum()

    st.subheader("📋 Cartera generada")
    st.dataframe(df_sel[["ticker", "spread", "duracion_modificada", "convexidad", "score", "peso"]].round(4))

    fig3, ax3 = plt.subplots()
    ax3.pie(df_sel["peso"], labels=df_sel["ticker"], autopct="%1.1f%%", startangle=90)
    ax3.axis("equal")
    st.pyplot(fig3)

    st.download_button("📥 Descargar cartera avanzada", data=df_sel.to_csv(index=False), file_name="cartera_avanzada.csv")

elif menu == "Análisis comparativo":
    st.markdown("## Análisis Comparativo de Bonos")
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Precios", "Curvas", "Tasas", "Convexidad", "Forward", "Mapa de calor", "Ficha de bono", "Shock personalizado", "Comparación bonos", "Optimización"])

    with tab1:
        st.subheader("Comparación de Precios")
        fig, ax = plt.subplots(figsize=(12, 6))
        df_plot = df.sort_values("ticker")
        ax.plot(df_plot["ticker"], df_plot["precio_teorico"], label="Precio Teórico", marker="o")
        ax.plot(df_plot["ticker"], df_plot["precio_mercado"], label="Precio Mercado", marker="x")
        ax.plot(df_plot["ticker"], df_plot["precio_teorico"] - df_plot["spread"], label="Precio por TIR", linestyle="--")
        ax.set_xticklabels(df_plot["ticker"], rotation=90)
        ax.legend()
        ax.set_title("Precios comparados por bono")
        st.pyplot(fig)

    with tab2:
        st.subheader("Curva interpolada (lineal)")
        x = df["duracion_modificada"].values
        y = df["spread"].values
        from scipy.interpolate import interp1d
        curva = interp1d(x, y, kind='linear', fill_value='extrapolate')
        x_vals = np.linspace(x.min(), x.max(), 200)
        y_vals = curva(x_vals)
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="Curva lineal")
        ax.scatter(x, y, alpha=0.6)
        ax.set_xlabel("Duración Modificada")
        ax.set_ylabel("Spread")
        ax.legend()
        st.pyplot(fig)

    with tab3:
        st.subheader("Sensibilidad a tasas (DV01 estimado)")
        fig, ax = plt.subplots()
        ax.bar(df["ticker"], -df["dv01"], color="orange")
        ax.axhline(0, color="black")
        ax.set_ylabel("DV01 (USD)")
        ax.set_xticklabels(df["ticker"], rotation=90)
        st.pyplot(fig)

    with tab4:
        st.subheader("Convexidad vs Duración")
        fig, ax = plt.subplots()
        scatter = ax.scatter(df["duracion_modificada"], df["convexidad"], c=df["spread"], cmap="coolwarm", s=100, edgecolor='k')
        plt.colorbar(scatter, label="Spread")
        ax.set_xlabel("Duración Modificada")
        ax.set_ylabel("Convexidad")
        st.pyplot(fig)

    with tab5:
        st.subheader("Estimación de Precio Forward (6 meses)")
        try:
            df_forward = pd.read_csv("df_forward.csv")
            df_merged = df.merge(df_forward, on="ticker")
            forward_col = [c for c in df_forward.columns if "forward" in c.lower()][0]
            df_merged["retorno"] = 100 * (df_merged[forward_col] - df_merged["precio_mercado"]) / df_merged["precio_mercado"]
            df_merged = df_merged.sort_values("retorno", ascending=False)
            st.dataframe(df_merged[["ticker", "precio_mercado", forward_col, "retorno"]].round(2))
        except Exception as e:
            st.warning(f"No se pudo cargar el archivo 'df_forward.csv'. Detalle: {e}")

    with tab6:
        st.subheader("Mapa de Calor de Métricas")
        columnas_heatmap = ["spread", "dv01", "duracion_modificada", "convexidad"]
        df_heat = df.set_index("ticker")[columnas_heatmap].copy()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df_heat.T, cmap="coolwarm", annot=True, fmt=".2f")
        st.pyplot(fig)

    with tab7:
        st.subheader("Ficha Técnica de un Bono")
        selected_ticker = st.selectbox("Seleccioná un bono", df["ticker"].unique())
        bono = df[df["ticker"] == selected_ticker].round(4).T
        st.dataframe(bono.rename(columns={bono.columns[0]: "Valor"}))

    with tab8:
        st.subheader("Simulación de Shock de Tasas")
        shock = st.slider("Shock de tasa (en bps)", -300, 300, 100)
        df["precio_shock"] = df["precio_teorico"] - df["dv01"] * shock
        df["impacto_pct"] = 100 * (df["precio_shock"] - df["precio_teorico"]) / df["precio_teorico"]
        st.dataframe(df[["ticker", "precio_teorico", "precio_shock", "impacto_pct"]].round(2))
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(df["ticker"], df["impacto_pct"], color="seagreen")
        ax.axhline(0, color="black")
        ax.set_ylabel("Impacto %")
        ax.set_title(f"Impacto en precio teórico con shock de {shock} bps")
        plt.xticks(rotation=90)
        st.pyplot(fig)

    with tab9:
        st.subheader("Comparación entre Bonos")
        bonos_comp = st.multiselect("Seleccioná 2 o más bonos para comparar", df["ticker"].unique(), default=df["ticker"].unique()[:2])
        df_comp = df[df["ticker"].isin(bonos_comp)].set_index("ticker")
        st.dataframe(df_comp[["precio_teorico", "precio_mercado", "spread", "duracion_modificada", "convexidad", "dv01"]].round(2))
        fig, ax = plt.subplots(figsize=(10, 5))
        df_comp[["precio_mercado", "precio_teorico"]].plot.bar(ax=ax)
        ax.set_title("Comparación de precios")
        ax.set_ylabel("USD")
        st.pyplot(fig)

    with tab10:
        st.subheader("Optimización de Cartera")
        from scipy.optimize import minimize

        max_duracion = st.slider("Duración promedio máxima", 1.0, 15.0, 8.0)
        max_peso = st.slider("Peso máximo por bono (%)", 5, 100, 30)
        n_bonos_opt = st.slider("Cantidad de bonos a evaluar", 3, min(20, len(df)), 8)

        df_opt = df.sort_values("score", ascending=False).head(n_bonos_opt).copy()
        scores = df_opt["score"].values
        duraciones = df_opt["duracion_modificada"].values

        def objective(w):
            return -np.dot(w, scores)  # maximizar score total

        def duracion_media(w):
            return np.dot(w, duraciones) - max_duracion

        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "ineq", "fun": lambda w: max_duracion - np.dot(w, duraciones)}
        )

        bounds = [(0, max_peso/100) for _ in range(n_bonos_opt)]
        w0 = np.ones(n_bonos_opt) / n_bonos_opt

        result = minimize(objective, w0, bounds=bounds, constraints=constraints)

        if result.success:
            df_opt["peso_opt"] = result.x
            st.dataframe(df_opt[["ticker", "score", "duracion_modificada", "peso_opt"]].round(4))
            fig, ax = plt.subplots()
            ax.pie(df_opt["peso_opt"], labels=df_opt["ticker"], autopct="%1.1f%%")
            ax.set_title("Distribución Óptima de Pesos")
            ax.axis("equal")
            st.pyplot(fig)
        else:
            st.error("La optimización no fue exitosa. Proba ajustar las restricciones.")

