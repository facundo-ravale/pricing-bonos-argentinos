import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Crear DataFrame con bonos duales ===
data = {
    "Ticker": ["TTM26", "TTJ26", "TTS26", "TTD26"],
    "TIR": [0.2863, 0.2911, 0.2921, 0.2873],  # decimal
    "MD": [0.78, 1.07, 1.28, 1.53]            # duración modificada
}
df = pd.DataFrame(data)

# === 2. Calcular todos los spreads forward ===
resultados = []
for i in range(len(df)):
    for j in range(len(df)):
        if i >= j:
            continue
        bono_corto = df.iloc[i]
        bono_largo = df.iloc[j]
        delta_t = bono_largo["MD"] - bono_corto["MD"]
        if delta_t <= 0:
            continue
        tasa_corta = (1 + bono_corto["TIR"]) ** bono_corto["MD"]
        tasa_larga = (1 + bono_largo["TIR"]) ** bono_largo["MD"]
        tasa_forward = (tasa_larga / tasa_corta) ** (1 / delta_t) - 1
        tasa_media = (bono_corto["TIR"] + bono_largo["TIR"]) / 2
        resultados.append({
            "Bono_corto": bono_corto["Ticker"],
            "Bono_largo": bono_largo["Ticker"],
            "ΔDuración": round(delta_t, 2),
            "TIR_corto (%)": round(bono_corto["TIR"] * 100, 2),
            "TIR_largo (%)": round(bono_largo["TIR"] * 100, 2),
            "Tasa_forward (%)": round(tasa_forward * 100, 2),
            "Spread_extra (%)": round((tasa_forward - tasa_media) * 100, 2)
        })

df_spreads = pd.DataFrame(resultados).sort_values("Spread_extra (%)", ascending=False)

# === 3. Generar recomendaciones automáticas ===
recomendaciones = []
for _, row in df_spreads.iterrows():
    spread = row["Spread_extra (%)"]
    corto = row["Bono_corto"]
    largo = row["Bono_largo"]
    if spread > 0.5:
        recomendaciones.append(f"✅ LONG {largo} / SHORT {corto}  | Spread extra: {spread:.2f}%")
    elif spread < -0.5:
        recomendaciones.append(f"⚠️ SHORT {largo} / LONG {corto}  | Spread extra: {spread:.2f}%")
    else:
        recomendaciones.append(f"🔍 NEUTRAL entre {corto} y {largo}  | Spread extra: {spread:.2f}%")

# === 4. Crear heatmap ===
tickers = df["Ticker"].tolist()
spread_matrix = pd.DataFrame(index=tickers, columns=tickers, dtype=float)
for _, row in df_spreads.iterrows():
    corto = row["Bono_corto"]
    largo = row["Bono_largo"]
    spread = row["Spread_extra (%)"]
    spread_matrix.loc[corto, largo] = spread

plt.figure(figsize=(8, 6))
sns.heatmap(spread_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, cbar_kws={'label': 'Spread Extra (%)'})
plt.title("Heatmap de Spread Forward Implícito entre Bonos Duales")
plt.xlabel("Bono Largo")
plt.ylabel("Bono Corto")
plt.tight_layout()
plt.show()

# === 5. Recomendación long-only: mejor bono para estar posicionado largo ===
ranking_long = df_spreads.groupby("Bono_largo")["Spread_extra (%)"].sum().sort_values(ascending=False)
mejor_bono_long = ranking_long.idxmax()
ganancia_estimada = ranking_long.max()

print(f"\n📈 Mejor bono para posicionamiento LONG: {mejor_bono_long} "
      f"(suma de spreads recibidos: {ganancia_estimada:.2f}%)")

