import time
import glob
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess
def init_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--start-maximized")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
def scrapear_flujos_bono_rava(ticker="AL30D"):
    url = f"https://www.rava.com/perfil/{ticker}"
    print(f"🔗 Abriendo URL: {url}")

    try:
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--start-maximized")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "scroll-flujo"))
        )
        print("✅ Tabla de flujos detectada")

        tabla = driver.find_element(By.ID, "scroll-flujo")
        filas = tabla.find_elements(By.TAG_NAME, "tr")
        flujos = []
        for fila in filas:
            celdas = fila.find_elements(By.TAG_NAME, "td")
            if len(celdas) >= 4:
                fecha = celdas[0].text.strip()
                monto = celdas[3].text.strip().replace('.', '').replace(',', '.')
                if fecha and monto and monto != "-":
                    flujos.append({"fecha": fecha, "monto": float(monto)})
        print(f"📄 Se extrajeron {len(flujos)} flujos para {ticker}")

        df = pd.DataFrame(flujos)
        nombre_csv = f"flujos_{ticker}.csv"
        df.to_csv(nombre_csv, index=False)
        print(f"✅ Guardado: {nombre_csv}")

    except Exception as e:
        print(f"⚠️ Error con {ticker}: {e}")

    finally:
        try:
            driver.quit()
        except:
            pass
def scrapear_info_adicional(ticker="GD30"):
    url = f"https://www.rava.com/perfil/{ticker}"
    print(f"🔗 Abriendo URL: {url}")
    driver = init_driver()

    try:
        driver.get(url)
        time.sleep(5)

        # TIR y Duration Modificada
        tir = None
        duracion_mod = None
        try:
            tir_label = driver.find_element(By.XPATH, "//*[contains(text(), 'Tasa interna de retorno')]")
            tir = tir_label.text.split(':')[-1].strip()
            dur_label = driver.find_element(By.XPATH, "//*[contains(text(), 'Duration modificada')]")
            duracion_mod = dur_label.text.split(':')[-1].strip()
        except:
            pass

        # Prospecto
        prospecto = ""
        try:
            perfil_section = driver.find_element(By.XPATH, "//*[contains(text(), 'Perfil')]/following-sibling::*")
            prospecto = perfil_section.text.strip()
        except:
            pass

        # Cotizaciones históricas
        historico_data = []
        try:
            hist_table = driver.find_element(By.XPATH, "//*[contains(text(), 'COTIZACIONES HISTÓRICAS')]/following::table[1]")
            rows = hist_table.find_elements(By.TAG_NAME, "tr")
            for row in rows[1:]:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 6:
                    historico_data.append({
                        "fecha": cols[0].text.strip(),
                        "apertura": cols[1].text.strip(),
                        "max": cols[2].text.strip(),
                        "min": cols[3].text.strip(),
                        "cierre": cols[4].text.strip(),
                        "vol": cols[5].text.strip()
                    })
        except:
            pass

        # Guardar archivos
        df_main = pd.DataFrame([{
            "ticker": ticker,
            "tir": tir,
            "duracion_modificada": duracion_mod,
            "prospecto": prospecto
        }])
        df_main.to_csv(f"info_completa_{ticker}.csv", index=False)

        if historico_data:
            df_hist = pd.DataFrame(historico_data)
            df_hist.to_csv(f"cotizaciones_historicas_{ticker}.csv", index=False)

        print(f"✅ Datos guardados para {ticker}")

    except Exception as e:
        print(f"⚠️ Error al scrapear {ticker}: {e}")

    finally:
        driver.quit()
def scrapear_cotizacion_historica(ticker="GD30"):
    url = f"https://www.rava.com/perfil/{ticker}"
    print(f"🔗 Abriendo URL: {url}")
    driver = init_driver()

    try:
        driver.get(url)
        time.sleep(5)

        # Cargar más resultados si hay botón "Ver más"
        while True:
            try:
                ver_mas = driver.find_element(By.XPATH, "//p[contains(text(), 'Ver más')]")
                driver.execute_script("arguments[0].scrollIntoView();", ver_mas)
                ver_mas.click()
                time.sleep(1.5)
            except:
                break

        # Buscar tabla dentro de Coti-hist-c
        historico_data = []
        try:
            tabla = driver.find_element(By.CSS_SELECTOR, '#Coti-hist-c table')
            rows = tabla.find_elements(By.TAG_NAME, "tr")
            for row in rows[1:]:
                cols = row.find_elements(By.TAG_NAME, "td")
                if len(cols) >= 6:
                    historico_data.append({
                        "fecha": cols[0].text.strip(),
                        "apertura": cols[1].text.strip(),
                        "max": cols[2].text.strip(),
                        "min": cols[3].text.strip(),
                        "cierre": cols[4].text.strip(),
                        "vol": cols[5].text.strip()
                    })
        except Exception as e:
            print(f"❌ Error al extraer la tabla de cotización histórica: {e}")

        if historico_data:
            df_hist = pd.DataFrame(historico_data)
            df_hist.to_csv(f"cotizaciones_historicas_{ticker}.csv", index=False)
            print(f"✅ Cotizaciones históricas guardadas: cotizaciones_historicas_{ticker}.csv")
        else:
            print("⚠️ No se encontraron datos de cotización histórica")

    except Exception as e:
        print(f"⚠️ Error general al scrapear {ticker}: {e}")

    finally:
        driver.quit()
def unir_datasets_por_bono(ticker):
    try:
        flujos = pd.read_csv(f"flujos_{ticker}.csv")
    except:
        flujos = pd.DataFrame()

    try:
        info = pd.read_csv(f"info_completa_{ticker}.csv")
    except:
        info = pd.DataFrame()

    try:
        hist = pd.read_csv(f"cotizaciones_historicas_{ticker}.csv")
    except:
        hist = pd.DataFrame()

    # Expandir info en cada fila del flujo o cotización
    if not info.empty:
        for col in info.columns:
            if col != "ticker":
                flujos[col] = info[col][0] if not flujos.empty else None
                hist[col] = info[col][0] if not hist.empty else None

    # Guardar archivos combinados por separado
    if not flujos.empty:
        flujos.to_csv(f"flujos_completo_{ticker}.csv", index=False)
    if not hist.empty:
        hist.to_csv(f"cotizaciones_completo_{ticker}.csv", index=False)

    print(f"✅ Datos unificados para {ticker}")
def limpiar_datos_bono(ticker):
    try:
        df_flujos = pd.read_csv(f"flujos_completo_{ticker}.csv")
        df_cotiz = pd.read_csv(f"cotizaciones_completo_{ticker}.csv")
    except:
        print(f"⚠️ No se encontraron archivos completos para {ticker}")
        return

    # Convertir fechas
    df_flujos['fecha'] = pd.to_datetime(df_flujos['fecha'], dayfirst=True, errors='coerce')
    df_cotiz['fecha'] = pd.to_datetime(df_cotiz['fecha'], dayfirst=True, errors='coerce')

    # Convertir montos a float
    df_flujos['monto'] = pd.to_numeric(df_flujos['monto'], errors='coerce')
    for col in ['apertura', 'max', 'min', 'cierre', 'vol']:
        df_cotiz[col] = pd.to_numeric(df_cotiz[col].str.replace('.', '', regex=False).str.replace(',', '.', regex=False), errors='coerce')

    # Dejar prospecto solo en la primera fila
    if 'prospecto' in df_flujos.columns:
        prospecto = df_flujos['prospecto'].iloc[0]
        df_flujos['prospecto'] = ''
        df_flujos.loc[0, 'prospecto'] = prospecto
    if 'prospecto' in df_cotiz.columns:
        prospecto = df_cotiz['prospecto'].iloc[0]
        df_cotiz['prospecto'] = ''
        df_cotiz.loc[0, 'prospecto'] = prospecto

    # Guardar sobrescribiendo
    df_flujos.to_csv(f"flujos_completo_{ticker}.csv", index=False)
    df_cotiz.to_csv(f"cotizaciones_completo_{ticker}.csv", index=False)
    print(f"✅ Datos limpiados y normalizados para {ticker}")
tickers = [
    "AE38", "AE38D", "AL29", "AL29D", "AL30", "AL30C", "AL30D", "AL35", "AL35D",
    "AL41", "AL41D",  "CUAP", "DICP", "GD29", "GD29D",
    "GD30", "GD30C", "GD30D", "GD35", "GD35D", "GD38", "GD38D", "GD41",
    "GD41D", "GD46", "GD46D"
]
'''
for ticker in tickers:
    print(f"\n🟢 Scrapeando flujos de fondos de {ticker}...")
    scrapear_flujos_bono_rava(ticker)
    time.sleep(2)
for ticker in tickers:
    print(f"\n🟢 Scrapeando data de {ticker}...")
    scrapear_info_adicional(ticker)
    time.sleep(2)

for ticker in tickers:
    print(f"\n🟢 Scrapeando data de {ticker}...")
    scrapear_cotizacion_historica(ticker)
    time.sleep(2)

for ticker in tickers:
    print(f"\n🔄 Unificando datasets de {ticker}...")
    unir_datasets_por_bono(ticker)
    time.sleep(1)

for ticker in tickers:
    limpiar_datos_bono(ticker)
    time.sleep(1)
    
'''
def get_ccl():
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    import time

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--start-maximized")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    url = "https://dolarhoy.com/cotizaciondolarcontadoconliqui"
    driver.get(url)
    time.sleep(5)

    try:
        # Buscar todos los bloques con precios
        tiles = driver.find_elements(By.CSS_SELECTOR, "div.tile.is-child")
        for tile in tiles:
            try:
                topic = tile.find_element(By.CLASS_NAME, "topic").text
                if "Venta" in topic:
                    valor_div = tile.find_element(By.CLASS_NAME, "value")
                    ccl_text = valor_div.text
                    ccl = float(ccl_text.replace("$", "").replace(".", "").replace(",", "."))
                    driver.quit()
                    return ccl
            except:
                continue
        driver.quit()
        print("❌ No se encontró el bloque de 'Venta'")
        return None
    except Exception as e:
        driver.quit()
        print(f"❌ Error obteniendo CCL: {e}")
        return None
def construir_curva_tir_real(path_csv):
    df = pd.read_csv(path_csv)
    df.columns = df.columns.str.strip()
    
    if not {'Ticker', 'TIR', 'MD'}.issubset(df.columns):
        raise ValueError("El CSV debe tener columnas: 'Ticker', 'TIR', 'MD'")

    # Filtrar bonos CER (ajustar si querés incluir otros)
    df_cer = df[df['Ticker'].str.contains("TX|T2X|PARP", na=False)].copy()

    # Limpiar datos
    df_cer['TIR'] = df_cer['TIR'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False).astype(float) / 100
    df_cer['MD'] = df_cer['MD'].astype(str).str.replace(',', '.', regex=False).astype(float)
    df_cer = df_cer.sort_values('MD')

    # Interpolación lineal
    x_dm = df_cer['MD'].values
    y_tir = df_cer['TIR'].values
    curva = interp1d(x_dm, y_tir, kind='linear', fill_value="extrapolate")

    return curva
def calcular_metricas_tecnicas_bono(ticker: str, ccl: float) -> dict:
    try:
        flujos = pd.read_csv(f"flujos_completo_{ticker}.csv")
        cotizaciones = pd.read_csv(f"cotizaciones_completo_{ticker}.csv")
    except FileNotFoundError:
        return {"ticker": ticker, "error": "Archivos no encontrados"}

    try:
        tir_str = flujos['tir'].dropna().iloc[0]
        tir_anual = float(tir_str.replace('%', '').replace(',', '.')) / 100
    except:
        return {"ticker": ticker, "error": "TIR inválida"}

    flujos['fecha'] = pd.to_datetime(flujos['fecha'], errors='coerce')
    flujos = flujos[flujos['fecha'] > datetime.today()].sort_values('fecha')
    if flujos.empty:
        return {"ticker": ticker, "error": "Sin flujos futuros"}

    # Determinar frecuencia y tasa por periodo
    freq_days = flujos['fecha'].diff().dt.days.dropna().mode()[0]
    frecuencia = round(365 / freq_days)
    r_periodo = (1 + tir_anual) ** (1 / frecuencia) - 1

    hoy = datetime.today()
    flujos['t'] = (flujos['fecha'] - hoy).dt.days / (365 / frecuencia)
    flujos = flujos[flujos['monto'] > 0]
    flujos['vp'] = flujos['monto'] / (1 + r_periodo) ** flujos['t']
    precio_teorico = flujos['vp'].sum()

    # Precio de mercado
    try:
        cierre = pd.to_numeric(cotizaciones['cierre'].iloc[0], errors='coerce')
        if ticker.endswith("D"):
            precio_mercado = cierre
        else:
            precio_mercado = cierre / ccl
    except:
        return {"ticker": ticker, "error": "Cierre inválido"}

    desviacion = precio_teorico - precio_mercado

    # Duración Macaulay
    flujos['peso'] = flujos['vp'] / precio_teorico
    duracion_macaulay = (flujos['peso'] * flujos['t']).sum()

    # Duración modificada
    duracion_mod = duracion_macaulay / (1 + r_periodo)

    # Convexidad
    flujos['conv'] = flujos['peso'] * flujos['t'] * (flujos['t'] + 1)
    convexidad = flujos['conv'].sum() / (1 + r_periodo) ** 2

    # DV01
    dv01 = -duracion_mod * precio_teorico * 0.0001

    return {
        "ticker": ticker,
        "precio_teorico": round(precio_teorico, 2),
        "precio_mercado": round(precio_mercado, 2),
        "desviacion": round(desviacion, 2),
        "duracion_macaulay": round(duracion_macaulay, 4),
        "duracion_modificada": round(duracion_mod, 4),
        "convexidad": round(convexidad, 4),
        "dv01": round(dv01, 6)
    }
def scrapear_tabla_bonistas():
    url = "https://bonistas.com/bonos-cer-hoy"

    # Configuración del navegador
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(url)
    time.sleep(5)

    # Buscar todas las filas de la tabla principal
    rows = driver.find_elements(By.CSS_SELECTOR, "table.table-auto.w-full tbody tr")

    data = []
    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) >= 14:
            fila = [col.text.strip() for col in cols[:14]]
            data.append(fila)

    driver.quit()

    # Columnas manualmente extraídas de tu captura
    columnas = ["Ticker", "Precio", "Dif", "TIR", "TEM", "TNA", "MD", "Vol(M)", "Paridad", "VT", "TTir", "upTTir", "", ""]

    df = pd.DataFrame(data, columns=columnas[:len(data[0])])  # ajustar columnas si cambian
    df.to_csv("bonos_cer_bonistas.csv", index=False)
    print(df.head())
    return df
def proyectar_precio_forward(ticker, curva_real, ccl, meses_forward=6):
    try:
        flujos = pd.read_csv(f"flujos_completo_{ticker}.csv")
        flujos['fecha'] = pd.to_datetime(flujos['fecha'], errors='coerce')
        flujos = flujos[flujos['fecha'] > datetime.today()].sort_values('fecha')
        flujos = flujos[flujos['monto'] > 0]
        if flujos.empty:
            return {"ticker": ticker, "error": "Sin flujos futuros"}
        
        hoy = datetime.today() + relativedelta(months=meses_forward)
        freq_days = flujos['fecha'].diff().dt.days.dropna().mode()[0]
        frecuencia = round(365 / freq_days)
        flujos['t'] = (flujos['fecha'] - hoy).dt.days / (365 / frecuencia)

        tasas_t = curva_real(flujos['t'].values)
        r_periodo = (1 + tasas_t) ** (1 / frecuencia) - 1
        flujos['vp'] = flujos['monto'].values / (1 + r_periodo) ** flujos['t'].values
        precio_forward = flujos['vp'].sum()

        return {"ticker": ticker, f"precio_forward_{meses_forward}m": round(precio_forward, 2)}
    
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}
def calcular_metricas_con_curva(ticker: str, curva_real, ccl: float) -> dict:
    try:
        flujos = pd.read_csv(f"flujos_completo_{ticker}.csv")
        cotizaciones = pd.read_csv(f"cotizaciones_completo_{ticker}.csv")
    except FileNotFoundError:
        return {"ticker": ticker, "error": "Archivos no encontrados"}

    flujos['fecha'] = pd.to_datetime(flujos['fecha'], errors='coerce')
    flujos = flujos[flujos['fecha'] > datetime.today()].sort_values('fecha')
    flujos = flujos[flujos['monto'] > 0].copy()
    if flujos.empty:
        return {"ticker": ticker, "error": "Sin flujos futuros"}

    hoy = datetime.today()
    freq_days = flujos['fecha'].diff().dt.days.dropna().mode()[0]
    frecuencia = round(365 / freq_days)
    flujos['t'] = (flujos['fecha'] - hoy).dt.days / (365 / frecuencia)
    flujos = flujos.dropna(subset=['t'])

    try:
        tasas_t = curva_real(flujos['t'].values)
        if any(pd.isna(tasas_t)) or any(tasas_t <= -1):
            return {"ticker": ticker, "error": "Tasa negativa o inválida en la curva"}
        r_periodo = (1 + tasas_t) ** (1 / frecuencia) - 1
    except Exception as e:
        return {"ticker": ticker, "error": f"Curva fallo: {e}"}

    if len(flujos) != len(r_periodo):
        return {"ticker": ticker, "error": "Tamaño de tasas y flujos no coincide"}

    flujos['vp'] = flujos['monto'].values / (1 + r_periodo) ** flujos['t'].values
    precio_teorico = flujos['vp'].sum()

    try:
        cierre = pd.to_numeric(cotizaciones['cierre'].iloc[0], errors='coerce')
        if ticker.endswith("D"):
            precio_mercado = cierre
        else:
            precio_mercado = cierre / ccl
    except:
        return {"ticker": ticker, "error": "Cierre inválido"}

    # ✅ Nuevo cálculo explícito del spread
    spread = precio_teorico - precio_mercado

    flujos['peso'] = flujos['vp'] / precio_teorico
    duracion_macaulay = (flujos['peso'] * flujos['t']).sum()
    tasa_media = tasas_t.mean()
    if tasa_media <= -1 or pd.isna(tasa_media):
        return {"ticker": ticker, "error": "Tasa media inválida"}
    r_media = (1 + tasa_media) ** (1 / frecuencia) - 1
    duracion_mod = duracion_macaulay / (1 + r_media)

    flujos['conv'] = flujos['peso'] * flujos['t'] * (flujos['t'] + 1)
    convexidad = flujos['conv'].sum() / (1 + r_media) ** 2
    dv01 = -duracion_mod * precio_teorico * 0.0001

    return {
        "ticker": ticker,
        "precio_teorico": round(precio_teorico, 2),
        "precio_mercado": round(precio_mercado, 2),
        "spread": round(spread, 2),  # ✅ agregado acá
        "duracion_macaulay": round(duracion_macaulay, 4),
        "duracion_modificada": round(duracion_mod, 4),
        "convexidad": round(convexidad, 4),
        "dv01": round(dv01, 6)
    }


ccl = get_ccl()
print("CCL:", ccl)

df_scrappeado = scrapear_tabla_bonistas()

curva_real = construir_curva_tir_real("bonos_cer_bonistas.csv")
print("Tasa a 2 años:", curva_real(2))

df = pd.read_csv("bonos_cer_bonistas.csv")
df.columns = df.columns.str.strip()
df['TIR'] = df['TIR'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False).astype(float) / 100
df['MD'] = df['MD'].astype(str).str.replace(',', '.', regex=False).astype(float)

# Filtrar TIR <= 15%
df = df[df['TIR'] <= 0.15]

# Ordenar por duración
df = df.sort_values('MD')

# Interpolación
x = df['MD'].values
y = df['TIR'].values
curva = interp1d(x, y, kind='linear', fill_value='extrapolate')
x_interp = np.linspace(x.min(), x.max(), 200)
y_interp = curva(x_interp)

# Graficar
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'o', label='TIRs observadas')
plt.plot(x_interp, y_interp, '-', label='Curva interpolada')
plt.xlabel("Duración Modificada (años)")
plt.ylabel("TIR Anual")
plt.title("Curva de Bonos CER - sin outliers (TIR > 15%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --------- Cálculo de métricas ---------

tickers_erroneos = ["AL30C", "CUAP", "DICP", "GD30C"]
tickers_filtrados = [t for t in tickers if t not in tickers_erroneos]

# Calcular métricas con TIR scrappeada
resultados_tir = [calcular_metricas_tecnicas_bono(t, ccl) for t in tickers_filtrados]
df_tir = pd.DataFrame(resultados_tir)
df_tir.to_csv("metricas_tir.csv", index=False)

# Calcular métricas con curva benchmark
resultados_benchmark = [calcular_metricas_con_curva(t, curva_real, ccl) for t in tickers_filtrados]
df_benchmark = pd.DataFrame(resultados_benchmark)
df_benchmark.to_csv("metricas_curva.csv", index=False)

# --------- Análisis comparativo ---------

merged = pd.merge(df_tir, df_benchmark, on="ticker", suffixes=("_tir", "_benchmark"))
merged["spread"] = merged["precio_teorico_tir"] - merged["precio_teorico_benchmark"]

# --------- Gráfico de comparación ---------

plt.figure(figsize=(12, 6))
plt.plot(merged["ticker"], merged["precio_teorico_tir"], label="Precio teórico (TIR)", marker="o")
plt.plot(merged["ticker"], merged["precio_teorico_benchmark"], label="Precio teórico (Curva)", marker="x")
plt.plot(merged["ticker"], merged["precio_mercado_tir"], label="Precio mercado", marker="s", linestyle="--", alpha=0.6)
plt.xticks(rotation=90)
plt.title("Comparación de precios teóricos vs mercado")
plt.ylabel("Precio")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("comparacion_precios_teoricos.png")
plt.close()

# --------- Gráfico de spread ---------

plt.figure(figsize=(12, 4))
plt.bar(merged["ticker"], merged["spread"], color="skyblue")
plt.axhline(0, color="black", linewidth=1)
plt.xticks(rotation=90)
plt.title("Spread entre precios teóricos: TIR - Benchmark")
plt.ylabel("Spread")
plt.tight_layout()
plt.savefig("spread_precios_teoricos.png")
plt.close()


# --------- Gráfico de la curva de tasas reales (curva_real) ---------

# Definir un rango razonable de duración para graficar la curva
x_vals = np.linspace(0.1, 20, 200)
y_vals = curva_real(x_vals)

plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals * 100, label="Curva de tasas reales (bonos CER)", color="green")
plt.xlabel("Duración modificada (años)")
plt.ylabel("Tasa real anual (%)")
plt.title("Curva de tasas reales del mercado (bonistas.com)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("curva_tasas_reales.png")
plt.show()  # Comentá esta línea si no querés que se abra el gráfico al correr

# --------- Gráfico: Duración vs Precio teórico ---------

# Usamos precios teóricos calculados con curva benchmark
df_benchmark = pd.DataFrame([calcular_metricas_con_curva(t, curva_real, ccl) for t in tickers_filtrados])
df_benchmark = df_benchmark.dropna(subset=["duracion_modificada", "precio_teorico"])

plt.figure(figsize=(10, 5))
plt.scatter(df_benchmark["duracion_modificada"], df_benchmark["precio_teorico"], color="dodgerblue", label="Precio teórico")
plt.xlabel("Duración modificada (años)")
plt.ylabel("Precio teórico (USD)")
plt.title("Precio teórico vs Duración modificada")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("precio_teorico_vs_duracion.png")
plt.show()

# --------- Gráfico: Desviación entre TIR y curva vs Duración ---------
# Tickers válidos (sin errores previos)
tickers_erroneos = ["AL30C", "CUAP", "DICP", "GD30C"]
tickers_filtrados = [t for t in tickers if t not in tickers_erroneos]

# Calcular precios teóricos con TIR
df_resultados = pd.DataFrame([calcular_metricas_tecnicas_bono(t, ccl) for t in tickers_filtrados])
df_resultados = df_resultados.dropna(subset=["precio_teorico", "duracion_modificada"])

# Calcular precios teóricos con curva benchmark
df_benchmark = pd.DataFrame([calcular_metricas_con_curva(t, curva_real, ccl) for t in tickers_filtrados])
df_benchmark = df_benchmark.dropna(subset=["precio_teorico", "duracion_modificada"])

# df_resultados ya tiene el precio_teorico usando TIR
df_merge = df_resultados.merge(df_benchmark, on="ticker", suffixes=("_tir", "_benchmark"))
df_merge["spread"] = df_merge["precio_teorico_tir"] - df_merge["precio_teorico_benchmark"]

plt.figure(figsize=(10, 5))
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.scatter(df_merge["duracion_modificada_benchmark"], df_merge["spread"], color="orange", label="Spread")
plt.xlabel("Duración modificada (años)")
plt.ylabel("Spread de precio teórico (TIR - Benchmark)")
plt.title("Desviación de precios teóricos vs Duración")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("desviacion_vs_duracion.png")
plt.show()

# Asegurarnos de tener el DataFrame combinado
df_merge = df_resultados.merge(df_benchmark, on="ticker", suffixes=("_tir", "_benchmark"))
df_merge['spread'] = df_merge['precio_teorico_tir'] - df_merge['precio_teorico_benchmark']

# Scatterplot: Duración Modificada vs Convexidad, color = spread
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df_merge['duracion_modificada_tir'],
    df_merge['convexidad_tir'],
    c=df_merge['spread'],
    cmap='coolwarm',
    s=100,
    edgecolor='k'
)
plt.colorbar(scatter, label="Spread (TIR - Curva)")
plt.xlabel("Duración Modificada")
plt.ylabel("Convexidad")
plt.title("Sensibilidad a tasas vs Spread de valuación")
plt.grid(True)
plt.tight_layout()
plt.show()

# Asegurate de tener x = duración modificada y y = TIR ya limpios
x = df['MD'].values
y = df['TIR'].values

# --- SPLINE CÚBICO ---
spline = UnivariateSpline(x, y, s=0.001)  # s=0 ajusta exactamente los puntos
x_spline = np.linspace(x.min(), x.max(), 200)
y_spline = spline(x_spline)

# --- LOWESS ---
lowess_result = lowess(y, x, frac=0.4)  # frac controla la suavidad
x_lowess, y_lowess = lowess_result[:, 0], lowess_result[:, 1]

# --- PLOTEO COMPARATIVO ---
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='TIR observada', alpha=0.6)
plt.plot(x_interp, y_interp, '-', label='Lineal (interp1d)')
plt.plot(x_spline, y_spline, '--', label='Spline cúbico', linewidth=2)
plt.plot(x_lowess, y_lowess, '-.', label='LOWESS', linewidth=2)
plt.xlabel("Duración Modificada (años)")
plt.ylabel("TIR Anual")
plt.title("Comparación de métodos de curva de tasas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Aplicar a todos los bonos
forwards = [proyectar_precio_forward(t, curva_real, ccl, meses_forward=6) for t in tickers if t not in ["AL30C", "CUAP", "DICP", "GD30C"]]
df_forward = pd.DataFrame(forwards)
df_forward.to_csv("df_forward.csv", index=False)
print(df_forward)

# === 4. RANKING DE BONOS POR SPREAD CONTRA CURVA ===
df_merge = df_resultados.merge(df_benchmark, on="ticker", suffixes=("_tir", "_benchmark"))
df_merge['spread'] = df_merge['precio_teorico_benchmark'] - df_merge['precio_mercado_benchmark']

df_ranking = df_merge[['ticker', 'spread']].sort_values('spread', ascending=False)

# Mostrar por consola
print("\n🏆 Bonos más subvaluados (spread positivo):")
print(df_ranking[df_ranking['spread'] > 0].head(10).to_string(index=False))

print("\n📉 Bonos más sobrevaluados (spread negativo):")
print(df_ranking[df_ranking['spread'] < 0].tail(10).to_string(index=False))

# Graficar
plt.figure(figsize=(12, 5))
plt.bar(df_ranking['ticker'], df_ranking['spread'], color='steelblue')
plt.axhline(0, color='black', linewidth=0.8)
plt.title("Ranking de bonos por spread (precio teórico - mercado) usando curva benchmark")
plt.xticks(rotation=90)
plt.ylabel("Spread ($)")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


# === 5. SIMULACIÓN DE SENSIBILIDAD A TASAS ±100bps ===
df_merge['precio_up'] = df_merge['precio_teorico_benchmark'] - df_merge['dv01_benchmark'] * 100
df_merge['precio_down'] = df_merge['precio_teorico_benchmark'] + df_merge['dv01_benchmark'] * 100
df_merge['impacto_up_pct'] = 100 * (df_merge['precio_up'] - df_merge['precio_teorico_benchmark']) / df_merge['precio_teorico_benchmark']
df_merge['impacto_down_pct'] = 100 * (df_merge['precio_down'] - df_merge['precio_teorico_benchmark']) / df_merge['precio_teorico_benchmark']

# Mostrar tabla resumen
print("\n📊 Sensibilidad de precios teóricos ante ±100bps:")
print(df_merge[['ticker', 'precio_teorico_benchmark', 'precio_up', 'precio_down', 'impacto_up_pct', 'impacto_down_pct']].round(2).to_string(index=False))

# Graficar impacto porcentual
plt.figure(figsize=(12, 5))
plt.bar(df_merge['ticker'], df_merge['impacto_up_pct'], label='▲ +100bps', color='indianred')
plt.bar(df_merge['ticker'], df_merge['impacto_down_pct'], label='▼ -100bps', color='mediumseagreen')
plt.axhline(0, color='black', linewidth=0.8)
plt.ylabel("Variación % en precio teórico")
plt.title("Impacto en precios teóricos ante shock de ±100bps")
plt.xticks(rotation=90)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Umbral de alerta
umbral = 0.02  # 2%

df_merge['desviacion_pct'] = (df_merge['precio_teorico_benchmark'] - df_merge['precio_mercado_benchmark']) / df_merge['precio_mercado_benchmark']

df_alertas = df_merge[df_merge['desviacion_pct'].abs() > umbral]

print("\n🚨 Bonos con desvío > ±2% entre precio mercado y benchmark:")
print(df_alertas[['ticker', 'precio_teorico_benchmark', 'precio_mercado_benchmark', 'desviacion_pct']].round(3).to_string(index=False))

# Ponderaciones arbitrarias (pueden ajustarse)
df_merge['score'] = (
    0.4 * df_merge['spread'] +                # spread positivo
    0.3 * df_merge['convexidad_benchmark'] +  # mayor convexidad
    0.3 * (1 / (1 + abs(df_merge['duracion_modificada_benchmark'] - 5)))  # preferencia por duración media (~5)
)

df_recomendados = df_merge.sort_values('score', ascending=False).head(10)

print("\n✅ Recomendación de bonos ordenados por score:")
print(df_recomendados[['ticker', 'spread', 'convexidad_benchmark', 'duracion_modificada_benchmark', 'score']].round(3).to_string(index=False))

# MERGE para comparar precios teóricos con curva
df_merge = df_resultados.merge(df_benchmark, on="ticker", suffixes=("_tir", "_benchmark"))

# CALCULAR Z-SPREAD (precio teórico con TIR - precio con curva)
df_merge["z_spread"] = df_merge["precio_teorico_tir"] - df_merge["precio_teorico_benchmark"]

# RANKING DE BONOS POR Z-SPREAD
df_ranking = df_merge[["ticker", "precio_teorico_tir", "precio_teorico_benchmark", "z_spread"]].copy()
df_ranking = df_ranking.dropna().sort_values("z_spread", ascending=False)

print("\n🟢 Ranking de bonos por spread positivo (potencial subvaluación):")
print(df_ranking.head(10).to_string(index=False))

print("\n🔴 Ranking de bonos por spread negativo (potencial sobrevaluación):")
print(df_ranking.tail(10).to_string(index=False))

# ALERTA: Bonos con desviación > ±2%
alertas = df_ranking[abs(df_ranking["z_spread"]) > 2]
if not alertas.empty:
    print("\n⚠️ ALERTA: Bonos con spread mayor a ±2%")
    print(alertas.to_string(index=False))
else:
    print("\n✅ Todos los spreads están dentro del rango aceptable (±2%)")

# GRAFICAR SPREADS
plt.figure(figsize=(12, 5))
plt.bar(df_ranking["ticker"], df_ranking["z_spread"], color='green')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=90)
plt.title("Z-Spread por bono (Precio con TIR - Precio con curva)")
plt.ylabel("Spread contra curva (USD)")
plt.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.show()

df = pd.read_csv("metricas_curva.csv")

# Verificar que tenga las columnas necesarias
if "precio_teorico" in df.columns and "precio_mercado" in df.columns:
    # Calcular la columna spread
    df["spread"] = df["precio_teorico"] - df["precio_mercado"]
    
    # Sobrescribir el archivo original
    df.to_csv("metricas_curva.csv", index=False)
    print("✅ Spread agregado y archivo sobrescrito correctamente.")
else:
    print("❌ No se pudo calcular el spread: faltan columnas necesarias.")