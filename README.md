# Fixed Income Valuation

Este script integra scraping automatizado, construcción de curva de tasas reales y análisis técnico para valuar bonos soberanos argentinos en dólares y CER.

🔍 **¿Qué hace?**

* Scrapea flujos de fondos, TIR, duración y precios desde Rava y Bonistas.com
* Construye curva de tasas reales interpolada con spline, LOWESS y lineal
* Calcula precios teóricos por TIR y por curva benchmark
* Estima spreads entre valuaciones y precio de mercado
* Calcula duración Macaulay, duración modificada, convexidad y DV01
* Proyecta precios forward con simulación a 6 meses
* Grafica desviaciones, sensibilidad a tasas (±100bps) y curvas comparativas
* Genera alertas de desalineamiento entre pricing y fundamentals
* Incluye un **dashboard interactivo con Streamlit** para explorar datos, armar carteras y analizar curvas
* Incluye un **modelo de forward spread implícito entre bonos duales**, con recomendaciones automáticas y heatmap visual

📈 **¿Para qué sirve?**

* Para detectar oportunidades de arbitraje y bonos mal valuados
* Para construir carteras de renta fija balanceando convexidad y duración
* Para monitorear el impacto de shocks de tasas en precios teóricos
* Para tener una curva implícita del mercado local sin depender de fuentes pagas

️ **Stack técnico**

* Python 3, Selenium, Pandas, Numpy, Matplotlib, Scipy, Statsmodels, Streamlit, Seaborn
* Web scraping en headless mode
* Dashboard visual e interactivo en tiempo real
* Análisis de pares, spreads forward y ranking long-only para bonos duales
* Cálculo financiero desde cero sin usar librerías de pricing externas

---

📌 Proyecto creado para automatizar y profesionalizar el análisis de bonos soberanos argentinos. Ideal para traders de fixed income, analistas de crédito o desks que busquen alfa en mercados emergentes.
