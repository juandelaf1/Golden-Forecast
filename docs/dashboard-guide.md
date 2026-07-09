# Golden Forecast — Guía del Dashboard

## Visión General

El dashboard de Golden Forecast es una herramienta interactiva de 9 pestañas que combina machine learning, análisis técnico y macroeconomía para generar señales de trading diarias sobre el oro (GC=F).

**Stack:** Plotly Dash + Flask + scikit-learn + XGBoost + Yahoo Finance  
**Despliegue:** Docker + Render  
**Documentación:** Mintlify (auto-deploy desde main)

---

## 1. Panel de Control (Summary)

**Objetivo:** Vista ejecutiva de 5 segundos. El usuario entra y sabe si debe comprar, vender o esperar.

### Tarjetas superiores (4 métricas clave)

<table>
<tr><th>Métrica</th><th>Valor</th><th>Propósito</th></tr>
<tr><td>Precio actual</td><td>Cotización spot del oro (GC=F)</td><td>Referencia de precio del día</td></tr>
<tr><td>Señal de mercado</td><td>ALZA / PRECAUCIÓN / ESTABLE</td><td>Decisión principal del modelo para mañana</td></tr>
<tr><td>Certeza</td><td>XX% (0-100%)</td><td>Confianza del modelo en su señal actual</td></tr>
<tr><td>Actualizado</td><td>Fecha y hora UTC</td><td>Trazabilidad de la última actualización</td></tr>
</table>

**Señales:**
- **ALZA** — confianza ≥ 58%. El modelo espera subida.
- **PRECAUCIÓN** — confianza ≤ 42%. El modelo espera bajada.
- **ESTABLE** — confianza entre 42-58%. Señal no concluyente.

### Gráfico: Aciertos y Confianza del Modelo

**Tipo:** Barras + línea superpuesta  
**X:** Fecha (últimos ~90 puntos de test)  
**Y (barras):** Retorno diario del oro  
**Y (línea, eje secundario):** Confianza del modelo (0-100%)

**Color de barras:**
- 🟢 Verde = el modelo acertó la dirección
- 🔴 Rojo = el modelo falló

**Por qué existe:** Muestra visualmente dónde acierta y dónde falla el modelo. El patrón deseable es barras verdes altas (aciertos con buen retorno) y rojas bajas (errores con pérdida mínima). La línea de confianza revela si el modelo es humilde (confianza baja cuando falla) o arrogante (confianza alta cuando falla).

**Límite:** 90 puntos máximos para evitar congelación del navegador.

### Gráfico: Rendimiento del Modelo

**Tipo:** Barras horizontales  
**Métricas del modelo principal (lr_strong_reg_binary):**

<table>
<tr><th>Métrica</th><th>Valor actual</th><th>Qué mide</th></tr>
<tr><td>Accuracy</td><td>56.9%</td><td>% de predicciones correctas sobre el total</td></tr>
<tr><td>Precision</td><td>58.3%</td><td>De las veces que dijo "sube", % de aciertos</td></tr>
<tr><td>Recall</td><td>87.5%</td><td>De las subidas reales, % capturadas</td></tr>
<tr><td>F1 Score</td><td>0.700</td><td>Media armónica Precision + Recall</td></tr>
<tr><td>ROC-AUC</td><td>0.500</td><td>Capacidad de distinguir subida/bajada</td></tr>
</table>

**Color de cada barra según valor:**
- < 0.35 → rojo
- 0.35-0.50 → naranja
- 0.50-0.65 → amarillo
- 0.65-0.80 → verde claro
- ≥ 0.80 → verde oscuro

**Por qué existe:** Resume las 5 métricas clave del modelo principal en un solo vistazo. La coloración permite detectar fortalezas (Recall alto en verde) y debilidades (ROC-AUC en amarillo) al instante.

### Gráfico: Rendimiento Acumulado

**Tipo:** Línea con área sombreada  
**X:** Operación (orden cronológico en test)  
**Y:** Accuracy acumulada

- **Línea dorada:** Accuracy acumulada operación a operación
- **Línea verde discontinua:** Accuracy total del modelo (referencia plana)

**Por qué existe:** Responde "¿el modelo es consistentemente bueno o tuvo rachas?". Una línea plana cerca del accuracy total = consistencia. Una línea volátil = suerte o rachas.

---

## 2. Precio y Señales (Price)

**Objetivo:** Contexto histórico. El inversor ve dónde está el precio hoy y cómo se alinean las señales del modelo con la tendencia real.

### Gráfico principal

- **Línea dorada:** Precio del oro indexado a base 100. La indexación permite ver cambios relativos en una escala manejable.
- **Línea azul discontinua:** Media Móvil de 21 días (MA21). Indica tendencia a corto plazo:
  - Precio > MA21 → tendencia alcista
  - Precio < MA21 → tendencia bajista
- **Estrellas:** Señales del modelo en el período de test:
  - ⭐ Verde = predicción de subida
  - ⭐ Roja = predicción de bajada
- **Línea vertical:** División train/test (80/20)
- **Selector de rango:** 1D, 5D, 1M, 3M, 6M, 1A, HIST
- **MAX_POINTS=500:** Límite de puntos para rendimiento

**Por qué existe:** Es el gráfico más intuitivo para cualquier inversor. Responde visualmente "¿las señales del modelo tienen sentido con lo que pasó después?" y "¿está el oro en tendencia alcista o bajista?".

---

## 3. Indicadores Técnicos (Indicators)

**Objetivo:** Validar la señal del modelo con indicadores técnicos clásicos.

### RSI (Relative Strength Index, 14 días)

**Tipo:** Línea + bandas horizontales  
**Rango:** 0-100

- **Línea verde:** Valor del RSI
- **Línea roja (y=70):** Sobrecompra. RSI > 70 → activo caro, posible corrección bajista.
- **Línea verde (y=30):** Sobreventa. RSI < 30 → activo barato, posible rebote alcista.

**Por qué existe:** Es el indicador más usado del mundo. Un RSI > 70 cuando el modelo dice ALZA es contradictorio y requiere cautela. RSI < 30 cuando el modelo dice BAJA sugiere posible rebote.

### MACD (Moving Average Convergence Divergence)

**Tipo:** Líneas + histograma  
**Componentes:**
- **Línea verde (MACD):** Diferencia entre EMA rápida (12) y EMA lenta (26). Mide momentum.
- **Línea roja discontinua (Señal):** EMA de 9 del MACD. Confirma cambios de tendencia.
- **Barras azules (Histograma):** MACD - Señal. Crecen = momentum alcista. Decrecen = momentum bajista.

**Por qué existe:** El cruce del MACD sobre la línea de señal anticipa cambios de tendencia. Complementa al RSI y a la señal del modelo.

---

## 4. Correlaciones Macro (Macro)

**Objetivo:** Mostrar las 3 fuerzas macroeconómicas que mueven el oro y cómo interactúan.

### Cuadrícula 2×2

<table>
<tr><th>Subplot</th><th>Contenido</th><th>Relación con oro</th></tr>
<tr><td>(1,1) Oro vs DXY</td><td>Scatter + recta regresión + ρ (Pearson)</td><td>Inversa. DXY sube → dólar fuerte → oro más caro → baja demanda</td></tr>
<tr><td>(1,2) Oro vs VIX</td><td>Scatter + recta regresión + ρ</td><td>Directa. VIX alto → incertidumbre → refugio en oro</td></tr>
<tr><td>(2,1) Oro vs TNX</td><td>Scatter + recta regresión + ρ</td><td>Inversa. TNX alto → bonos rentables → oro menos atractivo</td></tr>
<tr><td>(2,2) Indicadores normalizados</td><td>Líneas Gold, DXY, VIX, TNX en base 100</td><td>Visión conjunta de las 4 series</td></tr>
</table>

### Panel de ayuda expandible
Explicación de cada indicador macro con su valor actual y guía de interpretación.

**Por qué existe:** El oro no cotiza en el vacío. DXY, VIX y TNX explican gran parte de su movimiento. El inversor necesita ver estas fuerzas actuando simultáneamente para contextualizar la señal del modelo.

---

## 5. Backtest y Estrategia (Backtest)

**Objetivo:** Transparencia total. El usuario ve cómo se comportó cada modelo en el pasado, sin filtros ni sesgo.

### Estrategia ML vs Buy & Hold

**Tipo:** Líneas múltiples  
- **Línea verde:** Rentabilidad acumulada de la estrategia del modelo (compra/vende según señales con shift(1))
- **Línea roja discontinua:** Rentabilidad de comprar y mantener (benchmark mínimo)
- **Línea dorada (eje secundario):** Alpha = diferencia estrategia − benchmark. Positivo = supera al benchmark.
- **Anotación de advertencia:** Si alpha negativo, se muestra con fondo rojo.

**Fórmula:** `equity = capital * cumprod(1 + returns * signal.shift(1))` — `shift(1)` asegura que la señal de hoy se ejecuta mañana (sin forward-looking bias).

**Por qué existe:** Cualquier estrategia debe superar al benchmark de comprar y mantener. Si no lo hace, operar no tiene sentido.

### Precisión en el Tiempo
Curva de accuracy acumulada (mismo gráfico que en Summary).

### Aciertos y Confianza
Barras de retorno diario + confianza (mismo gráfico que en Summary).

### Métricas del Modelo
Barras de Accuracy, Precision, Recall, F1, ROC-AUC (mismo gráfico que en Summary).

### Matriz de Confusión

**Tipo:** Heatmap 2×2  
**Filas:** Real (Bajada/Subida)  
**Columnas:** Predicho (Bajada/Subida)

<table>
<tr><th></th><th>Pred: Bajada</th><th>Pred: Subida</th></tr>
<tr><td><strong>Real: Bajada</strong></td><td>Verdadero Negativo</td><td>Falso Positivo</td></tr>
<tr><td><strong>Real: Subida</strong></td><td>Falso Negativo</td><td>Verdadero Positivo</td></tr>
</table>

**Color:** marrón oscuro → dorado (más aciertos = más dorado)

### Curva ROC

**Tipo:** Línea + diagonal  
- **Línea dorada:** Tasa de verdaderos positivos vs tasa de falsos positivos
- **Diagonal punteada:** Clasificador aleatorio (AUC = 0.50)
- **AUC en leyenda:** Área bajo la curva
- La curva debe estar por encima de la diagonal. AUC ~0.500 actual indica rendimiento aleatorio.

### Tabla Comparativa de 12 Modelos

<table>
<tr><th>Columna</th><th>Descripción</th><th>Modelo líder</th></tr>
<tr><td>Modelo</td><td>Nombre (LR/RF/XGB × binario/multiclase × regular/deep)</td><td>—</td></tr>
<tr><td>Accuracy</td><td>% de aciertos totales</td><td>lr_strong_reg_binary (56.9%)</td></tr>
<tr><td>Precision</td><td>% de aciertos en predicciones de subida</td><td>xgb_binary (59.6%)</td></tr>
<tr><td>Recall</td><td>% de subidas reales capturadas</td><td>lr_strong_reg_binary (87.5%)</td></tr>
<tr><td>F1</td><td>Equilibrio precision-recall</td><td>lr_strong_reg_binary (0.700)</td></tr>
<tr><td>ROC-AUC</td><td>Capacidad discriminativa</td><td>xgb_binary (0.526)</td></tr>
<tr><td>Rentab.</td><td>Rentabilidad acumulada en test</td><td>xgb_binary (+34.5%)</td></tr>
</table>

**Rentabilidad coloreada:** verde si positiva, roja si negativa.

**Por qué existe:** La tabla es el corazón de la transparencia. Revela que:
- El mejor modelo por F1/accuracy es **lr_strong_reg_binary** (LR con regularización fuerte)
- El más rentable es **xgb_binary** (+34.5%), pero con ROC-AUC 0.526
- ROC-AUC ~0.50 para casi todos → **ninguno es fiable todavía**
- Las rentabilidades positivas pueden ser ruido estadístico

---

## 6. Simulación (Simulation)

**Objetivo:** Responder "¿cuánto habría ganado/perdido siguiendo las señales en un período concreto?"

### ⚠️ Banner de advertencia
> **Modelo en fase beta — ROC-AUC ~0.50 (equivalente a aleatorio).** Los resultados de esta simulación no son representativos. No usar para decisiones de inversión reales.

### Controles

<table>
<tr><th>Control</th><th>Tipo</th><th>Default</th></tr>
<tr><td>Fecha inicio</td><td>DatePicker</td><td>1 mes atrás desde el último dato</td></tr>
<tr><td>Fecha fin</td><td>DatePicker</td><td>Último dato disponible</td></tr>
<tr><td>Capital inicial</td><td>Input numérico</td><td>$10,000 (rango: $1,000 - $1,000,000)</td></tr>
<tr><td>Botón</td><td>"EJECUTAR SIMULACIÓN"</td><td>—</td></tr>
</table>

### Resultados

- **Valor final:** Patrimonio total tras el período
- **Retorno (%):** Rentabilidad total generada
- **Operaciones:** Número de trades ejecutados
- **Gráfico:** Evolución del portfolio en el período

### Fórmula de simulación
```
equity = capital * cumprod(1 + returns * signal.shift(1))
```
- `signal.shift(1)`: la señal de hoy se ejecuta mañana (sin sesgo de mirar hacia adelante)
- Si señal=1 y retorno positivo → ganas
- Si señal=0 y retorno negativo → evitas la pérdida (no operas)

**Por qué el default es 1 mes:** Con ROC-AUC ~0.50, períodos largos solo muestran ruido. 1 mes (~22 días hábiles) da una visión reciente sin arrastrar regímenes antiguos.

---

## 7. Métricas (Métricas — lazy load)

**Objetivo:** Análisis profundo para usuarios técnicos. Se carga bajo demanda (al visitar la pestaña) para no ralentizar el inicio del dashboard.

### Factores que Influyen en la Predicción

**Tipo:** Barras horizontales  
**Datos:** Feature importances del modelo Random Forest

- **Barras:** Importancia relativa de cada feature (0-1)
- **Color por categoría:**
  - 🟢 Precio (retornos diarios, lags)
  - 🟡 Técnico (RSI, MACD, medias móviles, volatilidad)
  - 🟠 Macro (DXY, VIX, TNX returns y rangos)
- **Filtro:** Dropdown para filtrar por categoría
- **Click interactivo:** Muestra descripción detallada de la feature

**Por qué existe:** Responde "¿qué está mirando el modelo para decidir?" Ayuda a alinear o cuestionar las decisiones del modelo basándose en conocimiento de dominio.

### Matriz de Confusión y Curva ROC
Ídem pestaña Backtest.

### Tabla Comparativa (12 modelos + regresión)
Ídem pestaña Backtest, incluyendo modelos de regresión si existen.

### Sección Experimental

**Horizontes múltiples:**
Evalúa si el modelo funciona mejor a diferentes plazos (1 día, 2 días, 5 días). Tabla con Accuracy, Precision, Recall, F1, ROC-AUC por horizonte.

**Voting Classifier:**
Rendimiento del ensemble por votación (LR + RF + XGB) a 1 día.

**Umbral óptimo:**
Calcula el umbral de probabilidad que maximiza F1 (en lugar del default 0.5). Muestra si un umbral diferente mejora el rendimiento.

**Por qué existe:** Investiga si el problema no es el modelo sino el horizonte de predicción o el umbral de decisión. A veces el mismo modelo funciona mejor a 3 días que a 1 día.

---

## 8. Valor y Riesgo (Regresión — lazy load)

**Objetivo:** Complementar la clasificación (sube/baja) con análisis de valoración y riesgo basado en regresión.

### Hasta 4 tarjetas de métricas

<table>
<tr><th>Métrica</th><th>Qué mide</th><th>Interpretación</th></tr>
<tr><td>Valor Justo</td><td>Distancia del precio a su media 200d en desviaciones típicas</td><td>&gt;2σ → sobrevalorado. &lt;-2σ → infravalorado</td></tr>
<tr><td>Volatilidad Esperada</td><td>Volatilidad anualizada a 20 días</td><td>15% = ±15% al año de media</td></tr>
<tr><td>Rango de Precio (ATR)</td><td>Rango verdadero medio 20 días / precio</td><td>2% = ±2% al día de media</td></tr>
<tr><td>Riesgo de Caída</td><td>Máximo drawdown a 20 días</td><td>-5% = mayor caída esperada</td></tr>
</table>

Cada tarjeta muestra: modelo usado, R², MAE, IC (Information Coefficient). Solo se muestra si el mejor modelo tiene R² > 0.

**Panel de ayuda:** Explicación de cada métrica, cómo interpretarla y cómo usarla junto a la señal de clasificación.

**Por qué existe:** La clasificación solo dice "sube o baja". La regresión dice cuánto, con qué volatilidad y qué riesgo. Un inversor necesita ambas.

---

## 9. Metodología (Methodology)

**Objetivo:** Pestaña de confianza y transparencia. Explica cómo funciona todo sin asumir conocimiento técnico.

### Secciones

<table>
<tr><th>Sección</th><th>Contenido</th><th>Propósito</th></tr>
<tr><td>Pipeline de Datos</td><td>5 pasos: Extracción (Yahoo Finance) → Preprocesamiento (limpieza) → Feature Engineering (35 features + targets) → Modelado (12 modelos) → Evaluación (métricas + backtest)</td><td>Demostrar que hay un proceso reproducible, no una caja negra</td></tr>
<tr><td>Stack Tecnológico</td><td>Python 3.10+, pandas, numpy, scikit-learn, XGBoost, Plotly Dash, Docker, Render</td><td>Demostrar solvencia técnica y stack moderno</td></tr>
<tr><td>Modelos Implementados</td><td>LR (C=0.1), RF (100 árboles, max_depth=5), XGB (100 estimadores, max_depth=3), LR multiclase</td><td>Transparencia total — cualquiera puede reproducir</td></tr>
<tr><td>Señales de Trading</td><td>ALZA (≥58% confianza), PRECAUCIÓN (≤42%), ESTABLE (42-58%)</td><td>El usuario entiende cómo se toman las decisiones</td></tr>
<tr><td>Equipo</td><td>María (PO), Juan (SM), Jose (Dev), Gema (Dev), Joel (Dev)</td><td>Atribución de roles (presentación colaborativa)</td></tr>
<tr><td>Limitaciones</td><td>5 puntos clave</td><td>Honestidad intelectual — el usuario sabe qué NO esperar</td></tr>
<tr><td>QR Repositorio</td><td>Enlace al GitHub</td><td>Acceso al código fuente completo</td></tr>
</table>

### Limitaciones declaradas

1. Datos de cierre diario — no apto para trading de alta frecuencia
2. Señales probabilísticas — no certezas absolutas
3. Rendimiento pasado no garantiza resultados futuros
4. Modelos se re-entrenan automáticamente con cada actualización
5. Gobernanza Scrum — ramas feature/, PRs con revisión, main protegido

**Por qué existe:** Sin esta pestaña, el dashboard es una caja negra. Con ella, el usuario entiende, confía y puede cuestionar adecuadamente las señales.

---

## Notas Técnicas

### MAX_POINTS = 500
Todos los gráficos limitan a 500 puntos de datos para evitar congelación del navegador. Esto es suficiente para visualizar tendencias sin sobrecargar el frontend.

### Lazy Loading
Las pestañas Métricas y Valor y Riesgo cargan sus modelos bajo demanda (solo cuando el usuario las visita). Esto reduce el tiempo de carga inicial del dashboard.

### Date Range Selector
Todas las gráficas temporales incluyen botones de selección de rango: 1D, 5D, 1M, 3M, 6M, 1A, HIST. El default visual son los últimos 90 puntos.

### Actualización de datos
Los datos se actualizan con cada despliegue. El pipeline automático (GitHub Actions) ejecuta extract → preprocess → feature_engineering → predict diariamente.
