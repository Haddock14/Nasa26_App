# Explorador de rotaciones de cultivos (v2) — NASA Space Apps

Herramienta de apoyo para que un agricultor explore rotaciones que mejoren la salud del suelo y la resiliencia
al clima y al agua, usando observaciones NASA + suelo, acuífero y rendimientos locales.
**No reemplaza el juicio del agricultor ni la revisión de un asesor local.**

## Cómo correrlo

```bash
pip install -r requirements.txt          # rasterio es opcional (Landsat/SRTM)
python -m rotacion.api                   # abre http://localhost:5000
python -m pytest -q                      # 18 pruebas sin conexión
```

Variables opcionales:

| Variable | Para qué |
|---|---|
| `EARTHDATA_TOKEN` | SMAP y ECOSTRESS (AppEEARS) y GPM IMERG (GES DISC). Se genera en urs.earthdata.nasa.gov |
| `NASS_API_KEY` | rendimientos de condados fuera de Finney |
| `OPENTOPO_KEY` | descargar SRTM si no hay DEM local |
| `EN_VIVO=0` | trabajar solo con datos locales |
| `DATA_DIR` | carpeta de datos (por defecto `./data`) |

## Versión 2.8: español / inglés

- Botón **ES | EN** arriba a la derecha. Cambia todo el texto visible al inglés (títulos, explicaciones, tablas,
  gráficos, avisos del mapa y los textos que arma el servidor) y vuelve al español sin recargar. La página recuerda el
  idioma elegido.
- El catálogo de traducción se edita en `scripts/catalogo_en.py` y se genera con `python scripts/catalogo_en.py`
  (escribe `web/i18n_en.json`). Tiene frases exactas, plantillas con números (`#`) y textos variables (`@`), y
  convierte el formato de números (1.450,5 -> 1,450.5) y fechas (14 abr -> Apr 14).
- `rotacion/i18n.py` aplica el mismo algoritmo que la página, para probarlo (`tests/test_rotacion.py`). Se revisó la
  página completa en inglés (todas las pestañas, campos, rotaciones y predicción): no quedan textos en español.
- Los datos y la API siguen en español; solo cambia lo que se muestra. En "Opciones avanzadas", las rotaciones a
  comparar se siguen escribiendo con los nombres de cultivo en español (trigo, sorgo...).

## Versión 2.7: predicción 2027-2029 (modelo FieldShift del notebook)

- `rotacion/prediccion.py` porta `modelo_hackaton.ipynb` paso por paso:
  1. **Clima:** lluvia trimestral de NASA POWER en el punto 38° N, 101° O, clasificada como seca, normal o húmeda (terciles 1981-2001). Con eso se arma una cadena de Markov con suavizado de Laplace, validada con Brier contra la climatología en 2002-2023.
  2. **Rendimientos USDA de Finney en kg/ha:** regresión Ridge (alpha = 10) para trigo y sorgo; los 7 años con clima más parecido para el maíz.
  3. **300 escenarios 2026-2029** (semilla 42, mismo orden aleatorio que el notebook), más el error histórico de cada cultivo.
  4. **Optimización:** maximiza (1 − a)·promedio + a·CVaR 20 %, con presupuesto de agua, sin repetir cultivo en años seguidos y con un máximo de 2 años de maíz.
     - Usa **Gurobi** si `gurobipy` está instalado (la misma formulación del notebook); si no, **HiGHS** (`scipy.optimize.milp`). Ambos dan el óptimo exacto; un test lo comprueba enumerando las 27 rotaciones.
- **Reproduce el notebook:**
  - la matriz de Markov y la validación son idénticas;
  - las tres estrategias dan la misma rotación: Productividad sorgo → trigo → sorgo, Equilibrada maíz → trigo → sorgo, Resiliencia maíz → sorgo → trigo;
  - las cifras difieren levemente (por ejemplo, CVaR 0,812 frente a 0,807) porque los rendimientos NASS del proyecto terminan en 2020-2021.

  Si se copia `datos_unidos_finney.csv` (el archivo del notebook) a `data/`, se usa ese archivo.
- **API:** `GET /api/prediccion` y `POST /api/prediccion`. El POST acepta:
  - `aversion` (0-1): agrega la estrategia "Personalizada";
  - `presupuesto_agua_mm`;
  - `max_maiz`;
  - `excluir`;
  - `agua_por_cultivo_mm`.

  La simulación queda en memoria, así que recalcular la optimización tarda unos 0,1 s.
- `python -m rotacion.prediccion` genera `resultados_fieldshift/fieldshift_web.json` y `comparacion_rotaciones.csv`, igual que el notebook.
- **Interfaz:** nueva pestaña "Predicción 2027-2029" (las demás no cambian). Incluye:
  - la elección de estrategia y la rotación de cada año con su rango;
  - la comparación de estrategias;
  - la distribución de los 300 escenarios;
  - el clima del modelo: probabilidades seco/normal/húmedo, historia 1981-2025 y qué tan confiable es el pronóstico;
  - los rendimientos por cultivo y cómo se estiman;
  - los ajustes de tolerancia al riesgo, agua, máximo de maíz y cultivos excluidos;
  - la lista de las 27 combinaciones.
- Los valores de agua por cultivo (trigo 450, sorgo 350, maíz 650 mm; presupuesto 1600 mm) siguen siendo los **ilustrativos** del notebook, y la app lo indica.

## Versión 2.6: salud y desempeño de los cultivos (USDA NASS Kansas 1981-2026)

- `scripts/preparar_nass_ks.py` compacta las 46 descargas anuales de QuickStats (~400 MB) en `data/nass_ks/` (menos de 1 MB):
  - rendimiento y superficie por condado y distrito;
  - estado semanal de los cultivos;
  - humedad del suelo;
  - avance de siembra y cosecha;
  - precios recibidos.
- `rotacion/desempeno.py` calcula:
  - **Por condado:** rendimiento típico (secano y riego), tendencia por década, frecuencia de años malos (menos del 75 % de lo esperado) y superficie sembrada que no se cosechó. En Finney, el trigo pierde en promedio el 12 % de lo sembrado, y el 47 % en 1996.
  - **Relación con el clima local de NASA POWER:** por ejemplo, el trigo de Finney rinde cerca de 6 % más por cada 25 mm de lluvia entre marzo y mayo.
  - **Estado de los cultivos en Kansas semana a semana:** curva normal, años malos y años buenos, y relación con la lluvia y el calor. Por ejemplo, cada 5 días sobre 35 °C en julio y agosto restan cerca de 5 puntos de maíz en buen estado (relación fuerte).
  - **Comparación satélite y terreno:** la humedad del suelo de NASA POWER coincide con los informes semanales del USDA (correlación 0,83 en 286 meses).
  - **Fechas reales de siembra y cosecha**, comparadas con las de hace 35 a 40 años.
- **Interfaz:**
  - nueva pregunta "¿Cómo les va a los cultivos en esta zona?";
  - nueva sección "Desempeño de los cultivos" en Mi campo, con el mismo formato de explicación y escala de referencia;
  - los riesgos de cada rotación citan la superficie no cosechada del condado.
- **Modelo:** los precios por defecto son el promedio recibido en Kansas en las últimas 3 campañas (USDA NASS) en lugar de supuestos.

## Versión 2.5

- **Corregido:** después de seleccionar un campo ya no se podían elegir otros (solo puntos). El contorno de la selección y las capas de clima y suelo creaban su propio lienzo de dibujo encima del de los campos y tapaban los clics. Ahora todas las capas usan una sola superficie de dibujo, así que siempre se puede seleccionar otro campo, volver a seleccionar el mismo o quitar la selección.

## Versión 2.4

- En "Lo más importante para usted" se pueden marcar todas las opciones (antes había un máximo de tres).
- **Superficie de los campos:** si el dato oficial (`csbacres`) no llega, se calcula con la forma del campo, así ningún campo queda con 0 acres.
- **Consulta al USDA más robusta:**
  - prueba las capas 2, 0 y 1 del servicio *Crop Sequence Boundaries 2024*;
  - pide todos los atributos y los normaliza;
  - elimina campos duplicados al paginar.

  Si antes quedó guardado el respaldo de OpenStreetMap (`data/campos/finney_osm.geojson`), en cada inicio vuelve a intentar el USDA y, si responde, guarda `finney.geojson` y lo usa desde entonces.
- **Historial de cada campo garantizado:** si un campo no trae su historial, al seleccionarlo se consulta el mapa de cultivos del USDA (CropScape, Cropland Data Layer) en el centro del campo para 2017-2024. Con eso se detecta la rotación y se carga en "Comparar rotaciones". La conversión a coordenadas Albers (EPSG:5070) está verificada contra el ejemplo numérico de Snyder (1987).

## Versión 2.3: interfaz sobria con navegación libre

- Vuelven las **pestañas libres** (Mi campo, Comparar rotaciones, Ayuda y glosario): el usuario va donde quiere, sin pasos obligatorios. La comparación funciona aunque no haya un campo elegido (usa el centro del condado).
- **Sin emojis ni animaciones** (también se desactivan las animaciones de zoom del mapa). Estados con texto y color (Bien / Atención / Problema), contraste AA, foco visible, pestañas accesibles por teclado (flechas) y tablas con encabezados.
- **Quitar la selección:** botón "Quitar selección" en el mapa y en el panel, volver a hacer clic sobre el mismo campo, hacer clic en un lugar vacío del mapa o presionar Esc.
- **Cada valor explicado:** al abrir un tema se ve cada dato con qué significa, en palabras simples, y una escala de referencia de tres tramos que marca "Su zona" en el que corresponde (por ejemplo, lluvia: menos de 400 mm muy seco · 400 a 600 semiárido · 600 o más suficiente).
- Calendario mensual como tabla (un mes por fila) y comparación de rotaciones con "Mejor / Igual / Peor" respecto de la rotación actual, con la definición de cada columna.

## Versión 2.2: todo Finney en el mapa y recorrido guiado

- **Todos los campos de Finney County se ven desde la vista del condado** (no hace falta acercarse). La primera vez se descargan del USDA (*Crop Sequence Boundaries*, por páginas de 2000 y con geometría simplificada) y se guardan solos en `data/campos/finney.geojson`; después funciona sin conexión. Si el USDA no responde, se usan los campos de OpenStreetMap (`landuse=farmland`, sin historial). Se dibujan en canvas para que miles de campos no pongan lento el mapa.
- **Fondo del mapa:** 🛰️ Satélite, ⛰️ Topográfico (Esri World Topo) o 🗺️ Calles.
- **Recorrido en 4 pasos** con botón "Siguiente": 1 Su campo · 2 Cómo está · 3 Lo que busca · 4 Resultado.
  - Las prioridades se eligen con fichas (hasta 3).
  - El resultado compara cada opción con **lo que hace hoy** (▲ Mejor / = Igual / ▼ Peor) en agua, tierra, dinero, años malos y plagas.
  - Lo técnico queda en secciones plegadas "para asesores".
  - La ayuda y el glosario se abren con un botón aparte.

## Versión 2.1: campos de cultivo y vista sencilla

- **Mapa de campos reales en lugar de pozos.** Se usa la imagen satelital de fondo (Esri) y encima los límites de cada campo del servicio público USDA NASS *Crop Sequence Boundaries* (2017-2024). Los campos se pueden colorear por:
  - cultivo 2024;
  - años que quedaron sin cultivo;
  - riego por pivote probable (se deduce de la forma circular).

  Al hacer clic en un campo:
  - se analiza ese polígono;
  - se muestra qué se sembró cada año;
  - la app detecta la rotación actual y la carga sola en el comparador;
  - si el campo parece tener pivote, propone riego moderado.

  El agua subterránea sigue apareciendo en el panel como una de las 6 preguntas.
  Para usar los campos sin conexión: `python scripts/descargar_campos.py` (por defecto descarga Finney County).
- **Vista sencilla, con el detalle técnico plegado:**
  - 6 preguntas con semáforo, cada una con "qué significa para su rotación" (¿Alcanza la lluvia? ¿Hace demasiado calor? ¿Heladas tardías? ¿Se vuela o se lava el suelo? ¿Qué tan sano está el suelo? ¿Está bajando el agua de los pozos?).
  - "Antes vs. ahora" (1981-2002 frente a 2003-2025).
  - Calendario mes a mes con íconos.
  - Prioridades como botones Poco/Algo/Mucho, riego como No/Poca/Moderada/Mucha y escenarios como botones.
  - Resultado como "lo que mejor encaja", una comparación rápida con puntos (agua, suelo, dinero, años sin pérdida, plagas) y cada opción explicada en palabras: en pocas palabras, lo bueno, lo que hay que cuidar, qué preguntarle al asesor.
  - Los números, gráficos, rangos de incertidumbre y la validación contra NASS siguen disponibles en "Detalles técnicos".
- **Modelo:** el riego se aplica solo a los cultivos donde se paga (el valor del grano extra por mm supera el costo de bombeo y manejo), y el costo del manejo con riego crece con el agua usada.

## Estructura

```
rotacion/config.py        catálogo de cultivos y supuestos agronómicos (editable por un agrónomo)
rotacion/fuentes.py       datos por lat/lon o polígono: local -> API en vivo -> "no disponible"
rotacion/indicadores.py   ET0, balance hídrico, extremos, heladas, tendencias, suelo, acuífero, caudal
rotacion/motor.py         simulación año a año de cada rotación + Monte Carlo
rotacion/explicaciones.py qué significa cada número, umbrales y efecto en la rotación
rotacion/servicio.py      une todo; capas del mapa (clima, campos, suelo)
rotacion/simple.py        traducción a lenguaje simple: semáforos, antes/ahora, mes a mes, rotación detectada
scripts/descargar_campos.py  guarda los campos USDA de una zona para uso sin conexión
rotacion/api.py           API Flask + sirve la web
web/index.html            mapa (Leaflet) y panel: Mi zona / Comparar rotaciones / ¿Qué significa?
data/                     power_kansas/ (112 puntos, 1981-2025), nwis/, nass/, ssurgo/, _cache/
```

## Qué se corrigió (los 8 puntos débiles)

1. **Mapa e interfaz.** `web/index.html` tiene un mapa con capas:
   - Clima por celda POWER (déficit hídrico, calor, tendencia del calor, heladas tardías, lluvia extrema, sequedad de primavera, humedad).
   - Pozos USGS coloreados según cuánto baja el acuífero.
   - Polígonos de suelo SSURGO del campo con riesgo de erosión eólica e hídrica y estrés hídrico.

   Para elegir la zona: clic en el mapa o "Dibujar campo" (rectángulo).
2. **Coordenadas y polígonos.** Toda función recibe una `Zona` (lat/lon o polígono GeoJSON). Primero usa los datos locales que cubren la zona y, si no hay, consulta en vivo:
   - NASA POWER API.
   - Soil Data Access (consulta espacial, ponderada por área).
   - Landsat C2 L2 vía STAC (Planetary Computer).
   - AppEEARS (SMAP, ECOSTRESS).
   - Data rods de GES DISC (IMERG).
   - USGS Water Data API (pozos) y NWIS (caudal).
   - NASS QuickStats.

   Todo se guarda en `data/_cache`. Si una fuente falla, aparece como brecha de información.
3. **Explicación de cada parámetro.** Cada número trae:
   - qué es y su unidad;
   - el tipo de dato (satélite, reanálisis, terreno, encuesta o cálculo);
   - un nivel (favorable, intermedio o desfavorable) con los umbrales usados;
   - cómo afecta a la rotación.

   El glosario explica ET, NDVI, riesgo de helada, anomalías, tendencias, la incertidumbre y la diferencia entre satélite y medición terrestre.
4. **Series largas.** Se usan 45 años de POWER (1981-2025). Las tendencias se calculan con pendiente de Sen y prueba de Mann-Kendall, y se compara 1981-2002 con 2003-2025. También hay un gráfico con media móvil por variable.
5. **Campo desnudo y épocas vulnerables.**
   - Calendario mensual de cada rotación: cultivo, rastrojo o suelo desnudo, con el riesgo de viento.
   - Tabla de meses vulnerables por cultivo según el clima del sitio (déficit, calor en floración o llenado, heladas).
   - Landsat en vivo de los últimos 3 años (NDVI mensual y meses de suelo desnudo).
6. **SMAP, GPM, ECOSTRESS, GRACE, acuífero y riego.**
   - SMAP, ECOSTRESS e IMERG se consultan en vivo con token.
   - El acuífero Ogallala se mide directamente con los niveles de pozos USGS: tasa de descenso por pozo y serie regional desde 1960. GRACE, con ~300 km de resolución, es demasiado grueso para un campo; se acepta como serie opcional.
   - El riego es una variable del agricultor (mm/año disponibles). Se aplica mes a mes y tiene costo de bombeo.
   - NASS separa secano y riego.
7. **Modelo agronómico secuencial.** Cada rotación se simula año a año con el clima real del sitio:
   - Balance hídrico mensual: Kc de FAO-56 sobre ET0 de Hargreaves, almacenamiento en barbecho y agua útil SSURGO.
   - Pérdida de rendimiento por déficit (Ky de FAO-33), por calor en etapas sensibles y por heladas.
   - Efecto del cultivo anterior: crédito de N de la soja, trigo después de sorgo, repetir el mismo cultivo.
   - Cobertura de residuo, de la que salen la erosión por viento y agua y la materia orgánica.
   - Plagas: intervalo entre hospederos, alternancia invierno/verano y diversidad.
   - Margen con rendimientos NASS.

   El orden importa (hay un test que lo comprueba) y sitios distintos dan rankings distintos. La app muestra **cuánto se parece lo simulado a NASS**. En Finney, por ejemplo, el trigo tras barbecho da 37,6 bu/acre simulados frente a 34,8 reales.
8. **Escenarios y restricciones.**
   - Deslizadores de lluvia (−40 a +30 %) y temperatura (+0 a +4 °C), aplicados a los 45 años diarios.
   - Restricciones: agua de riego, labranza, cultivos sin maquinaria o sin comprador, precios propios y margen mínimo.

**Errores corregidos del código original:**
- NASS mezclaba maíz grano (bu/acre) con ensilaje (t/acre) y "net planted acre", así que el promedio de maíz salía mal.
- El suelo se ponderaba solo por el % de componente; ahora también por el área de la unidad.

**Fuentes del código original que se reemplazaron:**
- PRISM (solo 2023 fijo) se reemplazó por la serie diaria de POWER 1981-2025.
- GAEZ se dejó de usar: escalaba los puntajes por un índice global (alfalfa = 0 en toda la zona). La aptitud ahora sale de la simulación con el clima del sitio.
- OpenET sigue como fuente local opcional: `data/openet/openet_et_monthly.json`, en el mismo formato de antes.

## Supuestos y límites (también se muestran en la app)

- Kc, Ky, costos directos, precios, residuo y efectos del cultivo previo son **supuestos orientativos** (`config.py`).
- POWER tiene ~50 km de resolución: no distingue lomas de bajos dentro del campo.
- El escenario climático es tipo "delta": no cambia la frecuencia de tormentas ni el viento.
- Las consultas en vivo (POWER, SDA, STAC, AppEEARS, IMERG, USGS, campos USDA) siguen la documentación pública, pero **no se probaron contra los servicios reales** en el entorno donde se escribió el código, que no tenía acceso a esos dominios. Se probaron con respuestas simuladas y con los datos locales de Finney.
- Hay que validar con extensión de K-State y con NRCS antes de cambiar la rotación.
