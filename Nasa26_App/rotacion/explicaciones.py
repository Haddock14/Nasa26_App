"""Qué significa cada número, si es bueno o malo y cómo afecta la elección de rotación.

Los umbrales son orientativos para agricultura de secano/riego en las Grandes Llanuras; se muestran al
usuario para que pueda juzgarlos (y un asesor local corregirlos).
"""
from __future__ import annotations

import math

TIPOS = {
    "satelite": "Observación satelital: mide desde el espacio una propiedad de la superficie (reflectancia, temperatura, "
                "microondas). Cubre todo el campo de forma pareja, pero depende de nubes, de la resolución del píxel y de "
                "modelos para convertir la señal en la variable agronómica.",
    "reanalisis": "Modelo/reanálisis (NASA POWER-MERRA-2): combina satélites, estaciones y un modelo físico en una grilla de "
                  "~50 km. Es continuo y largo (1981-hoy) pero suaviza diferencias locales: una tormenta puede no aparecer.",
    "terreno": "Medición en terreno: pozos, estaciones de caudal o encuestas de suelo. Es precisa en el punto medido pero "
               "no siempre representa el campo vecino.",
    "encuesta": "Estadística agrícola (USDA NASS): promedio de rendimientos declarados en el condado; no es su campo.",
    "modelo_local": "Cálculo de esta herramienta a partir de los datos anteriores (fórmulas FAO y supuestos editables).",
}

SATELITE_VS_TERRENO = (
    "Los satélites (Landsat, SMAP, ECOSTRESS, GPM, GRACE) ven grandes áreas de forma repetida y uniforme, pero miden "
    "señales indirectas y con limitaciones: SMAP mide solo los primeros ~5 cm de suelo en píxeles de 9 km; GRACE detecta "
    "cambios de agua en celdas de ~300 km; Landsat no ve a través de nubes y pasa cada 8-16 días. Las mediciones de "
    "terreno (pozos USGS, aforos, perfiles de suelo SSURGO) son exactas donde se toman, pero son puntuales. Combinarlas "
    "mejora la precisión: el satélite aporta el patrón espacial y temporal y el terreno lo calibra. Por eso esta "
    "herramienta usa los pozos USGS para el acuífero Ogallala (GRACE es demasiado grueso para un campo) y POWER/IMERG "
    "para clima, y avisa cuando una fuente falta.")

CONCEPTOS = {
    "evapotranspiracion": "Evapotranspiración (ET): agua que sale del campo por evaporación del suelo más transpiración de "
                          "las plantas. La ET de referencia (ET0) es lo que consumiría un pasto bien regado con el clima del "
                          "día; un cultivo consume ET0 x Kc (coeficiente de cultivo). Si la lluvia más el agua del suelo no "
                          "cubren ese consumo, el cultivo sufre y rinde menos.",
    "balance_hidrico": "Balance hídrico: lluvia menos ET0. Un déficit grande y persistente indica que en secano conviene "
                       "barbecho o cultivos tolerantes (sorgo, trigo) y que el riego depende del acuífero.",
    "ndvi": "NDVI: índice de verdor calculado con luz roja e infrarroja (Landsat). Suelo desnudo ~0.1-0.2, cultivo en "
            "pleno crecimiento 0.6-0.9. Sirve para ver cuándo el campo está desnudo y cuándo el cultivo se estresa.",
    "riesgo_helada": "Riesgo de heladas: frecuencia de noches bajo 0 °C (o -2 °C) en meses sensibles. Una helada tardía "
                     "en abril daña el trigo en espigazón; una en mayo puede obligar a resembrar maíz o soja.",
    "anomalia_humedad": "Anomalía de humedad del suelo: cuánto se aleja la humedad de un mes de su promedio histórico, "
                        "en desviaciones estándar. -1 o menos es un mes claramente más seco de lo normal.",
    "tendencia": "Tendencia: pendiente de Sen (cambio típico por década, robusta a años raros) con prueba de "
                 "Mann-Kendall. 'Significativa' (p<0.05) quiere decir que es poco probable que el cambio sea azar.",
    "incertidumbre": "Incertidumbre: cada rotación se simula 300 veces con años de clima distintos del registro 1981-2025, "
                     "precios y parámetros agronómicos variados y pesos de prioridad perturbados. El rango p10-p90 muestra "
                     "dónde cae el puntaje 8 de cada 10 veces; 'probabilidad de ser la mejor' es la fracción de "
                     "simulaciones en que esa rotación quedó primera. Rangos que se solapan = diferencia no concluyente.",
    "fijadoras_n": "Leguminosas (soja, alfalfa, vicia) fijan nitrógeno del aire con bacterias en sus raíces; parte queda "
                   "para el cultivo siguiente y reduce fertilizante.",
    "coberturas": "Cultivos de cobertura: protegen el suelo cuando no hay cultivo comercial, frenan la erosión por viento "
                  "y agua y aportan materia orgánica. En clima seco consumen agua que el cultivo siguiente no tendrá.",
    "raices_profundas": "Cultivos de raíz profunda (girasol, alfalfa, sorgo) extraen agua y nutrientes de capas hondas y "
                        "dejan canales que mejoran la infiltración.",
    "tolerantes_sequia": "Cultivos tolerantes a sequía y calor (sorgo, trigo, girasol) pierden menos rendimiento por "
                         "unidad de déficit hídrico (Ky menor) que el maíz.",
}

# clave: (nombre, unidad, tipo, qué es, efecto en la rotación, [(limite, nivel, texto)], sentido)
# sentido "alto_malo": valores mayores son peores; "alto_bueno": al revés
G = {
    "lluvia_mm": ("Lluvia anual", "mm/año", "reanalisis", "Precipitación total del año (POWER PRECTOTCORR).",
                  "Menos de ~500 mm limita el secano a cultivos tolerantes y favorece el barbecho para acumular agua.",
                  [(400, "desfavorable", "muy seco para cultivo continuo en secano"), (600, "intermedio", "semiárido: secano posible con manejo del agua"),
                   (math.inf, "favorable", "suficiente para intensificar la rotación")], "alto_bueno"),
    "et0_mm": ("Evapotranspiración de referencia (ET0)", "mm/año", "modelo_local",
               "Demanda de agua de la atmósfera, calculada con temperaturas máx/mín (Hargreaves-Samani).",
               "Una ET0 alta aumenta el consumo de cualquier cultivo; con la misma lluvia, más ET0 = más estrés.",
               [(1100, "favorable", "demanda moderada"), (1400, "intermedio", "demanda alta"), (math.inf, "desfavorable", "demanda muy alta")], "alto_malo"),
    "deficit_mm": ("Déficit hídrico (ET0 - lluvia)", "mm/año", "modelo_local", "Diferencia entre la demanda y la lluvia del año.",
                   "Es el principal indicador para decidir cuánta intensificación aguanta el secano y cuánto riego se requeriría.",
                   [(300, "favorable", "déficit bajo"), (700, "intermedio", "déficit importante"), (math.inf, "desfavorable", "déficit severo: la lluvia cubre menos de la mitad de la demanda")], "alto_malo"),
    "evap_suelo_mm": ("Evapotranspiración real modelada", "mm/año", "reanalisis",
                      "Agua que efectivamente se evaporó/transpiró según el modelo terrestre de MERRA-2 (EVLAND).",
                      "Si es casi igual a la lluvia, el suelo no recarga el acuífero ni guarda reservas entre campañas.", [], None),
    "dias_calor_35": ("Días con máxima ≥ 35 °C", "días/año", "reanalisis", "Cantidad de días muy calurosos.",
                      "El maíz pierde rendimiento si coinciden con la floración (julio); el sorgo tolera más.",
                      [(20, "favorable", "pocos días extremos"), (45, "intermedio", "frecuentes: evitar floración en julio"), (math.inf, "desfavorable", "calor extremo habitual")], "alto_malo"),
    "dias_calor_32_may_jun": ("Días ≥ 32 °C en mayo-junio", "días/año", "reanalisis", "Calor durante el llenado de grano del trigo.",
                              "Acelera la madurez del trigo y reduce el peso del grano.",
                              [(5, "favorable", "llenado de grano sin estrés"), (15, "intermedio", "estrés moderado en llenado"), (math.inf, "desfavorable", "llenado de grano comprometido")], "alto_malo"),
    "heladas_tardias_abr_may": ("Heladas tardías (≤ -2 °C, abril-mayo)", "eventos/año", "reanalisis",
                                "Noches frías cuando el trigo está encañando/espigando y se siembran cultivos de verano.",
                                "Riesgo para el trigo temprano y para siembras de maíz/soja adelantadas.",
                                [(0.5, "favorable", "riesgo bajo"), (2, "intermedio", "riesgo ocasional"), (math.inf, "desfavorable", "riesgo frecuente")], "alto_malo"),
    "periodo_libre_heladas_dias": ("Período libre de heladas", "días", "reanalisis", "Días entre la última helada de primavera y la primera de otoño (0 °C).",
                                   "Define qué cultivos de verano y qué fechas de siembra caben; también si entra un cultivo de cobertura tras la cosecha.",
                                   [(150, "desfavorable", "temporada corta"), (180, "intermedio", "temporada media"), (math.inf, "favorable", "temporada larga")], "alto_bueno"),
    "ultima_helada_doy": ("Última helada de primavera", "día del año", "reanalisis", "Fecha media de la última noche ≤ 0 °C.",
                          "Sembrar maíz/soja antes de esta fecha arriesga resiembra.", [], None),
    "racha_seca_abr_sep_dias": ("Racha seca más larga (abr-sep)", "días", "reanalisis", "Máximo de días seguidos con menos de 1 mm.",
                                "Rachas largas en floración son críticas; favorece cultivos con floración flexible (sorgo) o barbecho previo.",
                                [(15, "favorable", "rachas cortas"), (25, "intermedio", "rachas largas ocasionales"), (math.inf, "desfavorable", "rachas muy largas")], "alto_malo"),
    "dias_lluvia_extrema_25mm": ("Días de lluvia intensa (≥ 25 mm)", "días/año", "reanalisis", "Tormentas que generan escurrimiento.",
                                 "Con suelo desnudo producen erosión hídrica y pérdida de agua; residuo y coberturas la reducen.",
                                 [(2, "favorable", "pocas tormentas intensas"), (5, "intermedio", "tormentas intensas habituales"), (math.inf, "desfavorable", "muy frecuentes")], "alto_malo"),
    "lluvia_p95_mm": ("Lluvia diaria extrema (percentil 95)", "mm/día", "reanalisis", "Uno de cada 20 días lluviosos supera este valor.",
                      "Indica la intensidad típica de las tormentas fuertes (POWER suaviza los picos; IMERG los ve mejor).", [], None),
    "humedad_raiz_abr_sep": ("Humedad en zona de raíces", "fracción 0-1", "reanalisis",
                             "Fracción de saturación del suelo en la zona de raíces (POWER GWETROOT, modelo). 0 = seco, 1 = saturado.",
                             "Por debajo de ~0.3 el cultivo de verano suele estar en estrés; si es baja al sembrar, conviene barbecho o cultivo tolerante.",
                             [(.3, "desfavorable", "suelo seco en la temporada"), (.5, "intermedio", "humedad moderada"), (math.inf, "favorable", "buena humedad")], "alto_bueno"),
    "awc_100cm_mm": ("Agua útil del suelo (0-100 cm)", "mm", "terreno", "Agua que el suelo puede guardar y entregar a las plantas (SSURGO).",
                     "Suelos con mucha agua útil aprovechan el barbecho y permiten intensificar; los arenosos no guardan la lluvia.",
                     [(100, "desfavorable", "baja: suelo arenoso o somero"), (160, "intermedio", "media"), (math.inf, "favorable", "alta: buen almacenamiento")], "alto_bueno"),
    "materia_organica_pct": ("Materia orgánica (0-30 cm)", "%", "terreno", "Carbono orgánico del suelo superficial (SSURGO, valor representativo).",
                             "Baja MO = menos agregación, infiltración y nutrientes; rotaciones con más residuo y menos barbecho la mantienen.",
                             [(1.0, "desfavorable", "baja"), (2.0, "intermedio", "media para clima semiárido"), (math.inf, "favorable", "alta")], "alto_bueno"),
    "arena_pct": ("Arena", "%", "terreno", "Porcentaje de arena del suelo superficial.", "Más arena = más erosión eólica y menos agua útil.",
                  [(30, "favorable", "textura media/fina"), (60, "intermedio", "franco-arenosa"), (math.inf, "desfavorable", "arenosa")], "alto_malo"),
    "pendiente_pct": ("Pendiente media", "%", "terreno", "Inclinación del terreno (SSURGO o SRTM).",
                      "Sobre ~3 % la erosión hídrica crece rápido con suelo desnudo.",
                      [(2, "favorable", "plano"), (5, "intermedio", "ondulado"), (math.inf, "desfavorable", "pendiente fuerte")], "alto_malo"),
    "kffact": ("Erodabilidad del suelo (factor K)", "t·ha·h/(ha·MJ·mm)", "terreno", "Facilidad con que el agua arrastra el suelo (RUSLE).",
               "K alto + pendiente + suelo desnudo = mayor pérdida de suelo.",
               [(.2, "favorable", "poco erodable"), (.35, "intermedio", "moderado"), (math.inf, "desfavorable", "muy erodable")], "alto_malo"),
    "frac_escurrimiento_alto": ("Suelos con escurrimiento alto (grupos C/D)", "fracción", "terreno", "Parte del área con infiltración lenta.",
                                "Más escurrimiento = menos lluvia útil; coberturas y residuo mejoran la infiltración.",
                                [(.3, "favorable", "buena infiltración"), (.6, "intermedio", "infiltración mixta"), (math.inf, "desfavorable", "infiltración lenta")], "alto_malo"),
    "grupo_erosion_eolica_medio": ("Grupo de erosión eólica (WEG)", "1-8", "terreno", "Clasificación NRCS: 1 = muy susceptible al viento, 8 = poco.",
                                   "En grupos 1-3 dejar el suelo desnudo en primavera es muy riesgoso: priorizar residuo y coberturas.",
                                   [(3, "desfavorable", "muy susceptible al viento"), (5, "intermedio", "susceptible"), (math.inf, "favorable", "poco susceptible")], "alto_bueno"),
    "frac_erosion_eolica_alta": ("Área muy susceptible al viento (WEG 1-3)", "fracción", "terreno", "Parte de la zona con suelos que vuelan fácilmente.",
                                 "Esas partes del campo necesitan cobertura todo el año.", [(.1, "favorable", "poca"), (.3, "intermedio", "parte importante"), (math.inf, "desfavorable", "mayoría")], "alto_malo"),
    "clase_capacidad_secano": ("Clase de capacidad de uso (secano)", "1-8", "terreno", "Clasificación NRCS: 1-2 sin limitaciones fuertes, 3-4 limitaciones (clima, erosión), 5+ no apto para cultivo continuo.",
                               "Clases altas sugieren rotaciones con más cobertura o pasturas.", [(2.5, "favorable", "pocas limitaciones"), (4.5, "intermedio", "limitaciones moderadas"), (math.inf, "desfavorable", "limitaciones severas")], "alto_malo"),
    "tasa_mediana_m_anio": ("Descenso del acuífero (pozos USGS)", "m/año", "terreno",
                            "Cuánto baja por año la profundidad al agua en los pozos cercanos (mediciones de invierno, desde 2000). Positivo = el nivel baja.",
                            "Si el acuífero baja, el riego futuro es incierto: conviene rotaciones que rindan en secano o usen poco riego.",
                            [(.1, "favorable", "estable"), (.5, "intermedio", "descenso moderado"), (math.inf, "desfavorable", "descenso rápido: el riego es insostenible a largo plazo")], "alto_malo"),
    "prof_mediana_ultima_ft": ("Profundidad al agua subterránea", "pies", "terreno", "Profundidad mediana en la última medición de los pozos cercanos.",
                               "Más profundidad = bombeo más caro y menos caudal por pozo.", [], None),
    "frac_dias_sin_flujo": ("Días sin caudal en el río", "fracción", "terreno", "Fracción de días con caudal cero en la estación USGS.",
                            "Un río seco indica que el agua superficial no es una alternativa al acuífero.", [(.1, "favorable", "flujo casi permanente"), (.4, "intermedio", "seco con frecuencia"), (math.inf, "desfavorable", "seco la mayor parte del tiempo")], "alto_malo"),
    "cv_pct": ("Variabilidad del rendimiento (CV)", "%", "encuesta", "Desvío estándar / promedio de los rendimientos del condado año a año.",
               "CV alto = cultivo riesgoso en ese condado; conviene diversificar o elegir cultivos más estables.",
               [(15, "favorable", "estable"), (30, "intermedio", "variable"), (math.inf, "desfavorable", "muy variable")], "alto_malo"),
    "ndvi": ("NDVI (verdor)", "-1 a 1", "satelite", CONCEPTOS["ndvi"], "Meses con NDVI < 0.25 son meses de suelo desnudo o casi desnudo.",
             [(.25, "desfavorable", "suelo desnudo o vegetación muy escasa"), (.5, "intermedio", "vegetación parcial"), (math.inf, "favorable", "cobertura vegetal densa")], "alto_bueno"),
    "lst_c": ("Temperatura de superficie (Landsat/ECOSTRESS)", "°C", "satelite", "Temperatura del suelo/canopeo vista desde el satélite.",
              "Un canopeo mucho más caliente que el aire indica cierre de estomas por falta de agua (estrés).", [], None),
    "smap": ("Humedad superficial (SMAP)", "m³/m³", "satelite", "Agua en los primeros ~5 cm del suelo, píxel de 9 km.",
             "Complementa el modelo POWER con observación directa: valores < 0.15 indican suelo seco al sembrar.",
             [(.15, "desfavorable", "seco"), (.25, "intermedio", "moderado"), (math.inf, "favorable", "húmedo")], "alto_bueno"),
}


def interpretar(clave, valor):
    g = G.get(clave)
    if not g or valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return None
    nombre, unidad, tipo, que, efecto, umbrales, _ = g
    nivel = texto = None
    for lim, niv, txt in umbrales:
        if valor < lim:
            nivel, texto = niv, txt
            break
    return dict(nombre=nombre, unidad=unidad, valor=valor, tipo_dato=tipo, que_es=que, efecto_rotacion=efecto,
                nivel=nivel, lectura=texto,
                umbrales=[dict(hasta=None if math.isinf(l) else l, nivel=n, texto=t) for l, n, t in umbrales])


def anotar(d: dict):
    """Devuelve {clave: {valor, nombre, unidad, nivel, lectura, ...}} para las claves con glosario."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = interpretar(k, float(v)) or dict(valor=v)
        else:
            out[k] = v
    return out


def glosario():
    return dict(conceptos=CONCEPTOS, tipos_de_dato=TIPOS, satelite_vs_terreno=SATELITE_VS_TERRENO,
                indicadores={k: dict(nombre=v[0], unidad=v[1], tipo_dato=v[2], que_es=v[3], efecto_rotacion=v[4]) for k, v in G.items()})
