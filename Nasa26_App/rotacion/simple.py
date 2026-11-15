"""Traducción a lenguaje simple: semáforos, preguntas y comparaciones para personas sin formación técnica.

Cada resumen conserva los números técnicos en "numeros" para quien quiera verlos.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C

NOMBRE_MES = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
MES_LARGO = ["", "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _n(v, d=0):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ------------------------------------------------------------------ campos
def inferir_rotacion(seq):
    """seq: cultivos del catálogo por año (None = pasto u otro uso). Devuelve (ciclo, explicación)."""
    usados = [c for c in seq if c]
    if len(usados) < max(3, len(seq) // 2):
        return None, "La mayoría de los años no tuvo un cultivo anual (pastura, heno u otro uso)."
    s = [c or "barbecho" for c in seq]
    mejor = None
    for p in range(1, 5):
        pares = [(s[i], s[i - p]) for i in range(p, len(s))]
        coincide = sum(a == b for a, b in pares) / len(pares)
        if coincide >= .7:
            mejor = p
            break
    if mejor is None:
        ciclo = list(dict.fromkeys(s[-3:]))
        return ciclo, "Los cultivos cambiaron sin un patrón fijo; tomamos los últimos años como referencia."
    ciclo = s[-mejor:]
    if "trigo" in ciclo:
        i = ciclo.index("trigo")
        ciclo = ciclo[i:] + ciclo[:i]
    if mejor == 1:
        txt = f"Se sembró {C.CULTIVOS[ciclo[0]]['nombre'].lower()} casi todos los años (monocultivo)."
    else:
        txt = f"Se repite un ciclo de {mejor} años: " + " → ".join(C.CULTIVOS[c]["nombre"].lower() for c in ciclo) + "."
    return ciclo, txt


def circularidad(anillo, lat):
    """4πA/P²: 1 = círculo perfecto (pivote), 0.785 = cuadrado."""
    k = math.cos(math.radians(lat))
    pts = [(x * 111.32 * k, y * 111.32) for x, y in anillo]
    a = p = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        a += x1 * y2 - x2 * y1
        p += math.hypot(x2 - x1, y2 - y1)
    a = abs(a) / 2
    return 4 * math.pi * a / p ** 2 if p else 0


def area_m2(anillo, lat):
    k = math.cos(math.radians(lat))
    pts = [(x * 111320 * k, y * 110574) for x, y in anillo]
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]))) / 2


def analizar_campo(props, geom):
    from .fuentes import _anillo
    anillo = _anillo(geom)
    ys = [y for _, y in anillo]
    lat = sum(ys) / len(ys)
    hist = []
    for a in C.CSB_ANIOS:
        cod = props.get(f"cdl{a}", props.get(f"CDL{a}"))
        try:
            cod = int(cod)
        except (TypeError, ValueError):
            cod = None
        nombre, clave, color = C.CDL.get(cod, ("Otro uso", None, "#cccccc")) if cod else ("sin dato", None, "#eeeeee")
        hist.append(dict(anio=a, codigo=cod, nombre=nombre, cultivo=clave, color=color))
    if not any(h["codigo"] for h in hist):      # p. ej. campos de OpenStreetMap: sin historial
        ciclo, txt = None, "No tenemos el historial de cultivos de este campo: indique usted qué siembra en el paso 3."
        for h in hist:
            h.update(nombre="sin dato", color="#8fbf6a")
    else:
        ciclo, txt = inferir_rotacion([h["cultivo"] for h in hist])
    circ = circularidad(anillo, lat)
    ult = hist[-1]
    barbechos = sum(h["cultivo"] == "barbecho" for h in hist)
    xs = [x for x, _ in anillo]
    acres = props.get("csbacres")
    try:
        acres = float(acres)
    except (TypeError, ValueError):
        acres = 0
    if not acres:      # sin dato oficial: se calcula con la forma del campo
        acres = area_m2(anillo, lat) / 4046.86
    return dict(id=props.get("csbid"), acres=round(acres), historial=hist,
                rotacion_detectada=ciclo, rotacion_texto=txt, ultimo_cultivo=ult["nombre"], color=ult["color"],
                anios_barbecho=barbechos, circularidad=round(circ, 2), pivote_probable=circ >= .86,
                lat=round(lat, 5), lon=round(sum(xs) / len(xs), 5))


# ------------------------------------------------------------------ evidencia en preguntas
def nivel(bueno, malo, v, alto_es_malo=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "sin dato"
    if alto_es_malo:
        return "bien" if v <= bueno else ("problema" if v >= malo else "atención")
    return "bien" if v >= bueno else ("problema" if v <= malo else "atención")


NIVEL_SIMPLE = {"favorable": "bien", "intermedio": "atención", "desfavorable": "problema"}


def _f(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def dato(nombre, valor, unidad, explicacion, umbrales=None, alto_bueno=None, dec=0, texto=None, clave=None):
    """Un valor con: qué significa, una escala de referencia en palabras y en qué tramo cae.
    umbrales: [(limite, nivel, texto)] como en explicaciones.G (o se toman de G con `clave`)."""
    from . import explicaciones as X
    if clave and umbrales is None and clave in X.G:
        umbrales = X.G[clave][5]
    escala, nivel, lectura, prev = [], None, None, None
    for lim, niv, txt in (umbrales or []):
        fmt = lambda x: _n(x, 0 if float(x).is_integer() else max(dec, 1))
        if prev is None:
            rango = f"menos de {fmt(lim)}"
        elif math.isinf(lim):
            rango = f"{fmt(prev)} o más"
        else:
            rango = f"{fmt(prev)} a {fmt(lim)}"
        actual = _f(valor) and (prev is None or valor >= prev) and valor < lim
        niv_s = NIVEL_SIMPLE.get(niv, niv)
        escala.append(dict(rango=rango, texto=txt, nivel=niv_s, actual=bool(actual)))
        if actual:
            nivel, lectura = niv_s, txt
        prev = lim
    return dict(nombre=nombre, valor=valor if _f(valor) else None, texto_valor=texto or (f"{_n(valor, dec)} {unidad}".strip() if _f(valor) else "sin dato"),
                unidad=unidad, explicacion=explicacion, escala=escala, nivel=nivel, lectura=lectura)


def tarjetas(cl, su, acu, cau):
    r, t, per = cl["resumen"], cl["tendencias"], cl["periodos"]
    inf = math.inf
    cobertura = 100 * r["lluvia_mm"] / r["et0_mm"]
    out = []
    out.append(dict(
        clave="agua", pregunta="¿Alcanza la lluvia para los cultivos?",
        respuesta=f"La lluvia cubre cerca del {cobertura:.0f}% del agua que necesita un cultivo en pleno crecimiento.",
        nivel=nivel(70, 40, cobertura, alto_es_malo=False),
        para_usted=("Sin riego, el suelo tiene que guardar agua entre un cultivo y otro (barbecho o rastrojo). Convienen cultivos "
                    "que resisten la sequía, como sorgo y trigo, más que maíz o soja." if cobertura < 50 else
                    "Hay agua para sembrar más seguido, pero los años secos siguen siendo un riesgo."),
        detalles=[
            dato("Parte del agua necesaria que cubre la lluvia", cobertura, "%",
                 "Compara la lluvia del año con el agua que pierde un cultivo sano por el sol, el calor y el viento. 100% significa que la lluvia alcanza justo.",
                 [(40, "desfavorable", "la lluvia cubre menos de la mitad: el secano es difícil"), (70, "intermedio", "alcanza en parte: depende de guardar agua en el suelo"),
                  (inf, "favorable", "la lluvia cubre casi todo lo necesario")]),
            dato("Lluvia promedio del año", r["lluvia_mm"], "mm", "Toda el agua que cae en un año (lluvia y nieve), promedio de 1981 a 2025. 1 mm equivale a 1 litro por metro cuadrado.", clave="lluvia_mm"),
            dato("Agua que pide el ambiente", r["et0_mm"], "mm",
                 "Cantidad de agua que un cultivo bien regado perdería en un año por el sol, el calor y el viento. Los técnicos la llaman evapotranspiración de referencia (ET0).", clave="et0_mm"),
            dato("Agua que falta", r["deficit_mm"], "mm", "Diferencia entre el agua que pide el ambiente y la lluvia. Es lo que tendría que aportar el suelo o el riego.", clave="deficit_mm"),
            dato("Racha seca más larga (abril a septiembre)", r["racha_seca_abr_sep_dias"], "días", "Promedio del mayor número de días seguidos sin lluvia útil (menos de 1 mm) durante la temporada.", clave="racha_seca_abr_sep_dias"),
            dato("Años muy secos", 100 * r["frac_anios_deficit_severo"], "%", "Porcentaje de años en que llovió menos del 70% de lo normal.",
                 [(10, "favorable", "pocos años muy secos"), (25, "intermedio", "uno de cada cuatro a diez años es muy seco"), (inf, "desfavorable", "los años muy secos son frecuentes")]),
        ], fuente="NASA POWER, 1981-2025 (modelo climático con datos de satélite)"))
    ca, ch = (per["dias_calor_35"]["media_1"], per["dias_calor_35"]["media_2"]) if per.get("dias_calor_35") else (None, None)
    tc = t["dias_calor_35"]
    out.append(dict(
        clave="calor", pregunta="¿Hace demasiado calor para los cultivos?",
        respuesta=f"Hay unos {_n(r['dias_calor_35'])} días al año con más de 35 °C" + (f" (eran {_n(ca)} antes de 2003 y ahora son {_n(ch)})." if ca else "."),
        nivel=nivel(20, 45, r["dias_calor_35"]),
        para_usted="El maíz pierde rendimiento si florece con ese calor (julio). El sorgo y el girasol lo resisten mejor; el trigo lo evita porque se cosecha en junio.",
        detalles=[
            dato("Días de calor extremo", r["dias_calor_35"], "días por año", "Días en que la temperatura máxima llega a 35 °C o más. Con ese calor muchas plantas cierran sus poros y dejan de crecer.", clave="dias_calor_35"),
            dato("Días de calor en mayo y junio", r["dias_calor_32_may_jun"], "días por año", "Días con 32 °C o más cuando el trigo está llenando el grano. Más días significan granos más chicos.", clave="dias_calor_32_may_jun"),
            dato("Cambio por década", tc["pendiente_por_decada"], "días", "Cuántos días de calor extremo se suman cada 10 años según la tendencia desde 1981. " +
                 ("El cambio está confirmado por la estadística (no es casualidad)." if tc["significativa"] else "Todavía no está confirmado: puede ser variación natural entre años."),
                 [(1, "favorable", "sin aumento claro"), (3, "intermedio", "aumento moderado"), (inf, "desfavorable", "aumento rápido")], dec=1),
        ], fuente="NASA POWER, 1981-2025"))
    out.append(dict(
        clave="heladas", pregunta="¿Hay riesgo de heladas tardías?",
        respuesta=f"La última helada de primavera llega en promedio el {r.get('ultima_helada') or '—'} y hay {_n(r['heladas_tardias_abr_may'], 1)} noches muy frías por año en abril y mayo.",
        nivel=nivel(.5, 2, r["heladas_tardias_abr_may"]),
        para_usted="No siembre maíz ni soja antes de esa fecha. Una helada en abril puede dañar el trigo cuando está espigando.",
        detalles=[
            dato("Noches muy frías en abril y mayo", r["heladas_tardias_abr_may"], "por año", "Noches con -2 °C o menos cuando los cultivos ya están creciendo o recién sembrados.", clave="heladas_tardias_abr_may", dec=1),
            dato("Días seguidos sin heladas", r["periodo_libre_heladas_dias"], "días", "Tiempo entre la última helada de primavera y la primera de otoño. Define qué cultivos de verano alcanzan a madurar.", clave="periodo_libre_heladas_dias"),
            dato("Fecha de la última helada", None, "", "Fecha promedio de la última noche bajo 0 °C en primavera.", texto=r.get("ultima_helada") or "sin dato"),
            dato("Fecha de la primera helada", None, "", "Fecha promedio de la primera noche bajo 0 °C en otoño.", texto=r.get("primera_helada") or "sin dato"),
        ], fuente="NASA POWER, 1981-2025"))
    weg = su.get("grupo_erosion_eolica_medio")
    weg = 5 if not _f(weg) else weg
    fea = su.get("frac_erosion_eolica_alta") or 0
    out.append(dict(
        clave="erosion", pregunta="¿Se vuela o se lava el suelo?",
        respuesta=("Parte de los suelos se vuela con facilidad" if fea > .2 or weg <= 4 else "El suelo resiste razonablemente el viento") +
                  f"; hay {_n(r['dias_lluvia_extrema_25mm'], 1)} tormentas fuertes por año que pueden arrastrarlo si está desnudo.",
        nivel="problema" if weg <= 3 else ("atención" if weg <= 5 or fea > .2 else "bien"),
        para_usted="Los meses de mayor riesgo son de marzo a mayo. Conviene dejar rastrojo en pie o un cultivo de cobertura en esa época.",
        detalles=[
            dato("Resistencia del suelo al viento", weg, "en escala de 1 a 8", "Clasificación oficial de los suelos de EE. UU. (grupo de erosión eólica): 1 significa que el viento se lo lleva muy fácil; 8, que casi no se vuela.", clave="grupo_erosion_eolica_medio", dec=1),
            dato("Parte de la zona con suelos que se vuelan fácil", 100 * fea, "%", "Porcentaje del área con suelos de los grupos 1 a 3.",
                 [(10, "favorable", "poca superficie sensible"), (30, "intermedio", "una parte importante es sensible"), (inf, "desfavorable", "la mayoría es sensible")]),
            dato("Arena en la capa de arriba", su.get("arena_pct"), "%", "Los suelos con mucha arena se vuelan más y guardan menos agua.", clave="arena_pct"),
            dato("Pendiente del terreno", su.get("pendiente_pct"), "%", "Cuánto se inclina el terreno: 1% es un metro de bajada cada 100 metros. Con más pendiente, la lluvia arrastra más suelo.", clave="pendiente_pct", dec=1),
            dato("Tormentas fuertes", r["dias_lluvia_extrema_25mm"], "por año", "Días con 25 mm de lluvia o más. Si el suelo está desnudo, esa agua corre y se lleva tierra.", clave="dias_lluvia_extrema_25mm", dec=1),
        ], fuente="USDA SSURGO (mapa de suelos) y NASA POWER"))
    mo = su.get("materia_organica_pct")
    out.append(dict(
        clave="suelo", pregunta="¿Qué tan sano está el suelo?",
        respuesta=f"Tiene {_n(mo, 1)}% de materia orgánica y puede guardar unos {_n(su.get('awc_100cm_mm'))} mm de agua en el primer metro.",
        nivel=nivel(2, 1, mo or 0, alto_es_malo=False),
        para_usted="Dejar más rastrojo y menos meses de suelo desnudo aumenta la materia orgánica; así el suelo absorbe y guarda mejor el agua.",
        detalles=[
            dato("Materia orgánica", mo, "%", "Restos de plantas y raíces ya descompuestos en los primeros 30 cm. Mejora la estructura, la entrada de agua y los nutrientes.", clave="materia_organica_pct", dec=1),
            dato("Agua que el suelo puede guardar", su.get("awc_100cm_mm"), "mm", "Agua que el primer metro de suelo retiene y que las raíces pueden usar. Sirve de reserva entre lluvias.", clave="awc_100cm_mm"),
            dato("Arcilla", su.get("arcilla_pct"), "%", "Partículas muy finas que retienen agua y nutrientes. Mucha arcilla puede hacer el suelo pesado y lento para drenar.",
                 [(10, "intermedio", "poca arcilla: suelo liviano"), (35, "favorable", "proporción equilibrada"), (inf, "intermedio", "mucha arcilla: suelo pesado")]),
            dato("Aptitud para cultivar sin riego", su.get("clase_capacidad_secano"), "en escala de 1 a 8", "Clasificación oficial de aptitud del suelo: 1 y 2 casi sin limitaciones; 3 y 4 con limitaciones (clima, erosión); 5 o más, no apto para cultivo continuo.", clave="clase_capacidad_secano", dec=1),
        ], fuente="USDA SSURGO" + (" (valores por defecto: no hubo datos de esta zona)" if su.get("supuesto") else "")))
    if acu and acu.get("n_pozos"):
        m = acu["tasa_mediana_m_anio"]
        sin_flujo = max([e["frac_dias_sin_flujo"] for e in cau], default=None)
        det = [dato("Cuánto baja el agua subterránea", m, "m por año", f"Promedio de {acu['n_pozos']} pozos cercanos medidos en invierno desde el año 2000. Un valor positivo significa que cada año el agua está más honda.", clave="tasa_mediana_m_anio", dec=2),
               dato("Profundidad del agua", acu["prof_mediana_ultima_ft"] * .3048, "m", "Distancia desde la superficie hasta el agua en la última medición. Más profundidad significa bombeo más caro.",
                    [(15, "favorable", "agua cerca de la superficie"), (40, "intermedio", "profundidad media"), (inf, "desfavorable", "agua profunda")]),
               dato("Pozos donde el agua baja", 100 * acu["frac_pozos_en_descenso"], "%", "Porcentaje de pozos cercanos donde el nivel baja más de 6 cm por año.",
                    [(30, "favorable", "la mayoría estables"), (70, "intermedio", "muchos bajan"), (inf, "desfavorable", "casi todos bajan")])]
        if sin_flujo is not None:
            det.append(dato("Días con el río seco", 100 * sin_flujo, "%", "Porcentaje de días sin agua en la estación de aforo más cercana. Indica si el agua de río puede ser una alternativa.",
                            [(10, "favorable", "casi siempre tiene agua"), (40, "intermedio", "se seca con frecuencia"), (inf, "desfavorable", "pasa seco la mayor parte del tiempo")]))
        out.append(dict(
            clave="pozos", pregunta="¿Está bajando el agua de los pozos?",
            respuesta=f"Sí: el agua subterránea baja unos {_n(m, 1)} m por año en {acu['n_pozos']} pozos cercanos." if m > .1 else "El nivel de los pozos cercanos está bastante estable.",
            nivel=nivel(.1, .5, m),
            para_usted="Si riega, cada año el agua estará más honda y costará más bombearla. Convienen rotaciones que rindan con poco riego." if m > .3 else
                       "El riego parece sostenible por ahora; revise sus derechos de agua.",
            detalles=det, fuente="Mediciones del USGS en pozos y ríos"))
    return out


def antes_ahora(cl):
    p = cl["periodos"]
    t = cl["tendencias"]
    items = [("lluvia_mm", "Lluvia del año", "mm", False), ("dias_calor_35", "Días de calor extremo (≥ 35 °C)", "días", True),
             ("deficit_mm", "Agua que falta para los cultivos", "mm", True), ("humedad_raiz_abr_sep", "Humedad del suelo en verano", "0-1", False)]
    out = []
    for k, nombre, u, sube_malo in items:
        x = p.get(k)
        if not x:
            continue
        cambio = x["media_2"] - x["media_1"]
        rel = cambio / x["media_1"] * 100 if x["media_1"] else 0
        if abs(rel) < 3:
            frase, tono = "Casi igual", "igual"
        else:
            sube = cambio > 0
            frase = ("Aumentó" if sube else "Bajó") + f" {abs(rel):.0f}%"
            tono = "peor" if sube == sube_malo else "mejor"
        seguro = t.get(k, {}).get("significativa")
        out.append(dict(clave=k, nombre=nombre, unidad=u, antes=x["media_1"], ahora=x["media_2"], periodo_antes=x["periodo_1"],
                        periodo_ahora=x["periodo_2"], frase=frase, tono=tono,
                        certeza="cambio confirmado" if seguro else "puede ser variación natural"))
    return out


def mes_a_mes(cl):
    clim = cl["climatologia"]
    out = []
    for m in [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        r = clim.loc[m]
        falta = max(0.0, r.et0 * .8 - r.lluvia)
        avisos = []
        if falta > 60:
            avisos.append("falta agua")
        if r.calor35 >= 4:
            avisos.append(f"{r.calor35:.0f} días de calor extremo")
        if r.prob_helada >= .5 and m in (3, 4, 5, 10):
            avisos.append("heladas probables")
        if m in (3, 4, 5):
            avisos.append("viento fuerte: no deje el suelo desnudo")
        out.append(dict(mes=m, nombre=NOMBRE_MES[m], lluvia=round(float(r.lluvia)), necesidad=round(float(r.et0 * .8)), falta=round(falta),
                        calor=round(float(r.calor35), 1), prob_helada=round(float(r.prob_helada), 2), avisos=avisos,
                        nivel="problema" if len(avisos) >= 2 or falta > 100 else ("atención" if avisos else "bien")))
    return out


# ------------------------------------------------------------------ rotaciones en palabras simples
def estrellas(v):
    return int(max(1, min(5, round(1 + 4 * v))))


def rotacion_simple(r, act):
    cr, me, ma = r["criterios"], r["metricas"], act["metricas"]
    suelo = (cr["eros"] + cr["mo"]) / 2
    tramo = lambda v, cortes: 1 + sum(v >= c for c in cortes)
    # agua: qué tan cubierta queda la necesidad del cultivo, castigando el riego que se saca del acuífero
    agua = .5 * np.clip((me.get("agua_satisfecha", .7) - .45) / .5, 0, 1) + .5 * (1 - min(1, me["riego_mm_anio"] / 450))
    puntos = dict(agua=estrellas(agua), suelo=estrellas(suelo),
                  dinero=tramo(me["margen_usd_acre"], [0, 40, 80, 150]),
                  seguridad=tramo(-r["prob_margen_negativo"], [-.5, -.35, -.2, -.1]),
                  plagas=estrellas(cr["plagas"]))
    es_actual = r is act
    frases = []
    if not es_actual:
        dm = me["margen_usd_acre"] - ma["margen_usd_acre"]
        if abs(dm) >= 5:
            frases.append(f"{'gana' if dm > 0 else 'pierde'} unos {abs(dm):.0f} USD por acre al año")
        de = (ma["eros_relativa"] - me["eros_relativa"]) / max(ma["eros_relativa"], .01)
        if abs(de) >= .1:
            frases.append(f"{'protege mejor' if de > 0 else 'protege peor'} el suelo ({abs(de) * 100:.0f}% {'menos' if de > 0 else 'más'} erosión)")
        dmo = me["mo_aporte_t_ha"] - ma["mo_aporte_t_ha"]
        if abs(dmo) >= .3:
            frases.append(f"deja {'más' if dmo > 0 else 'menos'} rastrojo para la materia orgánica")
        dr = me["riego_mm_anio"] - ma["riego_mm_anio"]
        if abs(dr) >= 25:
            frases.append(f"usa {abs(dr):.0f} mm {'más' if dr > 0 else 'menos'} de riego por año")
        dn = cr["n"] - act["criterios"]["n"]
        if dn > .1:
            frases.append("ahorra fertilizante nitrogenado")
        dp = r["prob_margen_negativo"] - act["prob_margen_negativo"]
        if abs(dp) >= .08:
            frases.append(f"tiene {'más' if dp > 0 else 'menos'} años con pérdida")
    ciclo = " → ".join(C.CULTIVOS[c]["nombre"].split(" (")[0].lower() for c in r["ciclo"])
    if es_actual:
        resumen = f"Es lo que hace hoy: {ciclo}."
    elif frases:
        resumen = f"Comparado con lo que hace hoy, {', '.join(frases[:-1]) + (' y ' if len(frases) > 1 else '') + frases[-1]}."
    else:
        resumen = "Resultados muy parecidos a lo que hace hoy."
    perdida = r["prob_margen_negativo"]
    dinero = (f"Ganancia típica: {me['margen_usd_acre']:.0f} USD por acre al año (después de semillas, fertilizante, químicos y cosecha; "
              f"sin contar arriendo). En {round(perdida * 10)} de cada 10 años se pierde dinero.")
    p = r["prob_ser_primera"]
    seg = "alta" if p >= .7 else ("media" if p >= .35 else "baja")
    cs = r["ciclo"]
    bueno = []
    if any(C.CULTIVOS[c]["n_fija_kg_ha"] for c in cs):
        bueno.append("Deja nitrógeno en el suelo (soja, alfalfa o vicia): se compra menos fertilizante para el cultivo que sigue.")
    if "cobertura" in cs or "triticale" in cs:
        bueno.append("Mantiene el suelo cubierto en primavera, cuando más sopla el viento.")
    desn = sum(m["estado"] == "desnudo" for m in r.get("calendario", []))
    desn_act = sum(m["estado"] == "desnudo" for m in act.get("calendario", []))
    if not es_actual and desn < desn_act:
        bueno.append(f"Menos meses con el suelo desnudo que hoy ({desn} en vez de {desn_act} por ciclo).")
    if any(C.CULTIVOS[c]["raiz_cm"] >= 180 for c in cs):
        bueno.append("Tiene cultivos de raíz profunda que usan agua de abajo y abren caminos para que entre la lluvia.")
    if any(c in cs for c in ("sorgo", "girasol", "trigo")):
        bueno.append("Usa cultivos que aguantan bien la sequía y el calor.")
    if cr["plagas"] >= .85:
        bueno.append("Cambia de cultivo de un año a otro: corta plagas, malezas y enfermedades.")
    if cr["forraje"] >= .3:
        bueno.append("Produce forraje: ingreso extra si tiene animales o un feedlot cerca.")
    if me["riego_mm_anio"] == 0 and ma["riego_mm_anio"] > 0:
        bueno.append("No necesita agua del pozo.")
    valores = dict(agua=float(agua), suelo=float(suelo), dinero=float(me["margen_usd_acre"]),
                   seguridad=float(-r["prob_margen_negativo"]), plagas=float(cr["plagas"]))
    return dict(puntos=puntos, valores=valores, ciclo_texto=ciclo, resumen=resumen, dinero=dinero, lo_bueno=bueno,
                certeza=seg, es_actual=es_actual)


def certeza_texto(viables):
    if not viables:
        return ""
    a = viables[0]
    p = a["prob_ser_primera"]
    if p >= .7:
        return f"Con bastante seguridad: en {round(p * 10)} de cada 10 escenarios de clima y precios, esta opción sale primera."
    if p >= .35:
        return f"Con seguridad media: sale primera en {round(p * 10)} de cada 10 escenarios. La segunda opción también es razonable."
    return "Con poca seguridad: varias opciones quedan muy parejas. Decida según lo que más le importa y consulte a su asesor."


# ------------------------------------------------------------------ desempeño de los cultivos (USDA NASS) en palabras simples
def _pct(v):
    return f"{100 * v:.0f}%"


def desempeno_simple(des):
    if not des or des.get("modo") == "no disponible":
        return None
    inf = math.inf
    out = dict(cultivos=[], salud=[], humedad=None, calendario=des.get("calendario") or [], precios=des.get("precios") or {})
    cond = des.get("condado")
    if cond:
        for c in cond["cultivos"]:
            lugar = f"el condado de {c['lugar']}" if c["nivel_dato"] == "condado" else f"el distrito {c['lugar']}"
            frases = []
            if c["frac_anios_malos"] >= .1:
                frases.append(f"{int(c['frac_anios_malos'] * 10 + .5)} de cada 10 años rinde mucho menos de lo normal")
            if (c["abandono_medio"] or 0) >= .05:
                frases.append(f"en promedio no se cosecha el {_pct(c['abandono_medio'])} de lo sembrado")
            if c["tendencia_confirmada"] and abs(c["tendencia_pct_decada"]) >= 3:
                frases.append(f"el rendimiento {'sube' if c['tendencia_decada'] > 0 else 'baja'} cerca de {abs(c['tendencia_pct_decada']):.0f}% por década")
            resumen = (f"En {lugar}, " + ", ".join(frases) + ".") if frases else f"En {lugar} el rendimiento es bastante estable."
            det = [dato("Rendimiento típico (últimos 10 años con dato)", c["rend_reciente"], c["unidad"],
                        f"Promedio del condado entre {c['periodo_reciente']}, sumando campos con y sin riego." +
                        (" Sin riego: " + _n(c['practicas']['secano']['media'], 1) + f" {c['unidad']} ({c['practicas']['secano']['periodo']})." if "secano" in c["practicas"] else "") +
                        (" Con riego: " + _n(c['practicas']['riego']['media'], 1) + f" {c['unidad']} ({c['practicas']['riego']['periodo']})." if "riego" in c["practicas"] else ""), dec=1),
                   dato("Años malos", 100 * c["frac_anios_malos"], "% de los años",
                        f"Años en que el rendimiento quedó por debajo del 75% de lo esperado para ese año (según la tendencia {c['periodo']}). El peor fue {c['peor_anio']} con {_n(c['peor_valor'], 1)} {c['unidad']}.",
                        [(10, "favorable", "pocos años malos"), (25, "intermedio", "uno o dos años malos de cada diez"), (inf, "desfavorable", "años malos frecuentes")])]
            if c["abandono_medio"] is not None:
                det.append(dato("Superficie sembrada que no se cosechó", 100 * c["abandono_medio"], "% en promedio",
                                f"Parte de lo sembrado que se abandonó o se usó para pastoreo porque no valía la pena cosecharlo (sequía, granizo, heladas). El peor año fue {c['abandono_peor_anio']}, con {_pct(c['abandono_max'])}.",
                                [(5, "favorable", "casi todo se cosecha"), (15, "intermedio", "se pierde una parte en años malos"), (inf, "desfavorable", "pérdidas grandes y frecuentes")]))
            det.append(dato("Cambio del rendimiento por década", c["tendencia_pct_decada"], "%",
                            "Cuánto subió o bajó el rendimiento cada 10 años. Las mejoras suelen venir de variedades y manejo; las bajas, de más sequía o calor. " +
                            ("El cambio está confirmado estadísticamente." if c["tendencia_confirmada"] else "No está confirmado: puede ser variación entre años."),
                            [(-3, "desfavorable", "el rendimiento baja"), (3, "intermedio", "sin cambio claro"), (inf, "favorable", "el rendimiento sube")], dec=1))
            clima = []
            for k, x in (c.get("clima") or {}).items():
                if x["fuerza"] == "débil":
                    continue
                if k == "lluvia":
                    clima.append(f"Con {x['paso']} mm más de lluvia {x['meses']}, el rendimiento sube cerca de {abs(x['efecto_pct']):.0f}% (relación {x['fuerza']}).")
                else:
                    clima.append(f"Con {x['paso']} días más sobre {x['umbral']} °C {x['meses']}, el rendimiento {'baja' if x['efecto_pct'] < 0 else 'sube'} cerca de {abs(x['efecto_pct']):.0f}% (relación {x['fuerza']}).")
            out["cultivos"].append(dict(cultivo=c["cultivo"], nombre=c["nombre"], lugar=c["lugar"], nivel_dato=c["nivel_dato"], periodo=c["periodo"],
                                        estado=c["estado"], resumen=resumen, detalles=det, clima=clima, serie=c["serie"], unidad=c["unidad"],
                                        sembrado=c["sembrado_reciente"]))
    for k, s in (des.get("salud") or {}).items():
        frases = [f"En una temporada normal, cerca del {s['media_temporada']:.0f}% de los campos de Kansas está en condición buena o excelente.",
                  f"El momento más difícil suele ser alrededor del {s['semana_mas_baja']} (promedio {s['valor_mas_bajo']:.0f}%).",
                  "Peores temporadas: " + ", ".join(f"{p['anio']} ({p['buena']:.0f}%)" for p in s["peores_anios"]) + "."]
        for kk, x in (s.get("relacion_clima") or {}).items():
            if x["fuerza"] == "débil":
                continue
            if kk == "lluvia":
                frases.append(f"Cada {x['paso']} mm más de lluvia {x['meses']} suma cerca de {abs(x['efecto_puntos']):.0f} puntos de campos en buen estado (relación {x['fuerza']}).")
            else:
                frases.append(f"Cada {x['paso']} días más sobre {x['umbral']} °C {x['meses']} resta cerca de {abs(x['efecto_puntos']):.0f} puntos (relación {x['fuerza']}).")
        t = s.get("tendencia") or {}
        if t.get("significativa"):
            frases.append(f"La proporción de campos en buen estado {'baja' if t['pendiente_por_decada'] < 0 else 'sube'} unos {abs(t['pendiente_por_decada']):.0f} puntos por década (cambio confirmado).")
        out["salud"].append(dict(cultivo=k, nombre=s["nombre"], periodo=s["periodo"], frases=frases, semanas=s["semanas"], ultimo_anio=s["ultimo_anio"]))
    h = des.get("humedad") or {}
    if "subsuelo" in h:
        x = h["subsuelo"]
        out["humedad"] = dict(capa="subsuelo", r=x["r"], fuerza=x["fuerza"], periodo=x["periodo"], mensual=x["mensual"], anual=x["anual"], r_anual=x["r_anual"],
                              frase=(f"La humedad del suelo que calcula NASA POWER (con datos satelitales) y la que informan agricultores y técnicos de Kansas cada semana "
                                     f"suben y bajan juntas: la coincidencia es {x['fuerza']} (correlación {x['r']:.2f} en {x['n_meses']} meses, {x['periodo']}). "
                                     "Por eso podemos usar el dato satelital donde no hay observadores, sabiendo que suaviza las diferencias entre campos vecinos."))
    return out


def tarjeta_desempeno(ds):
    """Pregunta resumen sobre cómo les va a los cultivos de la zona (los 3 con más superficie)."""
    if not ds or not ds["cultivos"]:
        return None
    cs = sorted(ds["cultivos"], key=lambda c: -(c["sembrado"] or 0))[:3]
    orden = {"bien": 0, "atención": 1, "problema": 2}
    peor = max(cs, key=lambda c: orden[c["estado"]])
    det = []
    for c in cs:
        for d in c["detalles"]:
            if d["nombre"] in ("Años malos", "Superficie sembrada que no se cosechó"):
                det.append(dict(d, nombre=f"{c['nombre']}: {d['nombre'].lower()}"))
    return dict(clave="desempeno", pregunta="¿Cómo les va a los cultivos en esta zona?",
                respuesta=f"{peor['nombre']}: {peor['resumen'][0].lower() + peor['resumen'][1:]}",
                nivel=peor["estado"],
                para_usted="Los cultivos con muchos años malos o mucha superficie sin cosechar son los más riesgosos aquí. "
                           "Diversificar la rotación reparte ese riesgo entre cultivos que sufren en épocas distintas.",
                detalles=det, fuente=f"USDA NASS, rendimientos y superficies del condado ({cs[0]['periodo']})")
