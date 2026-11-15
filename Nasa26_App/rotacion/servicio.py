"""Une fuentes + indicadores + motor + explicaciones. Es lo que consume la API."""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from . import explicaciones as X
from . import fuentes as F
from . import indicadores as I
from . import motor as M
from . import simple as SP
from . import desempeno as D

NOMBRE_MES = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
SUELO_POR_DEFECTO = dict(awc_100cm_mm=150, materia_organica_pct=1.5, arcilla_pct=25, arena_pct=35, pendiente_pct=1.5, kffact=.32,
                         frac_escurrimiento_alto=.5, frac_erosion_eolica_alta=.2, grupo_erosion_eolica_medio=5)


def limpio(o):
    """Convierte a tipos JSON (NaN -> None, numpy -> python)."""
    if isinstance(o, dict):
        return {str(k): limpio(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [limpio(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 3)
    if isinstance(o, np.ndarray):
        return limpio(o.tolist())
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


# ------------------------------------------------------------------ carga del sitio (con caché)
@lru_cache(maxsize=32)
def _base(clave_json: str):
    z = F.Zona.desde(json.loads(clave_json))
    pw = F.power(z)
    st = F.suelo_tabla(z)
    pz = F.pozos(z)
    cq = F.caudal(z)
    ns = F.nass(z)
    return z, pw, st, pz, cq, ns, F.srtm(z), F.grace(z), F.openet(z)


@lru_cache(maxsize=64)
def _sitio(clave_json: str, dl: float, dt: float, riego_mm: float):
    z, pw, st, pz, cq, ns, dem, gr, oe = _base(clave_json)
    if pw["modo"] == "no disponible":
        raise RuntimeError(pw["motivo"])
    cl = I.clima(I.aplicar_escenario(pw["datos"], dl, dt), z.lat)
    su = I.suelo(st["tabla"]) if st["modo"] != "no disponible" else dict(SUELO_POR_DEFECTO, supuesto=True)
    if dem.get("pendiente_media_pct") is not None:
        su["pendiente_pct"] = dem["pendiente_media_pct"]
    rd = I.rendimientos(ns["datos"]) if ns["modo"] != "no disponible" else {}
    sitio = M.preparar_sitio(cl, su, rd, {"riego_mm": riego_mm})
    return cl, su, rd, sitio


def _clave(zona: dict):
    return json.dumps({k: zona.get(k) for k in ["lat", "lon", "poligono", "radio_km"] if zona.get(k) is not None}, sort_keys=True)


# ------------------------------------------------------------------ evidencia
def evidencia(zona: dict, escenario: dict | None = None):
    esc = {**C.ESCENARIO, **(escenario or {})}
    k = _clave(zona)
    z, pw, st, pz, cq, ns, dem, gr, oe = _base(k)
    cl, su, rd, sitio = _sitio(k, float(esc["delta_lluvia_pct"]), float(esc["delta_temp_c"]), 0.0)
    acu = I.acuifero(pz["datos"]) if pz["modo"] != "no disponible" else None
    cau = I.caudal(cq["estaciones"]) if cq["modo"] != "no disponible" else []
    anual = cl["anual"]
    clim = cl["climatologia"]
    fuentes = {n: dict(modo=x.get("modo"), fuente=x.get("fuente"), motivo=x.get("motivo") or x.get("nota"))
               for n, x in [("clima_power", pw), ("suelo_ssurgo", st), ("pozos_usgs", pz), ("caudal_usgs", cq),
                            ("rendimientos_nass", ns), ("relieve_srtm", dem), ("grace", gr), ("et_openet", oe)]}
    resumen = {k2: v for k2, v in cl["resumen"].items()}
    resumen["ultima_helada"] = _doy_a_fecha(resumen.get("ultima_helada_doy"))
    resumen["primera_helada"] = _doy_a_fecha(resumen.get("primera_helada_doy"))
    out = dict(
        zona=dict(lat=z.lat, lon=z.lon, poligono=z.poligono, celda_power=pw.get("celda"), distancia_celda_km=pw.get("distancia_km")),
        escenario=esc,
        fuentes=fuentes,
        clima=dict(resumen=X.anotar(resumen),
                   tendencias={k2: dict(v, nombre=X.G.get(k2, [k2])[0]) for k2, v in cl["tendencias"].items()},
                   periodos=cl["periodos"],
                   serie_anual=anual.round(2).reset_index().to_dict("records"),
                   mensual=[dict(mes=int(m), nombre=NOMBRE_MES[int(m)], **r) for m, r in clim.round(2).iterrows()]),
        suelo=X.anotar(su),
        acuifero=(X.anotar({k2: v for k2, v in acu.items() if k2 not in ("pozos", "serie_anomalia_ft")}) | dict(
            serie_anomalia_ft=acu["serie_anomalia_ft"], pozos=acu["pozos"])) if acu and acu.get("n_pozos") else None,
        caudal=[dict(e, frac_dias_sin_flujo=X.interpretar("frac_dias_sin_flujo", e["frac_dias_sin_flujo"])) for e in cau],
        rendimientos_nass={p: {k2: dict(v, cv=X.interpretar("cv_pct", v["cv_pct"]) if v.get("cv_pct") else None) for k2, v in r.items()} for p, r in rd.items()},
        relieve=dem if dem.get("modo") != "no disponible" else None,
        et_openet=oe.get("et_real_anual_mm"),
        vulnerabilidad=vulnerabilidad(cl, sitio),
        alertas=alertas(cl, su, acu, cau),
        brechas=brechas(fuentes),
        glosario=X.glosario(),
        simple=dict(tarjetas=SP.tarjetas(dict(cl, resumen=resumen), su, acu, cau), antes_ahora=SP.antes_ahora(cl),
                    mes_a_mes=SP.mes_a_mes(cl)),
    )
    des = D.resumen(z, _sitio(k, 0.0, 0.0, 0.0)[0]["diario"])
    ds = SP.desempeno_simple(des)
    out["desempeno"] = ds
    out["fuentes"]["desempeno_nass"] = dict(modo=des.get("modo"), fuente=des.get("fuente", "USDA NASS Kansas"),
                                           motivo=des.get("motivo") or (None if des.get("condado") else "La zona no está en un condado de Kansas con datos"))
    t7 = SP.tarjeta_desempeno(ds)
    if t7:
        out["simple"]["tarjetas"].append(t7)
    return limpio(out)


def _doy_a_fecha(doy):
    if doy is None or (isinstance(doy, float) and math.isnan(doy)):
        return None
    d = pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(round(doy)) - 1)
    return f"{d.day} {NOMBRE_MES[d.month]}"


def vulnerabilidad(cl, sitio):
    """Para cada cultivo: en qué meses de su ciclo el clima del sitio lo pone en riesgo."""
    clim = cl["climatologia"]
    out = {}
    for cult, c in C.CULTIVOS.items():
        if not c["meses"] or c["ky"] == 0:
            continue
        meses = []
        for m, kc in zip(c["meses"], c["kc"]):
            r = clim.loc[m]
            deficit = kc * r.et0 - r.lluvia
            riesgos = []
            if deficit > 60:
                riesgos.append(f"déficit hídrico ~{deficit:.0f} mm")
            if m in c["calor"][0]:
                dias = cl["mensual"].xs(m, level="mes").calor35.mean() if c["calor"][1] <= 35 else None
                if dias and dias > 3:
                    riesgos.append(f"{dias:.0f} días ≥35 °C en etapa sensible")
            if m in c["helada"][0] and r.prob_helada > .1:
                riesgos.append(f"helada en {r.prob_helada * 100:.0f}% de los años")
            if r.get("anomalia_humedad_ult5", 0) < -.5:
                riesgos.append("suelo más seco que lo normal en los últimos 5 años")
            meses.append(dict(mes=int(m), nombre=NOMBRE_MES[m], deficit_mm=round(float(deficit)), riesgos=riesgos,
                              nivel="alto" if len(riesgos) >= 2 or deficit > 120 else ("medio" if riesgos else "bajo")))
        out[cult] = dict(nombre=c["nombre"], meses=meses)
    return out


def alertas(cl, su, acu, cau):
    a = []
    r = cl["resumen"]
    t = cl["tendencias"]
    if r["deficit_mm"] > 700:
        a.append(f"La lluvia ({r['lluvia_mm']:.0f} mm) cubre {100 * r['lluvia_mm'] / r['et0_mm']:.0f}% de la demanda atmosférica: "
                 "el secano depende de guardar agua (barbecho, residuo).")
    if t["dias_calor_35"]["pendiente_por_decada"] and t["dias_calor_35"]["pendiente_por_decada"] > 1:
        s = "significativa" if t["dias_calor_35"]["significativa"] else "no significativa todavía (p={:.2f})".format(t["dias_calor_35"]["p_valor"])
        a.append(f"Los días ≥35 °C aumentan ~{t['dias_calor_35']['pendiente_por_decada']:.1f} por década desde 1981 (tendencia {s}).")
    if acu and acu.get("n_pozos") and acu["tasa_mediana_m_anio"] > .3:
        a.append(f"El acuífero baja ~{acu['tasa_mediana_m_anio']:.1f} m/año en {acu['n_pozos']} pozos cercanos: el riego actual no es sostenible a largo plazo.")
    for e in cau:
        if e["frac_dias_sin_flujo"] > .4:
            a.append(f"{e['nombre'].title()}: sin caudal el {100 * e['frac_dias_sin_flujo']:.0f}% de los días ({e['periodo']}).")
    if (su.get("grupo_erosion_eolica_medio") or 9) <= 4 or (su.get("frac_erosion_eolica_alta") or 0) > .2:
        a.append("Hay suelos muy susceptibles a erosión por viento: evitar suelo desnudo en marzo-mayo.")
    return a


def brechas(fuentes):
    b = []
    for n, f in fuentes.items():
        if f["modo"] == "no disponible":
            b.append(f"{f['fuente']}: {f['motivo']}")
        elif f.get("motivo"):
            b.append(f"{f['fuente']}: {f['motivo']}")
    b += ["Los costos son supuestos editables; los precios son el promedio recibido en Kansas en las últimas 3 campañas (USDA NASS), no el precio del día.",
          "No hay datos de manejo del lote (historial, fertilización, malezas, equipo).",
          "POWER tiene resolución ~50 km: no distingue lomas de bajos dentro del campo.",
          "El escenario de clima escala lluvia y temperatura pero no cambia la frecuencia de tormentas ni el viento."]
    return b


# ------------------------------------------------------------------ satélite (lento, se pide aparte)
def satelite(zona: dict, cuales=("landsat", "smap", "ecostress", "imerg")):
    z = F.Zona.desde(zona)
    out = {}
    if "landsat" in cuales:
        ls = F.landsat(z)
        out["landsat"] = dict(ls, resumen=I.landsat(ls["escenas"])) if ls.get("escenas") else ls
    for c in ("smap", "ecostress"):
        if c in cuales:
            r = F.appeears_punto(z, c)
            if r.get("serie"):
                s = pd.DataFrame(r["serie"])
                s["mes"] = pd.to_datetime(s.fecha).dt.month
                r["mensual"] = s.groupby("mes").valor.mean().round(3).to_dict()
            out[c] = r
    if "imerg" in cuales:
        r = F.imerg(z)
        if "serie" in r:
            s = r.pop("serie")
            r["resumen"] = dict(anios=f"{s.index.min().year}-{s.index.max().year}",
                                lluvia_media_mm=round(float(s.groupby(s.index.year).sum().mean())),
                                p95_mm=round(float(s[s >= 1].quantile(.95)), 1), p99_mm=round(float(s[s >= 1].quantile(.99)), 1),
                                dias_25mm_anio=round(float((s >= 25.4).groupby(s.index.year).sum().mean()), 1))
        out["imerg"] = r
    out["nota"] = X.SATELITE_VS_TERRENO
    return limpio(out)


# ------------------------------------------------------------------ rotaciones
def rotaciones(zona: dict, prioridades=None, actual=None, alternativas=None, restricciones=None, escenario=None, n_sim=300):
    esc = {**C.ESCENARIO, **(escenario or {})}
    restr = {**C.RESTRICCIONES, **(restricciones or {})}
    pri = {**C.PRIORIDADES, **(prioridades or {})}
    actual = actual or C.ACTUAL
    alt = alternativas or C.ROTACIONES
    for nombre, r in list(alt.items()) + [("actual", actual)]:
        desc = [c for c in r if c not in C.CULTIVOS]
        if desc:
            raise ValueError(f"Rotación '{nombre}': cultivos desconocidos {desc}. Válidos: {list(C.CULTIVOS)}")
    rots = {"A (actual): " + "-".join(actual): actual, **alt}
    # precios: los que indique el usuario; si no, el promedio recibido en Kansas (USDA NASS, últimas 3 campañas)
    restr["precios"] = {**{c: v["precio"] for c, v in D.precios_referencia().items()}, **(restr.get("precios") or {})}
    k = _clave(zona)
    cl, su, rd, sitio = _sitio(k, float(esc["delta_lluvia_pct"]), float(esc["delta_temp_c"]), float(restr["riego_mm"]))
    R = M.ranking(sitio, rots, pri, restr, n_sim=n_sim)
    z = F.Zona.desde(zona)
    cond = D.cultivos_condado(D.condado_de(z)) if D.disponible() else None
    perf = {c["cultivo"]: c for c in (cond or {}).get("cultivos", [])}
    ref = R["ranking"][0]
    act = next(r for r in R["ranking"] if r["rotacion"].startswith("A (actual)"))
    for r in R["ranking"]:
        r["calendario"] = calendario(r)
        r.update(explicar(r, act, R["pesos"], cl, su, restr, sitio))
        for cult in dict.fromkeys(r["ciclo"]):
            p = perf.get(cult)
            if p and (p["abandono_medio"] or 0) >= .05:
                r["riesgos"].append(f"En {p['lugar']} no se cosecha en promedio el {100 * p['abandono_medio']:.0f}% del {p['nombre'].lower()} sembrado "
                                    f"(en {p['abandono_peor_anio']}, el {100 * p['abandono_max']:.0f}%). Dato real USDA NASS.")
            elif p and p["frac_anios_malos"] >= .2:
                r["riesgos"].append(f"En {p['lugar']}, {int(p['frac_anios_malos'] * 10 + .5)} de cada 10 años el {p['nombre'].lower()} rindió mucho menos de lo normal (USDA NASS).")
        r["simple"] = SP.rotacion_simple(r, act)
    viables = [r for r in R["ranking"] if not r["problemas"]]
    viables.sort(key=lambda r: -r["puntaje"])
    no_viables = [r for r in R["ranking"] if r["problemas"]]
    for i, r in enumerate(viables):
        r["posicion"] = i + 1
    concl = _conclusion(viables, act)
    return limpio(dict(pesos=R["pesos"], escenario=esc, restricciones=restr, ranking=viables + no_viables,
                       conclusion=concl, certeza=SP.certeza_texto(viables), validacion=validacion(sitio),
                       criterios=C.CRITERIOS, recordatorio=("Herramienta de apoyo: los puntajes ordenan opciones con los datos y supuestos "
                                                            "mostrados. No reemplazan el juicio del agricultor ni la revisión de un asesor local "
                                                            "(extensión K-State, NRCS).")))


def _conclusion(viables, act):
    if not viables:
        return "Ninguna rotación cumple las restricciones indicadas."
    a, b = viables[0], viables[1] if len(viables) > 1 else None
    txt = f"{a['rotacion']} obtiene el mayor puntaje ({a['puntaje']}) y queda primera en el {a['prob_ser_primera'] * 100:.0f}% de las simulaciones."
    if b and a["rango_p10_p90"][0] < b["rango_p10_p90"][1]:
        txt += f" Su rango se solapa con {b['rotacion']}: la diferencia no es concluyente y depende de sus prioridades y del manejo."
    if a is not act:
        txt += f" Frente a la rotación actual cambia {a['puntaje'] - act['puntaje']:+.1f} puntos."
    return txt


def explicar(r, act, pesos, cl, su, restr, sitio):
    cr, met = r["criterios"], r["metricas"]
    crits = list(C.CRITERIOS)
    aporte = sorted(((c, pesos[c] * cr[c]) for c in crits), key=lambda t: -t[1])
    dif = {c: cr[c] - act["criterios"][c] for c in crits}
    mejoras = [c for c in sorted(dif, key=lambda c: -dif[c] * pesos[c]) if dif[c] > .03][:3]
    peores = [c for c in sorted(dif, key=lambda c: dif[c] * pesos[c]) if dif[c] < -.03][:2]
    ciclo = r["ciclo"]
    res = cl["resumen"]
    por_que = (f"Lo más fuerte de esta rotación, según sus prioridades, es {C.CRITERIOS[aporte[0][0]]} y {C.CRITERIOS[aporte[1][0]]}.")
    if not r["rotacion"].startswith("A (actual)"):
        if mejoras:
            por_que += " Frente a la actual mejora en " + ", ".join(f"{C.CRITERIOS[c]} ({dif[c]:+.2f})" for c in mejoras) + "."
        if peores:
            por_que += " Pierde en " + ", ".join(f"{C.CRITERIOS[c]} ({dif[c]:+.2f})" for c in peores) + "."
    det = pd.DataFrame(r["_det"])
    porcult = det.groupby("cultivo").agg(rend=("rend", "mean"), yr=("yr", "mean"), margen=("margen", "mean"), riego=("riego_mm", "mean"))
    evidencia = [f"Clima {res['anios']}: lluvia {res['lluvia_mm']:.0f} mm/año vs. demanda ET0 {res['et0_mm']:.0f} mm; "
                 f"{res['dias_calor_35']:.0f} días/año ≥35 °C; lluvia muy variable ({res['frac_anios_deficit_severo'] * 100:.0f}% de años con <70% de lo normal).",
                 f"Suelo: agua útil {su['awc_100cm_mm']} mm en 1 m, MO {su['materia_organica_pct']}%, arena {su['arena_pct']}%" +
                 (" (valores por defecto: no hubo datos de suelo)." if su.get("supuesto") else ".")]
    for cult, f in porcult.iterrows():
        c = C.CULTIVOS[cult]
        if c["unidad"]:
            evidencia.append(f"{c['nombre']}: rendimiento simulado {f.rend:.0f} {c['unidad']}/acre ({f.yr * 100:.0f}% del potencial), "
                             f"margen directo {f.margen:+.0f} USD/acre" + (f", riego {f.riego:.0f} mm" if f.riego > 0 else "") + ".")
    evidencia.append(f"Erosión estimada: {met['eros_relativa'] * 100:.0f}% de la de un suelo siempre desnudo; aporte de residuo "
                     f"{met['mo_aporte_t_ha']:.1f} t/ha/año.")
    beneficios = []
    if any(C.CULTIVOS[c]["n_fija_kg_ha"] for c in ciclo):
        beneficios.append(f"Incluye leguminosa: aporta ~{met['n_fijado_kg_ha']:.0f} kg N/ha/año al cultivo siguiente. " + X.CONCEPTOS["fijadoras_n"])
    if "cobertura" in ciclo or "triticale" in ciclo:
        beneficios.append("La cobertura protege el suelo en primavera, cuando el viento erosiona más. " + X.CONCEPTOS["coberturas"])
    if any(C.CULTIVOS[c]["raiz_cm"] >= 180 for c in ciclo):
        beneficios.append(X.CONCEPTOS["raices_profundas"])
    if any(c in ciclo for c in ("sorgo", "girasol", "trigo")):
        beneficios.append(X.CONCEPTOS["tolerantes_sequia"])
    if cr["plagas"] >= .85:
        beneficios.append("Alterna cultivos de invierno y verano y no repite hospederos seguidos: corta ciclos de malezas, plagas y enfermedades.")
    if cr["forraje"] >= .3:
        beneficios.append("Produce forraje (pastoreo o ensilaje): ingreso extra si hay ganado o feedlots cerca.")
    riesgos = []
    if "barbecho" in ciclo:
        meses = sum(1 for m in r["calendario"] if m["estado"] == "desnudo")
        riesgos.append(f"Deja el suelo desnudo o con poco residuo ~{meses} meses por ciclo; riesgo de voladura en primavera.")
    if "maíz" in ciclo and restr["riego_mm"] < 300:
        riesgos.append("El maíz en secano tiene pérdidas grandes en años secos (rendimiento muy variable).")
    if met["riego_mm_anio"] > 0:
        riesgos.append(f"Usa ~{met['riego_mm_anio']:.0f} mm/año de riego del acuífero; revisar derechos de agua y el descenso de los pozos.")
    if "cobertura" in ciclo:
        riesgos.append("La cobertura consume agua: en años secos puede bajar el rendimiento del cultivo siguiente. Terminarla temprano.")
    if "soja" in ciclo and restr["riego_mm"] == 0:
        riesgos.append("La soja en secano en el SO de Kansas rinde poco (NASS ~20 bu/acre): validar con ensayos locales.")
    if r["prob_margen_negativo"] > .3:
        riesgos.append(f"En el {r['prob_margen_negativo'] * 100:.0f}% de los años simulados el margen directo es negativo.")
    if restr.get("ingreso_minimo_usd_acre") is not None and met["margen_usd_acre"] < restr["ingreso_minimo_usd_acre"]:
        riesgos.append(f"No alcanza el ingreso mínimo pedido ({restr['ingreso_minimo_usd_acre']} USD/acre).")
    riesgos += [f"Restricción: {p}." for p in r["problemas"]]
    validar = ["Fechas de siembra/cosecha y variedades con el agente de extensión.",
               "Precios locales (elevador, feedlot) y costos reales del productor.",
               "Disponibilidad de equipo y mano de obra para los cultivos nuevos."]
    if "cobertura" in ciclo or "triticale" in ciclo:
        validar.append("Especies y fecha de terminación de la cobertura (NRCS puede cofinanciar con EQIP).")
    return dict(por_que=por_que, evidencia=evidencia, beneficios=beneficios, riesgos=riesgos,
                validar_localmente=validar,
                rendimiento_por_cultivo={k2: dict(rend=round(v.rend, 1), unidad=C.CULTIVOS[k2]["unidad"], frac_potencial=round(v.yr, 2),
                                                   margen=round(v.margen), riego_mm=round(v.riego)) for k2, v in porcult.iterrows()})


def calendario(r):
    """Estado del campo mes a mes durante un ciclo de la rotación (de la 1a simulación)."""
    out = []
    L = len(r["ciclo"])
    for reg in r["_reg"][L:2 * L]:
        c = C.CULTIVOS[reg["cultivo"]]
        for j, m in enumerate(M.MESES):
            cov = float(reg["cobertura"][j])
            est = "cultivo" if m in c["meses"] else ("rastrojo" if cov >= .3 else "desnudo")
            out.append(dict(ranura=reg["cultivo"], mes=m, nombre=NOMBRE_MES[m], estado=est, cobertura=round(cov, 2),
                            riesgo_viento=round(M.VIENTO_MES[m] * (1 - cov), 2)))
    return out


def validacion(sitio, riego_tipico_mm=350):
    """Prueba de realismo: rendimiento simulado vs. promedio NASS del condado, en secano y con riego típico.
    Si difieren mucho, los supuestos del cultivo deben revisarse para este sitio."""
    out = []
    rd = sitio["rend"]
    casos = [("trigo", ["trigo", "barbecho"], "secano_tras_barbecho"), ("trigo", ["trigo"], "secano_continuo"),
             ("sorgo", ["sorgo"], "secano"), ("maíz", ["maíz"], "secano"), ("soja", ["soja"], "secano"),
             ("maíz", ["maíz"], "riego"), ("sorgo", ["sorgo"], "riego"), ("trigo", ["trigo"], "riego"), ("soja", ["soja"], "riego")]
    for cult, rot, prac in casos:
        n = rd.get(C.CULTIVOS[cult]["nass"], {})
        if prac not in n:
            continue
        pot, _ = M.rendimiento_potencial(cult, rd, True)
        rmm = riego_tipico_mm if prac == "riego" else 0
        sim = [x["yr"] * pot for ini in range(sitio["n"]) for x in M.simular(rot, sitio, ini, len(rot) * 2, riego_mm=rmm) if x["cultivo"] == cult]
        s, v = float(np.mean(sim)), n[prac]["media"]
        out.append(dict(cultivo=cult, practica=prac.replace("_", " "), simulado=round(s, 1), nass=v, periodo_nass=n[prac]["periodo"],
                        unidad=C.CULTIVOS[cult]["unidad"] + "/acre", error_pct=round(100 * (s - v) / v)))
    return out


# ------------------------------------------------------------------ capas de mapa
def capa_grilla(forzar=False):
    """Indicadores por celda de la grilla POWER local (para colorear el mapa). Se guarda en caché."""
    f = F._cache("grilla.json")
    if f.exists() and not forzar:
        return json.loads(f.read_text())
    feats = []
    for p in F.grilla_power_local():
        df = F._leer_power_csv(open(p["archivo"], encoding="utf-8").read())
        df["ET0"] = I.et0_hargreaves(df, p["lat"])
        y = df.groupby(df.index.year)
        n = y.size()
        ok = n[n >= 360].index
        lluvia = y.PRECTOTCORR.sum()[ok]
        et0 = y.ET0.sum()[ok]
        calor = y.T2M_MAX.apply(lambda x: (x >= 35).sum())[ok]
        hel = df[(df.T2M_MIN <= -2) & df.index.month.isin([4, 5])].groupby(lambda d: d.year).size().reindex(ok, fill_value=0)
        prim = df[df.index.month.isin([3, 4, 5])]
        sec_prim = 1 - (prim.groupby(prim.index.year).PRECTOTCORR.sum() / prim.groupby(prim.index.year).ET0.sum())[ok]
        ext = df[df.PRECTOTCORR >= 25.4].groupby(lambda d: d.year).size().reindex(ok, fill_value=0)
        gw = df[df.index.month.isin(range(4, 10))].groupby(lambda d: d.year).GWETROOT.mean()[ok]
        tc = I.tendencia(calor)
        tl = I.tendencia(lluvia)
        prop = dict(lat=p["lat"], lon=p["lon"], lluvia_mm=round(float(lluvia.mean())), et0_mm=round(float(et0.mean())),
                    deficit_mm=round(float((et0 - lluvia).mean())), indice_aridez=round(float((lluvia / et0).mean()), 2),
                    dias_calor_35=round(float(calor.mean()), 1), tendencia_calor_decada=tc["pendiente_por_decada"],
                    tendencia_calor_signif=tc["significativa"], tendencia_lluvia_decada=tl["pendiente_por_decada"],
                    heladas_tardias=round(float(hel.mean()), 2), dias_lluvia_extrema=round(float(ext.mean()), 1),
                    humedad_raiz=round(float(gw.mean()), 3), sequedad_primavera=round(float(sec_prim.mean()), 2),
                    cv_lluvia_pct=round(float(100 * lluvia.std() / lluvia.mean()), 1))
        d = .25
        feats.append({"type": "Feature", "properties": prop, "geometry": {"type": "Polygon", "coordinates": [[
            [p["lon"] - d, p["lat"] - d], [p["lon"] + d, p["lat"] - d], [p["lon"] + d, p["lat"] + d], [p["lon"] - d, p["lat"] + d], [p["lon"] - d, p["lat"] - d]]]}})
    out = {"type": "FeatureCollection", "features": feats, "fuente": "NASA POWER 1981-2025 (grilla local 0.5°)"}
    f.write_text(json.dumps(limpio(out)))
    return out


def capa_pozos():
    f = F._cache("pozos_capa.json")
    if f.exists():
        return json.loads(f.read_text())
    local = C.DATA_DIR / "nwis" / "gwlevels_finney.json"
    if not local.exists():
        return {"type": "FeatureCollection", "features": []}
    d = F._pozos_df_desde_features(json.load(open(local)))
    a = I.acuifero(d)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
              "properties": {k: v for k, v in p.items() if k not in ("lat", "lon")}} for p in a.get("pozos", [])]
    out = limpio({"type": "FeatureCollection", "features": feats, "fuente": "USGS niveles de pozos (desde 2000)"})
    f.write_text(json.dumps(out))
    return out


_CAMPOS_CONDADO = {}


def capa_campos_condado(forzar=False):
    """Todos los campos de Finney procesados (se guardan en memoria: es la capa principal del mapa)."""
    if "fc" in _CAMPOS_CONDADO and not forzar:
        return _CAMPOS_CONDADO["fc"]
    fc = _procesar_campos(F.campos_condado())
    if fc.get("modo") != "no disponible":
        _CAMPOS_CONDADO["fc"] = fc
    return fc


def historial_punto(zona: dict):
    """Historial de cultivos 2017-2024 en el centro de un campo (para campos sin historial, p. ej. de OpenStreetMap)."""
    z = F.Zona.desde(zona)
    r = F.cdl_historial(z.lat, z.lon)
    if r.get("modo") == "no disponible":
        return r
    geom = z.poligono or {"type": "Polygon", "coordinates": [[[z.lon, z.lat]] * 4]}
    p = SP.analizar_campo({"csbid": None, **r["codigos"]}, geom)
    return limpio(dict(modo=r["modo"], fuente=r["fuente"], historial=p["historial"], rotacion_detectada=p["rotacion_detectada"],
                       rotacion_texto=p["rotacion_texto"], anios_barbecho=p["anios_barbecho"], ultimo_cultivo=p["ultimo_cultivo"]))


def capa_campos(bbox):
    """Campos de cultivo con su historial 2017-2024, rotación detectada y pivote probable."""
    return _procesar_campos(F.campos(bbox))


def _procesar_campos(r):
    if r.get("modo") == "no disponible":
        return r
    feats = []
    for f in r["features"]:
        g = f.get("geometry")
        if not g or g["type"] not in ("Polygon", "MultiPolygon"):
            continue
        feats.append({"type": "Feature", "geometry": g, "properties": SP.analizar_campo(f.get("properties", {}), g)})
    leyenda = {}
    for f in feats:
        h = f["properties"]["historial"][-1]
        leyenda[h["nombre"]] = h["color"]
    con_historial = any(f["properties"]["historial"][-1]["codigo"] for f in feats)
    return limpio({"type": "FeatureCollection", "features": feats, "modo": r["modo"], "fuente": r["fuente"], "motivo": r.get("motivo"),
                   "anios": C.CSB_ANIOS, "leyenda": dict(sorted(leyenda.items())), "con_historial": con_historial})


def capa_suelo(zona: dict, km=2.0):
    """Zonas de vulnerabilidad dentro del campo: riesgo de erosión eólica/hídrica y agua útil por polígono SSURGO."""
    z = F.Zona.desde(zona)
    fc = F.poligonos_suelo(z, km)
    if fc.get("modo") == "no disponible":
        return fc
    for f in fc["features"]:
        p = f["properties"]
        num = lambda v, d: float(v) if v not in (None, "", "None") and not (isinstance(v, float) and math.isnan(v)) else d
        weg = num(str(p.get("weg") or "5")[0] if str(p.get("weg") or "5")[0].isdigit() else 5, 5)
        pend = num(p.get("pendiente"), 1)
        k = num(p.get("kffact"), .3)
        awc = num(p.get("awc_100cm_cm"), 15) * 10
        ls = .065 + .0456 * pend + .006541 * pend ** 2
        p["riesgo_viento"] = round((9 - weg) / 8, 2)
        p["riesgo_agua"] = round(min(1, k * ls / .25), 2)
        p["agua_util_mm"] = round(awc)
        p["estres_hidrico"] = round(max(0, min(1, 1 - (awc - 60) / 160)), 2)
    return fc
