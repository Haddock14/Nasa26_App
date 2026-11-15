"""Motor agronómico: simula cada rotación año a año con el clima real del sitio.

Diferencias con la versión anterior:
- Los cultivos ya no tienen puntajes fijos de agua/calor: el déficit hídrico, el estrés por calor y las
  heladas se calculan mes a mes con el clima del punto (POWER 1981-2025) y el agua disponible del suelo.
- La rotación se evalúa como SECUENCIA: el agua guardada en el barbecho pasa al cultivo siguiente,
  la soja deja N al cereal que sigue, el residuo cubre el suelo los meses posteriores y repetir el mismo
  hospedero penaliza rendimiento y el criterio de plagas.
- La incertidumbre viene de (1) los 44 años de clima histórico re-muestreados, (2) precios y parámetros
  agronómicos inciertos y (3) pesos de prioridades inciertos.
- Escenarios (lluvia/temperatura) y restricciones (riego, maquinaria, mercado, labranza) se aplican antes
  de simular, así que cambian los resultados de verdad y no solo los pesos.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C

MESES = C.MESES_ANIO_AGRICOLA
UMBRALES_CALOR = sorted({c["calor"][1] for c in C.CULTIVOS.values()})
UMBRALES_HELADA = sorted({c["helada"][1] for c in C.CULTIVOS.values()})
# riesgo de erosión eólica por mes (en el SO de Kansas los vientos fuertes con suelo seco se concentran en primavera)
VIENTO_MES = {1: .6, 2: .7, 3: 1.0, 4: 1.0, 5: .8, 6: .5, 7: .3, 8: .3, 9: .3, 10: .4, 11: .5, 12: .6}
PRECIO_N_USD_KG = 1.2
COSTO_BOMBEO_USD_MM_ACRE = .35       # ~9 USD por acre-pulgada
EFIC_RIEGO = .85
LB_AC_POR_KG_HA = .892


# ------------------------------------------------------------------ preparar clima del sitio
def preparar_sitio(clima: dict, suelo: dict, rend: dict, restr: dict) -> dict:
    """Pasa el clima diario a matrices [año agrícola x mes] y fija parámetros del sitio."""
    d = clima["diario"]
    anios = sorted(set(d.index.year))
    agro = [a for a in anios[1:] if a - 1 in anios]       # año agrícola a = oct(a-1) .. sep(a)
    n = len(agro)
    M = {k: np.zeros((n, 12)) for k in ["P", "ET0", "Pext"]}
    for t in UMBRALES_CALOR:
        M[f"calor{t}"] = np.zeros((n, 12))
    for t in UMBRALES_HELADA:
        M[f"helada{t}"] = np.zeros((n, 12))
    g = d.groupby([d.index.year, d.index.month])
    P, E = g.PRECTOTCORR.sum(), g.ET0.sum()
    Pext = g.PRECTOTCORR.apply(lambda x: x[x >= 12.7].sum())
    calor = {t: g.T2M_MAX.apply(lambda x, t=t: (x >= t).sum()) for t in UMBRALES_CALOR}
    hel = {t: g.T2M_MIN.apply(lambda x, t=t: (x <= t).sum()) for t in UMBRALES_HELADA}
    for i, a in enumerate(agro):
        for j, m in enumerate(MESES):
            k = (a - 1 if m >= 10 else a, m)
            M["P"][i, j], M["ET0"][i, j], M["Pext"][i, j] = P.get(k, 0), E.get(k, 0), Pext.get(k, 0)
            for t in UMBRALES_CALOR:
                M[f"calor{t}"][i, j] = calor[t].get(k, 0)
            for t in UMBRALES_HELADA:
                M[f"helada{t}"][i, j] = hel[t].get(k, 0)
    awc = suelo.get("awc_100cm_mm") or 150
    weg = suelo.get("grupo_erosion_eolica_medio") or 5
    pend = suelo.get("pendiente_pct") or 1
    ls = (0.065 + 0.0456 * pend + 0.006541 * pend ** 2) * 1.0          # factor LS de RUSLE para ~22 m de ladera
    et0_mes = {m: float(M["ET0"][:, j].mean()) for j, m in enumerate(MESES)}
    return dict(M=M, anios=agro, n=n, awc=awc, s_max=awc * 1.8, et0_mes=et0_mes,
                escurrimiento=.10 + .10 * (suelo.get("frac_escurrimiento_alto") or .5),
                k_viento=max(.1, (9 - weg) / 8), k_agua=(suelo.get("kffact") or .3) * ls,
                rend=rend, restr=restr)


def rendimiento_potencial(cult: str, rend: dict, riego: bool):
    """Rendimiento alcanzable sin falta de agua: percentil 90 de los años con riego del condado (NASS).
    El modelo de agua/calor/heladas lo reduce año a año. Sin NASS se usa un supuesto regional."""
    c = C.CULTIVOS[cult]
    if c.get("rend_ref"):
        return c["rend_ref"], "supuesto"
    r = rend.get(c["nass"]) if c["nass"] else None
    if r:
        for k in ["riego", "todas"]:
            if k in r and r[k]["anios"] >= 5:
                base = r[k]["p90"]
                if k == "todas" and cult == "girasol":
                    base *= 1.3          # NASS girasol es casi todo secano
                return base, f"NASS {k} p90 {r[k]['periodo']}"
    return C.POTENCIAL_DEFECTO.get(cult, 0), "supuesto regional"


def vale_regar(cult, sitio, precios=None):
    """Regla simple del productor: riega un cultivo solo si 1 mm de agua vale más (en grano) que lo que cuesta
    (bombeo + manejo más intensivo). Valor marginal ≈ precio x potencial x Ky / demanda de agua de la temporada."""
    c = C.CULTIVOS[cult]
    if not c["meses"] or c["ky"] == 0:
        return False
    ypot, _ = rendimiento_potencial(cult, sitio["rend"], True)
    precio = (precios or {}).get(cult, c["precio"])
    etm = sum(k * e for k, e in zip(c["kc"], [sitio["et0_mes"][m] for m in c["meses"]]))
    valor_mm = precio * ypot * c["ky"] / max(etm, 1)
    costo_mm = COSTO_BOMBEO_USD_MM_ACRE + (c["costo_riego"] - c["costo_secano"]) / 300
    return valor_mm > costo_mm


# ------------------------------------------------------------------ simulación de una secuencia
def simular(rot, sitio, inicio, horizonte, par=None, riego_mm=0, labranza="reducida", precios=None):
    par = par or {}
    M, n = sitio["M"], sitio["n"]
    efic_b = C.EFIC_BARBECHO[labranza]
    decai = C.DECAIMIENTO_RESIDUO_MES[labranza]
    S = sitio["s_max"] * .5
    cobertura_res = .3        # cobertura por residuo al inicio
    meses_desde_cosecha = 0
    L = len(rot)
    reg = []
    for k in range(horizonte):
        cult = rot[k % L]
        prev = rot[(k - 1) % L]
        c = C.CULTIVOS[cult]
        y = (inicio + k) % n
        activos = c["meses"]
        kc = dict(zip(activos, c["kc"]))
        etm = eta = riego_usado = 0.0
        riego_disp = riego_mm if riego_mm and vale_regar(cult, sitio, precios) else 0
        cobertura = np.zeros(12)
        perd_viento = perd_agua = perd_viento_base = perd_agua_base = 0.0
        for j, m in enumerate(MESES):
            p, e0 = M["P"][y, j], M["ET0"][y, j]
            if m in kc:
                dem = kc[m] * e0
                acceso = min(1, c["raiz_cm"] / 180)
                oferta = p * (1 - sitio["escurrimiento"]) + S * acceso
                falta = max(0, dem - oferta)
                r = min(falta / EFIC_RIEGO, riego_disp)
                riego_disp -= r
                riego_usado += r
                oferta += r * EFIC_RIEGO
                a = min(dem, oferta)
                S = min(sitio["s_max"], max(0, S + p * (1 - sitio["escurrimiento"]) + r * EFIC_RIEGO - a))
                etm += dem
                eta += a
                idx = activos.index(m)
                cobertura[j] = max([.25, .55, .85][min(idx, 2)], cobertura_res * .5)
                meses_desde_cosecha = 0
            else:
                S = min(sitio["s_max"], S + p * efic_b)     # almacenamiento en barbecho
                meses_desde_cosecha += 1
                cobertura[j] = cobertura_res * math.exp(-decai * meses_desde_cosecha)
            seco = 1 - min(1, p / max(e0 * .5, 1))
            perd_viento += VIENTO_MES[m] * seco * (1 - cobertura[j]) ** 2
            perd_agua += M["Pext"][y, j] * (1 - cobertura[j])
            perd_viento_base += VIENTO_MES[m] * seco
            perd_agua_base += M["Pext"][y, j]
        # rendimiento relativo (FAO-33) + calor + heladas + efecto del cultivo previo
        ky = c["ky"] * par.get("ky", 1)
        yr_agua = max(0.0, 1 - ky * (1 - eta / etm)) if etm else 0
        mc, umb, a_cal = c["calor"]
        dias = sum(M[f"calor{umb}"][y, MESES.index(m)] for m in mc)
        f_calor = max(.4, 1 - a_cal * max(0, dias - 3))
        mh, umh, a_hel = c["helada"]
        ev = sum(M[f"helada{umh}"][y, MESES.index(m)] for m in mh)
        f_hel = max(.5, 1 - a_hel * ev)
        f_prev = C.EFECTO_PREVIO.get((prev, cult), 1.0) * par.get("prev", 1) if (prev, cult) in C.EFECTO_PREVIO else 1.0
        yr = min(1.15, yr_agua * f_calor * f_hel * f_prev)
        # residuo que queda para los meses siguientes
        if activos:
            cobertura_res = min(.95, 1 - math.exp(-.5 * c["residuo_t_ha"] * max(yr, .3)))
            if labranza == "convencional":
                cobertura_res *= .6
        reg.append(dict(cultivo=cult, previo=prev, anio=sitio["anios"][y], yr=yr, yr_agua=yr_agua, f_calor=f_calor,
                        f_helada=f_hel, f_previo=f_prev, etm=etm, eta=eta, riego=riego_usado, cobertura=cobertura,
                        viento=perd_viento, agua_eros=perd_agua, viento_base=perd_viento_base, agua_base=perd_agua_base,
                        S_fin=S))
    return reg


# ------------------------------------------------------------------ de la simulación a criterios 0-1
def evaluar(rot, reg, sitio, par=None, riego_mm=0, precios=None):
    par = par or {}
    precios = precios or {}
    L = len(rot)
    riego = riego_mm > 0
    margen, n_fijo, n_req, forr, mo_in, calor_f = [], 0.0, 0.0, [], [], []
    agua_sat, riego_tot, detalles = [], 0.0, []
    for r in reg:
        c = C.CULTIVOS[r["cultivo"]]
        cp = C.CULTIVOS[r["previo"]]
        ypot, _ = rendimiento_potencial(r["cultivo"], sitio["rend"], riego)
        rend = ypot * r["yr"]
        precio = precios.get(r["cultivo"], c["precio"]) * par.get("precio", 1)
        # el costo extra del manejo con riego (más semilla, fertilizante) crece con el agua aplicada
        costo = c["costo_secano"] + (c["costo_riego"] - c["costo_secano"]) * min(1.0, r["riego"] / 300)
        credito_n = min(cp["n_fija_kg_ha"], c["n_req_kg_ha"])
        m = rend * precio - costo - r["riego"] * COSTO_BOMBEO_USD_MM_ACRE + credito_n * PRECIO_N_USD_KG / 2.47
        margen.append(m)
        n_fijo += credito_n
        n_req += c["n_req_kg_ha"]
        forr.append(c["forraje"] * (r["yr"] if c["meses"] else 0))
        mo_in.append(c["residuo_t_ha"] * 1.4 * (max(r["yr"], .3) if c["meses"] else 0))
        if c["meses"] and c["ky"] > 0:
            calor_f.append(r["f_calor"] * r["f_helada"])
            agua_sat.append(r["eta"] / r["etm"] if r["etm"] else 1)
        riego_tot += r["riego"]
        detalles.append(dict(cultivo=r["cultivo"], anio=r["anio"], rend=round(rend, 1), unidad=c["unidad"],
                             margen=round(m), yr=round(r["yr"], 2), riego_mm=round(r["riego"])))
    H = len(reg)
    eros_v = sum(r["viento"] for r in reg) / max(sum(r["viento_base"] for r in reg), 1e-6)
    eros_a = sum(r["agua_eros"] for r in reg) / max(sum(r["agua_base"] for r in reg), 1e-6)
    peso_v = sitio["k_viento"] / (sitio["k_viento"] + sitio["k_agua"] * 2)
    eros = peso_v * eros_v + (1 - peso_v) * eros_a        # 1 = como suelo siempre desnudo
    mo_t_ha = sum(mo_in) / H
    riego_anual = riego_tot / H
    crit = dict(
        agua=.6 * (np.mean(agua_sat) if agua_sat else 1) + .4 * (1 - min(1, riego_anual / 450)),
        calor=float(np.mean(calor_f)) if calor_f else 1.0,
        n=min(1, n_fijo / n_req) if n_req else 1.0,
        eros=float(np.clip(1 - eros, 0, 1)),
        mo=float(np.clip(mo_t_ha / 7, 0, 1)),
        ingreso=float(np.clip((np.mean(margen) + 100) / 400, 0, 1)),
        plagas=plagas(rot),
        forraje=float(np.mean(forr)),
    )
    met = dict(margen_usd_acre=float(np.mean(margen)), riego_mm_anio=riego_anual,
               agua_satisfecha=float(np.mean(agua_sat)) if agua_sat else 1.0, eros_relativa=eros,
               eros_viento=eros_v, eros_agua=eros_a, mo_aporte_t_ha=mo_t_ha, n_fijado_kg_ha=n_fijo / H)
    return crit, met, detalles


def plagas(rot):
    """Corte de ciclos: (a) años entre repeticiones del mismo hospedero vs. el intervalo recomendado,
    (b) alternancia invierno/verano (rompe ciclos de malezas), (c) puente verde trigo->trigo."""
    L = len(rot)
    sc = []
    for i, cult in enumerate(rot):
        c = C.CULTIVOS[cult]
        if not c["intervalo_min"]:
            continue
        fam = c["familia"]
        gap = next((d for d in range(1, L + 1) if C.CULTIVOS[rot[(i + d) % L]]["familia"] == fam), L) - 1
        sc.append(min(1, gap / c["intervalo_min"]))
    hospedero = float(np.mean(sc)) if sc else 1
    estac = lambda c: "inv" if 10 in C.CULTIVOS[c]["meses"] or 11 in C.CULTIVOS[c]["meses"] else ("ver" if C.CULTIVOS[c]["meses"] else "nada")
    trans = [(estac(rot[i]), estac(rot[(i + 1) % L])) for i in range(L)] if L > 1 else []
    alterna = np.mean([a != b for a, b in trans]) if trans else 0
    fams = len({C.CULTIVOS[c]["familia"] for c in rot if C.CULTIVOS[c]["meses"]})
    divers = min(1, (fams - 1) / 2)
    return float(.45 * hospedero + .3 * alterna + .25 * divers)


# ------------------------------------------------------------------ restricciones
def chequear(rot, restr):
    prob = []
    for cult in set(rot):
        if cult not in C.CULTIVOS:
            prob.append(f"cultivo desconocido: {cult}")
        if cult in restr.get("sin_maquinaria", []):
            prob.append(f"sin maquinaria para {C.CULTIVOS[cult]['nombre'].lower()}")
        if cult in restr.get("sin_mercado", []):
            prob.append(f"sin mercado cercano para {C.CULTIVOS[cult]['nombre'].lower()}")
    return prob


# ------------------------------------------------------------------ ranking con incertidumbre
def ranking(sitio, rotaciones: dict, prioridades: dict, restr: dict, n_sim=300, seed=0):
    riego_mm = float(restr.get("riego_mm") or 0)
    lab = restr.get("labranza", "reducida")
    precios = restr.get("precios", {})
    crits = list(C.CRITERIOS)
    w0 = np.array([max(0, float(prioridades.get(c, 0))) for c in crits])
    if w0.sum() == 0:
        w0 = np.ones(len(crits))
    n = sitio["n"]
    rng = np.random.default_rng(seed)
    nombres = list(rotaciones)
    base, extra = {}, {}
    for k in nombres:
        rot = rotaciones[k]
        L = len(rot)
        H = max(12, 3 * L)
        crs, mets, regs = [], [], []
        for ini in range(0, n, max(1, n // 15)):      # recorre el registro histórico con distintos inicios
            reg = simular(rot, sitio, ini, H, riego_mm=riego_mm, labranza=lab, precios=precios)
            cr, me, det = evaluar(rot, reg, sitio, riego_mm=riego_mm, precios=precios)
            crs.append(cr), mets.append(me), regs.append((reg, det))
        base[k] = {c: float(np.mean([x[c] for x in crs])) for c in crits}
        extra[k] = dict(met={m: float(np.mean([x[m] for x in mets])) for m in mets[0]},
                        margen_anios=[d["margen"] for _, det in regs for d in det],
                        reg=regs[0][0], det=[d for _, det in regs for d in det])
    # Monte Carlo: clima re-muestreado + parámetros + pesos
    sims = np.zeros((n_sim, len(nombres)))
    margen_sim = np.zeros((n_sim, len(nombres)))
    for i in range(n_sim):
        ini = int(rng.integers(n))
        par = dict(ky=float(rng.normal(1, .1)), prev=float(rng.normal(1, .03)), precio=float(rng.lognormal(0, .15)))
        w = w0 * rng.lognormal(0, .3, len(w0))
        for j, k in enumerate(nombres):
            rot = rotaciones[k]
            reg = simular(rot, sitio, ini, max(6, 2 * len(rot)), par=par, riego_mm=riego_mm, labranza=lab, precios=precios)
            cr, me, _ = evaluar(rot, reg, sitio, par=par, riego_mm=riego_mm, precios=precios)
            cr["plagas"] = base[k]["plagas"]
            x = np.array([cr[c] for c in crits])
            sims[i, j] = 100 * (w @ x) / w.sum()
            margen_sim[i, j] = me["margen_usd_acre"]
    top = np.bincount(sims.argmax(1), minlength=len(nombres)) / n_sim
    res = []
    for j, k in enumerate(nombres):
        x = np.array([base[k][c] for c in crits])
        res.append(dict(rotacion=k, ciclo=rotaciones[k], puntaje=round(float(100 * (w0 @ x) / w0.sum()), 1),
                        rango_p10_p90=[round(float(np.percentile(sims[:, j], 10)), 1), round(float(np.percentile(sims[:, j], 90)), 1)],
                        prob_ser_primera=round(float(top[j]), 2),
                        criterios={c: round(base[k][c], 2) for c in crits},
                        metricas={m: round(v, 2) for m, v in extra[k]["met"].items()},
                        margen_p10_p90=[round(float(np.percentile(margen_sim[:, j], 10))), round(float(np.percentile(margen_sim[:, j], 90)))],
                        prob_margen_negativo=round(float((np.array(extra[k]["margen_anios"]) < 0).mean()), 2),
                        problemas=chequear(rotaciones[k], restr),
                        _reg=extra[k]["reg"], _det=extra[k]["det"]))
    return dict(pesos={c: round(float(w), 2) for c, w in zip(crits, w0 / w0.sum())}, ranking=res)
