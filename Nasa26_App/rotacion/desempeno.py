"""Salud y desempeño de los cultivos (USDA NASS, Kansas 1981-2026) y su relación con el clima NASA.

- Por condado: rendimiento típico, tendencia, años malos y superficie sembrada que no se cosechó (abandono).
- Estado: condición semanal de los cultivos (encuesta a agricultores y técnicos), humedad del suelo medida en
  terreno, fechas reales de siembra y cosecha y precios recibidos.
- Fusión con NASA POWER: cómo cambian la condición y el rendimiento con la lluvia y el calor, y comparación de la
  humedad del suelo del modelo satelital con la observada en terreno.
Las tablas se generan con scripts/preparar_nass_ks.py.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from . import fuentes as F
from . import indicadores as I

DIR = C.DATA_DIR / "nass_ks"
NOMBRE = {"trigo": "Trigo de invierno", "sorgo": "Sorgo", "maíz": "Maíz", "soja": "Soja", "girasol": "Girasol", "alfalfa": "Alfalfa",
          "pastura": "Pasturas"}
UNIDAD = {"trigo": "bu/acre", "sorgo": "bu/acre", "maíz": "bu/acre", "soja": "bu/acre", "girasol": "lb/acre", "alfalfa": "ton/acre"}
# meses de lluvia y de calor que más pesan en cada cultivo (para relacionar con el clima)
VENTANA = {"trigo": dict(lluvia=[3, 4, 5], calor=[5, 6], umbral=32, txt_lluvia="de marzo a mayo", txt_calor="en mayo y junio"),
           "maíz": dict(lluvia=[6, 7, 8], calor=[7, 8], umbral=35, txt_lluvia="de junio a agosto", txt_calor="en julio y agosto"),
           "sorgo": dict(lluvia=[7, 8, 9], calor=[8], umbral=35, txt_lluvia="de julio a septiembre", txt_calor="en agosto"),
           "soja": dict(lluvia=[7, 8], calor=[7, 8], umbral=35, txt_lluvia="en julio y agosto", txt_calor="en julio y agosto"),
           "girasol": dict(lluvia=[7, 8], calor=[7, 8], umbral=35, txt_lluvia="en julio y agosto", txt_calor="en julio y agosto"),
           "alfalfa": dict(lluvia=[5, 6, 7, 8], calor=[7, 8], umbral=35, txt_lluvia="de mayo a agosto", txt_calor="en julio y agosto"),
           "pastura": dict(lluvia=[4, 5, 6, 7], calor=[7, 8], umbral=35, txt_lluvia="de abril a julio", txt_calor="en julio y agosto")}
ETAPA = {"planted": "siembra", "harvested": "cosecha", "headed": "espigazón", "silking": "floración (estigmas)", "blooming": "floración",
         "jointing": "encañazón"}
MESES = ["", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def disponible():
    return (DIR / "condado_anual.csv.gz").exists()


@lru_cache(maxsize=1)
def tablas():
    if not disponible():
        return None
    return {k: pd.read_csv(DIR / f"{k}.csv.gz") for k in ["condado_anual", "condicion_semanal", "humedad_semanal", "avance_semanal", "precios_anuales"]
            if (DIR / f"{k}.csv.gz").exists()}


def fuerza(r):
    a = abs(r)
    return "fuerte" if a >= .5 else ("moderada" if a >= .3 else "débil")


def fecha_doy(doy):
    if doy is None or (isinstance(doy, float) and math.isnan(doy)):
        return None
    d = pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(round(doy)) - 1)
    return f"{d.day} {MESES[d.month]}"


# ------------------------------------------------------------------ precios
@lru_cache(maxsize=1)
def precios_referencia(n=3):
    t = tablas()
    if not t or "precios_anuales" not in t:
        return {}
    p = t["precios_anuales"]
    ult = p.groupby("cultivo").anio.max()
    out = {}
    for c, a in ult.items():
        x = p[(p.cultivo == c) & (p.anio > a - n)]
        out[c] = dict(precio=round(float(x.precio.mean()), 3), unidad=x.unidad.iloc[0], periodo=f"{int(x.anio.min())}-{int(x.anio.max())}")
    return out


# ------------------------------------------------------------------ condado
def condado_de(z: F.Zona):
    x0, y0, x1, y1 = F.FINNEY_BBOX
    if x0 <= z.lon <= x1 and y0 <= z.lat <= y1:
        return "FINNEY"
    c = F.condado_fips(z)
    if c and c.get("estado") == "KS":
        return c["condado"].upper()
    return None


def _serie(c, lugar, cultivo, practica, variable):
    s = c[(c.lugar == lugar) & (c.cultivo == cultivo) & (c.practica == practica) & (c.variable == variable)]
    return s.set_index("anio").valor.sort_index()


def cultivos_condado(condado, clima_diario=None):
    t = tablas()
    if not t or not condado:
        return None
    c = t["condado_anual"]
    if condado not in set(c.lugar):
        return None
    distrito = c[c.lugar == condado].distrito.dropna()
    distrito = distrito.iloc[0] if len(distrito) else None
    out = []
    for cult in ["trigo", "sorgo", "maíz", "soja", "girasol", "alfalfa"]:
        lugar, nivel = condado, "condado"
        r = _serie(c, lugar, cult, "todas", "rendimiento")
        if len(r) < 8 and distrito:
            lugar, nivel = distrito, "distrito"
            r = _serie(c, lugar, cult, "todas", "rendimiento")
        if len(r) < 8:
            continue
        x = r.index.values.astype(float)
        pend, inter = stats.theilslopes(r.values, x)[:2]
        tau, p = stats.kendalltau(x, r.values)
        esperado = inter + pend * x
        rel = r.values / esperado
        reciente = r[r.index > r.index.max() - 10]
        sem = _serie(c, lugar, cult, "todas", "sembrado")
        cos = _serie(c, lugar, cult, "todas", "cosechado")
        ab = (1 - cos / sem).dropna()
        ab = ab[(sem.reindex(ab.index) > 500) & (ab >= -0.05)].clip(lower=0)
        pr = {}
        for k in ["secano", "riego", "secano_tras_barbecho", "secano_continuo"]:
            s = _serie(c, lugar, cult, k, "rendimiento")
            if len(s) >= 5:
                ss = s[s.index > s.index.max() - 10]
                pr[k] = dict(media=round(float(ss.mean()), 1), periodo=f"{int(ss.index.min())}-{int(ss.index.max())}")
        clima = relacion_clima_local(cult, r, esperado, clima_diario) if clima_diario is not None else None
        peor = r.idxmin()
        item = dict(cultivo=cult, nombre=NOMBRE[cult], unidad=UNIDAD[cult], lugar=lugar.title(), nivel_dato=nivel,
                    periodo=f"{int(r.index.min())}-{int(r.index.max())}", n_anios=len(r),
                    rend_reciente=round(float(reciente.mean()), 1), periodo_reciente=f"{int(reciente.index.min())}-{int(reciente.index.max())}",
                    tendencia_decada=round(float(pend * 10), 1), tendencia_pct_decada=round(float(100 * pend * 10 / r.mean()), 1),
                    tendencia_confirmada=bool(p < .05), frac_anios_malos=round(float((rel < .75).mean()), 2),
                    peor_anio=int(peor), peor_valor=round(float(r[peor]), 1),
                    cv_pct=round(float(100 * np.std(rel)), 1),
                    abandono_medio=round(float(ab.mean()), 3) if len(ab) else None,
                    abandono_max=round(float(ab.max()), 3) if len(ab) else None,
                    abandono_peor_anio=int(ab.idxmax()) if len(ab) else None,
                    sembrado_reciente=round(float(sem[sem.index > sem.index.max() - 5].mean())) if len(sem) else None,
                    practicas=pr, clima=clima,
                    serie=[dict(anio=int(a), valor=round(float(v), 1), esperado=round(float(e), 1)) for (a, v), e in zip(r.items(), esperado)])
        malos, aband = item["frac_anios_malos"], item["abandono_medio"] or 0
        item["estado"] = "problema" if malos >= .25 or aband >= .15 else ("atención" if malos >= .1 or aband >= .05 else "bien")
        out.append(item)
    return dict(condado=condado.title(), distrito=(distrito or "").title(), cultivos=out)


def relacion_clima_local(cult, r, esperado, diario):
    """Correlación entre el rendimiento del condado (relativo a su tendencia) y la lluvia/calor locales de NASA POWER."""
    v = VENTANA[cult]
    d = diario
    lluvia = d[d.index.month.isin(v["lluvia"])].groupby(lambda x: x.year).PRECTOTCORR.sum()
    calor = d[d.index.month.isin(v["calor"]) & (d.T2M_MAX >= v["umbral"])].groupby(lambda x: x.year).size()
    calor = calor.reindex(lluvia.index, fill_value=0)
    rel = pd.Series(r.values / esperado, index=r.index)
    out = {}
    for nombre, x, paso, unidad in [("lluvia", lluvia, 25, "mm"), ("calor", calor, 5, "días")]:
        k = rel.index.intersection(x.index)
        if len(k) < 10 or x[k].std() == 0:
            continue
        rr, p = stats.pearsonr(x[k], rel[k])
        b = stats.linregress(x[k], rel[k]).slope * paso * 100
        out[nombre] = dict(r=round(float(rr), 2), p=round(float(p), 3), fuerza=fuerza(rr), n=len(k), efecto_pct=round(float(b), 1),
                           paso=paso, unidad=unidad, meses=v["txt_lluvia" if nombre == "lluvia" else "txt_calor"], umbral=v["umbral"])
    return out or None


# ------------------------------------------------------------------ estado: clima promedio de Kansas (grilla POWER)
@lru_cache(maxsize=1)
def clima_estado():
    cache = F._cache("clima_estado_kansas.json")
    if cache.exists():
        j = json.loads(cache.read_text())
        return pd.DataFrame(j["mensual"])
    filas = []
    for p in F.grilla_power_local():
        df = F._leer_power_csv(open(p["archivo"], encoding="utf-8").read())
        g = df.groupby([df.index.year, df.index.month])
        m = pd.DataFrame(dict(lluvia=g.PRECTOTCORR.sum(), humedad=g.GWETROOT.mean(),
                              calor35=g.T2M_MAX.apply(lambda x: (x >= 35).sum()), calor32=g.T2M_MAX.apply(lambda x: (x >= 32).sum())))
        m.index.names = ["anio", "mes"]
        filas.append(m)
    if not filas:
        return pd.DataFrame()
    m = pd.concat(filas).groupby(["anio", "mes"]).mean().reset_index()
    cache.write_text(json.dumps({"mensual": m.round(3).to_dict("records")}))
    return m


# ------------------------------------------------------------------ estado: salud semanal
def _semana_a_fecha(sem):
    d = pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(sem) * 7 - 4)
    return f"{d.day} {MESES[d.month]}"


@lru_cache(maxsize=1)
def salud_temporada():
    t = tablas()
    if not t or "condicion_semanal" not in t:
        return None
    k = t["condicion_semanal"].copy()
    k["buena"] = k[["good", "excellent"]].sum(axis=1, min_count=1)
    k["mala"] = k[["poor", "very_poor"]].sum(axis=1, min_count=1)
    k["indice"] = (k.very_poor * 1 + k.poor * 2 + k.fair * 3 + k.good * 4 + k.excellent * 5) / 100
    clima = clima_estado()
    out = {}
    for cult, g in k.groupby("cultivo"):
        if g.anio.nunique() < 8:
            continue
        g = g[(g.semana >= 10) & (g.semana <= 46)]
        sem = g.groupby("semana").buena.agg(media="mean", p10=lambda x: x.quantile(.1), p90=lambda x: x.quantile(.9), n="count")
        sem = sem[sem.n >= 5]
        ultimo = int(g.anio.max())
        ult = g[g.anio == ultimo].set_index("semana").buena
        anual = g.groupby("anio").buena.mean()
        anual_ok = anual[g.groupby("anio").size() >= 6]
        mala_sem = g.groupby("semana").mala.mean().reindex(sem.index)
        nucleo = sem[sem.n >= max(10, .5 * sem.n.max())]
        caida = nucleo.media.idxmin()
        tend = I.tendencia(anual_ok)
        rel = None
        if len(clima) and cult in VENTANA:
            v = VENTANA[cult]
            ll = clima[clima.mes.isin(v["lluvia"])].groupby("anio").lluvia.sum()
            ca = clima[clima.mes.isin(v["calor"])].groupby("anio")["calor35" if v["umbral"] >= 35 else "calor32"].sum()
            rel = {}
            for nombre, x, paso, unidad in [("lluvia", ll, 25, "mm"), ("calor", ca, 5, "días")]:
                kk = anual_ok.index.intersection(x.index)
                if len(kk) >= 10 and x[kk].std() > 0:
                    rr, p = stats.pearsonr(x[kk], anual_ok[kk])
                    rel[nombre] = dict(r=round(float(rr), 2), p=round(float(p), 3), fuerza=fuerza(rr), n=len(kk),
                                       efecto_puntos=round(float(stats.linregress(x[kk], anual_ok[kk]).slope * paso), 1), paso=paso, unidad=unidad,
                                       meses=v["txt_lluvia" if nombre == "lluvia" else "txt_calor"], umbral=v["umbral"])
        peores = anual_ok.nsmallest(3)
        out[cult] = dict(
            nombre=NOMBRE.get(cult, cult), periodo=f"{int(anual_ok.index.min())}-{int(anual_ok.index.max())}",
            semanas=[dict(semana=int(s), fecha=_semana_a_fecha(s), media=round(float(r.media), 1), p10=round(float(r.p10), 1), p90=round(float(r.p90), 1),
                          mala=round(float(mala_sem[s]), 1), ultimo=None if s not in ult.index else round(float(ult[s]), 1)) for s, r in sem.iterrows()],
            ultimo_anio=ultimo, media_temporada=round(float(anual_ok.mean()), 1),
            semana_mas_baja=_semana_a_fecha(caida), valor_mas_bajo=round(float(sem.media[caida]), 1),
            peores_anios=[dict(anio=int(a), buena=round(float(v), 1)) for a, v in peores.items()],
            tendencia=tend, relacion_clima=rel or None,
            serie_anual=[dict(anio=int(a), buena=round(float(v), 1)) for a, v in anual_ok.items()])
    return out


# ------------------------------------------------------------------ humedad: terreno vs modelo satelital
@lru_cache(maxsize=1)
def humedad_validacion():
    t = tablas()
    clima = clima_estado()
    if not t or "humedad_semanal" not in t or not len(clima):
        return None
    h = t["humedad_semanal"].copy()
    h["fecha"] = pd.to_datetime(h.week_ending)
    h["mes"] = h.fecha.dt.month
    h["adecuada"] = h[["adequate", "surplus"]].sum(axis=1, min_count=1)
    h["seca"] = h[["short", "very_short"]].sum(axis=1, min_count=1)
    out = {}
    for capa, g in h.groupby("capa"):
        m = g.groupby(["anio", "mes"]).adecuada.mean().reset_index()
        x = m.merge(clima[["anio", "mes", "humedad"]], on=["anio", "mes"])
        if len(x) < 24:
            continue
        # anomalías respecto del promedio de cada mes (quita la estacionalidad)
        x["a_obs"] = x.adecuada - x.groupby("mes").adecuada.transform("mean")
        x["a_mod"] = x.humedad - x.groupby("mes").humedad.transform("mean")
        r, p = stats.pearsonr(x.a_obs, x.a_mod)
        clim = x.groupby("mes").agg(obs=("adecuada", "mean"), modelo=("humedad", "mean")).reset_index()
        seca = g.groupby("mes").seca.mean()
        ver = x[x.mes.between(4, 9)].groupby("anio")[["adecuada", "humedad"]].mean()
        z = (ver - ver.mean()) / ver.std()
        ra = stats.pearsonr(ver.adecuada, ver.humedad)[0] if len(ver) >= 8 else float("nan")
        out[capa] = dict(r=round(float(r), 2), p=round(float(p), 4), fuerza=fuerza(r), n_meses=len(x),
                         periodo=f"{int(x.anio.min())}-{int(x.anio.max())}",
                         mensual=[dict(mes=int(q.mes), nombre=MESES[int(q.mes)], obs=round(float(q.obs), 1), modelo=round(float(q.modelo), 3),
                                       seca=round(float(seca.get(q.mes, np.nan)), 1)) for _, q in clim.iterrows()],
                         r_anual=round(float(ra), 2),
                         anual=[dict(anio=int(a), obs=round(float(q.adecuada), 2), modelo=round(float(q.humedad), 2)) for a, q in z.iterrows()])
    return out


# ------------------------------------------------------------------ calendario real de siembra y cosecha
@lru_cache(maxsize=1)
def calendario_real():
    t = tablas()
    if not t or "avance_semanal" not in t:
        return None
    a = t["avance_semanal"].copy()
    a["fecha"] = pd.to_datetime(a.week_ending, errors="coerce")
    a = a.dropna(subset=["fecha"])
    filas = []
    for (cult, etapa), g in a[a.etapa.isin(ETAPA)].groupby(["cultivo", "etapa"]):
        doys = {}
        for anio, h in g.groupby("anio"):
            h = h.sort_values("fecha")
            dd, pp = h.fecha.dt.dayofyear.values, h.pct.values
            if cult == "trigo" and etapa in ("planted",):
                dd = np.where(h.fecha.dt.year.values > anio, dd + 365, dd)
            i = np.argmax(pp >= 50) if (pp >= 50).any() else None
            if i is None or i == 0:
                continue
            d = dd[i - 1] + (50 - pp[i - 1]) * (dd[i] - dd[i - 1]) / max(pp[i] - pp[i - 1], 1)
            doys[int(anio)] = float(d)
        if len(doys) < 8:
            continue
        s = pd.Series(doys).sort_index()
        rec, ant = s[s.index > s.index.max() - 10], s[s.index < s.index.min() + 10]
        filas.append(dict(cultivo=cult, nombre=NOMBRE.get(cult, cult), etapa=ETAPA[etapa], reciente=fecha_doy(rec.median() % 365 or 365),
                          periodo_reciente=f"{rec.index.min()}-{rec.index.max()}", antes=fecha_doy(ant.median() % 365 or 365),
                          periodo_antes=f"{ant.index.min()}-{ant.index.max()}", cambio_dias=int(round(rec.median() - ant.median())),
                          temprano=fecha_doy(s.quantile(.1) % 365 or 365), tarde=fecha_doy(s.quantile(.9) % 365 or 365)))
    orden = {"trigo": 0, "maíz": 1, "sorgo": 2, "soja": 3, "girasol": 4}
    return sorted(filas, key=lambda f: (orden.get(f["cultivo"], 9), f["etapa"]))


# ------------------------------------------------------------------ todo junto para una zona
def resumen(z: F.Zona, clima_diario=None):
    if not disponible():
        return dict(modo="no disponible", motivo="Faltan las tablas NASS de Kansas (ejecute scripts/preparar_nass_ks.py)")
    cond = condado_de(z)
    return dict(modo="local", fuente="USDA NASS QuickStats, Kansas 1981-2026",
                condado=cultivos_condado(cond, clima_diario) if cond else None,
                salud=salud_temporada(), humedad=humedad_validacion(), calendario=calendario_real(),
                precios=precios_referencia())
