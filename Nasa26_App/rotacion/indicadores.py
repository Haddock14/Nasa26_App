"""Indicadores de clima, agua y suelo calculados a partir de las fuentes.

Todo se calcula sobre la serie larga (POWER 1981-2025) para poder mostrar tendencias y no solo promedios.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ escenario
def aplicar_escenario(df: pd.DataFrame, delta_lluvia_pct=0.0, delta_temp_c=0.0) -> pd.DataFrame:
    """Escenario simple tipo 'delta': escala la lluvia diaria y desplaza las temperaturas.
    No cambia la frecuencia de días de lluvia (limitación documentada)."""
    d = df.copy()
    if delta_lluvia_pct:
        d["PRECTOTCORR"] = d.PRECTOTCORR * (1 + delta_lluvia_pct / 100)
    if delta_temp_c:
        d["T2M_MAX"] = d.T2M_MAX + delta_temp_c
        d["T2M_MIN"] = d.T2M_MIN + delta_temp_c
    return d


# ------------------------------------------------------------------ ET0
def radiacion_extraterrestre(lat, doy):
    """Ra en mm/día equivalentes (FAO-56 ec. 21, x 0.408)."""
    phi = math.radians(lat)
    dr = 1 + .033 * np.cos(2 * np.pi * doy / 365)
    dlt = .409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dlt), -1, 1))
    ra = 24 * 60 / np.pi * .082 * dr * (ws * np.sin(phi) * np.sin(dlt) + np.cos(phi) * np.cos(dlt) * np.sin(ws))
    return ra * .408


def et0_hargreaves(df: pd.DataFrame, lat: float) -> pd.Series:
    """Evapotranspiración de referencia (pasto bien regado), Hargreaves-Samani 1985."""
    tx, tn = df.T2M_MAX, df.T2M_MIN
    ra = radiacion_extraterrestre(lat, df.index.dayofyear.values)
    return (.0023 * ra * ((tx + tn) / 2 + 17.8) * np.sqrt((tx - tn).clip(lower=0))).clip(lower=0)


# ------------------------------------------------------------------ utilidades
def racha_max(mask: pd.Series) -> int:
    m = mask.astype(int)
    return int(m.groupby((m == 0).cumsum()).sum().max()) if len(m) else 0


def tendencia(serie: pd.Series, por=10):
    """Pendiente de Sen (por década) y prueba de Mann-Kendall (vía tau de Kendall)."""
    s = serie.dropna()
    if len(s) < 8:
        return dict(pendiente_por_decada=None, p_valor=None, significativa=False, n=len(s))
    x = np.asarray(s.index, dtype=float)
    pend = stats.theilslopes(s.values, x)[0] * por
    tau, p = stats.kendalltau(x, s.values)
    return dict(pendiente_por_decada=round(float(pend), 2), p_valor=round(float(p), 3), significativa=bool(p < .05),
                n=len(s), tau=round(float(tau), 2))


def comparar_periodos(serie: pd.Series, corte=None):
    s = serie.dropna()
    if len(s) < 10:
        return None
    corte = corte or int(s.index[len(s) // 2])
    a, b = s[s.index < corte], s[s.index >= corte]
    return dict(periodo_1=f"{int(a.index.min())}-{int(a.index.max())}", media_1=round(float(a.mean()), 1),
                periodo_2=f"{int(b.index.min())}-{int(b.index.max())}", media_2=round(float(b.mean()), 1),
                cambio=round(float(b.mean() - a.mean()), 1))


# ------------------------------------------------------------------ clima
def clima(df: pd.DataFrame, lat: float) -> dict:
    d = df.copy()
    d["ET0"] = et0_hargreaves(d, lat)
    anios_completos = d.groupby(d.index.year).size()
    anios_completos = anios_completos[anios_completos >= 360].index
    d = d[d.index.year.isin(anios_completos)]
    filas = {}
    for yr, g in d.groupby(d.index.year):
        gs = g[(g.index.month >= 4) & (g.index.month <= 9)]
        primavera = g[(g.index.month >= 1) & (g.index.month <= 6)]
        otono = g[g.index.month >= 7]
        ult = primavera[primavera.T2M_MIN <= 0].index.dayofyear.max() if (primavera.T2M_MIN <= 0).any() else np.nan
        pri = otono[otono.T2M_MIN <= 0].index.dayofyear.min() if (otono.T2M_MIN <= 0).any() else np.nan
        filas[int(yr)] = dict(
            lluvia_mm=g.PRECTOTCORR.sum(), lluvia_abr_sep_mm=gs.PRECTOTCORR.sum(), et0_mm=g.ET0.sum(),
            deficit_mm=g.ET0.sum() - g.PRECTOTCORR.sum(), evap_suelo_mm=g.EVLAND.sum(),
            dias_calor_35=int((g.T2M_MAX >= 35).sum()),
            dias_calor_32_may_jun=int(((g.T2M_MAX >= 32) & g.index.month.isin([5, 6])).sum()),
            heladas_tardias_abr_may=int(((g.T2M_MIN <= -2) & g.index.month.isin([4, 5])).sum()),
            ultima_helada_doy=ult, primera_helada_doy=pri,
            periodo_libre_heladas_dias=(pri - ult) if not (np.isnan(ult) or np.isnan(pri)) else np.nan,
            racha_seca_abr_sep_dias=racha_max(gs.PRECTOTCORR < 1),
            dias_lluvia_extrema_25mm=int((g.PRECTOTCORR >= 25.4).sum()), lluvia_max_1dia_mm=g.PRECTOTCORR.max(),
            humedad_raiz_abr_sep=gs.GWETROOT.mean(),
        )
    anual = pd.DataFrame(filas).T
    anual.index.name = "anio"

    mens = d.groupby([d.index.year, d.index.month]).agg(
        lluvia=("PRECTOTCORR", "sum"), et0=("ET0", "sum"), calor35=("T2M_MAX", lambda x: (x >= 35).sum()),
        helada=("T2M_MIN", lambda x: (x <= 0).sum()), helada_fuerte=("T2M_MIN", lambda x: (x <= -2).sum()),
        humedad=("GWETROOT", "mean"), tmax=("T2M_MAX", "mean"), tmin=("T2M_MIN", "mean"),
        lluvia_ext=("PRECTOTCORR", lambda x: x[x >= 12.7].sum()))
    mens.index.names = ["anio", "mes"]
    clim = mens.groupby("mes").mean()
    clim["prob_helada"] = mens.groupby("mes").helada.apply(lambda x: (x > 0).mean())
    # anomalía de humedad en zona de raíces: últimos 5 años vs climatología mensual
    z = (mens.humedad - mens.groupby("mes").humedad.transform("mean")) / mens.groupby("mes").humedad.transform("std")
    ult = sorted(mens.index.get_level_values(0).unique())[-5:]
    anom = z[z.index.get_level_values(0).isin(ult)].groupby("mes").mean()
    clim["anomalia_humedad_ult5"] = anom
    # extremos
    p = d.PRECTOTCORR[d.PRECTOTCORR >= 1]
    resumen = {k: round(float(anual[k].mean()), 2) for k in anual.columns}
    resumen.update(lluvia_p95_mm=round(float(p.quantile(.95)), 1), lluvia_p99_mm=round(float(p.quantile(.99)), 1),
                   anios=f"{anual.index.min()}-{anual.index.max()}", n_anios=len(anual),
                   frac_anios_deficit_severo=round(float((anual.lluvia_mm < .7 * anual.lluvia_mm.mean()).mean()), 2))
    tend = {k: tendencia(anual[k]) for k in ["lluvia_mm", "et0_mm", "deficit_mm", "dias_calor_35", "heladas_tardias_abr_may",
                                             "periodo_libre_heladas_dias", "racha_seca_abr_sep_dias",
                                             "dias_lluvia_extrema_25mm", "humedad_raiz_abr_sep"]}
    periodos = {k: comparar_periodos(anual[k]) for k in ["lluvia_mm", "dias_calor_35", "deficit_mm", "humedad_raiz_abr_sep"]}
    return dict(resumen=resumen, anual=anual, mensual=mens, climatologia=clim, tendencias=tend, periodos=periodos, diario=d)


# ------------------------------------------------------------------ suelo
def suelo(t: pd.DataFrame) -> dict:
    """Promedios ponderados por área de la unidad (muacres o área intersectada) x % del componente."""
    s = t.copy()
    for c in ["hzdept_r", "hzdepb_r", "awc_r", "om_r", "claytotal_r", "sandtotal_r", "kffact", "comppct_r", "slope_r", "area"]:
        s[c] = pd.to_numeric(s.get(c), errors="coerce")
    s["esp"] = (s.hzdepb_r.clip(upper=100) - s.hzdept_r.clip(upper=100)).clip(lower=0)
    awc = (s.awc_r * s.esp * 10).groupby(s.cokey).sum()          # mm en 100 cm
    top = s[s.hzdept_r < 30].copy()
    top["e"] = (top.hzdepb_r.clip(upper=30) - top.hzdept_r).clip(lower=0)

    def pond(g):
        r = {}
        for c in ["om_r", "claytotal_r", "sandtotal_r", "kffact"]:
            m = g[c].notna() & (g.e > 0)
            r[c] = np.average(g.loc[m, c], weights=g.loc[m, "e"]) if m.any() else np.nan
        return pd.Series(r)
    comp = top.groupby("cokey")[["om_r", "claytotal_r", "sandtotal_r", "kffact", "e"]].apply(pond)
    meta_cols = [c for c in ["comppct_r", "slope_r", "hydgrp", "weg", "irrcapcl", "nirrcapcl", "area", "mukey", "muname"] if c in s]
    meta = s.drop_duplicates("cokey").set_index("cokey")[meta_cols]
    c = comp.join(meta).join(awc.rename("awc_mm"))
    c["w"] = c.comppct_r.fillna(0) * c.area.fillna(1)

    def m(col):
        k = pd.to_numeric(c[col], errors="coerce").notna() & (c.w > 0)
        return float(np.average(pd.to_numeric(c.loc[k, col]), weights=c.w[k])) if k.any() else float("nan")
    esc = c.hydgrp.fillna("").astype(str).str[0].isin(["C", "D"])
    weg = pd.to_numeric(c.weg.astype(str).str.extract(r"(\d)")[0], errors="coerce") if "weg" in c else pd.Series(dtype=float)
    frac_weg_alta = float(c.w[weg <= 3].sum() / c.w.sum()) if len(weg) else float("nan")
    # unidad dominante
    dom = c.groupby("muname").w.sum().sort_values(ascending=False) if "muname" in c else pd.Series(dtype=float)
    return dict(awc_100cm_mm=round(m("awc_mm")), materia_organica_pct=round(m("om_r"), 2), arcilla_pct=round(m("claytotal_r")),
                arena_pct=round(m("sandtotal_r")), pendiente_pct=round(m("slope_r"), 1), kffact=round(m("kffact"), 2),
                frac_escurrimiento_alto=round(float(c.w[esc].sum() / c.w.sum()), 2),
                frac_erosion_eolica_alta=round(frac_weg_alta, 2),
                grupo_erosion_eolica_medio=round(float(np.average(weg[weg.notna()], weights=c.w[weg.notna()])), 1) if weg.notna().any() else None,
                clase_capacidad_secano=round(m("nirrcapcl"), 1), clase_capacidad_riego=round(m("irrcapcl"), 1),
                suelos_dominantes=[dict(nombre=k, frac=round(float(v / c.w.sum()), 2)) for k, v in dom.head(4).items()])


# ------------------------------------------------------------------ agua subterránea
def acuifero(d: pd.DataFrame, desde=2000) -> dict:
    """Tendencia de profundidad al agua por pozo (pies/año, + = el nivel baja)."""
    res = []
    for pid, g in d.groupby("id"):
        g = g[g.fecha.dt.year >= desde]
        inv = g[g.fecha.dt.month.isin([12, 1, 2, 3])]       # mediciones de invierno: sin bombeo de riego
        g = inv if inv.fecha.dt.year.nunique() >= 5 else g
        anual = g.groupby(g.fecha.dt.year).prof_ft.median()
        if len(anual) < 5:
            continue
        pend = stats.theilslopes(anual.values, anual.index.values)[0]
        res.append(dict(id=pid, lat=g.lat.iloc[0], lon=g.lon.iloc[0], anios=len(anual), desde=int(anual.index.min()),
                        hasta=int(anual.index.max()), prof_ultima_ft=round(float(anual.iloc[-1]), 1),
                        tasa_ft_anio=round(float(pend), 2)))
    if not res:
        return dict(n_pozos=0)
    r = pd.DataFrame(res)
    # serie regional: mediana anual de la anomalía de cada pozo
    dd = d[d.id.isin(r.id)].copy()
    dd["anio"] = dd.fecha.dt.year
    a = dd.groupby(["id", "anio"]).prof_ft.median().reset_index()
    a["anom"] = a.prof_ft - a.groupby("id").prof_ft.transform("mean")
    serie = a[a.anio >= 1960].groupby("anio").anom.median()
    return dict(n_pozos=len(r), tasa_mediana_ft_anio=round(float(r.tasa_ft_anio.median()), 2),
                tasa_mediana_m_anio=round(float(r.tasa_ft_anio.median()) * .3048, 2),
                frac_pozos_en_descenso=round(float((r.tasa_ft_anio > .2).mean()), 2),
                prof_mediana_ultima_ft=round(float(r.prof_ultima_ft.median()), 1),
                serie_anomalia_ft={int(k): round(float(v), 1) for k, v in serie.items()}, pozos=r.to_dict("records"))


def caudal(estaciones) -> list:
    out = []
    for e in estaciones:
        s = e["serie"]
        anual = s.groupby(s.index.year).mean()
        anual = anual[s.groupby(s.index.year).size() > 300]
        out.append(dict(nombre=e["nombre"], lat=e["lat"], lon=e["lon"], km=round(e["km"], 1),
                        periodo=f"{s.index.min().year}-{s.index.max().year}", media_cfs=round(float(s.mean()), 1),
                        frac_dias_menor_10cfs=round(float((s < 10).mean()), 2),
                        frac_dias_sin_flujo=round(float((s <= .01).mean()), 2),
                        tendencia_media_anual=tendencia(anual), serie_anual={int(k): round(float(v), 1) for k, v in anual.items()}))
    return out


def rendimientos(n: pd.DataFrame) -> dict:
    """Rendimiento de referencia por cultivo: secano y riego (grano, no ensilaje)."""
    out = {}
    base = n[~n.short_desc.str.contains("SILAGE|NET PLANTED|OIL TYPE|EXCL ALFALFA")]
    for prod in base.commodity_desc.unique():
        g = base[base.commodity_desc == prod]
        if prod == "HAY":
            g = g[g.class_desc == "ALFALFA"]
        r = {}
        for nombre, filtro in [("todas", "ALL PRODUCTION PRACTICES"), ("riego", "IRRIGATED"), ("secano", "NON-IRRIGATED"),
                               ("secano_tras_barbecho", "NON-IRRIGATED, FOLLOWING SUMMER FALLOW"),
                               ("secano_continuo", "NON-IRRIGATED, CONTINUOUS CROP")]:
            h = g[g.prodn_practice_desc == filtro].drop_duplicates("year")
            if prod == "WHEAT":
                h = g[(g.prodn_practice_desc == filtro) & (g.class_desc == "ALL CLASSES")].drop_duplicates("year")
            if len(h):
                r[nombre] = dict(anios=len(h), media=round(float(h.v.mean()), 1), p90=round(float(h.v.quantile(.9)), 1),
                                 cv_pct=round(float(100 * h.v.std() / h.v.mean()), 1) if len(h) > 2 else None,
                                 periodo=f"{int(h.year.min())}-{int(h.year.max())}", unidad=h.unit_desc.iloc[0])
        if r:
            out[prod] = r
    return out


def landsat(escenas: list) -> dict:
    d = pd.DataFrame(escenas)
    d["mes"] = pd.to_datetime(d.fecha).dt.month
    m = d.groupby("mes").agg(ndvi=("ndvi", "mean"), suelo_desnudo=("frac_suelo_desnudo", "mean"), lst_c=("lst_c", "mean"))
    return dict(n_escenas=len(d), periodo=f"{d.fecha.min()} a {d.fecha.max()}",
                meses_suelo_desnudo=[int(k) for k, v in m.suelo_desnudo.items() if v > .5],
                mensual=m.round(2).reset_index().to_dict("records"), escenas=escenas)
