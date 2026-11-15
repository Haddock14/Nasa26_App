"""Modelo de predicción FieldShift (port del notebook `modelo_hackaton.ipynb`).

Etapas (iguales al notebook):
  1. Clima NASA POWER (punto 38.0 N, -101.0 O, 1981-2025) -> lluvia trimestral -> estados seco/normal/húmedo
     (terciles por trimestre con 1981-2001) -> cadena de Markov (suavizado de Laplace), validada con Brier
     contra la climatología en 2002-2023.
  2. Rendimientos USDA NASS del condado (trigo, sorgo, maíz; kg/ha) relacionados con la proporción del ciclo de
     cada cultivo en estado seco/húmedo: Ridge (alpha=10) para trigo y sorgo; vecinos climáticos (7) para maíz.
  3. 300 escenarios de clima 2026-2029 simulados con la cadena de Markov + error histórico -> rendimiento
     relativo (1 = promedio histórico) de cada cultivo y año 2027-2029.
  4. Optimización: elige un cultivo por año maximizando (1-a)·promedio + a·CVaR20%, con presupuesto de agua,
     sin repetir cultivo en años seguidos y con un máximo de años de maíz. Se resuelve con Gurobi si está
     instalado (igual que el notebook) o con HiGHS (scipy.optimize.milp); ambos dan el óptimo exacto.

Datos: usa data/datos_unidos_finney.csv (el archivo del notebook) si existe; si no, data/power_kansas y
data/nass_ks (las mismas fuentes NASA POWER y USDA NASS).
"""
from __future__ import annotations

import itertools
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from . import fuentes as F

LAT, LON = 38.0, -101.0
CULTIVOS = ["trigo", "sorgo", "maiz"]
CLAVE_CATALOGO = {"trigo": "trigo", "sorgo": "sorgo", "maiz": "maíz"}
ESTADOS = ["seco", "normal", "humedo"]
ANIOS_PLAN = [2027, 2028, 2029]
ENTRENAMIENTO = (1981, 2001)
VALIDACION = (2002, 2023)
KG_BUSHEL = {"trigo": 27.2155, "sorgo": 25.4012, "maiz": 25.4012}
AGUA_CULTIVO = {"trigo": 450, "sorgo": 350, "maiz": 650}      # ilustrativo, como en el notebook
PRESUPUESTO_AGUA = 1600
ESTRATEGIAS = {"Productividad": 0.0, "Equilibrada": 0.5, "Resiliencia": 1.0}


# ------------------------------------------------------------------ datos
@lru_cache(maxsize=1)
def cargar_datos():
    unido = C.DATA_DIR / "datos_unidos_finney.csv"
    if unido.exists():
        clima, rend = _desde_archivo_unido(unido)
        fuente = "datos_unidos_finney.csv (archivo del notebook)"
    else:
        clima, rend = _desde_proyecto()
        fuente = "NASA POWER (data/power_kansas) y USDA NASS (data/nass_ks)"
    return clima, rend, fuente


def _desde_archivo_unido(ruta):
    cols = ["fuente", "lat", "lon", "fecha", "anio", "lluvia_mm", "temp_max_C", "temp_min_C", "humedad_raiz_0a1",
            "nass_cultivo", "nass_clase", "nass_practica", "nass_uso", "nass_estadistica", "nass_valor", "nass_unidad"]
    nasa, usda = [], []
    for b in pd.read_csv(ruta, usecols=cols, chunksize=100000, low_memory=False):
        f = (b.fuente == "NASA_POWER_diario") & np.isclose(pd.to_numeric(b.lat, errors="coerce"), LAT) & np.isclose(pd.to_numeric(b.lon, errors="coerce"), LON)
        nasa.append(b.loc[f, ["fecha", "lluvia_mm", "temp_max_C", "temp_min_C", "humedad_raiz_0a1"]])
        usda.append(b.loc[b.fuente == "USDA_NASS_rendimiento_condado", ["anio", "nass_cultivo", "nass_clase", "nass_practica", "nass_uso",
                                                                        "nass_estadistica", "nass_valor", "nass_unidad"]])
    clima = pd.concat(nasa, ignore_index=True)
    r = pd.concat(usda, ignore_index=True)
    for c in ["nass_cultivo", "nass_clase", "nass_practica", "nass_uso", "nass_estadistica", "nass_unidad"]:
        r[c] = r[c].astype(str).str.upper().str.strip()
    filtro = (r.nass_cultivo.isin(["WHEAT", "SORGHUM", "CORN"]) & (r.nass_practica == "ALL PRODUCTION PRACTICES")
              & (r.nass_estadistica == "YIELD") & (r.nass_unidad == "BU / ACRE")
              & (((r.nass_cultivo == "WHEAT") & (r.nass_clase == "WINTER") & (r.nass_uso == "ALL UTILIZATION PRACTICES"))
                 | (r.nass_cultivo.isin(["SORGHUM", "CORN"]) & (r.nass_clase == "ALL CLASSES") & (r.nass_uso == "GRAIN"))))
    r = r[filtro].copy()
    r["cultivo"] = r.nass_cultivo.map({"WHEAT": "trigo", "SORGHUM": "sorgo", "CORN": "maiz"})
    r["anio"] = pd.to_numeric(r.anio, errors="coerce")
    r["bu"] = pd.to_numeric(r.nass_valor.astype(str).str.replace(",", ""), errors="coerce")
    r = r.dropna(subset=["anio", "bu"])
    return _limpiar_clima(clima), _a_kg(r[["anio", "cultivo", "bu"]])


def _desde_proyecto():
    pts = F.grilla_power_local()
    p = min(pts, key=lambda q: (q["lat"] - LAT) ** 2 + (q["lon"] - LON) ** 2)
    d = F._leer_power_csv(open(p["archivo"], encoding="utf-8").read()).reset_index()
    clima = d.rename(columns={"date": "fecha", "PRECTOTCORR": "lluvia_mm", "T2M_MAX": "temp_max_C", "T2M_MIN": "temp_min_C",
                              "GWETROOT": "humedad_raiz_0a1"})[["fecha", "lluvia_mm", "temp_max_C", "temp_min_C", "humedad_raiz_0a1"]]
    t = pd.read_csv(C.DATA_DIR / "nass_ks" / "condado_anual.csv.gz")
    t = t[(t.lugar == "FINNEY") & (t.practica == "todas") & (t.variable == "rendimiento") & t.cultivo.isin(["trigo", "sorgo", "maíz"])]
    r = pd.DataFrame(dict(anio=t.anio, cultivo=t.cultivo.replace({"maíz": "maiz"}), bu=t.valor))
    return _limpiar_clima(clima), _a_kg(r)


def _limpiar_clima(clima):
    c = clima.copy()
    c["fecha"] = pd.to_datetime(c.fecha, errors="coerce")
    v = ["lluvia_mm", "temp_max_C", "temp_min_C", "humedad_raiz_0a1"]
    for k in v:
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c[v] = c[v].replace([-999, -9999], np.nan)
    c.loc[c.lluvia_mm < 0, "lluvia_mm"] = np.nan
    c.loc[c.temp_min_C > c.temp_max_C, ["temp_max_C", "temp_min_C"]] = np.nan
    c["temp_media_C"] = (c.temp_max_C + c.temp_min_C) / 2
    c = c.dropna(subset=["fecha"]).drop_duplicates("fecha").sort_values("fecha").reset_index(drop=True)
    c = c[c.fecha.dt.year <= 2025]
    return c


def _a_kg(r):
    r = r.drop_duplicates(["anio", "cultivo"]).copy()
    r["anio"] = r.anio.astype(int)
    r["rendimiento_kg_ha"] = r.bu * r.cultivo.map(KG_BUSHEL) / 0.40468564224
    return r[["anio", "cultivo", "rendimiento_kg_ha"]].sort_values(["cultivo", "anio"]).reset_index(drop=True)


# ------------------------------------------------------------------ etapa 1: Markov trimestral
def trimestres(clima):
    c = clima.copy()
    c["anio"], c["trimestre"] = c.fecha.dt.year, c.fecha.dt.quarter
    c["dia_calor"] = (c.temp_max_C > 32).astype(int)
    t = c.groupby(["anio", "trimestre"]).agg(lluvia_mm=("lluvia_mm", "sum"), temperatura_C=("temp_media_C", "mean"),
                                            dias_calor=("dia_calor", "sum"), dias=("fecha", "count")).reset_index()
    t = t[t.dias >= 89].drop(columns="dias")
    base = t[t.anio.between(*ENTRENAMIENTO)]
    lim = base.groupby("trimestre").lluvia_mm.quantile([1 / 3, 2 / 3]).unstack()
    lim.columns = ["q33", "q67"]
    t = t.merge(lim, left_on="trimestre", right_index=True, how="left")
    t["estado"] = np.where(t.lluvia_mm < t.q33, "seco", np.where(t.lluvia_mm > t.q67, "humedo", "normal"))
    return t.sort_values(["anio", "trimestre"]).reset_index(drop=True), lim


def matriz_markov(estados_seq):
    conteos = pd.crosstab(np.asarray(estados_seq[:-1]), np.asarray(estados_seq[1:])).reindex(index=ESTADOS, columns=ESTADOS, fill_value=0)
    return (conteos + 1).div(conteos.sum(axis=1) + len(ESTADOS), axis=0), conteos


def validar_markov(t):
    """Brier de Markov vs. climatología del mismo trimestre, re-entrenando con el pasado (como el notebook)."""
    res = []
    for i in range(len(t)):
        if not VALIDACION[0] <= t.anio[i] <= VALIDACION[1]:
            continue
        tr = t.iloc[:i]
        P, _ = matriz_markov(tr.estado.to_numpy())
        pm = P.loc[tr.estado.iloc[-1], ESTADOS].to_numpy()
        cb = tr[tr.trimestre == t.trimestre[i]].estado.value_counts().reindex(ESTADOS, fill_value=0).to_numpy()
        pb = (cb + 1) / (cb.sum() + 3)
        real = np.array([e == t.estado[i] for e in ESTADOS], dtype=float)
        res.append((np.sum((pm - real) ** 2), np.sum((pb - real) ** 2)))
    r = np.array(res)
    bm, bc = r[:, 0].mean(), r[:, 1].mean()
    return dict(n=len(r), brier_markov=round(float(bm), 4), brier_climatologia=round(float(bc), 4), mejora_pct=round(float(100 * (1 - bm / bc)), 2))


# ------------------------------------------------------------------ etapa 2: rendimientos
def fechas_cultivo(cultivo, anio):
    return {"trigo": (f"{anio - 1}-10-01", f"{anio}-06-30"), "maiz": (f"{anio}-04-01", f"{anio}-09-30"),
            "sorgo": (f"{anio}-05-01", f"{anio}-09-30")}[cultivo]


def base_agricola(clima, t, rend):
    cc = clima.assign(anio=clima.fecha.dt.year, trimestre=clima.fecha.dt.quarter).merge(t[["anio", "trimestre", "estado"]], on=["anio", "trimestre"], how="left")
    filas = []
    for f in rend.itertuples(index=False):
        ini, fin = fechas_cultivo(f.cultivo, f.anio)
        ciclo = cc[cc.fecha.between(ini, fin)]
        if len(ciclo) != (pd.Timestamp(fin) - pd.Timestamp(ini)).days + 1 or ciclo[["lluvia_mm", "temp_max_C", "estado"]].isna().any().any():
            continue
        p = ciclo.estado.value_counts(normalize=True)
        filas.append(dict(anio=f.anio, cultivo=f.cultivo, rendimiento_kg_ha=f.rendimiento_kg_ha, lluvia_ciclo_mm=ciclo.lluvia_mm.sum(),
                          dias_calor=int((ciclo.temp_max_C > 32).sum()), prop_seco=p.get("seco", 0.0), prop_normal=p.get("normal", 0.0),
                          prop_humedo=p.get("humedo", 0.0)))
    return pd.DataFrame(filas)


def _ridge():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), Ridge(alpha=10))


def modelos_rendimiento(base):
    from sklearn.metrics import mean_absolute_error
    X = ["prop_seco", "prop_humedo"]
    modelos, evaluacion = {}, []
    for c in CULTIVOS:
        d = base[base.cultivo == c].sort_values("anio")
        corte = int(len(d) * .75)
        tr, te = d.iloc[:corte], d.iloc[corte:]
        m = _ridge().fit(tr[X], tr.rendimiento_kg_ha)
        mae_r = mean_absolute_error(te.rendimiento_kg_ha, np.maximum(0, m.predict(te[X])))
        mae_p = mean_absolute_error(te.rendimiento_kg_ha, np.full(len(te), tr.rendimiento_kg_ha.mean()))
        evaluacion.append(dict(cultivo=c, entrenamiento=len(tr), prueba=len(te), mae_ridge=round(float(mae_r)), mae_promedio=round(float(mae_p)),
                               mejora_pct=round(float(100 * (1 - mae_r / mae_p)), 1),
                               metodo="regresión Ridge" if c in ("trigo", "sorgo") else "años con clima parecido (7 vecinos)"))
        modelos[c] = _ridge().fit(d[X], d.rendimiento_kg_ha)
    return modelos, evaluacion


def predecir(c, seco, humedo, modelos, historial):
    if c in ("trigo", "sorgo"):
        p = modelos[c].predict(pd.DataFrame([{"prop_seco": seco, "prop_humedo": humedo}]))[0]
    else:
        h = historial[c]
        dist = (h.prop_seco - seco) ** 2 + (h.prop_humedo - humedo) ** 2
        p = h.loc[dist.nsmallest(7).index, "rendimiento_kg_ha"].mean()
    return max(0.0, float(p))


# ------------------------------------------------------------------ etapa 3: escenarios
@lru_cache(maxsize=4)
def preparar(n_escenarios=300, semilla=42):
    clima, rend, fuente = cargar_datos()
    t, lim = trimestres(clima)
    val = validar_markov(t)
    P, conteos = matriz_markov(t.estado.to_numpy())
    base = base_agricola(clima, t, rend)
    modelos, evaluacion = modelos_rendimiento(base)
    hist = {c: base[base.cultivo == c].copy() for c in CULTIVOS}
    promedios = base.groupby("cultivo").rendimiento_kg_ha.mean().to_dict()
    residuos = {}
    for c in CULTIVOS:
        est = [predecir(c, f.prop_seco, f.prop_humedo, modelos, hist) for f in hist[c].itertuples()]
        e = hist[c].rendimiento_kg_ha.to_numpy() - est
        residuos[c] = e - e.mean()
    # pronóstico de probabilidades trimestrales
    ult = t.iloc[-1]
    prob = np.zeros(3)
    prob[ESTADOS.index(ult.estado)] = 1
    pron = []
    for per in pd.period_range(pd.Period(year=int(ult.anio), quarter=int(ult.trimestre), freq="Q") + 1, f"{max(ANIOS_PLAN)}Q4", freq="Q"):
        prob = prob @ P.to_numpy()
        pron.append(dict(anio=per.year, trimestre=per.quarter, **{e: round(float(v), 4) for e, v in zip(ESTADOS, prob)}))
    # simulación (mismo orden de números aleatorios que el notebook)
    rng = np.random.default_rng(semilla)
    Pm = P.loc[ESTADOS, ESTADOS].to_numpy()
    pesos = {}
    for a in ANIOS_PLAN:
        for c in CULTIVOS:
            ini, fin = fechas_cultivo(c, a)
            pesos[(a, c)] = pd.Series(pd.date_range(ini, fin).to_period("Q")).value_counts(normalize=True).to_dict()
    periodos = pd.period_range("2026Q1", f"{max(ANIOS_PLAN)}Q4", freq="Q")
    filas, trayectorias = [], []
    for s in range(n_escenarios):
        e = ult.estado
        tray = {}
        for per in periodos:
            e = rng.choice(ESTADOS, p=Pm[ESTADOS.index(e)])
            tray[per] = e
        trayectorias.append([tray[p] for p in periodos])
        for a in ANIOS_PLAN:
            for c in CULTIVOS:
                w = pesos[(a, c)]
                ps = sum(v for q, v in w.items() if tray[q] == "seco")
                ph = sum(v for q, v in w.items() if tray[q] == "humedo")
                r = max(0.0, predecir(c, ps, ph, modelos, hist) + rng.choice(residuos[c]))
                filas.append(dict(escenario_id=s + 1, anio=a, cultivo=c, rendimiento_kg_ha=r, rendimiento_relativo=r / promedios[c],
                                  prop_seco=ps, prop_humedo=ph))
    esc = pd.DataFrame(filas)
    matriz = esc.pivot(index="escenario_id", columns=["anio", "cultivo"], values="rendimiento_relativo")
    return dict(clima=clima, trimestral=t, limites=lim, validacion=val, P=P, conteos=conteos, base=base, evaluacion=evaluacion,
                promedios=promedios, pronostico=pron, escenarios=esc, matriz=matriz, trayectorias=trayectorias,
                periodos=[str(p) for p in periodos], fuente=fuente, n=n_escenarios)


# ------------------------------------------------------------------ etapa 4: optimización
def _optimizar_gurobi(M, anios, cults, a, alfa, agua, presupuesto, max_maiz, no_repetir):
    import gurobipy as gp
    from gurobipy import GRB
    N = M.shape[0]
    m = gp.Model("FieldShift")
    m.Params.OutputFlag = 0
    x = m.addVars(anios, cults, vtype=GRB.BINARY)
    z = m.addVar(lb=-GRB.INFINITY)
    u = m.addVars(range(N), lb=0)
    for y in anios:
        m.addConstr(gp.quicksum(x[y, c] for c in cults) == 1)
    if no_repetir:
        for i in range(1, len(anios)):
            for c in cults:
                m.addConstr(x[anios[i - 1], c] + x[anios[i], c] <= 1)
    if "maiz" in cults:
        m.addConstr(gp.quicksum(x[y, "maiz"] for y in anios) <= max_maiz)
    m.addConstr(gp.quicksum(agua[c] * x[y, c] for y in anios for c in cults) <= presupuesto)
    R = [gp.quicksum(float(M[s, j * len(cults) + k]) * x[y, c] for j, y in enumerate(anios) for k, c in enumerate(cults)) / len(anios) for s in range(N)]
    for s in range(N):
        m.addConstr(u[s] >= z - R[s])
    m.setObjective((1 - a) * gp.quicksum(R) / N + a * (z - gp.quicksum(u.values()) / (alfa * N)), GRB.MAXIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None
    return [next(c for c in cults if x[y, c].X > .5) for y in anios]


def _optimizar_highs(M, anios, cults, a, alfa, agua, presupuesto, max_maiz, no_repetir):
    from scipy.optimize import Bounds, LinearConstraint, milp
    N, Y, K = M.shape[0], len(anios), len(cults)
    nx = Y * K
    nv = nx + 1 + N                    # x (binarias), z, u_s
    # R_s = sum_j M[s,j] x_j / Y
    Rcoef = M / Y                       # N x nx
    c = np.zeros(nv)
    c[:nx] = -(1 - a) * Rcoef.mean(axis=0)
    c[nx] = -a
    c[nx + 1:] = a / (alfa * N)
    A, lo, hi = [], [], []
    for j in range(Y):                  # un cultivo por año
        row = np.zeros(nv); row[j * K:(j + 1) * K] = 1
        A.append(row); lo.append(1); hi.append(1)
    if no_repetir:
        for j in range(1, Y):
            for k in range(K):
                row = np.zeros(nv); row[(j - 1) * K + k] = 1; row[j * K + k] = 1
                A.append(row); lo.append(-np.inf); hi.append(1)
    if "maiz" in cults:
        row = np.zeros(nv); row[[j * K + cults.index("maiz") for j in range(Y)]] = 1
        A.append(row); lo.append(-np.inf); hi.append(max_maiz)
    row = np.zeros(nv)
    for j in range(Y):
        for k, cu in enumerate(cults):
            row[j * K + k] = agua[cu]
    A.append(row); lo.append(-np.inf); hi.append(presupuesto)
    U = np.zeros((N, nv))               # u_s - z + R_s >= 0
    U[:, :nx] = Rcoef
    U[:, nx] = -1
    U[np.arange(N), nx + 1 + np.arange(N)] = 1
    cons = [LinearConstraint(np.array(A), lo, hi), LinearConstraint(U, 0, np.inf)]
    integr = np.r_[np.ones(nx), np.zeros(1 + N)]
    lb = np.r_[np.zeros(nx), -np.inf, np.zeros(N)]
    ub = np.r_[np.ones(nx), np.inf, np.full(N, np.inf)]
    r = milp(c, constraints=cons, integrality=integr, bounds=Bounds(lb, ub), options={"disp": False})
    if r.status != 0:
        return None
    xs = r.x[:nx].reshape(Y, K)
    return [cults[int(np.argmax(xs[j]))] for j in range(Y)]


def optimizar(prep, aversion, alfa=.2, presupuesto=PRESUPUESTO_AGUA, agua=None, max_maiz=2, excluir=(), no_repetir=True):
    agua = {**AGUA_CULTIVO, **(agua or {})}
    cults = [c for c in CULTIVOS if c not in set(excluir)]
    if not cults:
        return None, "sin cultivos"
    cols = [(y, c) for y in ANIOS_PLAN for c in cults]
    M = prep["matriz"][cols].to_numpy()
    try:
        import gurobipy  # noqa: F401
        sec, metodo = _optimizar_gurobi(M, ANIOS_PLAN, cults, aversion, alfa, agua, presupuesto, max_maiz, no_repetir), "Gurobi"
    except ImportError:
        sec, metodo = _optimizar_highs(M, ANIOS_PLAN, cults, aversion, alfa, agua, presupuesto, max_maiz, no_repetir), "HiGHS (scipy)"
    return sec, metodo


def indicadores(prep, sec, alfa=.2, presupuesto=PRESUPUESTO_AGUA, agua=None):
    agua = {**AGUA_CULTIVO, **(agua or {})}
    M = prep["matriz"]
    v = M[[(a, c) for a, c in zip(ANIOS_PLAN, sec)]].to_numpy().mean(axis=1)
    k = max(1, int(np.ceil(len(v) * alfa)))
    total = sum(agua[c] for c in sec)
    por_anio = [dict(anio=a, cultivo=c, cultivo_catalogo=CLAVE_CATALOGO[c], promedio=round(float(M[(a, c)].mean()), 3),
                     p10=round(float(M[(a, c)].quantile(.1)), 3), p90=round(float(M[(a, c)].quantile(.9)), 3),
                     prob_bajo=round(float((M[(a, c)] < 1).mean()), 3), agua_mm=agua[c]) for a, c in zip(ANIOS_PLAN, sec)]
    h, bordes = np.histogram(v, bins=np.arange(0.4, 1.65, .05))
    return dict(rotacion=sec, promedio=round(float(v.mean()), 3), cvar_20=round(float(np.sort(v)[:k].mean()), 3),
                p10=round(float(np.quantile(v, .1)), 3), p90=round(float(np.quantile(v, .9)), 3),
                prob_bajo_promedio=round(float((v < 1).mean()), 3), agua_total_mm=float(total),
                uso_agua_pct=round(100 * total / presupuesto, 1), por_anio=por_anio,
                histograma=dict(desde=[round(float(b), 2) for b in bordes[:-1]], cuenta=[int(x) for x in h]))


def todas_rotaciones(prep, alfa=.2, presupuesto=PRESUPUESTO_AGUA, agua=None, max_maiz=2, excluir=(), no_repetir=True, aversion=.5):
    """Evalúa por enumeración las 27 rotaciones posibles (sirve para comprobar el óptimo y para mostrar alternativas)."""
    agua = {**AGUA_CULTIVO, **(agua or {})}
    cults = [c for c in CULTIVOS if c not in set(excluir)]
    out = []
    for sec in itertools.product(cults, repeat=len(ANIOS_PLAN)):
        factible = (sum(agua[c] for c in sec) <= presupuesto and sec.count("maiz") <= max_maiz
                    and (not no_repetir or all(sec[i] != sec[i - 1] for i in range(1, len(sec)))))
        ind = indicadores(prep, list(sec), alfa, presupuesto, agua)
        out.append(dict(rotacion=list(sec), factible=factible, promedio=ind["promedio"], cvar_20=ind["cvar_20"],
                        objetivo=(1 - aversion) * ind["promedio"] + aversion * ind["cvar_20"], agua_total_mm=ind["agua_total_mm"]))
    return out


# ------------------------------------------------------------------ explicación simple
NOMBRE = {"trigo": "trigo", "sorgo": "sorgo", "maiz": "maíz"}


def explicar(nombre, aversion, ind, ref):
    sec = ind["rotacion"]
    txt = {"Productividad": "Busca el mayor rendimiento promedio, sin mirar especialmente los años malos.",
           "Equilibrada": "Da el mismo peso al rendimiento promedio y a protegerse de los años malos.",
           "Resiliencia": "Prioriza que los años malos sean lo menos malos posible, aunque el promedio baje un poco."}.get(
        nombre, f"Peso a los años malos: {aversion:.0%}; peso al promedio: {1 - aversion:.0%}.")
    partes = [txt]
    if ref and ref is not ind:
        d = 100 * (ind["cvar_20"] - ref["cvar_20"]) / ref["cvar_20"]
        dm = 100 * (ind["promedio"] - ref["promedio"]) / ref["promedio"]
        if abs(d) >= .5:
            f1 = lambda v: f"{abs(v):.1f}".replace(".", ",")
            partes.append(f"Frente a la estrategia de productividad, en los peores escenarios rinde {f1(d)} % {'más' if d > 0 else 'menos'}"
                          f" y en promedio {f1(dm)} % {'más' if dm > 0 else 'menos'}.")
    if "maiz" in sec:
        partes.append("Incluye maíz, que en los registros de Finney es el cultivo más estable entre años (se suele regar), pero el que más agua pide.")
    if "sorgo" in sec:
        partes.append("El sorgo pide menos agua y aguanta la sequía; por eso ayuda a cumplir el presupuesto de agua.")
    if "trigo" in sec:
        partes.append("El trigo crece en otoño-invierno-primavera y reparte el riesgo con los cultivos de verano.")
    return " ".join(partes)


def resultado(estrategias=None, aversion_propia=None, alfa=.2, presupuesto=PRESUPUESTO_AGUA, agua=None, max_maiz=2, excluir=(),
              no_repetir=True, n_escenarios=300, semilla=42):
    prep = preparar(int(n_escenarios), int(semilla))
    est = dict(estrategias or ESTRATEGIAS)
    if aversion_propia is not None:
        est["Personalizada"] = float(aversion_propia)
    res, metodo = [], None
    for nombre, a in est.items():
        sec, metodo = optimizar(prep, a, alfa, presupuesto, agua, max_maiz, excluir, no_repetir)
        if sec is None:
            res.append(dict(estrategia=nombre, aversion=a, factible=False,
                            motivo="No hay una rotación que cumpla el presupuesto de agua y las restricciones. Aumente el agua o permita más cultivos."))
            continue
        res.append(dict(estrategia=nombre, aversion=a, factible=True, **indicadores(prep, sec, alfa, presupuesto, agua)))
    ref = next((r for r in res if r["estrategia"] == "Productividad" and r["factible"]), None)
    for r in res:
        if r["factible"]:
            r["explicacion"] = explicar(r["estrategia"], r["aversion"], r, ref)
    t = prep["trimestral"]
    P = prep["P"]
    pron = [p for p in prep["pronostico"] if p["anio"] in ANIOS_PLAN]
    clima = dict(
        limites_mm={int(k): dict(seco_menos_de=round(float(v.q33)), humedo_mas_de=round(float(v.q67))) for k, v in prep["limites"].iterrows()},
        historial=[dict(anio=int(r.anio), trimestre=int(r.trimestre), lluvia_mm=round(float(r.lluvia_mm), 1), estado=r.estado) for r in t.itertuples()],
        matriz={e: {f: round(float(P.loc[e, f]), 3) for f in ESTADOS} for e in ESTADOS},
        ultimo=dict(anio=int(t.anio.iloc[-1]), trimestre=int(t.trimestre.iloc[-1]), estado=t.estado.iloc[-1]),
        pronostico=pron, validacion=prep["validacion"],
        frecuencia_escenarios={e: round(float(np.mean([tr.count(e) / len(tr) for tr in prep["trayectorias"]])), 3) for e in ESTADOS})
    esc = prep["escenarios"]
    cult = []
    for c in CULTIVOS:
        x = esc[esc.cultivo == c].rendimiento_relativo
        ev = next(e for e in prep["evaluacion"] if e["cultivo"] == c)
        cult.append(dict(cultivo=c, cultivo_catalogo=CLAVE_CATALOGO[c], promedio_hist_kg_ha=round(prep["promedios"][c]),
                         promedio_hist_bu_acre=round(prep["promedios"][c] * .40468564224 / KG_BUSHEL[c], 1),
                         relativo_medio=round(float(x.mean()), 3), p10=round(float(x.quantile(.1)), 3), p90=round(float(x.quantile(.9)), 3),
                         prob_bajo=round(float((x < 1).mean()), 3), agua_mm=({**AGUA_CULTIVO, **(agua or {})})[c], evaluacion=ev))
    return dict(
        proyecto="FieldShift Resilience Engine", ubicacion="Finney County, Kansas", punto_nasa=[LAT, LON], horizonte=ANIOS_PLAN,
        escenarios_simulados=prep["n"], metodo_climatico="Markov trimestral (seco/normal/húmedo)",
        metodo_optimizacion=f"CVaR {int(alfa * 100)}% con restricción hídrica ({metodo})", fuente_datos=prep["fuente"],
        presupuesto_agua_mm=presupuesto, demanda_agua_por_cultivo_mm={**AGUA_CULTIVO, **(agua or {})}, agua_es_ilustrativa=agua is None,
        parametros=dict(alfa=alfa, max_maiz=max_maiz, excluir=list(excluir), no_repetir=no_repetir),
        rotaciones={r["estrategia"]: {str(a): c for a, c in zip(ANIOS_PLAN, r["rotacion"])} for r in res if r["factible"]},
        resultados=res, clima=clima, cultivos=cult,
        alternativas=sorted(todas_rotaciones(prep, alfa, presupuesto, agua, max_maiz, excluir, no_repetir), key=lambda r: -r["objetivo"]))


def exportar(carpeta=None):
    """Genera resultados_fieldshift/fieldshift_web.json y comparacion_rotaciones.csv, como el notebook."""
    carpeta = carpeta or (C.DATA_DIR.parent / "resultados_fieldshift")
    carpeta.mkdir(parents=True, exist_ok=True)
    r = resultado()
    salida = {k: r[k] for k in ["proyecto", "ubicacion", "horizonte", "escenarios_simulados", "metodo_climatico", "metodo_optimizacion",
                                "presupuesto_agua_mm", "demanda_agua_por_cultivo_mm", "agua_es_ilustrativa", "rotaciones"]}
    salida["resultados"] = [dict(Estrategia=x["estrategia"], Rotacion=" → ".join(x["rotacion"]), Promedio=x["promedio"], CVaR_20=x["cvar_20"],
                                 Prob_bajo_promedio=x["prob_bajo_promedio"], Agua_total_mm=x["agua_total_mm"], **{"Uso_agua_%": x["uso_agua_pct"]})
                            for x in r["resultados"] if x["factible"]]
    (carpeta / "fieldshift_web.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(salida["resultados"]).to_csv(carpeta / "comparacion_rotaciones.csv", index=False)
    return carpeta


if __name__ == "__main__":
    print("Generado en", exportar())
