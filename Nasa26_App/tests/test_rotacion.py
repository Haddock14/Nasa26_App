"""Pruebas sin conexión: python -m pytest -q   (EN_VIVO=0 se fuerza aquí)."""
import os

os.environ["EN_VIVO"] = "0"

import numpy as np
import pandas as pd
import pytest

from rotacion import config as C
from rotacion import fuentes as F
from rotacion import indicadores as I
from rotacion import motor as M
from rotacion import servicio as S
from rotacion.api import crear_app

C.EN_VIVO = False
FINNEY = {"lat": 37.97, "lon": -100.87}


@pytest.fixture(scope="module")
def sitio():
    z = F.Zona(**FINNEY)
    cl = I.clima(F.power(z)["datos"], z.lat)
    su = I.suelo(F.suelo_tabla(z)["tabla"])
    rd = I.rendimientos(F.nass(z)["datos"])
    return M.preparar_sitio(cl, su, rd, {})


def media_yr(sitio, rot, cult, **kw):
    return np.mean([r["yr"] for i in range(sitio["n"]) for r in M.simular(rot, sitio, i, len(rot) * 3, **kw) if r["cultivo"] == cult])


def test_zona_poligono_centroide():
    z = F.Zona.desde({"poligono": {"type": "Polygon", "coordinates": [[[-101, 37], [-100, 37], [-100, 38], [-101, 38], [-101, 37]]]}})
    assert abs(z.lat - 37.4) < .3 and abs(z.lon + 100.6) < .3
    assert z.wkt.startswith("POLYGON((")


def test_clima_serie_larga():
    cl = I.clima(F.power(F.Zona(**FINNEY))["datos"], 37.97)
    assert cl["resumen"]["n_anios"] >= 40
    assert set(cl["tendencias"]["dias_calor_35"]) >= {"pendiente_por_decada", "p_valor", "significativa"}


def test_nass_no_mezcla_ensilaje():
    rd = I.rendimientos(F.nass(F.Zona(**FINNEY))["datos"])
    assert 120 < rd["CORN"]["todas"]["media"] < 220          # bu/acre de grano, no t/acre de ensilaje
    assert rd["WHEAT"]["secano_tras_barbecho"]["media"] > rd["WHEAT"]["secano_continuo"]["media"]


def test_orden_de_la_secuencia_importa(sitio):
    assert media_yr(sitio, ["trigo", "barbecho"], "trigo") > media_yr(sitio, ["trigo"], "trigo") * 1.05
    a = M.evaluar(["trigo", "soja", "sorgo"], M.simular(["trigo", "soja", "sorgo"], sitio, 0, 12), sitio)[0]
    b = M.evaluar(["trigo", "sorgo", "soja"], M.simular(["trigo", "sorgo", "soja"], sitio, 0, 12), sitio)[0]
    assert a != b                                             # mismos cultivos, distinto orden -> distinto resultado


def test_riego_y_escenario(sitio):
    assert media_yr(sitio, ["maíz"], "maíz", riego_mm=450) > media_yr(sitio, ["maíz"], "maíz") + .3
    seco = S.rotaciones(FINNEY, escenario={"delta_lluvia_pct": -30, "delta_temp_c": 2}, n_sim=40)
    base = S.rotaciones(FINNEY, n_sim=40)
    m = lambda R: np.mean([r["metricas"]["margen_usd_acre"] for r in R["ranking"]])
    assert m(seco) < m(base)


def test_sitios_distintos_dan_rankings_distintos():
    oeste = [r["rotacion"] for r in S.rotaciones(FINNEY, n_sim=40)["ranking"]]
    este = [r["rotacion"] for r in S.rotaciones({"lat": 38.5, "lon": -95.5}, n_sim=40)["ranking"]]
    assert oeste != este


def test_restricciones():
    R = S.rotaciones(FINNEY, restricciones={"sin_maquinaria": ["girasol"]}, n_sim=30)
    e = next(r for r in R["ranking"] if "girasol" in r["ciclo"])
    assert e["problemas"] and "posicion" not in e
    with pytest.raises(ValueError):
        S.rotaciones(FINNEY, actual=["trigo", "papa"], n_sim=10)


def test_explicaciones_y_brechas():
    ev = S.evidencia(FINNEY)
    d = ev["clima"]["resumen"]["deficit_mm"]
    assert d["nivel"] in ("favorable", "intermedio", "desfavorable") and d["que_es"] and d["efecto_rotacion"]
    assert ev["acuifero"]["n_pozos"]["valor"] > 5
    assert any("GRACE" in b for b in ev["brechas"])
    r = S.rotaciones(FINNEY, n_sim=30)["ranking"][0]
    for k in ("por_que", "evidencia", "beneficios", "riesgos", "validar_localmente", "calendario"):
        assert r[k]


def test_validacion_razonable():
    v = S.rotaciones(FINNEY, n_sim=10)["validacion"]
    assert v and all(abs(x["error_pct"]) < 60 for x in v)


def test_wkt():
    g = F.wkt_a_geojson("MULTIPOLYGON (((0 0, 1 0, 1 1, 0 0)), ((2 2, 3 2, 3 3, 2 2)))")
    assert g["type"] == "MultiPolygon" and len(g["coordinates"]) == 2
    assert F.wkt_a_geojson("POLYGON ((0 0, 1 0, 1 1, 0 0))")["type"] == "Polygon"


def test_power_en_vivo_simulado(monkeypatch, tmp_path):
    """Comprueba el camino 'en vivo' de POWER con una respuesta falsa (sin red)."""
    csv = "-BEGIN HEADER-\nx\n-END HEADER-\nYEAR,DOY,PRECTOTCORR,EVLAND,T2M_MAX,T2M_MIN,GWETROOT\n2020,1,0.5,0.1,10,-2,0.3\n2020,2,-999,0.1,11,-1,0.3\n"

    class R:
        text = csv
        def raise_for_status(self): pass

    class Sess:
        def get(self, *a, **k): return R()
    monkeypatch.setattr(F, "_http", lambda: Sess())
    monkeypatch.setattr(C, "EN_VIVO", True)
    monkeypatch.setattr(C, "CACHE_DIR", tmp_path)
    r = F.power(F.Zona(10.0, 10.0))
    assert r["modo"] == "en vivo" and len(r["datos"]) == 2 and np.isnan(r["datos"].PRECTOTCORR.iloc[1])


def test_api():
    c = crear_app().test_client()
    assert c.get("/api/evidencia?lat=37.97&lon=-100.87").status_code == 200
    assert c.get("/api/evidencia").status_code == 400
    r = c.post("/api/rotaciones", json={"zona": FINNEY, "n_sim": 20, "restricciones": {"riego_mm": 300}})
    assert r.status_code == 200 and r.get_json()["ranking"]
    assert c.post("/api/rotaciones", json={"zona": FINNEY, "actual": ["xx"]}).status_code == 400
    assert len(c.get("/api/capas/grilla").get_json()["features"]) > 100


def test_campos_rotacion_y_pivote(monkeypatch, tmp_path):
    import json, math
    from rotacion import simple as SP
    assert SP.inferir_rotacion(["trigo", "sorgo", "barbecho"] * 2 + ["trigo", "sorgo"])[0] == ["trigo", "sorgo", "barbecho"]
    assert SP.inferir_rotacion(["maíz", "soja"] * 4)[0] == ["maíz", "soja"]
    assert SP.inferir_rotacion([None] * 7 + ["trigo"])[0] is None
    k = math.cos(math.radians(38))
    circ = [[-100.9 + .004 * math.cos(t / 40 * 2 * math.pi) / k, 38 + .004 * math.sin(t / 40 * 2 * math.pi)] for t in range(41)]
    cuad = [[-100.9, 38], [-100.89, 38], [-100.89, 38.008], [-100.9, 38.008], [-100.9, 38]]
    assert SP.circularidad(circ, 38) > .95 and SP.circularidad(cuad, 38) < .8
    # capa de campos desde archivo local
    d = tmp_path / "campos"
    d.mkdir()
    props = {"csbid": "1", "csbacres": 130, **{f"cdl{a}": c for a, c in zip(range(2017, 2025), [24, 4, 61, 24, 4, 61, 24, 4])}}
    (d / "c.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [circ]}}]}))
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    fc = S.capa_campos((-101, 37.9, -100.8, 38.1))
    p = fc["features"][0]["properties"]
    assert p["rotacion_detectada"] == ["trigo", "sorgo", "barbecho"] and p["pivote_probable"] and p["anios_barbecho"] == 2


def test_resumen_simple():
    ev = S.evidencia(FINNEY)
    t = ev["simple"]["tarjetas"]
    assert len(t) >= 5 and all(x["nivel"] in ("bien", "atención", "problema", "sin dato") and x["para_usted"] for x in t)
    assert len(ev["simple"]["mes_a_mes"]) == 12 and ev["simple"]["antes_ahora"]
    R = S.rotaciones(FINNEY, n_sim=30)
    assert R["certeza"] and all(set(r["simple"]["puntos"]) == {"agua", "suelo", "dinero", "seguridad", "plagas"} for r in R["ranking"])


def test_campos_condado_local(monkeypatch, tmp_path):
    import json
    d = tmp_path / "campos"
    d.mkdir()
    feats = [{"type": "Feature", "properties": {"csbid": str(i), "csbacres": 100, **{f"cdl{a}": 24 for a in range(2017, 2025)}},
              "geometry": {"type": "Polygon", "coordinates": [[[-100.9 + i * .01, 38], [-100.89 + i * .01, 38], [-100.89 + i * .01, 38.01], [-100.9 + i * .01, 38]]]}}
             for i in range(5)]
    (d / "finney.geojson").write_text(json.dumps({"type": "FeatureCollection", "fuente": "prueba", "features": feats}))
    monkeypatch.setattr(C, "DATA_DIR", tmp_path)
    S._CAMPOS_CONDADO.clear()
    fc = S.capa_campos_condado()
    assert len(fc["features"]) == 5 and fc["con_historial"] and fc["fuente"] == "prueba"
    S._CAMPOS_CONDADO.clear()
    sin = {"type": "Feature", "properties": {"csbid": "osm1"}, "geometry": feats[0]["geometry"]}
    (d / "finney.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [sin]}))
    fc = S.capa_campos_condado()
    assert not fc["con_historial"] and "historial" in fc["features"][0]["properties"]["rotacion_texto"]
    S._CAMPOS_CONDADO.clear()


def test_area_y_historial_punto(monkeypatch):
    import math
    from rotacion import simple as SP
    sq = [[-100.9, 38], [-100.88, 38], [-100.88, 38.0145], [-100.9, 38.0145], [-100.9, 38]]
    p = SP.analizar_campo({"csbid": "x"}, {"type": "Polygon", "coordinates": [sq]})
    assert 650 < p["acres"] < 740                  # sin csbacres se calcula por la forma
    x, y = F.albers_conus(37.97, -100.87)
    assert -440000 < x < -410000 and 1.6e6 < y < 1.75e6

    class R:
        def __init__(self, v): self.text = f'<Result>{{x: 1, y: 2, value: "{v}", category: "x"}}</Result>'
        def raise_for_status(self): pass
    codigos = iter([24, 4, 61, 24, 4, 61, 24, 4])

    class Sess:
        def get(self, *a, **k): return R(next(codigos))
    monkeypatch.setattr(F, "_http", lambda: Sess())
    monkeypatch.setattr(C, "EN_VIVO", True)
    monkeypatch.setattr(F, "_cache", lambda n: __import__("pathlib").Path("/tmp/_no_existe_cache_test_" + n))
    h = S.historial_punto({"lat": 38.123, "lon": -100.456})
    assert h["rotacion_detectada"] == ["trigo", "sorgo", "barbecho"]


def test_desempeno_nass():
    from rotacion import desempeno as D
    if not D.disponible():
        return
    c = D.cultivos_condado("FINNEY")
    trigo = next(x for x in c["cultivos"] if x["cultivo"] == "trigo")
    assert 20 < trigo["rend_reciente"] < 60 and 0 < trigo["abandono_medio"] < .5
    p = D.precios_referencia()
    assert 3 < p["trigo"]["precio"] < 12 and p["sorgo"]["unidad"] == "bu"
    h = D.humedad_validacion()
    assert h["subsuelo"]["r"] > .5          # el modelo NASA y la observación en terreno coinciden
    s = D.salud_temporada()
    assert s["maíz"]["relacion_clima"]["calor"]["r"] < 0
    ev = S.evidencia(FINNEY)
    assert ev["desempeno"]["cultivos"] and any(t["clave"] == "desempeno" for t in ev["simple"]["tarjetas"])


def test_prediccion_fieldshift():
    from rotacion import prediccion as PR
    prep = PR.preparar()
    P = prep["P"]
    # mismas probabilidades de transición que el notebook (datos NASA POWER 38N -101O)
    assert abs(P.loc["seco", "seco"] - .5263) < .002 and abs(P.loc["humedo", "humedo"] - .4242) < .002
    assert abs(prep["validacion"]["mejora_pct"] - .55) < .05
    r = PR.resultado()
    rot = {x["estrategia"]: x["rotacion"] for x in r["resultados"]}
    assert rot["Productividad"] == ["sorgo", "trigo", "sorgo"]
    assert rot["Equilibrada"] == ["maiz", "trigo", "sorgo"] and rot["Resiliencia"] == ["maiz", "sorgo", "trigo"]
    for x in r["resultados"]:                      # el optimizador alcanza el óptimo exacto (comprobado por enumeración)
        a = x["aversion"]
        mejor = max(q["objetivo"] for q in PR.todas_rotaciones(prep, aversion=a) if q["factible"])
        assert abs((1 - a) * x["promedio"] + a * x["cvar_20"] - mejor) < 1e-3
        assert x["agua_total_mm"] <= 1600 and all(x["rotacion"][i] != x["rotacion"][i - 1] for i in (1, 2))
    sin = PR.resultado(presupuesto=800)
    assert not any(x["factible"] for x in sin["resultados"])
    sm = PR.resultado(excluir=("maiz",))
    assert all("maiz" not in x["rotacion"] for x in sm["resultados"] if x["factible"])


def test_traduccion_ingles():
    """El catálogo traduce textos fijos, textos con números y frases compuestas del servidor (mismo algoritmo que la página)."""
    from rotacion.i18n import traducir
    casos = {
        "Comparar rotaciones": "Compare rotations",
        "Campo de 1.450 acres": "Field of 1,450 acres",
        "La última helada de primavera llega en promedio el 14 abr y hay 2,0 noches muy frías por año en abril y mayo.":
            "The last spring frost arrives on average on Apr 14, and there are 2.0 very cold nights a year in April and May.",
        "En su zona: déficit bajo.": "In your area: low deficit.",
        "falta agua, heladas probables": "water shortage, frost likely",
        "Trigo de invierno: años malos": "Winter wheat: bad years",
    }
    for es, en in casos.items():
        assert traducir(es) == en, (es, traducir(es))
    # una frase compuesta por el servidor no debe quedar con palabras en español
    r = traducir("Comparado con lo que hace hoy, pierde unos 5 USD por acre al año y protege peor el suelo (17% más erosión).")
    assert r == "Compared with what you do today, it loses about 5 USD per acre per year and protects the soil worse (17% more erosion)."
