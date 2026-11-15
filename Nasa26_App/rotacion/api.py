"""API JSON + servidor de la interfaz web.

    python -m rotacion.api            -> http://localhost:5000  (mapa en /, API en /api/...)

GET  /api/evidencia?lat=37.97&lon=-100.87[&delta_lluvia_pct=-10&delta_temp_c=1.5]
POST /api/evidencia                 {"zona": {"lat":..,"lon":..} | {"poligono": GeoJSON}, "escenario": {...}}
POST /api/rotaciones                {"zona":..., "prioridades":{...}, "actual":[...], "alternativas":{...},
                                     "restricciones":{...}, "escenario":{...}}
GET  /api/satelite?lat=..&lon=..[&cuales=landsat,smap]   (lento: Landsat/SMAP/ECOSTRESS/IMERG en vivo)
GET  /api/capas/grilla              indicadores por celda POWER (GeoJSON)
GET  /api/capas/campos?bbox=x0,y0,x1,y1   campos de cultivo (USDA Crop Sequence Boundaries) con historial 2017-2024
GET  /api/capas/pozos               tendencia de pozos USGS (GeoJSON)
GET  /api/capas/suelo?lat=..&lon=..&km=2   polígonos SSURGO con riesgo de erosión y agua útil (en vivo)
GET|POST /api/prediccion            modelo FieldShift 2027-2029 (Markov + escenarios + optimización CVaR)
GET  /api/catalogo                  cultivos, criterios, rotaciones y restricciones por defecto
GET  /api/glosario                  explicación de parámetros y conceptos
"""
from __future__ import annotations

import os
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from . import config as C
from . import explicaciones as X
from . import servicio as S

WEB = Path(__file__).resolve().parent.parent / "web"


def _zona():
    b = request.get_json(silent=True) or {}
    if b.get("zona"):
        return b["zona"], b
    a = request.args
    if "lat" not in a or "lon" not in a:
        raise ValueError("Indique lat y lon (o 'zona' en el cuerpo JSON)")
    z = dict(lat=float(a["lat"]), lon=float(a["lon"]))
    if a.get("radio_km"):
        z["radio_km"] = float(a["radio_km"])
    return z, b


def _escenario(b):
    e = dict(b.get("escenario") or {})
    for k in ("delta_lluvia_pct", "delta_temp_c"):
        if k in request.args:
            e[k] = float(request.args[k])
    e = {**C.ESCENARIO, **e}
    if not (-60 <= e["delta_lluvia_pct"] <= 60 and -3 <= e["delta_temp_c"] <= 6):
        raise ValueError("Escenario fuera de rango (lluvia -60..60 %, temperatura -3..6 °C)")
    return e


def crear_app():
    app = Flask(__name__, static_folder=None)
    app.json.ensure_ascii = False

    @app.after_request
    def cors(r):
        r.headers.update({"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type"})
        return r

    @app.errorhandler(Exception)
    def error(e):
        if isinstance(e, (ValueError, KeyError)):
            return jsonify(error=str(e)), 400
        traceback.print_exc()
        return jsonify(error=f"{type(e).__name__}: {e}"), 500

    @app.get("/")
    def index():
        return send_from_directory(WEB, "index.html")

    @app.get("/web/<path:p>")
    def estaticos(p):
        return send_from_directory(WEB, p)

    @app.route("/api/evidencia", methods=["GET", "POST", "OPTIONS"])
    def evidencia():
        if request.method == "OPTIONS":
            return "", 204
        z, b = _zona()
        return jsonify(S.evidencia(z, _escenario(b)))

    @app.route("/api/rotaciones", methods=["POST", "OPTIONS"])
    def rotaciones():
        if request.method == "OPTIONS":
            return "", 204
        z, b = _zona()
        return jsonify(S.rotaciones(z, b.get("prioridades"), b.get("actual"), b.get("alternativas"), b.get("restricciones"),
                                    _escenario(b), n_sim=min(int(b.get("n_sim", 300)), 1000)))

    @app.get("/api/satelite")
    def satelite():
        z, _ = _zona()
        cuales = tuple(request.args.get("cuales", "landsat,smap,ecostress,imerg").split(","))
        return jsonify(S.satelite(z, cuales))

    @app.get("/api/capas/grilla")
    def grilla():
        return jsonify(S.capa_grilla())

    @app.get("/api/capas/campos")
    def campos():
        if "bbox" not in request.args:          # por defecto: todos los campos del condado
            return jsonify(S.capa_campos_condado())
        try:
            x0, y0, x1, y1 = (float(v) for v in request.args["bbox"].split(","))
        except (KeyError, ValueError):
            raise ValueError("Indique bbox=lon_min,lat_min,lon_max,lat_max")
        if x1 - x0 > .4 or y1 - y0 > .3 or x1 <= x0 or y1 <= y0:
            raise ValueError("Acérquese más en el mapa para ver los campos (área demasiado grande)")
        return jsonify(S.capa_campos((x0, y0, x1, y1)))

    @app.route("/api/campos/historial", methods=["GET", "POST"])
    def historial():
        z, _ = _zona()
        return jsonify(S.historial_punto(z))

    @app.route("/api/prediccion", methods=["GET", "POST", "OPTIONS"])
    def prediccion():
        """Modelo FieldShift: Markov trimestral + rendimientos + 300 escenarios 2027-2029 + optimización CVaR."""
        if request.method == "OPTIONS":
            return "", 204
        from . import prediccion as PR
        b = request.get_json(silent=True) or {}
        presupuesto = float(b.get("presupuesto_agua_mm", PR.PRESUPUESTO_AGUA))
        max_maiz = int(b.get("max_maiz", 2))
        excluir = [c.replace("maíz", "maiz") for c in b.get("excluir", [])]
        av = b.get("aversion")
        if not 300 <= presupuesto <= 3000 or not 0 <= max_maiz <= 3 or (av is not None and not 0 <= float(av) <= 1):
            raise ValueError("Parámetros fuera de rango (agua 300-3000 mm, maíz 0-3 años, aversión 0-1)")
        agua = {k.replace("maíz", "maiz"): float(v) for k, v in (b.get("agua_por_cultivo_mm") or {}).items()} or None
        return jsonify(PR.resultado(aversion_propia=None if av is None else float(av), presupuesto=presupuesto, agua=agua,
                                    max_maiz=max_maiz, excluir=tuple(excluir), no_repetir=bool(b.get("no_repetir", True))))

    @app.get("/api/capas/pozos")
    def pozos():
        return jsonify(S.capa_pozos())

    @app.get("/api/capas/suelo")
    def suelo():
        z, _ = _zona()
        return jsonify(S.capa_suelo(z, float(request.args.get("km", 2))))

    @app.get("/api/catalogo")
    def catalogo():
        from . import desempeno as D
        pn = D.precios_referencia()
        return jsonify(dict(cultivos={k: dict(nombre=v["nombre"], familia=v["familia"], meses=v["meses"],
                                              precio=pn.get(k, {}).get("precio", v["precio"]),
                                              precio_fuente=f"USDA NASS {pn[k]['periodo']}" if k in pn else "supuesto",
                                              unidad=v["unidad"], forraje=v["forraje"]) for k, v in C.CULTIVOS.items()},
                            criterios=C.CRITERIOS, prioridades=C.PRIORIDADES, rotaciones=C.ROTACIONES, actual=C.ACTUAL,
                            restricciones=C.RESTRICCIONES, escenario=C.ESCENARIO))

    @app.get("/api/glosario")
    def glosario():
        return jsonify(X.glosario())

    return app


app = crear_app()

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", 5000)))
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    app.run(host=a.host, port=a.port, threaded=True)
