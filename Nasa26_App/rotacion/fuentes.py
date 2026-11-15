"""Fuentes de datos por coordenada o polígono.

Cada función acepta una zona (`Zona`: punto lat/lon y opcionalmente un polígono GeoJSON) y:
  1) usa datos locales si cubren la zona (lo descargado para Finney County / grilla POWER de Kansas),
  2) si no, consulta la API en vivo y guarda el resultado en data/_cache,
  3) si tampoco es posible, devuelve {"modo": "no disponible", "motivo": ...} para que la interfaz
     lo muestre como brecha de información en vez de fallar.

APIs en vivo usadas (todas gratuitas; las de NASA Earthdata requieren token en EARTHDATA_TOKEN):
  NASA POWER ............ https://power.larc.nasa.gov/api/temporal/daily/point
  USDA Soil Data Access . https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest
  Landsat C2 L2 (STAC) .. https://planetarycomputer.microsoft.com/api/stac/v1
  SMAP / ECOSTRESS ...... https://appeears.earthdatacloud.nasa.gov/api   (token Earthdata)
  GPM IMERG ............. https://hydro1.gesdisc.eosdis.nasa.gov  data rods (token Earthdata)
  USGS pozos / caudal ... https://api.waterdata.usgs.gov/ogcapi/v0 , https://waterservices.usgs.gov/nwis/dv
  NASS QuickStats ....... https://quickstats.nass.usda.gov/api (clave en NASS_API_KEY)
"""
from __future__ import annotations

import glob
import hashlib
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

TIMEOUT = 60
POWER_PARAMS = ["PRECTOTCORR", "EVLAND", "T2M_MAX", "T2M_MIN", "GWETROOT"]
FINNEY_BBOX = (-101.11, 37.73, -100.22, 38.27)  # lon_min, lat_min, lon_max, lat_max


# ------------------------------------------------------------------ zona
@dataclass
class Zona:
    lat: float
    lon: float
    poligono: dict | None = None          # GeoJSON Polygon (coordenadas lon, lat)
    radio_km: float = 15.0                # para pozos y estaciones cercanas

    @classmethod
    def desde(cls, d: dict) -> "Zona":
        pol = d.get("poligono")
        if pol and pol.get("type") == "Feature":
            pol = pol["geometry"]
        if pol:
            xs, ys = zip(*pol["coordinates"][0])
            lat, lon = float(np.mean(ys)), float(np.mean(xs))
        else:
            lat, lon = float(d["lat"]), float(d["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("lat/lon fuera de rango")
        return cls(lat=lat, lon=lon, poligono=pol, radio_km=float(d.get("radio_km", 15)))

    @property
    def wkt(self) -> str:
        if self.poligono:
            anillo = ", ".join(f"{x} {y}" for x, y in self.poligono["coordinates"][0])
            return f"POLYGON(({anillo}))"
        return f"POINT({self.lon} {self.lat})"

    def bbox(self, km=None):
        km = km or self.radio_km
        if self.poligono:
            xs, ys = zip(*self.poligono["coordinates"][0])
            return min(xs), min(ys), max(xs), max(ys)
        dlat = km / 111.32
        dlon = km / (111.32 * math.cos(math.radians(self.lat)))
        return self.lon - dlon, self.lat - dlat, self.lon + dlon, self.lat + dlat

    def geojson(self):
        if self.poligono:
            return self.poligono
        return {"type": "Point", "coordinates": [self.lon, self.lat]}

    def clave(self):
        return hashlib.md5(json.dumps([round(self.lat, 3), round(self.lon, 3), self.poligono], sort_keys=True).encode()).hexdigest()[:12]


def dist_km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = .5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))


def _nd(motivo, fuente):
    return {"modo": "no disponible", "motivo": motivo, "fuente": fuente}


def _cache(nombre):
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return C.CACHE_DIR / nombre


def _http():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = "rotacion-finney/2.0 (herramienta educativa)"
    return s


def _earthdata_headers():
    if not C.EARTHDATA_TOKEN:
        return None
    return {"Authorization": f"Bearer {C.EARTHDATA_TOKEN}"}


# ------------------------------------------------------------------ NASA POWER (clima diario 1981-hoy)
def _leer_power_csv(texto: str) -> pd.DataFrame:
    lin = texto.splitlines()
    ini = next((i for i, l in enumerate(lin) if l.strip() == "-END HEADER-"), -1) + 1
    df = pd.read_csv(io.StringIO("\n".join(lin[ini:])))
    if "YEAR" in df and "DOY" in df:
        df["date"] = pd.to_datetime(df.YEAR.astype(str)) + pd.to_timedelta(df.DOY - 1, unit="D")
        df = df.drop(columns=["YEAR", "DOY"])
    elif "YEAR" in df and "MO" in df:
        df["date"] = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY))
        df = df.drop(columns=["YEAR", "MO", "DY"])
    else:
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df.replace(-999, np.nan).set_index("date").sort_index()


def grilla_power_local():
    """Puntos de la grilla POWER descargada (data/power_kansas/power_LAT_LON.csv)."""
    out = []
    for f in glob.glob(str(C.DATA_DIR / "power_kansas" / "power_*.csv")):
        m = re.search(r"power_(-?[\d.]+)_(-?[\d.]+)\.csv$", f)
        if m:
            out.append(dict(lat=float(m.group(1)), lon=float(m.group(2)), archivo=f))
    return out


def power(z: Zona, inicio=1981, fin=2025):
    pts = grilla_power_local()
    if pts:
        p = min(pts, key=lambda p: dist_km(z.lat, z.lon, p["lat"], p["lon"]))
        d = dist_km(z.lat, z.lon, p["lat"], p["lon"])
        if d <= 40:   # la resolución nativa de MERRA-2 es 0.5° x 0.625° (~50 km)
            df = _leer_power_csv(Path(p["archivo"]).read_text(encoding="utf-8"))
            return dict(modo="local", fuente="NASA POWER (MERRA-2), archivo local", celda=[p["lat"], p["lon"]],
                        distancia_km=round(d, 1), datos=df)
    cache = _cache(f"power_{z.lat:.2f}_{z.lon:.2f}_{inicio}_{fin}.csv")
    if cache.exists():
        return dict(modo="cache", fuente="NASA POWER API", celda=[z.lat, z.lon], datos=_leer_power_csv(cache.read_text()))
    if not C.EN_VIVO:
        return _nd("Sin datos POWER locales para esta zona y modo en vivo desactivado", "NASA POWER")
    try:
        r = _http().get("https://power.larc.nasa.gov/api/temporal/daily/point", timeout=TIMEOUT * 3, params=dict(
            parameters=",".join(POWER_PARAMS), community="AG", latitude=round(z.lat, 4), longitude=round(z.lon, 4),
            start=f"{inicio}0101", end=f"{fin}1231", format="CSV"))
        r.raise_for_status()
        cache.write_text(r.text)
        return dict(modo="en vivo", fuente="NASA POWER API", celda=[z.lat, z.lon], datos=_leer_power_csv(r.text))
    except Exception as e:  # noqa: BLE001
        return _nd(f"No se pudo consultar NASA POWER: {e}", "NASA POWER")


# ------------------------------------------------------------------ SSURGO (suelo)
SDA_URL = "https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest"
SDA_COLS = ("mu.mukey, mu.musym, mu.muname, co.cokey, co.compname, co.comppct_r, co.slope_r, co.hydgrp, co.drainagecl, "
            "co.irrcapcl, co.nirrcapcl, co.wei, ch.hzdept_r, ch.hzdepb_r, ch.awc_r, ch.om_r, ch.claytotal_r, "
            "ch.sandtotal_r, ch.kffact, mu.muacres, mu.weg")


def _sda(query):
    r = _http().post(SDA_URL, data={"query": query, "format": "JSON+COLUMNNAME"}, timeout=TIMEOUT)
    r.raise_for_status()
    t = r.json().get("Table", [])
    return pd.DataFrame(t[1:], columns=t[0]) if t else pd.DataFrame()


def suelo_tabla(z: Zona):
    """Tabla horizonte x componente con el peso (área) de cada unidad cartográfica dentro de la zona."""
    local = C.DATA_DIR / "ssurgo" / "ssurgo_tabular.csv"
    x0, y0, x1, y1 = FINNEY_BBOX
    dentro_finney = x0 <= z.lon <= x1 and y0 <= z.lat <= y1
    cache = _cache(f"sda_{z.clave()}.csv")
    if cache.exists():
        return dict(modo="cache", fuente="USDA SSURGO (Soil Data Access)", tabla=pd.read_csv(cache))
    if C.EN_VIVO:
        try:
            g = f"geometry::STGeomFromText('{z.wkt}', 4326)"
            if z.poligono:
                areas = _sda(f"SELECT mukey, SUM(mupolygongeo.STIntersection({g}).STArea()) AS area FROM mupolygon "
                             f"WHERE mupolygongeo.STIntersects({g}) = 1 GROUP BY mukey")
            else:
                areas = _sda(f"SELECT mukey, 1 AS area FROM SDA_Get_Mukey_from_intersection_with_WktWgs84('{z.wkt}')")
            if not areas.empty:
                keys = ",".join(str(int(k)) for k in areas.mukey)
                t = _sda(f"SELECT {SDA_COLS.replace('mu.weg', 'NULL AS weg')} FROM mapunit mu "
                         f"JOIN component co ON co.mukey = mu.mukey JOIN chorizon ch ON ch.cokey = co.cokey "
                         f"WHERE mu.mukey IN ({keys}) AND co.majcompflag = 'Yes'")
                weg = _sda(f"SELECT mukey, weg FROM muaggatt WHERE mukey IN ({keys})")
                t = t.drop(columns="weg").merge(weg, on="mukey", how="left").merge(areas, on="mukey")
                for col in t.columns:
                    conv = pd.to_numeric(t[col], errors="coerce")
                    if conv.notna().sum() >= t[col].notna().sum():
                        t[col] = conv
                t.to_csv(cache, index=False)
                return dict(modo="en vivo", fuente="USDA SSURGO (Soil Data Access)", tabla=t)
        except Exception as e:  # noqa: BLE001
            err = str(e)
        else:
            err = "la zona no intersecta unidades de suelo"
    else:
        err = "modo en vivo desactivado"
    if local.exists() and dentro_finney:
        t = pd.read_csv(local)
        t["area"] = t["muacres"]
        return dict(modo="local", fuente="USDA SSURGO, condado completo de Finney (sin consulta espacial)",
                    nota=f"No se pudo consultar SDA ({err}); se usa el promedio del condado ponderado por área.",
                    tabla=t)
    return _nd(f"Sin suelo para esta zona: {err}", "USDA SSURGO")


def poligonos_suelo(z: Zona, km=2.0):
    """Polígonos de unidades de suelo en un cuadro de `km` alrededor del punto (para el mapa de campo)."""
    zz = Zona(z.lat, z.lon, z.poligono)
    x0, y0, x1, y1 = zz.bbox(km)
    caja = f"POLYGON(({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"
    cache = _cache(f"sda_pol_{hashlib.md5(caja.encode()).hexdigest()[:10]}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    if not C.EN_VIVO:
        return _nd("modo en vivo desactivado", "USDA SSURGO")
    try:
        g = f"geometry::STGeomFromText('{caja}', 4326)"
        p = _sda(f"SELECT mukey, mupolygongeo.STIntersection({g}).STAsText() AS wkt FROM mupolygon "
                 f"WHERE mupolygongeo.STIntersects({g}) = 1")
        if p.empty:
            return _nd("sin polígonos", "USDA SSURGO")
        keys = ",".join(sorted({str(int(k)) for k in p.mukey}))
        a = _sda("SELECT mu.mukey, mu.muname, ag.weg, ag.slopegradwta AS pendiente, ag.aws0100wta AS awc_100cm_cm, "
                 "ag.hydgrpdcd AS hydgrp, ag.niccdcd AS nirrcapcl, ag.iccdcd AS irrcapcl, "
                 "(SELECT TOP 1 ch.kffact FROM component co JOIN chorizon ch ON ch.cokey = co.cokey "
                 " WHERE co.mukey = mu.mukey AND co.majcompflag = 'Yes' ORDER BY co.comppct_r DESC, ch.hzdept_r) AS kffact "
                 f"FROM mapunit mu JOIN muaggatt ag ON ag.mukey = mu.mukey WHERE mu.mukey IN ({keys})")
        attrs = {str(int(r.mukey)): r.to_dict() for _, r in a.iterrows()}
        feats = []
        for _, r in p.iterrows():
            geom = wkt_a_geojson(r.wkt)
            if geom:
                feats.append({"type": "Feature", "geometry": geom, "properties": attrs.get(str(int(r.mukey)), {"mukey": r.mukey})})
        out = {"type": "FeatureCollection", "features": feats, "modo": "en vivo", "fuente": "USDA SSURGO (Soil Data Access)"}
        cache.write_text(json.dumps(out, default=str))
        return out
    except Exception as e:  # noqa: BLE001
        return _nd(f"No se pudo consultar SDA: {e}", "USDA SSURGO")


def wkt_a_geojson(wkt: str):
    """Conversor mínimo WKT (POLYGON/MULTIPOLYGON) -> GeoJSON, sin dependencias."""
    if not wkt or "EMPTY" in wkt:
        return None
    def anillos(s):
        return [[[float(v) for v in par.split()] for par in a.strip().strip("()").split(",")] for a in re.findall(r"\([^()]+\)", s)]
    tipo = wkt.split("(", 1)[0].strip().upper()
    cuerpo = wkt[len(tipo):].strip()
    if tipo == "POLYGON":
        return {"type": "Polygon", "coordinates": anillos(cuerpo)}
    if tipo == "MULTIPOLYGON":
        polys = re.findall(r"\(\s*(\([^()]+\)(?:\s*,\s*\([^()]+\))*)\s*\)", cuerpo[1:-1])
        return {"type": "MultiPolygon", "coordinates": [anillos(p) for p in polys]}
    if tipo == "GEOMETRYCOLLECTION":
        partes = [wkt_a_geojson(m) for m in re.findall(r"((?:MULTI)?POLYGON\s*\((?:[^()]|\([^()]*\)|\((?:[^()]|\([^()]*\))*\))*\))", cuerpo)]
        partes = [p for p in partes if p]
        coords = [c for p in partes for c in ([p["coordinates"]] if p["type"] == "Polygon" else p["coordinates"])]
        return {"type": "MultiPolygon", "coordinates": coords} if coords else None
    return None


# ------------------------------------------------------------------ USGS: pozos (acuífero) y caudal
def _pozos_df_desde_features(feats):
    filas = []
    for f in feats:
        p = f["properties"]
        try:
            v = float(p["value"])
        except (TypeError, ValueError):
            continue
        filas.append(dict(id=p["monitoring_location_id"], fecha=p["time"][:10], prof_ft=v,
                          lon=f["geometry"]["coordinates"][0], lat=f["geometry"]["coordinates"][1]))
    d = pd.DataFrame(filas)
    if not d.empty:
        d["fecha"] = pd.to_datetime(d.fecha, errors="coerce")
    return d.dropna(subset=["fecha"])


def pozos(z: Zona):
    """Profundidad al agua subterránea (USGS 72019, pies bajo superficie). Proxy directo del acuífero Ogallala."""
    local = C.DATA_DIR / "nwis" / "gwlevels_finney.json"
    d = pd.DataFrame()
    modo = fuente = None
    if local.exists():
        d = _pozos_df_desde_features(json.load(open(local)))
        d["km"] = [dist_km(z.lat, z.lon, a, b) for a, b in zip(d.lat, d.lon)]
        d = d[d.km <= z.radio_km]
        modo, fuente = "local", "USGS NWIS niveles de pozos (archivo local Finney)"
    if d.empty and C.EN_VIVO:
        x0, y0, x1, y1 = z.bbox()
        cache = _cache(f"pozos_{z.clave()}.json")
        try:
            if cache.exists():
                feats = json.loads(cache.read_text())
            else:
                feats, url = [], "https://api.waterdata.usgs.gov/ogcapi/v0/collections/field-measurements/items"
                params = dict(bbox=f"{x0},{y0},{x1},{y1}", parameter_code="72019", f="json", limit=10000)
                s = _http()
                while url and len(feats) < 50000:
                    r = s.get(url, params=params, timeout=TIMEOUT)
                    r.raise_for_status()
                    j = r.json()
                    feats += j.get("features", [])
                    url = next((l["href"] for l in j.get("links", []) if l.get("rel") == "next"), None)
                    params = None
                cache.write_text(json.dumps(feats))
            d = _pozos_df_desde_features(feats)
            modo, fuente = "en vivo", "USGS Water Data API (field-measurements)"
        except Exception as e:  # noqa: BLE001
            return _nd(f"No se pudo consultar USGS: {e}", "USGS pozos")
    if d.empty:
        return _nd("No hay pozos con mediciones en el radio indicado", "USGS pozos")
    return dict(modo=modo, fuente=fuente, datos=d)


def caudal(z: Zona):
    local = C.DATA_DIR / "nwis" / "nwis_streamflow.json"
    j = None
    modo = "local"
    if local.exists():
        j = json.load(open(local))
    out = []
    def leer(j):
        res = []
        for ts in j["value"]["timeSeries"]:
            g = ts["sourceInfo"]["geoLocation"]["geogLocation"]
            vals = ts["values"][0]["value"]
            if len(vals) < 300:
                continue
            s = pd.Series([float(v["value"]) for v in vals], index=pd.to_datetime([v["dateTime"][:10] for v in vals]))
            s = s[s >= 0]
            res.append(dict(nombre=ts["sourceInfo"]["siteName"], lat=g["latitude"], lon=g["longitude"], serie=s,
                            km=dist_km(z.lat, z.lon, g["latitude"], g["longitude"])))
        return res
    if j:
        out = [e for e in leer(j) if e["km"] <= max(z.radio_km, 40)]
    if not out and C.EN_VIVO:
        x0, y0, x1, y1 = z.bbox(40)
        try:
            r = _http().get("https://waterservices.usgs.gov/nwis/dv/", timeout=TIMEOUT, params=dict(
                format="json", bBox=f"{x0:.4f},{y0:.4f},{x1:.4f},{y1:.4f}", parameterCd="00060", startDT="1986-01-01",
                siteStatus="all"))
            r.raise_for_status()
            out, modo = leer(r.json()), "en vivo"
        except Exception as e:  # noqa: BLE001
            return _nd(f"No se pudo consultar NWIS: {e}", "USGS caudal")
    if not out:
        return _nd("Sin estaciones de caudal cercanas", "USGS caudal")
    return dict(modo=modo, fuente="USGS NWIS caudal diario (00060)", estaciones=out)


# ------------------------------------------------------------------ NASS (rendimientos del condado)
def condado_fips(z: Zona):
    x0, y0, x1, y1 = FINNEY_BBOX
    if x0 <= z.lon <= x1 and y0 <= z.lat <= y1:
        return dict(estado="KS", condado="FINNEY", fips="20055")
    try:
        r = _http().get("https://geo.fcc.gov/api/census/block/find", timeout=20,
                        params=dict(latitude=z.lat, longitude=z.lon, format="json"))
        c = r.json()["County"]
        return dict(estado=r.json()["State"]["code"], condado=c["name"].upper().replace(" COUNTY", ""), fips=c["FIPS"])
    except Exception:  # noqa: BLE001
        return None


def nass(z: Zona):
    """Rendimientos por cultivo y práctica (secano/riego). Corrige el error anterior que mezclaba
    maíz grano (bu/acre) con ensilaje (t/acre) y 'net planted acre'."""
    c = condado_fips(z)
    local = C.DATA_DIR / "nass" / "quickstats_county.csv"
    if c and c["fips"] == "20055" and local.exists():
        n, modo = pd.read_csv(local), "local"
    elif c and os.getenv("NASS_API_KEY") and C.EN_VIVO:
        cache = _cache(f"nass_{c['fips']}.csv")
        try:
            if cache.exists():
                n = pd.read_csv(cache)
            else:
                r = _http().get("https://quickstats.nass.usda.gov/api/api_GET/", timeout=TIMEOUT, params=dict(
                    key=os.getenv("NASS_API_KEY"), source_desc="SURVEY", statisticcat_desc="YIELD",
                    agg_level_desc="COUNTY", state_alpha=c["estado"], county_ansi=c["fips"][-3:], format="CSV"))
                r.raise_for_status()
                cache.write_text(r.text)
                n = pd.read_csv(io.StringIO(r.text))
            modo = "en vivo"
        except Exception as e:  # noqa: BLE001
            return _nd(f"No se pudo consultar NASS: {e}", "USDA NASS")
    else:
        return _nd("Rendimientos NASS solo locales para Finney; para otro condado defina NASS_API_KEY", "USDA NASS")
    n = n[(n.statisticcat_desc == "YIELD") & n.unit_desc.str.contains("/ ACRE") & ~n.unit_desc.str.contains("NET")].copy()
    n["v"] = pd.to_numeric(n.Value.astype(str).str.replace(",", ""), errors="coerce")
    n = n.dropna(subset=["v"])
    return dict(modo=modo, fuente=f"USDA NASS QuickStats, condado {c['condado']}", datos=n)


# ------------------------------------------------------------------ Landsat (NDVI y temperatura superficial)
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"


def landsat(z: Zona, anios=3, max_escenas=48):
    """Serie NDVI/LST desde Landsat Collection 2 Level-2 (Planetary Computer). Requiere rasterio.
    Toma la escena menos nublada de cada mes de los últimos `anios` y lee una ventana pequeña."""
    cache = _cache(f"landsat_{z.clave()}_{anios}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    local = sorted(glob.glob(str(C.DATA_DIR / "landsat" / "*_red.tif")))
    try:
        import rasterio
        from rasterio.warp import transform as rtransform
        from rasterio.windows import from_bounds
    except ImportError:
        return _nd("Falta la librería rasterio (pip install rasterio)", "Landsat")
    if not C.EN_VIVO and not local:
        return _nd("modo en vivo desactivado", "Landsat")
    s = _http()
    hoy = pd.Timestamp.today()
    try:
        body = {"collections": ["landsat-c2-l2"], "intersects": z.geojson(), "limit": 500,
                "datetime": f"{hoy.year - anios}-01-01/{hoy.date()}", "query": {"eo:cloud_cover": {"lt": 50}}}
        items = s.post(f"{STAC}/search", json=body, timeout=TIMEOUT).json().get("features", [])
        tok = s.get("https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2", timeout=TIMEOUT).json()["token"]
    except Exception as e:  # noqa: BLE001
        return _nd(f"No se pudo consultar STAC: {e}", "Landsat")
    por_mes = {}
    for it in items:
        m = it["properties"]["datetime"][:7]
        if m not in por_mes or it["properties"]["eo:cloud_cover"] < por_mes[m]["properties"]["eo:cloud_cover"]:
            por_mes[m] = it
    filas = []
    x0, y0, x1, y1 = z.bbox(0.6)
    for mes, it in sorted(por_mes.items())[-max_escenas:]:
        try:
            bandas = {}
            for b in ["red", "nir08", "lwir11", "qa_pixel"]:
                href = it["assets"][b]["href"] + "?" + tok
                with rasterio.open(href) as src:
                    xs, ys = rtransform("EPSG:4326", src.crs, [x0, x1], [y0, y1])
                    w = from_bounds(min(xs), min(ys), max(xs), max(ys), src.transform)
                    bandas[b] = src.read(1, window=w).astype(float)
            qa = bandas["qa_pixel"].astype(int)
            limpio = ((qa & 0b111010) == 0) & (bandas["red"] > 0) & (bandas["nir08"] > 0)
            if limpio.mean() < .3:
                continue
            r, n = bandas["red"] * 2.75e-5 - .2, bandas["nir08"] * 2.75e-5 - .2
            ndvi = ((n - r) / (n + r))[limpio]
            lst = (bandas["lwir11"] * .00341802 + 149 - 273.15)[limpio & (bandas["lwir11"] > 0)]
            filas.append(dict(fecha=it["properties"]["datetime"][:10], ndvi=round(float(np.nanmean(ndvi)), 3),
                              frac_suelo_desnudo=round(float((ndvi < .25).mean()), 2),
                              lst_c=round(float(np.nanmean(lst)), 1) if lst.size else None,
                              frac_valida=round(float(limpio.mean()), 2)))
        except Exception:  # noqa: BLE001
            continue
    if not filas:
        return _nd("No hubo escenas Landsat válidas", "Landsat")
    out = dict(modo="en vivo", fuente="Landsat 8/9 Collection 2 L2 (Planetary Computer)", escenas=filas)
    cache.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------ AppEEARS: SMAP y ECOSTRESS (token Earthdata)
APPEEARS = "https://appeears.earthdatacloud.nasa.gov/api"
CAPAS_APPEEARS = {
    "smap": [{"product": "SPL3SMP_E.006", "layer": "Soil_Moisture_Retrieval_Data_AM_soil_moisture"}],
    "ecostress": [{"product": "ECO_L2T_LSTE.002", "layer": "LST"}],
}


def appeears_punto(z: Zona, cual: str, inicio="2015-04-01", fin=None, esperar_s=0):
    """Pide una serie de punto a AppEEARS. Es asíncrono: la 1a llamada crea la tarea y la guarda;
    las siguientes revisan si terminó. Con esperar_s > 0 sondea hasta ese tiempo."""
    h = _earthdata_headers()
    fuente = {"smap": "NASA SMAP L3 humedad de suelo 9 km", "ecostress": "NASA ECOSTRESS temperatura superficial 70 m"}[cual]
    cache = _cache(f"appeears_{cual}_{z.clave()}.json")
    estado = json.loads(cache.read_text()) if cache.exists() else {}
    if estado.get("serie"):
        return dict(modo="cache", fuente=fuente, serie=estado["serie"])
    if not h:
        return _nd("Defina EARTHDATA_TOKEN para consultar AppEEARS", fuente)
    if not C.EN_VIVO:
        return _nd("modo en vivo desactivado", fuente)
    s = _http()
    try:
        if not estado.get("task_id"):
            fin = fin or str(pd.Timestamp.today().date())
            tarea = {"task_type": "point", "task_name": f"rotacion_{cual}_{z.clave()}",
                     "params": {"dates": [{"startDate": pd.Timestamp(inicio).strftime("%m-%d-%Y"),
                                           "endDate": pd.Timestamp(fin).strftime("%m-%d-%Y")}],
                                "layers": CAPAS_APPEEARS[cual],
                                "coordinates": [{"latitude": z.lat, "longitude": z.lon, "id": "zona"}]}}
            r = s.post(f"{APPEEARS}/task", json=tarea, headers=h, timeout=TIMEOUT)
            r.raise_for_status()
            estado = {"task_id": r.json()["task_id"], "creada": time.time()}
            cache.write_text(json.dumps(estado))
        t0 = time.time()
        while True:
            st = s.get(f"{APPEEARS}/task/{estado['task_id']}", headers=h, timeout=TIMEOUT).json()
            if st.get("status") == "done" or time.time() - t0 >= esperar_s:
                break
            time.sleep(10)
        if st.get("status") != "done":
            return dict(modo="pendiente", fuente=fuente, motivo=f"Tarea AppEEARS en estado '{st.get('status')}'. Reintente en unos minutos.")
        archivos = s.get(f"{APPEEARS}/bundle/{estado['task_id']}", headers=h, timeout=TIMEOUT).json()["files"]
        csvf = next(f for f in archivos if f["file_name"].endswith(".csv") and "results" in f["file_name"])
        txt = s.get(f"{APPEEARS}/bundle/{estado['task_id']}/{csvf['file_id']}", headers=h, timeout=TIMEOUT).text
        df = pd.read_csv(io.StringIO(txt))
        col = next(c for c in df.columns if CAPAS_APPEEARS[cual][0]["layer"].split("_")[-1] in c and not c.endswith("QC"))
        df = df[["Date", col]].rename(columns={"Date": "fecha", col: "valor"})
        df = df[df.valor > -9000]
        if cual == "ecostress" and df.valor.median() > 200:
            df["valor"] = df.valor - 273.15
        estado["serie"] = df.to_dict("records")
        cache.write_text(json.dumps(estado))
        return dict(modo="en vivo", fuente=fuente, serie=estado["serie"])
    except Exception as e:  # noqa: BLE001
        return _nd(f"Error AppEEARS: {e}", fuente)


# ------------------------------------------------------------------ GPM IMERG (lluvia, data rods de GES DISC)
def imerg(z: Zona, inicio="2001-01-01", fin=None):
    fuente = "NASA GPM IMERG (data rods GES DISC)"
    cache = _cache(f"imerg_{z.lat:.2f}_{z.lon:.2f}.csv")
    if cache.exists():
        return dict(modo="cache", fuente=fuente, serie=pd.read_csv(cache, parse_dates=["fecha"]).set_index("fecha").mm)
    h = _earthdata_headers()
    if not h:
        return _nd("Defina EARTHDATA_TOKEN para consultar IMERG", fuente)
    fin = fin or str(pd.Timestamp.today().date())
    try:
        r = _http().get("https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/access/timeseries.cgi", headers=h, timeout=TIMEOUT * 5,
                        params=dict(variable="GPM:GPM_3IMERGHH_07:precipitation", type="asc2",
                                    location=f"GEOM:POINT({z.lon:.3f}, {z.lat:.3f})",
                                    startDate=f"{inicio}T00", endDate=f"{fin}T23"))
        r.raise_for_status()
        filas = [l.split() for l in r.text.splitlines() if re.match(r"^\d{4}-\d{2}-\d{2}", l)]
        df = pd.DataFrame(filas, columns=["d", "h", "mmh"][:len(filas[0])])
        df["mm"] = pd.to_numeric(df.mmh, errors="coerce").clip(lower=0) * .5          # 30 min
        s = df.groupby("d").mm.sum()
        s.index = pd.to_datetime(s.index)
        s.rename_axis("fecha").to_frame().to_csv(cache)
        return dict(modo="en vivo", fuente=fuente, serie=s)
    except Exception as e:  # noqa: BLE001
        return _nd(f"No se pudo consultar IMERG: {e}", fuente)


# ------------------------------------------------------------------ GRACE / GRACE-FO (archivo exportado)
def grace(z: Zona):
    """GRACE tiene ~300 km de resolución: sirve como tendencia regional, no por campo.
    Lee data/grace/grace_tws.csv (columnas: fecha, tws_cm) exportado p. ej. desde el
    GRACE Data Analysis Tool o PO.DAAC (TELLUS_GRAC-GRFO_MASCON)."""
    f = C.DATA_DIR / "grace" / "grace_tws.csv"
    if not f.exists():
        return _nd("Sin serie GRACE local (data/grace/grace_tws.csv). Se usan los pozos USGS como medición directa del acuífero.",
                   "NASA GRACE/GRACE-FO")
    d = pd.read_csv(f, parse_dates=["fecha"]).set_index("fecha").tws_cm
    return dict(modo="local", fuente="NASA GRACE/GRACE-FO JPL mascons", serie=d)


# ------------------------------------------------------------------ OpenET (ET real satelital, archivo local opcional)
def openet(z: Zona):
    """ET real mensual de OpenET (Landsat). Lee data/openet/openet_et_monthly.json (formato de la versión 1)."""
    f = C.DATA_DIR / "openet" / "openet_et_monthly.json"
    if not f.exists():
        return _nd("Sin archivo OpenET local (opcional; requiere clave de openetdata.org)", "OpenET")
    d = pd.DataFrame(json.load(open(f)))
    d["time"] = pd.to_datetime(d["time"])
    anual = d.groupby(d.time.dt.year).et.sum().round()
    return dict(modo="local", fuente="OpenET (ensamble Landsat)", et_real_anual_mm={int(k): float(v) for k, v in anual.items()})


# ------------------------------------------------------------------ campos de cultivo (Crop Sequence Boundaries)
def campos(bbox, max_campos=3000):
    """Polígonos de campos con el cultivo de cada año 2017-2024.
    1) data/campos/*.geojson (descargado con scripts/descargar_campos.py), 2) servicio USDA en vivo."""
    x0, y0, x1, y1 = bbox
    feats = []
    for f in glob.glob(str(C.DATA_DIR / "campos" / "*.geojson")):
        for ft in json.load(open(f)).get("features", []):
            cx, cy = ft["properties"].get("lon"), ft["properties"].get("lat")
            if cx is None:
                xs, ys = zip(*_anillo(ft["geometry"]))
                cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                feats.append(ft)
    if feats:
        return dict(modo="local", fuente="USDA NASS Crop Sequence Boundaries (archivo local)", features=feats[:max_campos])
    cache = _cache(f"campos_{x0:.3f}_{y0:.3f}_{x1:.3f}_{y1:.3f}.json")
    if cache.exists():
        return dict(modo="cache", fuente="USDA NASS Crop Sequence Boundaries 2017-2024", features=json.loads(cache.read_text()))
    if not C.EN_VIVO:
        return _nd("Sin campos locales (data/campos) y modo en vivo desactivado", "USDA Crop Sequence Boundaries")
    campos_cdl = ",".join(f"cdl{a}" for a in C.CSB_ANIOS)
    try:
        s = _http()
        while len(feats) < max_campos:
            r = s.get(C.CSB_URL, timeout=TIMEOUT, params=dict(
                f="geojson", where="1=1", geometry=f"{x0},{y0},{x1},{y1}", geometryType="esriGeometryEnvelope", inSR=4326,
                spatialRel="esriSpatialRelIntersects", outSR=4326, outFields=f"csbid,csbacres,{campos_cdl}",
                returnGeometry="true", geometryPrecision=5, resultOffset=len(feats), resultRecordCount=2000))
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                raise RuntimeError(j["error"].get("message"))
            nuevos = j.get("features", [])
            feats += nuevos
            if len(nuevos) < 2000:
                break
        cache.write_text(json.dumps(feats))
        return dict(modo="en vivo", fuente="USDA NASS Crop Sequence Boundaries 2017-2024", features=feats)
    except Exception as e:  # noqa: BLE001
        return _nd(f"No se pudo consultar el servicio de campos del USDA: {e}", "USDA Crop Sequence Boundaries")


_CSB_ERROR = {}


def _csb_consulta(bbox, max_campos=40000, simplificar=True):
    """Consulta el servicio USDA Crop Sequence Boundaries probando las capas conocidas (2, 0, 1).
    Normaliza nombres de atributos a minúsculas y elimina duplicados."""
    x0, y0, x1, y1 = bbox
    campos_cdl = ",".join(f"cdl{a}" for a in C.CSB_ANIOS)
    s = _http()
    base = C.CSB_URL.rsplit("/", 2)[0]          # .../FeatureServer
    capas = [C.CSB_URL] + [f"{base}/{k}/query" for k in (2, 0, 1) if f"{base}/{k}/query" != C.CSB_URL]
    errores = []
    for url in capas:
        feats, vistos = [], set()
        try:
            while len(feats) < max_campos:
                params = dict(f="geojson", where="1=1", geometry=f"{x0},{y0},{x1},{y1}", geometryType="esriGeometryEnvelope",
                              inSR=4326, spatialRel="esriSpatialRelIntersects", outSR=4326, outFields="*",
                              returnGeometry="true", geometryPrecision=5, resultOffset=len(vistos), resultRecordCount=2000)
                if simplificar:
                    params["maxAllowableOffset"] = .00008
                r = s.get(url, timeout=TIMEOUT * 2, params=params)
                r.raise_for_status()
                j = r.json()
                if "error" in j:
                    raise RuntimeError(j["error"].get("message") or str(j["error"]))
                nuevos = j.get("features", [])
                for f in nuevos:
                    p = {k.lower(): v for k, v in (f.get("properties") or {}).items()}
                    f["properties"] = {k: p.get(k) for k in ["csbid", "csbacres", *[f"cdl{a}" for a in C.CSB_ANIOS]]}
                    k = p.get("csbid") or p.get("objectid") or len(vistos)
                    if k not in vistos:
                        vistos.add(k)
                        feats.append(f)
                if len(nuevos) < 2000:
                    break
            if feats:
                return feats, None
            errores.append(f"{url.split('/FeatureServer/')[1]}: sin campos")
        except Exception as e:  # noqa: BLE001
            errores.append(f"{url.split('/FeatureServer/')[1]}: {e}")
    return None, "; ".join(errores)


def campos_condado(bbox=FINNEY_BBOX, nombre="finney"):
    """Todos los campos de un condado para verlos a escala de condado.
    Orden: archivo local del USDA -> servicio USDA Crop Sequence Boundaries (se guarda en data/campos/)
    -> archivo local o consulta de OpenStreetMap (sin historial ni superficie oficial) como último respaldo.
    Si solo hay respaldo OSM guardado, se vuelve a intentar el USDA una vez por sesión."""
    destino = C.DATA_DIR / "campos" / f"{nombre}.geojson"
    locales = sorted(glob.glob(str(C.DATA_DIR / "campos" / "*.geojson")))
    usda = [f for f in locales if not f.endswith("_osm.geojson")]
    osm = [f for f in locales if f.endswith("_osm.geojson")]
    if usda:
        feats = [ft for f in usda for ft in json.load(open(f)).get("features", [])]
        fuente = json.load(open(usda[0])).get("fuente", "USDA NASS Crop Sequence Boundaries (archivo local)")
        return dict(modo="local", fuente=fuente, features=feats)
    errores = []
    if C.EN_VIVO and not _CSB_ERROR.get(nombre):
        feats, err = _csb_consulta(bbox)
        if feats:
            fuente = "USDA NASS Crop Sequence Boundaries 2017-2024"
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(json.dumps({"type": "FeatureCollection", "fuente": fuente, "features": feats}))
            return dict(modo="en vivo", fuente=fuente, features=feats)
        _CSB_ERROR[nombre] = err
    if _CSB_ERROR.get(nombre):
        errores.append(f"USDA: {_CSB_ERROR[nombre]}")
    if osm:
        feats = [ft for f in osm for ft in json.load(open(f)).get("features", [])]
        return dict(modo="local", fuente="OpenStreetMap (sin historial; el servicio del USDA no respondió)", features=feats,
                    motivo="; ".join(errores))
    if not C.EN_VIVO:
        return _nd("Sin archivo de campos (data/campos) y modo en vivo desactivado", "Campos de cultivo")
    x0, y0, x1, y1 = bbox
    try:   # respaldo: polígonos de tierras de cultivo dibujados en OpenStreetMap
        q = (f"[out:json][timeout:120];(way[\"landuse\"=\"farmland\"]({y0},{x0},{y1},{x1});"
             f"relation[\"landuse\"=\"farmland\"]({y0},{x0},{y1},{x1}););out geom;")
        r = _http().post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=TIMEOUT * 3)
        r.raise_for_status()
        feats = []
        for el in r.json().get("elements", []):
            anillos = []
            if el["type"] == "way" and el.get("geometry"):
                anillos = [el["geometry"]]
            elif el["type"] == "relation":
                anillos = [m["geometry"] for m in el.get("members", []) if m.get("role") == "outer" and m.get("geometry")]
            for k, g in enumerate(anillos):
                c = [[p["lon"], p["lat"]] for p in g]
                if len(c) >= 4 and c[0] == c[-1]:
                    feats.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [c]},
                                  "properties": {"csbid": f"osm{el['id']}_{k}"}})
        if feats:
            fuente = "OpenStreetMap (sin historial; el servicio del USDA no respondió)"
            d = destino.with_name(f"{nombre}_osm.geojson")
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_text(json.dumps({"type": "FeatureCollection", "fuente": fuente, "features": feats}))
            return dict(modo="en vivo", fuente=fuente, features=feats, motivo="; ".join(errores))
        errores.append("OpenStreetMap no devolvió campos")
    except Exception as e:  # noqa: BLE001
        errores.append(f"OpenStreetMap: {e}")
    return _nd("; ".join(errores), "Campos de cultivo")


# ------------------------------------------------------------------ historial de un punto (CropScape, Cropland Data Layer)
def albers_conus(lat, lon):
    """WGS84/NAD83 -> USA Contiguous Albers Equal Area (EPSG:5070), en metros. Fórmulas de Snyder (1987)."""
    a, e2 = 6378137.0, 0.00669438002290
    e = math.sqrt(e2)
    f1, f2, f0, l0 = map(math.radians, (29.5, 45.5, 23.0, -96.0))
    def q(f):
        s = math.sin(f)
        return (1 - e2) * (s / (1 - e2 * s * s) - (1 / (2 * e)) * math.log((1 - e * s) / (1 + e * s)))
    def m(f):
        return math.cos(f) / math.sqrt(1 - e2 * math.sin(f) ** 2)
    m1, m2, q0, q1, q2 = m(f1), m(f2), q(f0), q(f1), q(f2)
    n = (m1 ** 2 - m2 ** 2) / (q2 - q1)
    Cc = m1 ** 2 + n * q1
    rho0 = a * math.sqrt(Cc - n * q0) / n
    rho = a * math.sqrt(Cc - n * q(math.radians(lat))) / n
    th = n * (math.radians(lon) - l0)
    return rho * math.sin(th), rho0 - rho * math.cos(th)


def cdl_historial(lat, lon, anios=None):
    """Código de cultivo de cada año en un punto, desde el servicio CropScape (USDA NASS / George Mason Univ.)."""
    anios = anios or C.CSB_ANIOS
    cache = _cache(f"cdl_{lat:.5f}_{lon:.5f}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    if not C.EN_VIVO:
        return _nd("modo en vivo desactivado", "CropScape")
    x, y = albers_conus(lat, lon)
    s, out, errores = _http(), {}, []
    for a in anios:
        try:
            r = s.get("https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLValue", timeout=30,
                      params=dict(year=a, x=f"{x:.1f}", y=f"{y:.1f}"))
            r.raise_for_status()
            m = re.search(r"value:\s*\"?(\d+)", r.text)
            out[f"cdl{a}"] = int(m.group(1)) if m else None
        except Exception as e:  # noqa: BLE001
            errores.append(f"{a}: {e}")
            out[f"cdl{a}"] = None
    if not any(out.values()):
        return _nd("CropScape no respondió: " + "; ".join(errores[:2]), "CropScape")
    res = dict(modo="en vivo", fuente="USDA NASS Cropland Data Layer (CropScape)", codigos=out)
    cache.write_text(json.dumps(res))
    return res


def _anillo(geom):
    if geom["type"] == "Polygon":
        return geom["coordinates"][0]
    return max((p[0] for p in geom["coordinates"]), key=len)


# ------------------------------------------------------------------ SRTM (relieve)
def srtm(z: Zona):
    f = C.DATA_DIR / "srtm" / "srtm_1arcsec_zona.tif"
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        return _nd("Falta rasterio para leer el DEM SRTM", "NASA SRTM")
    try:
        if f.exists():
            src_path = str(f)
        elif os.getenv("OPENTOPO_KEY") and C.EN_VIVO:
            x0, y0, x1, y1 = z.bbox(3)
            cache = _cache(f"srtm_{z.clave()}.tif")
            if not cache.exists():
                r = _http().get("https://portal.opentopography.org/API/globaldem", timeout=TIMEOUT, params=dict(
                    demtype="SRTMGL1", south=y0, north=y1, west=x0, east=x1, outputFormat="GTiff", API_Key=os.getenv("OPENTOPO_KEY")))
                r.raise_for_status()
                cache.write_bytes(r.content)
            src_path = str(cache)
        else:
            return _nd("Sin DEM local; defina OPENTOPO_KEY para descargar SRTM", "NASA SRTM")
        x0, y0, x1, y1 = z.bbox(2)
        with rasterio.open(src_path) as s:
            zz = s.read(1, window=from_bounds(x0, y0, x1, y1, s.transform)).astype(float)
            res = s.res
        if zz.size < 9:
            return _nd("La zona queda fuera del DEM local", "NASA SRTM")
        dy, dx = res[1] * 111320, res[0] * 111320 * np.cos(np.radians(z.lat))
        gy, gx = np.gradient(zz, dy, dx)
        pend = np.hypot(gx, gy) * 100
        return dict(modo="local" if f.exists() else "en vivo", fuente="NASA SRTM 1 arcsec",
                    elevacion_m=round(float(np.nanmean(zz))), pendiente_media_pct=round(float(np.nanmean(pend)), 2),
                    pendiente_p90_pct=round(float(np.nanpercentile(pend, 90)), 2), frac_pend_mayor_3=round(float((pend > 3).mean()), 2))
    except Exception as e:  # noqa: BLE001
        return _nd(f"No se pudo leer SRTM: {e}", "NASA SRTM")
