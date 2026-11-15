"""Compacta las descargas anuales de USDA NASS QuickStats para Kansas (nass_ks_AAAA.csv) en tablas pequeñas
que usa la herramienta (data/nass_ks/*.csv.gz).

    python scripts/preparar_nass_ks.py --origen ruta/datos_nass_kansas

Tablas generadas:
  condado_anual.csv.gz      rendimiento, superficie sembrada y cosechada por condado y distrito, por práctica
  condicion_semanal.csv.gz  % muy mala/mala/regular/buena/excelente por cultivo y semana (estado)
  humedad_semanal.csv.gz    % muy corta/corta/adecuada/sobrante, capa superficial y subsuelo (estado)
  avance_semanal.csv.gz     % sembrado/cosechado/etc. por cultivo y semana (estado)
  precios_anuales.csv.gz    precio recibido por campaña comercial (estado)
"""
import argparse
import glob
import re
from pathlib import Path

import pandas as pd

CULTIVO = {"CORN": "maíz", "SORGHUM": "sorgo", "WHEAT": "trigo", "SOYBEANS": "soja", "SUNFLOWER": "girasol",
           "HAY": "alfalfa", "PASTURELAND": "pastura", "OATS": "avena", "SOIL": "suelo"}
COLS = ["commodity_desc", "class_desc", "prodn_practice_desc", "util_practice_desc", "statisticcat_desc", "unit_desc", "short_desc",
        "agg_level_desc", "county_name", "asd_desc", "year", "reference_period_desc", "week_ending", "domain_desc", "Value", "freq_desc"]


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def main(origen, destino):
    partes = []
    for f in sorted(glob.glob(str(Path(origen) / "nass_ks_*.csv"))):
        for ch in pd.read_csv(f, usecols=lambda c: c in COLS, chunksize=200000, dtype=str, low_memory=False):
            ch = ch[ch.commodity_desc.isin(CULTIVO) & (ch.domain_desc == "TOTAL")]
            ch = ch[ch.statisticcat_desc.isin(["CONDITION", "PROGRESS", "MOISTURE", "YIELD", "AREA PLANTED", "AREA HARVESTED", "PRICE RECEIVED"])]
            partes.append(ch)
        print("leído", Path(f).name, flush=True)
    d = pd.concat(partes, ignore_index=True)
    d["year"] = d.year.astype(int)
    d["v"] = num(d.Value)
    d = d.dropna(subset=["v"])
    d["cultivo"] = d.commodity_desc.map(CULTIVO)
    destino.mkdir(parents=True, exist_ok=True)

    # 1) condado y distrito: rendimiento de grano (no ensilaje, no "net planted") y superficies
    c = d[d.agg_level_desc.isin(["COUNTY", "AGRICULTURAL DISTRICT"])].copy()
    rend = c[(c.statisticcat_desc == "YIELD") & c.unit_desc.str.contains("/ ACRE") & ~c.unit_desc.str.contains("NET")]
    rend = rend[~((rend.commodity_desc.isin(["CORN", "SORGHUM"])) & rend.unit_desc.str.startswith("TONS"))]
    rend = rend[~((rend.commodity_desc == "HAY") & (rend.class_desc != "ALFALFA"))]
    rend = rend[~((rend.commodity_desc == "WHEAT") & ~rend.class_desc.isin(["ALL CLASSES", "WINTER"]))]
    rend = rend[~((rend.commodity_desc == "SUNFLOWER") & (rend.class_desc != "ALL CLASSES"))]
    area = c[c.statisticcat_desc.isin(["AREA PLANTED", "AREA HARVESTED"]) & (c.unit_desc == "ACRES")]
    area = area[~((area.commodity_desc.isin(["CORN", "SORGHUM"])) & area.util_practice_desc.str.contains("SILAGE", na=False))]
    area = area[~((area.commodity_desc == "HAY") & (area.class_desc != "ALFALFA"))]
    area = area[~((area.commodity_desc == "WHEAT") & ~area.class_desc.isin(["ALL CLASSES", "WINTER"]))]
    area = area[~((area.commodity_desc == "SUNFLOWER") & (area.class_desc != "ALL CLASSES"))]
    x = pd.concat([rend, area])
    x["nivel"] = x.agg_level_desc.map({"COUNTY": "condado", "AGRICULTURAL DISTRICT": "distrito"})
    x["lugar"] = x.county_name.where(x.nivel == "condado", x.asd_desc)
    x["practica"] = x.prodn_practice_desc.map({"ALL PRODUCTION PRACTICES": "todas", "IRRIGATED": "riego", "NON-IRRIGATED": "secano",
                                                "NON-IRRIGATED, CONTINUOUS CROP": "secano_continuo",
                                                "NON-IRRIGATED, FOLLOWING SUMMER FALLOW": "secano_tras_barbecho"})
    x["variable"] = x.statisticcat_desc.map({"YIELD": "rendimiento", "AREA PLANTED": "sembrado", "AREA HARVESTED": "cosechado"})
    x["unidad"] = x.unit_desc.str.lower()
    x = x.dropna(subset=["practica"]).sort_values("class_desc")   # ALL CLASSES antes que WINTER
    x = x.drop_duplicates(["nivel", "lugar", "asd_desc", "cultivo", "practica", "variable", "year"])
    x[["nivel", "lugar", "asd_desc", "cultivo", "practica", "variable", "year", "v", "unidad"]].rename(columns={"asd_desc": "distrito", "year": "anio", "v": "valor"}) \
        .to_csv(destino / "condado_anual.csv.gz", index=False)

    e = d[d.agg_level_desc == "STATE"].copy()
    e["semana"] = e.reference_period_desc.str.extract(r"WEEK #(\d+)")[0].astype(float)
    # 2) condición semanal (excluye promedios de 5 años y año previo)
    k = e[(e.statisticcat_desc == "CONDITION") & e.unit_desc.str.startswith("PCT")].copy()
    k = k[~((k.commodity_desc == "WHEAT") & (k.class_desc != "WINTER"))]
    k = k[~((k.commodity_desc == "HAY") & (k.class_desc != "ALFALFA"))]
    k["cat"] = k.unit_desc.str.replace("PCT ", "").str.lower().str.replace(" ", "_")
    k = k.pivot_table(index=["cultivo", "year", "week_ending", "semana"], columns="cat", values="v", aggfunc="first").reset_index()
    k.rename(columns={"year": "anio"}).to_csv(destino / "condicion_semanal.csv.gz", index=False)
    # 3) humedad del suelo
    h = e[e.statisticcat_desc == "MOISTURE"].copy()
    h["capa"] = h.class_desc.map({"TOPSOIL": "superficial", "SUBSOIL": "subsuelo"})
    h["cat"] = h.unit_desc.str.replace("PCT ", "").str.lower().str.replace(" ", "_")
    h = h.pivot_table(index=["capa", "year", "week_ending", "semana"], columns="cat", values="v", aggfunc="first").reset_index()
    h.rename(columns={"year": "anio"}).to_csv(destino / "humedad_semanal.csv.gz", index=False)
    # 4) avance de etapas
    a = e[(e.statisticcat_desc == "PROGRESS") & e.unit_desc.str.startswith("PCT")].copy()
    a = a[~((a.commodity_desc == "WHEAT") & (a.class_desc != "WINTER"))]
    a = a[a.commodity_desc != "HAY"]
    a["etapa"] = a.unit_desc.str.replace("PCT ", "").str.lower()
    a = a[["cultivo", "etapa", "year", "week_ending", "semana", "v"]].drop_duplicates(["cultivo", "etapa", "year", "week_ending"])
    a.rename(columns={"year": "anio", "v": "pct"}).to_csv(destino / "avance_semanal.csv.gz", index=False)
    # 5) precios por campaña comercial
    p = e[(e.statisticcat_desc == "PRICE RECEIVED") & (e.reference_period_desc == "MARKETING YEAR")].copy()
    elegir = {"CORN, GRAIN - PRICE RECEIVED, MEASURED IN $ / BU": ("maíz", 1, "bu"),
              "SORGHUM, GRAIN - PRICE RECEIVED, MEASURED IN $ / CWT": ("sorgo", .56, "bu"),       # 1 bu de sorgo = 56 lb
              "WHEAT, WINTER - PRICE RECEIVED, MEASURED IN $ / BU": ("trigo", 1, "bu"),
              "SOYBEANS - PRICE RECEIVED, MEASURED IN $ / BU": ("soja", 1, "bu"),
              "SUNFLOWER - PRICE RECEIVED, MEASURED IN $ / CWT": ("girasol", .01, "lb"),
              "HAY, ALFALFA - PRICE RECEIVED, MEASURED IN $ / TON": ("alfalfa", 1, "ton")}
    p = p[p.short_desc.isin(elegir)]
    p["cultivo"] = p.short_desc.map(lambda s: elegir[s][0])
    p["precio"] = p.v * p.short_desc.map(lambda s: elegir[s][1])
    p["unidad"] = p.short_desc.map(lambda s: elegir[s][2])
    p[["cultivo", "year", "precio", "unidad"]].drop_duplicates(["cultivo", "year"]).rename(columns={"year": "anio"}) \
        .to_csv(destino / "precios_anuales.csv.gz", index=False)
    for f in sorted(destino.glob("*.gz")):
        print(f.name, f.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, help="carpeta con nass_ks_AAAA.csv")
    ap.add_argument("--destino", default=str(Path(__file__).resolve().parent.parent / "data" / "nass_ks"))
    a = ap.parse_args()
    main(a.origen, Path(a.destino))
