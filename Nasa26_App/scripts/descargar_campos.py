"""Descarga los campos de cultivo (USDA Crop Sequence Boundaries 2017-2024) de una zona a data/campos/
para usar la app sin conexión.   python scripts/descargar_campos.py --bbox -101.11,37.73,-100.22,38.27
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rotacion import config as C, fuentes as F

ap = argparse.ArgumentParser()
ap.add_argument("--bbox", default="-101.11,37.73,-100.22,38.27", help="lon_min,lat_min,lon_max,lat_max (por defecto Finney County)")
ap.add_argument("--salida", default=str(C.DATA_DIR / "campos" / "campos.geojson"))
a = ap.parse_args()
x0, y0, x1, y1 = map(float, a.bbox.split(","))
paso, feats, vistos = .1, [], set()
y = y0
while y < y1:
    x = x0
    while x < x1:
        r = F.campos((x, y, min(x + paso, x1), min(y + paso, y1)), max_campos=20000)
        if r.get("modo") == "no disponible":
            sys.exit(r["motivo"])
        for f in r["features"]:
            k = f["properties"].get("csbid")
            if k not in vistos:
                vistos.add(k); feats.append(f)
        print(f"{x:.2f},{y:.2f}: {len(feats)} campos", flush=True)
        x += paso
    y += paso
Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
Path(a.salida).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
print("guardado", a.salida, len(feats))
