"""Catálogo de cultivos, rotaciones por defecto y supuestos.

TODO lo que está aquí es un SUPUESTO ORIENTATIVO para el suroeste de Kansas (literatura FAO-56/FAO-33,
guías de K-State). Está separado del código para que un agrónomo local lo pueda revisar y editar.
Los valores que dependen del sitio (agua, calor, heladas, erosión, rendimiento) NO están aquí:
el motor los calcula con el clima y el suelo del punto elegido.
"""
import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", DATA_DIR / "_cache"))
EARTHDATA_TOKEN = os.getenv("EARTHDATA_TOKEN")      # https://urs.earthdata.nasa.gov -> Generate Token
EN_VIVO = os.getenv("EN_VIVO", "1") == "1"          # consultar APIs si no hay datos locales

# Año agrícola: octubre -> septiembre. Cada elemento de una rotación ocupa un año agrícola ("ranura").
MESES_ANIO_AGRICOLA = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Campos por cultivo
#   meses: meses calendario con cultivo vivo dentro de su ranura
#   kc: coeficiente de cultivo por mes activo (FAO-56, simplificado a mensual)
#   ky: sensibilidad del rendimiento al déficit hídrico (FAO-33)
#   calor: (meses sensibles, umbral Tmax °C, pérdida por día sobre el umbral)
#   helada: (meses sensibles, umbral Tmin °C, pérdida por evento)
#   residuo_t_ha: biomasa de residuo con rendimiento potencial (cobertura y aporte de C)
#   n_fija_kg_ha: N fijado aprovechable por el cultivo siguiente
#   n_req_kg_ha: necesidad de N con rendimiento potencial
#   raiz_cm: profundidad radicular efectiva (los cultivos profundos usan agua de capas hondas)
#   familia / plagas: para cortar ciclos; intervalo_min = años mínimos sin repetir el mismo hospedero
#   forraje: valor forrajero 0-1 (pastoreo, ensilaje, heno)
#   nass: (producto, unidad) para rendimiento de referencia del condado
#   precio: USD por unidad; costo_*: costos DIRECTOS por acre (semilla, fertilizante, químicos, cosecha contratada),
#   sin arriendo ni amortización. Orientativos según guías K-State; reemplazar por los del productor.
CULTIVOS = {
    "trigo": dict(nombre="Trigo de invierno", meses=[10, 11, 12, 1, 2, 3, 4, 5, 6],
                  kc=[.3, .4, .3, .3, .4, .7, 1.0, 1.05, .35], ky=0.85,
                  calor=([5, 6], 32, .010), helada=([4], -2, .03),
                  residuo_t_ha=4.0, n_fija_kg_ha=0, n_req_kg_ha=90, raiz_cm=150,
                  familia="gramínea de invierno", plagas=["virus del mosaico (WSMV)", "royas"], intervalo_min=1,
                  forraje=.4, nass="WHEAT", unidad="bu", precio=5.5, costo_secano=150, costo_riego=300),
    "sorgo": dict(nombre="Sorgo granífero", meses=[6, 7, 8, 9],
                  kc=[.5, 1.0, 1.05, .7], ky=1.0,
                  calor=([8], 38, .005), helada=([], 0, 0),
                  residuo_t_ha=5.0, n_fija_kg_ha=0, n_req_kg_ha=90, raiz_cm=180,
                  familia="gramínea de verano", plagas=["pulgón del sorgo"], intervalo_min=1,
                  forraje=.7, nass="SORGHUM", unidad="bu", precio=4.0, costo_secano=170, costo_riego=350),
    "maíz": dict(nombre="Maíz", meses=[5, 6, 7, 8, 9],
                 kc=[.4, .8, 1.2, 1.1, .6], ky=1.25,
                 calor=([7], 35, .012), helada=([5], 0, .05),
                 residuo_t_ha=6.5, n_fija_kg_ha=0, n_req_kg_ha=200, raiz_cm=150,
                 familia="gramínea de verano", plagas=["gusano de la raíz del maíz"], intervalo_min=1,
                 forraje=.7, nass="CORN", unidad="bu", precio=4.2, costo_secano=230, costo_riego=480),
    "soja": dict(nombre="Soja", meses=[5, 6, 7, 8, 9],
                 kc=[.4, .7, 1.1, 1.1, .6], ky=1.05,
                 calor=([8], 35, .008), helada=([5], 0, .05),
                 residuo_t_ha=2.5, n_fija_kg_ha=35, n_req_kg_ha=0, raiz_cm=120,
                 familia="leguminosa", plagas=["nematodo del quiste (SCN)"], intervalo_min=2,
                 forraje=.2, nass="SOYBEANS", unidad="bu", precio=10.2, costo_secano=170, costo_riego=300),
    "girasol": dict(nombre="Girasol", meses=[6, 7, 8, 9],
                    kc=[.4, .9, 1.05, .6], ky=0.95,
                    calor=([7, 8], 37, .005), helada=([], 0, 0),
                    residuo_t_ha=3.0, n_fija_kg_ha=0, n_req_kg_ha=70, raiz_cm=200,
                    familia="compuesta", plagas=["esclerotinia", "gorgojo del tallo"], intervalo_min=3,
                    forraje=.1, nass="SUNFLOWER", unidad="lb", precio=.22, costo_secano=160, costo_riego=300),
    "triticale": dict(nombre="Triticale forrajero", meses=[10, 11, 12, 1, 2, 3, 4, 5],
                      kc=[.4, .4, .4, .4, .5, .8, 1.05, 1.0], ky=0.9,
                      calor=([], 40, 0), helada=([], -10, 0),
                      residuo_t_ha=2.0, n_fija_kg_ha=0, n_req_kg_ha=60, raiz_cm=120,
                      familia="gramínea de invierno", plagas=["royas"], intervalo_min=1,
                      forraje=1.0, nass=None, unidad="ton", precio=70, rend_ref=3.0, costo_secano=110, costo_riego=220),
    "cobertura": dict(nombre="Cultivo de cobertura (mezcla avena-vicia en primavera)", meses=[3, 4, 5],
                      kc=[.5, .8, .9], ky=0.0,
                      calor=([], 40, 0), helada=([], -10, 0),
                      residuo_t_ha=2.5, n_fija_kg_ha=25, n_req_kg_ha=0, raiz_cm=90,
                      familia="cobertura", plagas=[], intervalo_min=0,
                      forraje=.5, nass=None, unidad=None, precio=0, costo_secano=45, costo_riego=45),
    "alfalfa": dict(nombre="Alfalfa (año de producción)", meses=[10, 11, 3, 4, 5, 6, 7, 8, 9],
                    kc=[.6, .5, .6, .9, 1.0, 1.05, 1.05, 1.0, .9], ky=1.1,
                    calor=([7, 8], 38, .004), helada=([], -15, 0),
                    residuo_t_ha=1.5, n_fija_kg_ha=80, n_req_kg_ha=0, raiz_cm=250,
                    familia="leguminosa perenne", plagas=["gorgojo de la alfalfa"], intervalo_min=0,
                    forraje=1.0, nass="HAY", nass_clase="ALFALFA", unidad="ton", precio=180, costo_secano=180, costo_riego=420),
    "barbecho": dict(nombre="Barbecho (sin cultivo)", meses=[], kc=[], ky=0.0,
                     calor=([], 40, 0), helada=([], -10, 0),
                     residuo_t_ha=0, n_fija_kg_ha=0, n_req_kg_ha=0, raiz_cm=0,
                     familia="suelo desnudo", plagas=[], intervalo_min=0,
                     forraje=0, nass=None, unidad=None, precio=0, costo_secano=40, costo_riego=40),
}

# Rendimiento alcanzable con buen agua si no hay datos NASS del condado (SUPUESTO regional, Kansas occidental)
POTENCIAL_DEFECTO = {"trigo": 60, "sorgo": 120, "maíz": 210, "soja": 60, "girasol": 2000, "alfalfa": 6.5}

# Efectos del cultivo anterior sobre el siguiente (multiplicador de rendimiento). Supuestos orientativos:
# - trigo después de barbecho: calibrado con NASS Finney (trigo secano tras barbecho / trigo continuo ≈ 1.08-1.14)
# - trigo después de sorgo: siembra tardía y poca agua residual (K-State reporta pérdidas de 5-15 %)
# - cereal después de leguminosa: crédito de N y corte de enfermedades
EFECTO_PREVIO = {
    ("trigo", "trigo"): .88, ("maíz", "maíz"): .90, ("soja", "soja"): .88, ("sorgo", "sorgo"): .93,
    ("girasol", "girasol"): .80, ("sorgo", "trigo"): .92,
    ("soja", "maíz"): 1.08, ("soja", "trigo"): 1.04, ("soja", "sorgo"): 1.05,
    ("alfalfa", "maíz"): 1.10, ("alfalfa", "trigo"): 1.06, ("cobertura", "sorgo"): 1.0,
}

# Eficiencia de barbecho: fracción de la lluvia de los meses sin cultivo que queda almacenada en el perfil.
EFIC_BARBECHO = {"cero": .25, "reducida": .20, "convencional": .15}   # Peterson et al.; K-State Tribune
# Cobertura por residuo que queda en meses sin cultivo (decae con el tiempo y con la labranza)
DECAIMIENTO_RESIDUO_MES = {"cero": .03, "reducida": .05, "convencional": .12}

ROTACIONES = {
    "B (cobertura en vez de barbecho)": ["trigo", "sorgo", "cobertura"],
    "C (con leguminosa)": ["trigo", "soja", "sorgo"],
    "D (maíz-soja)": ["maíz", "soja"],
    "E (con girasol de raíz profunda)": ["trigo", "sorgo", "girasol", "barbecho"],
    "F (forrajera para ganado)": ["triticale", "sorgo", "barbecho"],
}
ACTUAL = ["trigo", "sorgo", "barbecho"]

CRITERIOS = {
    "agua": "uso eficiente del agua",
    "calor": "tolerancia a calor y heladas",
    "n": "aporte de nitrógeno",
    "eros": "control de erosión (agua y viento)",
    "mo": "aporte de materia orgánica",
    "ingreso": "ingreso / mercado",
    "plagas": "corte de ciclos de plagas y malezas",
    "forraje": "valor forrajero",
}
PRIORIDADES = {"agua": .8, "calor": .5, "n": .5, "eros": .6, "mo": .6, "ingreso": .6, "plagas": .4, "forraje": .2}

# Restricciones por defecto del agricultor
RESTRICCIONES = dict(
    riego_mm=0,                 # agua de riego disponible por año (0 = secano)
    labranza="reducida",        # cero | reducida | convencional
    sin_maquinaria=[],          # cultivos que no puede sembrar/cosechar (p. ej. ["girasol"])
    sin_mercado=[],             # cultivos sin comprador cercano
    precios={},                 # {"soja": 11.0} reemplaza precio por defecto
    ingreso_minimo_usd_acre=None,
)
ESCENARIO = dict(delta_lluvia_pct=0.0, delta_temp_c=0.0)

# ------------------------------------------------------------------ campos de cultivo (USDA NASS Crop Sequence Boundaries)
# Servicio público con el polígono de cada campo y el cultivo de cada año (Cropland Data Layer) 2017-2024.
CSB_URL = os.getenv("CSB_URL", "https://pdi.scinet.usda.gov/hosting/rest/services/Hosted/Crop_Sequence_Boundaries_2024/FeatureServer/2/query")
CSB_ANIOS = list(range(2017, 2025))
# código CDL -> (nombre en español, cultivo del catálogo o None, color)
CDL = {
    1: ("Maíz", "maíz", "#f2c200"), 4: ("Sorgo", "sorgo", "#ff9e0a"), 5: ("Soja", "soja", "#3c8c2a"),
    6: ("Girasol", "girasol", "#ffe94d"), 21: ("Cebada", "triticale", "#e2007c"), 22: ("Trigo duro", "trigo", "#8a6a3d"),
    23: ("Trigo de primavera", "trigo", "#d8b56b"), 24: ("Trigo de invierno", "trigo", "#a87000"),
    26: ("Trigo + soja (doble cultivo)", "trigo", "#707000"), 27: ("Centeno", "triticale", "#ae017e"),
    28: ("Avena", "triticale", "#a05989"), 205: ("Triticale", "triticale", "#d69ebc"),
    225: ("Trigo + maíz (doble cultivo)", "trigo", "#ffd300"), 236: ("Trigo + sorgo (doble cultivo)", "trigo", "#d6a33a"),
    36: ("Alfalfa", "alfalfa", "#ffa5e2"), 37: ("Otros henos", None, "#a5f58c"), 59: ("Pastos para semilla", None, "#b4d78c"),
    61: ("Barbecho / sin cultivo", "barbecho", "#bfbf7a"), 176: ("Pastizal / pastura", None, "#e8ffbf"),
    31: ("Canola", None, "#d1ff00"), 42: ("Frijol", None, "#a80000"), 44: ("Otros cultivos", None, "#00af4d"),
    111: ("Agua", None, "#4970a3"), 121: ("Urbano", None, "#9c9c9c"), 122: ("Urbano", None, "#9c9c9c"),
    123: ("Urbano", None, "#9c9c9c"), 124: ("Urbano", None, "#9c9c9c"), 152: ("Matorral", None, "#c6d69e"),
    141: ("Bosque", None, "#93cc93"), 190: ("Humedal", None, "#7cafaf"), 195: ("Humedal", None, "#7cafaf"),
}
