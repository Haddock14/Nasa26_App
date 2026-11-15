"""Traductor español -> inglés del texto visible (misma lógica que web/index.html, para probarlo en Python)."""
import json, re
from functools import lru_cache
from pathlib import Path

CAT = Path(__file__).resolve().parent.parent / "web" / "i18n_en.json"
MES = dict(zip("ene feb mar abr may jun jul ago sep oct nov dic".split(), "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()))
RE_FECHA = re.compile(r"(^|[^\w])(\d{1,2}) (ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)(?!\w)")
NUM = r"([-+−]?\d[\d.,]*)"
ARROBA = r"((?:(?![.;!?]\s+[A-ZÁÉÍÓÚÑ¿\"(]).)+?)"   # texto que no cruza un fin de oración


@lru_cache(1)
def catalogo():
    c = json.loads(CAT.read_text(encoding="utf-8"))
    pl = []
    for es, en in c["plantillas"]:
        src, tipos = "", []
        for ch in es:
            if ch == "#": src += NUM; tipos.append("#")
            elif ch == "@": src += ARROBA; tipos.append("@")
            else: src += re.escape(ch)
        pl.append((re.compile("^" + src + "$", re.S), en, tipos, len(es.replace("#", "").replace("@", ""))))
    return c, pl


def num_en(x):
    x = re.sub(r"(\d),(\d)", "\\1\x01\\2", x)
    x = re.sub(r"(\d{1,3})((?:\.\d{3})+)(?!\d)", lambda m: m.group(1) + m.group(2).replace(".", ","), x)
    return x.replace("\x01", ".")


def fechas_en(x):
    return RE_FECHA.sub(lambda m: f"{m.group(1)}{MES[m.group(3)]} {m.group(2)}", x)


GENERICA = 4   # plantillas con 4 caracteres fijos o menos ("@ y @", "@: @"...) se prueban todas y gana la de menos español
RE_ES = re.compile(r"[áéíóúñ¿¡]|\b(de|del|la|las|los|el|que|con|para|por|en|un|una|y|se|su|sus|es|son|más|años|sin|cada|muy)\b", re.I)


def espanol(x):
    return len(RE_ES.findall(x))


def _aplicar(m, en, tipos, prof):
    def g(k):
        i = int(k.group(1)); v = m.group(i) or ""
        return trad_texto(v.strip(), prof + 1) if tipos[i - 1] == "@" else num_en(v)
    return re.sub(r"\{(\d+)\}", g, en)


def trad_texto(t, prof=0):
    c, pl = catalogo()
    ex = c["exactas"]
    if t in ex: return ex[t]
    if t[-1:] in ".:;" and t[:-1] in ex: return ex[t[:-1]] + t[-1]
    if t + "." in ex and ex[t + "."].endswith("."): return ex[t + "."][:-1]
    m = re.fullmatch(r"(\d{1,2}) (ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic)", t)
    if m: return f"{MES[m.group(2)]} {m.group(1)}"
    if prof < 6:
        for re_, en, tipos, lit in pl:
            if lit <= GENERICA: continue
            m = re_.match(t)
            if m: return _aplicar(m, en, tipos, prof)
        partes = re.split(r"(?<=[.;!?])\s+(?=[A-ZÁÉÍÓÚÑ¿\"(])", t)
        if len(partes) > 1: return " ".join(trad_texto(x, prof + 1) for x in partes)
        if t[-1:] in ".:;" and len(t) > 1: return trad_texto(t[:-1], prof + 1) + t[-1]
        mejor = None
        for re_, en, tipos, lit in pl:
            if lit > GENERICA: continue
            m = re_.match(t)
            if m:
                r = _aplicar(m, en, tipos, prof)
                if mejor is None or espanol(r) < espanol(mejor): mejor = r
                if espanol(r) == 0: break
        if mejor is not None and mejor != t and espanol(mejor) <= espanol(t): return mejor
    r = t
    for es, en in c["frases"]:
        if es in r: r = r.replace(es, en)
    return fechas_en(num_en(r))


def traducir(s):
    if not s or not re.search(r"[^\W_]", s): return s
    pre = re.match(r"^\s*", s).group(0); post = re.search(r"\s*$", s).group(0)
    t = re.sub(r"\s+", " ", s.strip())
    if not re.search(r"[^\W\d_]", t): return pre + num_en(t) + post
    return pre + trad_texto(t) + post
