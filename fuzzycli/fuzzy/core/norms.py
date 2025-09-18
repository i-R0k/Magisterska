from typing import Iterable
from .types import Float

# --- Helpers (fold) ---
def _fold(vals: Iterable[Float], init: Float, op):
    acc = init
    for v in vals:
        acc = op(acc, float(v))
    return acc

# --- T-normy ---
def t_min(vals: Iterable[Float]) -> Float:
    it = iter(vals)
    try:
        m = float(next(it))
    except StopIteration:
        return 1.0
    for v in it:
        if v < m: m = float(v)
    return m

def t_prod(vals: Iterable[Float]) -> Float:
    p = 1.0
    for v in vals:
        p *= float(v)
    return p

def t_lukasiewicz(vals: Iterable[Float]) -> Float:
    vals = list(float(v) for v in vals)
    if not vals: return 1.0
    s = sum(vals); k = len(vals)
    return max(0.0, s - (k - 1.0))

def _t_hamacher_pair(a: Float, b: Float) -> Float:
    a = float(a); b = float(b)
    denom = a + b - a * b
    if denom == 0.0:  # (0,0) -> 0
        return 0.0
    return (a * b) / denom

def t_hamacher(vals: Iterable[Float]) -> Float:
    vals = list(float(v) for v in vals)
    if not vals: return 1.0
    acc = vals[0]
    for v in vals[1:]:
        acc = _t_hamacher_pair(acc, v)
    return acc

# --- S-normy ---
def s_max(vals: Iterable[Float]) -> Float:
    it = iter(vals)
    try:
        m = float(next(it))
    except StopIteration:
        return 0.0
    for v in it:
        if v > m: m = float(v)
    return m

def s_prob(vals: Iterable[Float]) -> Float:
    acc = 0.0
    for v in vals:
        v = float(v)
        acc = acc + v - acc * v
    return acc

def s_lukasiewicz(vals: Iterable[Float]) -> Float:
    s = 0.0
    for v in vals:
        s += float(v)
        if s >= 1.0:
            return 1.0
    return s  # min(1, sum)

def s_sum(vals: Iterable[Float]) -> Float:
    it = iter(vals)
    try: s = float(next(it))
    except StopIteration: return 0.0
    for v in it:
        v = float(v)
        s = s + v - s*v  # algebraic sum
    return s

def s_bsum(vals: Iterable[Float]) -> Float:
    it = iter(vals)
    try: s = float(next(it))
    except StopIteration: return 0.0
    for v in it:
        s = min(1.0, s + float(v))  # bounded sum
    return s

def _s_hamacher_pair(a: Float, b: Float) -> Float:
    a = float(a); b = float(b)
    if a == 1.0 and b == 1.0:
        return 1.0
    denom = 1.0 - a * b
    if denom == 0.0:
        return 1.0
    return (a + b - 2.0 * a * b) / denom

def s_hamacher(vals: Iterable[Float]) -> Float:
    vals = list(float(v) for v in vals)
    if not vals: return 0.0
    acc = vals[0]
    for v in vals[1:]:
        acc = _s_hamacher_pair(acc, v)
    return acc

# --- Dubois–Prade (stub z parametrem; do ewentualnego rozwinięcia) ---
def t_dubois_prade(vals: Iterable[Float], lam: Float = 0.5) -> Float:
    raise NotImplementedError("Dubois–Prade wymaga parametru; dodamy w kolejnej iteracji.")

def s_dubois_prade(vals: Iterable[Float], lam: Float = 0.5) -> Float:
    raise NotImplementedError("Dubois–Prade wymaga parametru; dodamy w kolejnej iteracji.")

TNORMS = {
    "min": t_min,
    "prod": t_prod,
    "lukasiewicz": t_lukasiewicz,
    "hamacher": t_hamacher,
}
SNORMS = {
    "max": s_max,
    "prob": s_prob,
    "bsum": s_bsum,
    "prob": s_prob,
    "lukasiewicz": s_lukasiewicz,
    "hamacher": s_hamacher,
    # "dubois-prade": lambda vals: s_dubois_prade(vals)  # zostawione na pozniej
}
