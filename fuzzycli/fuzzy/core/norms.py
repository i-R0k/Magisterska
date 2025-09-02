# T-normy, S-normy, negacja

from typing import Iterable
from .types import Float

def t_min(vals: Iterable[Float]) -> Float:
    it = iter(vals)
    try:
        m = next(it)
    except StopIteration:
        return 1.0
    for v in it:
        if v < m: m = v
    return m

def t_prod(vals: Iterable[Float]) -> Float:
    p = 1.0
    for v in vals:
        p *= v
    return p

def t_lukasiewicz(vals: Iterable[Float]) -> Float:
    s = 0.0
    for v in vals:
        s += v
    return max(0.0, s - (len(list(vals)) or 1) + 1)  # note: vals consumed; use with pairs

def s_max(vals: Iterable[Float]) -> Float:
    it = iter(vals)
    try:
        m = next(it)
    except StopIteration:
        return 0.0
    for v in it:
        if v > m: m = v
    return m

def s_prob(vals: Iterable[Float]) -> Float:
    # probabilistic sum: a + b - ab (for multiple, fold)
    acc = 0.0
    for v in vals:
        acc = acc + v - acc * v
    return acc

TNORMS = {"min": t_min, "prod": t_prod}
SNORMS = {"max": s_max, "prob": s_prob}
