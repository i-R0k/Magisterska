# Defuzyfikacja: centroid, MOM, bisector (siatka)

from typing import Callable, Iterable, Tuple
from .types import Float

def centroid_on_grid(ymin: Float, ymax: Float, n: int, mu: Callable[[Float], Float]) -> Float:
    if n <= 1: return (ymin + ymax) / 2.0
    step = (ymax - ymin) / (n - 1)
    num = 0.0
    den = 0.0
    y = ymin
    for _ in range(n):
        w = mu(y)
        num += y * w
        den += w
        y += step
    return num / den if den > 0.0 else (ymin + ymax) / 2.0
