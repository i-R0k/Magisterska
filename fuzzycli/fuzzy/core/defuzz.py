from typing import Callable, Tuple, List
from .types import Float

def _linspace(ymin: Float, ymax: Float, n: int) -> List[Float]:
    if n <= 1:
        return [(ymin + ymax) / 2.0]
    step = (ymax - ymin) / (n - 1)
    return [ymin + i * step for i in range(n)]

def centroid_on_grid(ymin: Float, ymax: Float, n: int, mu: Callable[[Float], Float]) -> Float:
    if n <= 1:
        return (ymin + ymax) / 2.0
    ys = _linspace(ymin, ymax, int(n))
    num = 0.0
    den = 0.0
    for y in ys:
        w = mu(y)
        num += y * w
        den += w
    return num / den if den > 0.0 else (ymin + ymax) / 2.0

def centroid_adaptive(ymin: Float, ymax: Float, mu: Callable[[Float], Float],
                      n_base: int = 201, refine_per_peak: int = 5, window_frac: Float = 0.1) -> Float:
    ys = _linspace(ymin, ymax, int(n_base))
    ws = [mu(y) for y in ys]
    if not any(w > 0 for w in ws):
        return (ymin + ymax) / 2.0

    # znajdź lokalne maksima
    peaks = []
    for i in range(1, len(ws) - 1):
        if ws[i] >= ws[i - 1] and ws[i] >= ws[i + 1] and ws[i] > 0:
            peaks.append(ys[i])

    # dogęszczenie
    yall = ys[:]
    rng = ymax - ymin
    win = max(1e-9, window_frac * rng)
    for p in peaks:
        a = max(ymin, p - win)
        b = min(ymax, p + win)
        extra = _linspace(a, b, max(3, refine_per_peak * 10))
        yall.extend(extra)

    num = 0.0
    den = 0.0
    for y in yall:
        w = mu(y)
        num += y * w
        den += w
    return num / den if den > 0.0 else (ymin + ymax) / 2.0

def mom_on_grid(ymin: Float, ymax: Float, n: int, mu: Callable[[Float], Float]) -> Float:
    ys = _linspace(ymin, ymax, int(n))
    ws = [mu(y) for y in ys]
    m = max(ws) if ws else 0.0
    if m <= 0.0:
        return (ymin + ymax) / 2.0
    # tolerancja numeryczna
    tol = max(1e-12, 1e-6 * m)
    tops = [y for y, w in zip(ys, ws) if abs(w - m) <= tol]
    return sum(tops) / len(tops) if tops else (ymin + ymax) / 2.0

def bisector_on_grid(ymin: Float, ymax: Float, n: int, mu: Callable[[Float], Float]) -> Float:
    if n <= 1:
        return (ymin + ymax) / 2.0
    ys = _linspace(ymin, ymax, int(n))
    ws = [mu(y) for y in ys]
    dy = (ymax - ymin) / (len(ys) - 1)
    total = sum(w * dy for w in ws)
    if total <= 0.0:
        return (ymin + ymax) / 2.0
    half = total / 2.0
    acc = 0.0
    for i, w in enumerate(ws):
        acc += w * dy
        if acc >= half:
            return ys[i]
    return ys[-1]
