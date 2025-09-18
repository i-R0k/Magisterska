from __future__ import annotations
from typing import Dict, Callable, Iterable, Tuple, Any
from ..core import norms
from ..core.defuzz import centroid_on_grid, centroid_adaptive, mom_on_grid, bisector_on_grid
from ..core.types import Float
from .knowledge import KnowledgeBase

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

class MamdaniEngine:
    """
    Mamdani: FIT (agregacja reguł S-normą), FATI (najpierw S-norma po etykietach, potem implikacja).
    Defuzyfikacja: centroid | mom | bisector | centroid_adaptive.
    """
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.tnorm_fn: Callable = norms.TNORMS.get(self.kb.tnorm, norms.TNORMS["min"])
        self.snorm_fn: Callable = norms.SNORMS.get(self.kb.snorm, norms.SNORMS["max"])
        self._rules_by_output: Dict[str, list[Tuple[int, Any]]] = {}
        for i, rule in enumerate(getattr(self.kb, "rules", [])):
            if not getattr(rule, 'active', True):
                continue
            oname = rule.consequent[0]
            self._rules_by_output.setdefault(oname, []).append((i, rule))

    # ---------- helpers ----------

    def _tnorm(self, values: Iterable[float]) -> float:
        vals = [float(v) for v in values]
        if not vals:  # brak antecedentów => prawda neutralna
            return 1.0
        try:
            return float(self.tnorm_fn(vals))
        except Exception:
            return float(min(vals))

    def _snorm_pair(self, a: float, b: float) -> float:
        # inkrementalne łączenie S-normą (norms.SNORMS oczekują listy)
        try:
            return float(self.snorm_fn([float(a), float(b)]))
        except Exception:
            return max(float(a), float(b))

    def _safe_mu(self, mf, x: float) -> float:
        """Bezpieczne μ z fallbackiem dla tuple('tri',(a,b,c))."""
        try:
            return _clip01(float(mf.mu(x)))
        except Exception:
            pass
        if isinstance(mf, tuple) and len(mf) == 2 and mf[0] == "tri":
            a, b, c = mf[1]
            if x <= a or x >= c:
                return 0.0
            if x == b:
                return 1.0
            if x < b:
                return (x - a) / (b - a if b != a else 1e-12)
            return (c - x) / (c - b if c != b else 1e-12)
        return 0.0

    def _auto_grid(self, ovar) -> tuple[Float, Float, int]:
        # bezpieczne pobranie gridu (z domyślnymi wartościami)
        ymin, ymax, n = getattr(ovar, "grid", (ovar.vmin, ovar.vmax, 201))
        if ymin >= ymax or (ymin, ymax) == (0.0, 1.0):
            if getattr(ovar, "terms", None):
                # support może być przybliżony (np. Gauss) – zakładamy, że MF je implementują
                supports = [mf.support() for mf in ovar.terms.values()]
                ymin = min(a for a, _ in supports)
                ymax = max(b for _, b in supports)
            else:
                ymin, ymax = ovar.vmin, ovar.vmax
        if n is None or int(n) < 3:
            n = 201
        return float(ymin), float(ymax), int(n)

    # ---------- API ----------

    def predict(self, inputs: Dict[str, Float]) -> Dict[str, Float]:
        """
        FIT: α = T-norm(μ_i)*w; implikacja: min(α, μ_consequent(y)); agregacja: S-norma.
        FATI: grupujemy α per etykieta konsekwentu S-normą, potem implikacja i agregacja S-normą.
        Zwraca ostrą wartość dla każdego wyjścia (po defuzyfikacji), obciętą do [vmin, vmax].
        """
        out_values: Dict[str, Float] = {}
        mode = (getattr(self.kb, "mode", "FIT") or "FIT").upper()
        method = (getattr(self.kb, "defuzz", "centroid") or "centroid").lower()

        for oname, ovar in self.kb.outputs.items():
            # --- 1) policz α dla reguł z tym konsekwentem ---
            rule_alphas: list[tuple[str, float]] = []  # (label, alpha)
            for _idx, rule in self._rules_by_output.get(oname, []):
                if not getattr(rule, 'active', True):
                    continue
                acts: list[float] = []
                ok = True
                for vname, label in rule.antecedent:
                    v = inputs.get(vname, None)
                    if v is None:
                        ok = False; break  # brak wartości wejścia -> reguła nieaktywna
                    vin = self.kb.inputs.get(vname)
                    if vin is None or label not in vin.terms:
                        ok = False; break  # błąd modelu/etykiety -> pomijamy
                    mu = self._safe_mu(vin.terms[label], float(v))
                    acts.append(mu)
                if not ok:
                    continue
                alpha = _clip01(self._tnorm(acts) * float(getattr(rule, "weight", 1.0)))
                if alpha > 0.0:
                    rule_alphas.append((rule.consequent[1], alpha))

            # --- 2) zbuduj zagregowaną μ(y) wyjścia ---
            if mode == "FATI":
                # najpierw S-norma alf per etykieta
                per_label: Dict[str, float] = {}
                for lab, a in rule_alphas:
                    per_label[lab] = self._snorm_pair(per_label.get(lab, 0.0), a)

                def agg_mu(y: Float) -> Float:
                    acc = 0.0
                    for lab, a in per_label.items():
                        cmf = ovar.terms[lab]
                        mu_val = min(a, self._safe_mu(cmf, float(y)))  # mamdani-implication
                        acc = self._snorm_pair(acc, mu_val)           # agregacja S-normą
                    return acc
            else:
                # FIT: każda reguła osobno -> klip -> agregacja S-normą
                def agg_mu(y: Float) -> Float:
                    acc = 0.0
                    for lab, a in rule_alphas:
                        cmf = ovar.terms[lab]
                        mu_val = min(a, self._safe_mu(cmf, float(y)))
                        acc = self._snorm_pair(acc, mu_val)
                    return acc

            ymin, ymax, n = self._auto_grid(ovar)

            # --- 3) defuzyfikacja ---
            if method == "centroid":
                ystar = centroid_on_grid(ymin, ymax, n, agg_mu)
            elif method == "mom":
                ystar = mom_on_grid(ymin, ymax, n, agg_mu)
            elif method == "bisector":
                ystar = bisector_on_grid(ymin, ymax, n, agg_mu)
            elif method == "centroid_adaptive":
                ystar = centroid_adaptive(ymin, ymax, agg_mu, n_base=max(101, n))
            else:
                # fallback
                ystar = centroid_on_grid(ymin, ymax, n, agg_mu)

            # clamp do zakresu wyjścia
            ystar = float(ystar)
            if ystar < ovar.vmin:
                ystar = ovar.vmin
            elif ystar > ovar.vmax:
                ystar = ovar.vmax

            out_values[oname] = ystar

        return out_values
