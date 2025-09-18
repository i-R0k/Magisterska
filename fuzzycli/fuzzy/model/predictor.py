# fuzzycli/fuzzy/model/predictor.py
from typing import Dict, Callable
from ..core import defuzz
from .classifier import Classifier
try:
    from .knowledge import KnowledgeBase
except Exception as e:
    raise ImportError("Dopasuj import KnowledgeBase w predictor.py") from e


class Predictor:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.classifier = Classifier(kb)

    def _auto_grid(self, ovar) -> tuple[float, float, int]:
        ymin, ymax, n = getattr(ovar, "grid", (getattr(ovar, "vmin", 0.0), getattr(ovar, "vmax", 1.0), 201))
        if ymin >= ymax or (ymin, ymax) == (0.0, 1.0):
            if getattr(ovar, "terms", {}):
                supports = []
                for mf in getattr(ovar, "terms", {}).values():
                    try:
                        supports.append(tuple(mf.support()))
                    except Exception:
                        # skip if MF doesn't have support method
                        pass
                if supports:
                    ymin = min(a for a, _ in supports)
                    ymax = max(b for _, b in supports)
                else:
                    ymin, ymax = getattr(ovar, "vmin", 0.0), getattr(ovar, "vmax", 1.0)
            else:
                ymin, ymax = getattr(ovar, "vmin", 0.0), getattr(ovar, "vmax", 1.0)
        if n is None or int(n) < 3:
            n = 201
        return float(ymin), float(ymax), int(n)

    def predict(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """Pełny pipeline: inference -> implication (clip) -> aggregation -> defuzz."""
        outs: Dict[str, float] = {}
        # compute cache of mus once (Classifier has helper)
        cache = self.classifier._compute_mus_cache(inputs)

        for oname, ovar in getattr(self.kb, "outputs", {}).items():
            # collect rule alphas for this output as list of (alpha,label)
            rule_alphas = []
            for rule in getattr(self.kb, "rules", []):
                if rule.consequent[0] != oname:
                    continue
                mus = []
                ok = True
                for vname, lbl in rule.antecedent:
                    key = (vname, lbl)
                    if key not in cache:
                        ok = False
                        break
                    mus.append(cache[key])
                if not ok:
                    continue
                # alpha = tnorm(mus) * weight
                try:
                    alpha_base = float(self.classifier.tnorm_fn(mus)) if mus else 1.0
                except Exception:
                    alpha_base = float(min(mus)) if mus else 1.0
                alpha = alpha_base * float(getattr(rule, "weight", 1.0))
                rule_alphas.append((alpha, rule.consequent[1]))

            # build aggregated mu(y): FIT (clip per rule then OR)
            def agg_mu(y: float) -> float:
                best = 0.0
                for alpha, lab in rule_alphas:
                    cmf = getattr(ovar, "terms", {}).get(lab)
                    if cmf is None:
                        continue
                    try:
                        mu_val = min(alpha, cmf.mu(y))
                    except Exception:
                        # if stored as ("tri", (a,b,c)):
                        if isinstance(cmf, tuple) and cmf[0] == "tri":
                            a, b, c = cmf[1]
                            if y <= a or y >= c:
                                mval = 0.0
                            elif y == b:
                                mval = 1.0
                            elif y < b:
                                mval = (y - a) / (b - a if b != a else 1e-12)
                            else:
                                mval = (c - y) / (c - b if c != b else 1e-12)
                            mu_val = min(alpha, mval)
                        else:
                            mu_val = 0.0
                    if mu_val > best:
                        best = mu_val
                return best

            ymin, ymax, n = self._auto_grid(ovar)
            method = (getattr(self.kb, "defuzz", "centroid") or "centroid").lower()
            if method == "centroid":
                ystar = defuzz.centroid_on_grid(ymin, ymax, n, agg_mu)
            elif method == "centroid_adaptive":
                ystar = defuzz.centroid_adaptive(ymin, ymax, agg_mu, n_base=max(101, n))
            elif method == "mom":
                ystar = defuzz.mom_on_grid(ymin, ymax, n, agg_mu)
            elif method == "bisector":
                ystar = defuzz.bisector_on_grid(ymin, ymax, n, agg_mu)
            else:
                ystar = defuzz.centroid_on_grid(ymin, ymax, n, agg_mu)
            outs[oname] = float(ystar)

        return outs
