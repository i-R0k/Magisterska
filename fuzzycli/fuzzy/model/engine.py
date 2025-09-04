from __future__ import annotations
from typing import Dict, Callable
from ..core import norms
from ..core.defuzz import centroid_on_grid, centroid_adaptive, mom_on_grid, bisector_on_grid
from ..core.types import Float
from .knowledge import KnowledgeBase

class MamdaniEngine:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.tnorm_fn: Callable = norms.TNORMS[self.kb.tnorm]
        self.snorm_fn: Callable = norms.SNORMS[self.kb.snorm]

    def _auto_grid(self, ovar) -> tuple[Float, Float, int]:
        ymin, ymax, n = ovar.grid
        if ymin >= ymax or (ymin, ymax) == (0.0, 1.0):
            if ovar.terms:
                supports = [mf.support() for mf in ovar.terms.values()]
                ymin = min(a for a, _ in supports)
                ymax = max(b for _, b in supports)
            else:
                ymin, ymax = ovar.vmin, ovar.vmax
        if n is None or int(n) < 3:
            n = 201
        return float(ymin), float(ymax), int(n)

    def predict(self, inputs: Dict[str, Float]) -> Dict[str, Float]:
        """
        FIT: per-rule implication -> aggregate -> defuzz
        FATI: group by consequent label first (s-norm alphas), then implication -> aggregate labels -> defuzz
        Defuzz: centroid | mom | bisector (centroid ma tez wariant adaptacyjny, wewnetrznie sterowany)
        """
        out_values: Dict[str, Float] = {}

        for oname, ovar in self.kb.outputs.items():
            # --- policz aktywacje reguł (alfa) ---
            rule_alphas = []  # (consequent_label, alpha)
            for rule in self.kb.rules:
                if rule.consequent[0] != oname:
                    continue
                acts = []
                ok = True
                for vname, label in rule.antecedent:
                    if vname not in self.kb.inputs:
                        ok = False; break
                    x = float(inputs[vname])
                    mf = self.kb.inputs[vname].terms[label]
                    acts.append(mf.mu(x))
                if not ok:
                    continue
                alpha = self.tnorm_fn(acts) * float(rule.weight)
                rule_alphas.append((rule.consequent[1], alpha))

            # --- zbuduj funkcje agregowanej przynaleznosci wyjscia ---
            if (self.kb.mode or "FIT").upper() == "FATI":
                # FATI: najpierw agregujemy alfy per etykiete konsekwentu
                per_label = {}
                for lab, a in rule_alphas:
                    per_label.setdefault(lab, []).append(a)
                for lab in per_label:
                    per_label[lab] = self.snorm_fn(per_label[lab])  # s-agregacja alfa
                def agg_mu(y: Float) -> Float:
                    best = 0.0
                    for lab, a in per_label.items():
                        cmf = ovar.terms[lab]
                        mu_val = min(a, cmf.mu(y))
                        if mu_val > best: best = mu_val
                    return best
            else:
                # FIT: klipujemy per reguła i agregujemy s-norma (max)
                def agg_mu(y: Float) -> Float:
                    best = 0.0
                    for lab, a in rule_alphas:
                        cmf = ovar.terms[lab]
                        mu_val = min(a, cmf.mu(y))
                        if mu_val > best: best = mu_val
                    return best

            ymin, ymax, n = self._auto_grid(ovar)

            # --- wybór metody defuzyfikacji ---
            method = (self.kb.defuzz or "centroid").lower()
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

            out_values[oname] = ystar

        return out_values
