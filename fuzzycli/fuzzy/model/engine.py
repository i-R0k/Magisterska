# MamdaniEngine (API: predict, explain)

from __future__ import annotations
from typing import Dict, Callable
from ..core import norms
from ..core.defuzz import centroid_on_grid
from ..core.types import Float
from .knowledge import KnowledgeBase

class MamdaniEngine:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.tnorm_fn: Callable = norms.TNORMS[self.kb.tnorm]
        self.snorm_fn: Callable = norms.SNORMS[self.kb.snorm]

    def predict(self, inputs: Dict[str, Float]) -> Dict[str, Float]:
        """
        FIT: per-rule implication -> aggregate -> defuzz (centroid).
        Automatyczny dobor siatki:
          - jesli ov.grid ma domyslne (0.0, 1.0, 101) lub ymin>=ymax -> zakres z unii supportow MF wyjscia,
            a gdy brak MF – z (vmin, vmax) zmiennej wyjsciowej;
          - jesli n<3 lub None -> n=201.
        """
        out_values: Dict[str, Float] = {}

        for oname, ovar in self.kb.outputs.items():
            # zbuduj zagregowana mu(y) jako max z przycietych konsekwentow
            def agg_mu(y: Float) -> Float:
                best = 0.0
                for rule in self.kb.rules:
                    if rule.consequent[0] != oname:
                        continue
                    # antecedent activation
                    acts = []
                    ok = True
                    for vname, label in rule.antecedent:
                        if vname not in self.kb.inputs:
                            ok = False
                            break
                        x = float(inputs[vname])
                        mf = self.kb.inputs[vname].terms[label]
                        acts.append(mf.mu(x))
                    if not ok:
                        continue
                    alpha = self.tnorm_fn(acts) * float(rule.weight)
                    # Mamdani implication: clip
                    cmf = ovar.terms[rule.consequent[1]]
                    mu_val = min(alpha, cmf.mu(y))
                    if mu_val > best:
                        best = mu_val
                return best

            # --- auto siatka do centroidu ---
            ymin, ymax, n = ovar.grid
            # auto-zakres jesli domyslny/niepoprawny
            if ymin >= ymax or (ymin, ymax) == (0.0, 1.0):
                if ovar.terms:
                    supports = [mf.support() for mf in ovar.terms.values()]
                    ymin = min(a for a, _ in supports)
                    ymax = max(b for _, b in supports)
                else:
                    ymin, ymax = ovar.vmin, ovar.vmax
            # auto-n jesli zbyt male lub brak
            if n is None or int(n) < 3:
                n = 201

            ystar = centroid_on_grid(float(ymin), float(ymax), int(n), agg_mu)
            out_values[oname] = ystar

        return out_values
