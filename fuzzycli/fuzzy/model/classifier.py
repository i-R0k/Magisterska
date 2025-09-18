# fuzzycli/fuzzy/model/classifier.py
from __future__ import annotations
from typing import Dict, List, Any, Tuple, Iterable
from ..core import norms

try:
    from .knowledge import KnowledgeBase
except Exception as e:
    raise ImportError("Dopasuj import KnowledgeBase w classifier.py") from e


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class Classifier:
    """
    Regułowo-rozmyty klasyfikator (Mamdani, inference-only).
    - FIT: wybór etykiety przez max(alpha) per etykieta.
    - FATI: agregacja sił etykiet przez S-normę z KB (domyślnie max lub algebraic sum),
            a potem wybór przez argmax.
    Publiczne API pozostaje bez zmian:
      - explain(inputs: Dict[str, float], mode: str | None = None, threshold: float = 0.0)
      - classify(inputs: Dict[str, float], mode: str | None = None)
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        # T/S-normy z bezpiecznymi fallbackami
        self.tnorm_fn = norms.TNORMS.get(getattr(kb, "tnorm", "min"), norms.TNORMS["min"])
        self.snorm_fn = norms.SNORMS.get(getattr(kb, "snorm", "max"), norms.SNORMS["max"])
        # Indeks reguł per wyjście – przyspiesza wnioskowanie
        self._rules_by_output: Dict[str, List[Tuple[int, Any]]] = {}
        for i, rule in enumerate(getattr(self.kb, "rules", [])):
            oname = rule.consequent[0]
            self._rules_by_output.setdefault(oname, []).append((i, rule))

    # ---------- helpers ----------

    def _tnorm(self, values: Iterable[float]) -> float:
        vals = [float(v) for v in values]
        if not vals:
            return 1.0
        try:
            return float(self.tnorm_fn(vals))
        except Exception:
            # awaryjnie zachowaj się jak 'min'
            return float(min(vals))

    def _snorm(self, values: Iterable[float]) -> float:
        vals = [float(v) for v in values]
        if not vals:
            return 0.0
        try:
            return float(self.snorm_fn(vals))
        except Exception:
            # awaryjnie zachowaj się jak 'max'
            return float(max(vals))

    def _safe_mu(self, mf, x: float) -> float:
        """
        Bezpieczne wyliczenie przynależności. Preferuje mf.mu(x),
        ma fallback dla tuple('tri',(a,b,c)).
        """
        try:
            mu = float(mf.mu(x))  # typowe MF-y
            return _clip01(mu)
        except Exception:
            pass
        # Fallback: tuple MF
        if isinstance(mf, tuple) and len(mf) == 2 and mf[0] == "tri":
            a, b, c = mf[1]
            if x <= a or x >= c:
                return 0.0
            if x == b:
                return 1.0
            if x < b:
                return (x - a) / (b - a if b != a else 1e-12)
            return (c - x) / (c - b if c != b else 1e-12)
        # Ostatnia deska ratunku
        return 0.0

    def _compute_mus_cache(self, inputs: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        """Cache μ(var,label) dla wszystkich WEJŚĆ (inputs)."""
        cache: Dict[Tuple[str, str], float] = {}
        for vname, var in getattr(self.kb, "inputs", {}).items():
            x = float(inputs.get(vname, 0.0))
            for label, mf in getattr(var, "terms", {}).items():
                cache[(vname, label)] = self._safe_mu(mf, x)
        return cache

    # ---------- API ----------

    def explain(
        self,
        inputs: Dict[str, float],
        mode: str | None = None,
        threshold: float = 0.0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Zwraca szczegóły aktywacji reguł (bez defuzyfikacji).
        Struktura:
          {
            <output_name>: [
              (opcjonalnie) {"_fati_label_strengths": {label: strength, ...}},
              {
                "rule_index": int,
                "antecedent": [{"var": str, "label": str, "value": float, "mu": float}, ...],
                "alpha": float,
                "weight": float,
                "consequent": {"var": str, "label": str}
              }, ...
            ],
            ...
          }
        """
        mode = (mode or getattr(self.kb, "mode", "FIT")).upper()
        cache = self._compute_mus_cache(inputs)
        results: Dict[str, List[Dict[str, Any]]] = {}

        for oname, ovar in getattr(self.kb, "outputs", {}).items():
            infos: List[Dict[str, Any]] = []
            rule_alphas: List[Tuple[int, str, float]] = []

            for i, rule in self._rules_by_output.get(oname, []):
                # Antecedent
                mus = []
                antecedent_info = []
                ok = True
                for vname, lbl in rule.antecedent:
                    key = (vname, lbl)
                    if key not in cache:
                        ok = False
                        break
                    mu = cache[key]
                    mus.append(mu)
                    antecedent_info.append({
                        "var": vname,
                        "label": lbl,
                        "value": float(inputs.get(vname, 0.0)),
                        "mu": float(mu),
                    })
                if not ok:
                    continue

                alpha_base = self._tnorm(mus)
                alpha = _clip01(alpha_base * float(getattr(rule, "weight", 1.0)))
                if alpha < threshold:
                    continue

                infos.append({
                    "rule_index": i,
                    "antecedent": antecedent_info,
                    "alpha": float(alpha),
                    "weight": float(getattr(rule, "weight", 1.0)),
                    "consequent": {"var": rule.consequent[0], "label": rule.consequent[1]},
                })
                rule_alphas.append((i, rule.consequent[1], float(alpha)))

            # Meta dla FATI: agregacja etykiet S-normą z KB
            if mode == "FATI":
                label_buckets: Dict[str, List[float]] = {}
                for (_idx, lab, a) in rule_alphas:
                    label_buckets.setdefault(lab, []).append(a)
                label_strengths = {lab: self._snorm(vals) for lab, vals in label_buckets.items()}
                results[oname] = [{"_fati_label_strengths": label_strengths}] + infos
            else:
                results[oname] = infos

        return results

    def classify(self, inputs: Dict[str, float], mode: str | None = None) -> Dict[str, Dict[str, Any]]:
        """
        Zwraca dla każdego outputu:
          {"chosen": <label|None>, "strengths": {label: strength, ...}}
        FIT: strengths = max(alpha per label)
        FATI: strengths = S-norma po alpha per label (wg kb.snorm)
        """
        explain = self.explain(inputs, mode=mode)
        out: Dict[str, Dict[str, Any]] = {}

        for oname, infos in explain.items():
            # FATI: pierwszy element może zawierać meta
            if infos and isinstance(infos[0], dict) and "_fati_label_strengths" in infos[0]:
                meta = infos[0]["_fati_label_strengths"]
                chosen = max(meta.items(), key=lambda kv: kv[1])[0] if meta else None
                out[oname] = {"chosen": chosen, "strengths": meta}
                continue

            # FIT: max po alpha w ramach etykiety
            per: Dict[str, float] = {}
            for info in infos:
                lab = info["consequent"]["label"]
                a = float(info["alpha"])
                # max-aggregation dla FIT
                if a > per.get(lab, 0.0):
                    per[lab] = a
            chosen = max(per.items(), key=lambda kv: kv[1])[0] if per else None
            out[oname] = {"chosen": chosen, "strengths": per}

        return out
