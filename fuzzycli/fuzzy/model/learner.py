from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import csv
import math

from ..core import norms
from .knowledge import KnowledgeBase
from .variable import InputVariable, OutputVariable
from ..core.rule import Rule
from ..core.mfs import Triangular, Trapezoidal, Gaussian


# === MEMBERSHIP FUNCTIONS (μ) =================================================

def _mu_tri(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if b == a == c:
        return 1.0 if x == b else 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if (b - a) != 0 else 0.0
    return (c - x) / (c - b) if (c - b) != 0 else 0.0


def _mu_trap(x: float, a: float, b: float, c: float, d: float) -> float:
    # a <= b <= c <= d
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if (b - a) != 0 else 0.0
    # c < x < d
    return (d - x) / (d - c) if (d - c) != 0 else 0.0


def _mu_gauss(x: float, mu: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if x == mu else 0.0
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z)


def _mu(kind: str, params: Tuple[float, ...], x: float) -> float:
    k = kind.lower()
    if k == "tri":
        a, b, c = params
        return _mu_tri(x, a, b, c)
    if k == "trap":
        a, b, c, d = params
        return _mu_trap(x, a, b, c, d)
    if k == "gauss":
        mu, sigma = params
        return _mu_gauss(x, mu, sigma)
    # nieznany kształt
    return 0.0


# === AUTO PARTITION BUILDERS ==================================================

def _labels_for_terms(terms: int, custom: Optional[List[str]] = None) -> List[str]:
    if custom:
        return list(custom)
    if terms == 3:
        return ["small", "medium", "large"]
    return [f"t{i+1}" for i in range(terms)]


def _grid_centers(vmin: float, vmax: float, terms: int) -> List[float]:
    if terms <= 1:
        return [(vmin + vmax) / 2.0]
    step = (vmax - vmin) / (terms - 1)
    return [vmin + step * i for i in range(terms)]


def _sigma_from(step: float, mode: str, value: float) -> float:
    mode = (mode or "factor").lower()
    if mode == "factor":
        return max(1e-12, value * step)
    if mode == "fwhm":
        # value = FWHM w jednostkach "step"
        fwhm = max(1e-12, value * step)
        return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    if mode == "fixed":
        return max(1e-12, value)
    # domyślnie factor
    return max(1e-12, value * step)


def _build_auto_mfs(
    vmin: float,
    vmax: float,
    terms: int,
    shape: str = "tri",
    labels: Optional[List[str]] = None,
    plateau_ratio: float = 0.25,
    sigma_mode: str = "factor",
    sigma_value: float = 0.5,
) -> List[Tuple[str, Tuple[str, Tuple[float, ...]]]]:
    """
    Zwraca listę elementów: (label, (kind, params_tuple)).
    """
    shape = (shape or "tri").lower()
    labels = _labels_for_terms(terms, labels)
    centers = _grid_centers(vmin, vmax, terms)
    step = centers[1] - centers[0] if len(centers) > 1 else max(1e-9, (vmax - vmin))

    out: List[Tuple[str, Tuple[str, Tuple[float, ...]]]] = []

    if shape == "tri":
        for i, c in enumerate(centers):
            a = centers[i - 1] if i > 0 else vmin
            b = c
            d = centers[i + 1] if i < len(centers) - 1 else vmax
            out.append((labels[i], ("tri", (a, b, d))))
        return out

    if shape == "trap":
        half_plateau = max(0.0, plateau_ratio) * step * 0.5
        for i, c in enumerate(centers):
            left_knot = centers[i - 1] if i > 0 else vmin
            right_knot = centers[i + 1] if i < len(centers) - 1 else vmax
            b = max(vmin, c - half_plateau)
            d = min(vmax, c + half_plateau)
            out.append((labels[i], ("trap", (left_knot, b, d, right_knot))))
        return out

    if shape == "gauss":
        sigma = _sigma_from(step, sigma_mode, sigma_value)
        for i, c in enumerate(centers):
            out.append((labels[i], ("gauss", (c, sigma))))
        return out

    # fallback → tri
    for i, c in enumerate(centers):
        a = centers[i - 1] if i > 0 else vmin
        b = c
        d = centers[i + 1] if i < len(centers) - 1 else vmax
        out.append((labels[i], ("tri", (a, b, d))))
    return out


def _apply_margin(vmin: float, vmax: float, margin: float) -> Tuple[float, float]:
    if margin <= 0:
        return vmin, vmax
    span = vmax - vmin
    if span <= 0:
        return vmin, vmax
    return vmin - span * margin, vmax + span * margin


# === LEARNER (Wang–Mendel) ===================================================

def learn_from_csv(csv_path: str,
                   terms: int = 3,
                   partition: str = "grid",
                   tnorm: str = "min",
                   snorm: str = "max",
                   min_weight: float = 0.0,
                   mf_cfg: Optional[Dict[str, Any]] = None) -> KnowledgeBase:
    """
    Prosta implementacja Wang–Mendel z obsługą MF z konfiguracji.

    CSV: header: x1,...,xN,y

    MF konfigurujesz przez 'mf_cfg' (opcjonalnie). Struktura:
      mf_cfg = {
        "mode": "auto_from_data" | "manual",
        "default": {
          "shape": "tri" | "trap" | "gauss",
          "terms": 3,
          "range_margin": 0.02,
          "trapezoid": {"plateau_ratio": 0.25},
          "gaussian": {"sigma_mode": "factor", "sigma_value": 0.5},
          "labels": ["small","medium","large"]   # opcjonalne, globalne
        },
        "per_variable": {
          "VarName": {
             "shape": "...",
             "terms": 5,
             "labels": ["...","...","..."],
             "plateau_ratio": 0.3,               # skrót dla trap
             "trapezoid": {"plateau_ratio": 0.3},
             "gaussian": {"sigma_mode": "fwhm", "sigma_value": 1.0},
          },
          ...
        },
        "explicit": {
          "VarName": [
            {"label": "low",  "tri":   [a,b,c]},
            {"label": "mid",  "trap":  [a,b,c,d]},
            {"label": "high", "gauss": [mu,sigma]}
          ],
          ...
        }
      }

    Jeżeli 'mf_cfg' == None → wstecznie: trójkąty z parametrem 'terms'.
    """
    # 1) Wczytaj dane
    rows: List[List[float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header is None or len(header) < 2:
            raise ValueError("CSV musi mieć co najmniej 1 wejście i 1 wyjście (nagłówek).")
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])

    inputs = header[:-1]
    output = header[-1]

    # 2) Zakresy (z marginesem z configu)
    mins = [min(r[i] for r in rows) for i in range(len(header))]
    maxs = [max(r[i] for r in rows) for i in range(len(header))]

    # margin (globalny)
    range_margin = 0.0
    if mf_cfg and "default" in mf_cfg:
        range_margin = float(mf_cfg["default"].get("range_margin", 0.0))
    mins = list(mins)
    maxs = list(maxs)
    for j in range(len(header)):
        mins[j], maxs[j] = _apply_margin(mins[j], maxs[j], range_margin)

    # 3) MF-y: budowa definicji (label, (kind, params))
    mf_defs: Dict[str, List[Tuple[str, Tuple[str, Tuple[float, ...]]]]] = {}

    # back-compat: prosty trójkątny grid
    def _legacy_tris(name: str, j: int):
        tris = _build_auto_mfs(mins[j], maxs[j], terms, shape="tri")
        mf_defs[name] = tris

    mode = (mf_cfg or {}).get("mode", "auto_from_data").lower() if mf_cfg else "auto_from_data"
    default = (mf_cfg or {}).get("default", {}) if mf_cfg else {}
    per_var = (mf_cfg or {}).get("per_variable", {}) if mf_cfg else {}
    explicit = (mf_cfg or {}).get("explicit", {}) if mf_cfg else {}

    # inputs
    for j, name in enumerate(inputs):
        if mf_cfg and mode == "manual" and name in explicit:
            defs = []
            for mf in explicit[name]:
                lbl = mf["label"]
                if "tri" in mf:
                    a, b, c = mf["tri"]
                    defs.append((lbl, ("tri", (a, b, c))))
                elif "trap" in mf:
                    a, b, c, d = mf["trap"]
                    defs.append((lbl, ("trap", (a, b, c, d))))
                elif "gauss" in mf:
                    mu, s = mf["gauss"]
                    defs.append((lbl, ("gauss", (mu, s))))
            mf_defs[name] = defs
        else:
            # auto
            pv = per_var.get(name, {})
            shape = pv.get("shape", default.get("shape", "tri"))
            t = int(pv.get("terms", default.get("terms", terms)))
            labels = pv.get("labels", default.get("labels"))
            # trapez
            plateau_ratio = pv.get("plateau_ratio",
                                   pv.get("trapezoid", {}).get("plateau_ratio",
                                   default.get("trapezoid", {}).get("plateau_ratio", 0.25)))
            # gauss
            gcfg = default.get("gaussian", {})
            gpv = pv.get("gaussian", {})
            sigma_mode = gpv.get("sigma_mode", gcfg.get("sigma_mode", "factor"))
            sigma_value = float(gpv.get("sigma_value", gcfg.get("sigma_value", 0.5)))

            mf_defs[name] = _build_auto_mfs(
                mins[j], maxs[j], t,
                shape=shape, labels=labels,
                plateau_ratio=float(plateau_ratio),
                sigma_mode=sigma_mode, sigma_value=sigma_value
            )

        # fallback jeśli z jakiegoś powodu pusto
        if not mf_defs.get(name):
            _legacy_tris(name, j)

    # output
    j_out = len(header) - 1
    name = output
    if mf_cfg and mode == "manual" and name in explicit:
        defs = []
        for mf in explicit[name]:
            lbl = mf["label"]
            if "tri" in mf:
                a, b, c = mf["tri"]
                defs.append((lbl, ("tri", (a, b, c))))
            elif "trap" in mf:
                a, b, c, d = mf["trap"]
                defs.append((lbl, ("trap", (a, b, c, d))))
            elif "gauss" in mf:
                mu, s = mf["gauss"]
                defs.append((lbl, ("gauss", (mu, s))))
        mf_defs[name] = defs
    else:
        pv = per_var.get(name, {})
        shape = pv.get("shape", default.get("shape", "tri"))
        t = int(pv.get("terms", default.get("terms", terms)))
        labels = pv.get("labels", default.get("labels"))
        plateau_ratio = pv.get("plateau_ratio",
                               pv.get("trapezoid", {}).get("plateau_ratio",
                               default.get("trapezoid", {}).get("plateau_ratio", 0.25)))
        gcfg = default.get("gaussian", {})
        gpv = pv.get("gaussian", {})
        sigma_mode = gpv.get("sigma_mode", gcfg.get("sigma_mode", "factor"))
        sigma_value = float(gpv.get("sigma_value", gcfg.get("sigma_value", 0.5)))
        mf_defs[name] = _build_auto_mfs(
            mins[j_out], maxs[j_out], t,
            shape=shape, labels=labels,
            plateau_ratio=float(plateau_ratio),
            sigma_mode=sigma_mode, sigma_value=sigma_value
        )
    if not mf_defs.get(name):
        _legacy_tris(name, j_out)

    # 4) T-norma (dla WM)
    tnorm_fn = norms.TNORMS.get(tnorm, norms.TNORMS["min"])

    # 5) Liczenie sił reguł (Wang–Mendel) – μ zgodnie z typem MF
    weighted: Dict[Tuple[Tuple[Tuple[str, str], ...], Tuple[str, str]], float] = {}

    for r in rows:
        # antecedenty
        ante: List[Tuple[str, str]] = []
        mu_ante: List[float] = []
        for j, name in enumerate(inputs):
            x = r[j]
            # wybierz MF o max μ
            best_label = None
            best_kind = None
            best_params = None
            best_mu = -1.0
            for lbl, (kind, params) in mf_defs[name]:
                mu = _mu(kind, params, x)
                if mu > best_mu:
                    best_mu = mu
                    best_label = lbl
                    best_kind = kind
                    best_params = params
            if best_label is None:
                # skrajny fallback: bierz pierwszą MF
                lbl, (kind, params) = mf_defs[name][0]
                best_label, best_kind, best_params, best_mu = lbl, kind, params, _mu(kind, params, x)

            ante.append((name, best_label))
            mu_ante.append(best_mu)

        # konsekwent (wyjście)
        y = r[-1]
        out_label = None
        out_kind = None
        out_params = None
        out_mu = -1.0
        for lbl, (kind, params) in mf_defs[output]:
            mu = _mu(kind, params, y)
            if mu > out_mu:
                out_mu = mu
                out_label = lbl
                out_kind = kind
                out_params = params
        if out_label is None:
            lbl, (kind, params) = mf_defs[output][0]
            out_label, out_kind, out_params, out_mu = lbl, kind, params, _mu(kind, params, y)

        try:
            strength_ante = tnorm_fn(mu_ante) if mu_ante else 1.0
        except Exception:
            strength_ante = min(mu_ante) if mu_ante else 1.0

        strength = float(strength_ante) * float(out_mu)
        key = (tuple(ante), (output, out_label))
        weighted[key] = max(weighted.get(key, 0.0), strength)

    # 6) Budowa KB (z MF zgodnymi z definicją)
    kb = KnowledgeBase()
    kb.inputs = {}
    kb.outputs = {}
    kb.rules = []
    kb.tnorm = tnorm
    kb.snorm = snorm
    kb.mode = "FIT"
    kb.defuzz = "centroid"

    # zmienne wejściowe
    for j, name in enumerate(inputs):
        var = InputVariable(name=name, vmin=mins[j], vmax=maxs[j])
        for label, (kind, params) in mf_defs[name]:
            if kind == "tri":
                var.add_term(label, Triangular(*params))     # type: ignore[arg-type]
            elif kind == "trap":
                var.add_term(label, Trapezoidal(*params))    # type: ignore[arg-type]
            elif kind == "gauss":
                var.add_term(label, Gaussian(*params))       # type: ignore[arg-type]
        kb.inputs[name] = var

    # zmienna wyjściowa
    ov = OutputVariable(name=output, vmin=mins[-1], vmax=maxs[-1])
    for label, (kind, params) in mf_defs[output]:
        if kind == "tri":
            ov.add_term(label, Triangular(*params))
        elif kind == "trap":
            ov.add_term(label, Trapezoidal(*params))
        elif kind == "gauss":
            ov.add_term(label, Gaussian(*params))
    # domyślna siatka dla defuzz (jeśli klasa jej nie ustawia)
    if not hasattr(ov, "grid") or ov.grid is None:
        ov.grid = (ov.vmin, ov.vmax, 201)
    kb.outputs[output] = ov

    # reguły
    for (ante_tuple, (ov_name, olab)), w in weighted.items():
        if w < min_weight:
            continue
        rule = Rule(antecedent=list(ante_tuple), consequent=(ov_name, olab), weight=float(w))
        kb.rules.append(rule)

    return kb


def save_kb_to_fz(kb: KnowledgeBase, path: str) -> None:
    """
    Zapisuje KnowledgeBase do pliku .fz (zgodnego z parserem).
    Obsługuje zarówno MF-y jako obiekty (Triangular/Trapezoidal/Gaussian),
    jak i fallbackowe tuple ("tri", (a,b,c)).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as fz:
        # 1) Zmienne
        for name, var in getattr(kb, "inputs", {}).items():
            fz.write(f"var input {name} {var.vmin} {var.vmax}\n")
        for name, var in getattr(kb, "outputs", {}).items():
            fz.write(f"var output {name} {var.vmin} {var.vmax}\n")

        # 2) MF-y
        all_vars: Dict[str, object] = {}
        all_vars.update(getattr(kb, "inputs", {}))
        all_vars.update(getattr(kb, "outputs", {}))

        for vname, container in all_vars.items():
            terms = getattr(container, "terms", {})
            for label, mf in terms.items():
                # a) fallback: ("tri", (a,b,c))
                if isinstance(mf, tuple) and len(mf) == 2 and mf[0] == "tri":
                    a, b, c = mf[1]
                    fz.write(f"mf {vname} {label} tri {a} {b} {c}\n")
                    continue

                # b) obiekty MF
                try:
                    if isinstance(mf, Triangular):
                        if hasattr(mf, "a") and hasattr(mf, "b") and hasattr(mf, "c"):
                            a, b, c = mf.a, mf.b, mf.c
                        elif hasattr(mf, "params"):
                            a, b, c = mf.params
                        else:
                            a, c = mf.support()
                            b = (a + c) / 2.0
                        fz.write(f"mf {vname} {label} tri {a} {b} {c}\n")
                    elif isinstance(mf, Trapezoidal):
                        if all(hasattr(mf, x) for x in ("a", "b", "c", "d")):
                            a, b, c, d = mf.a, mf.b, mf.c, mf.d
                        elif hasattr(mf, "params"):
                            a, b, c, d = mf.params
                        else:
                            a, d = mf.support()
                            b = a
                            c = d
                        fz.write(f"mf {vname} {label} trap {a} {b} {c} {d}\n")
                    elif isinstance(mf, Gaussian):
                        # Spróbuj wyciągnąć środek i sigma z różnych możliwych atrybutów.
                        def _num_or_none(val):
                            if val is None or callable(val):
                                return None
                            try:
                                return float(val)
                            except Exception:
                                return None
                    
                        mu = None
                        # częste nazwy środka
                        for attr in ("mu", "mu0", "mean", "center", "c", "m"):
                            if hasattr(mf, attr):
                                mu = _num_or_none(getattr(mf, attr))
                                if mu is not None:
                                    break
                                
                        sigma = None
                        # częste nazwy odchylenia
                        for attr in ("sigma", "sd", "std", "stddev"):
                            if hasattr(mf, attr):
                                sigma = _num_or_none(getattr(mf, attr))
                                if sigma is not None:
                                    break
                                
                        # fallback: params
                        if (mu is None or sigma is None) and hasattr(mf, "params"):
                            try:
                                p_mu, p_sigma = mf.params
                                if mu is None: mu = _num_or_none(p_mu)
                                if sigma is None: sigma = _num_or_none(p_sigma)
                            except Exception:
                                pass
                            
                        # fallback: ze wsparcia
                        if mu is None or sigma is None:
                            a, b = mf.support()
                            mu = float((a + b) / 2.0)
                            sigma = float(max(1e-9, (b - a) / 6.0))
                    
                        fz.write(f"mf {vname} {label} gauss {mu} {sigma}\n")

                    else:
                        if hasattr(mf, "params"):
                            params = mf.params
                            if len(params) == 3:
                                a, b, c = params
                                fz.write(f"mf {vname} {label} tri {a} {b} {c}\n")
                            elif len(params) == 4:
                                a, b, c, d = params
                                fz.write(f"mf {vname} {label} trap {a} {b} {c} {d}\n")
                        elif hasattr(mf, "support"):
                            a, b = mf.support()
                            fz.write(f"mf {vname} {label} trap {a} {a} {b} {b}\n")
                except Exception:
                    # pomiń MF, którego nie umiemy bezpiecznie zserializować
                    pass

        # 3) Reguły
        for rule in getattr(kb, "rules", []):
            conds = " AND ".join(f"{vn} is {lbl}" for vn, lbl in rule.antecedent)
            w = getattr(rule, "weight", None)
            if w is not None:
                fz.write(f"rule IF {conds} THEN {rule.consequent[0]} is {rule.consequent[1]} weight {w}\n")
            else:
                fz.write(f"rule IF {conds} THEN {rule.consequent[0]} is {rule.consequent[1]}\n")

        # 4) Ustawienia silnika
        fz.write(f"tnorm {getattr(kb, 'tnorm', 'min')}\n")
        fz.write(f"snorm {getattr(kb, 'snorm', 'max')}\n")
        fz.write(f"mode {getattr(kb, 'mode', 'FIT')}\n")
        fz.write(f"defuzz {getattr(kb, 'defuzz', 'centroid')}\n")
