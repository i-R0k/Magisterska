import sys
from typing import Dict, List, Sequence, Union

from ...fuzzy.core import norms
from ...fuzzy.io.fz_parser import parse_fz


# ========= utils: ANSI / pretty =========

_RESET = "\x1b[0m"

def _use_ansi() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def _ansi_color(mu: float) -> str:
    """
    Kolor wg przynależności (μ):
      ≥ 0.50 → zielony
      ≥ 0.20 → żółty
      < 0.20 → szary
    """
    if not _use_ansi():
        return ""
    if mu >= 0.50:
        return "\x1b[32m"  # green
    if mu >= 0.20:
        return "\x1b[33m"  # yellow
    return "\x1b[90m"      # grey


# ========= utils: MF / μ =========

def _mu_tri(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        denom = (b - a) if b != a else 1e-12
        return (x - a) / denom
    denom = (c - b) if c != b else 1e-12
    return (c - x) / denom

def _mf_mu(mf, x: float) -> float:
    """
    Zwraca μ(x) dla obiektu MF:
      - preferuje .mu(x)
      - fallback: tuple ("tri", (a,b,c))
      - inaczej: 0.0
    """
    try:
        if hasattr(mf, "mu"):
            return float(mf.mu(float(x)))
        if isinstance(mf, tuple) and len(mf) == 2 and mf[0] == "tri":
            a, b, c = mf[1]
            return float(_mu_tri(float(x), a, b, c))
    except Exception:
        pass
    return 0.0


# ========= utils: args parsing =========

def _parse_at(at_arg: Union[str, Sequence[str], None]) -> Dict[str, float]:
    """
    Akceptuje:
      - None
      - "x=1,y=2"
      - ["x=1","y=2"] lub ["x=1, y=2"]
    """
    if not at_arg:
        return {}
    pairs: List[str] = []
    if isinstance(at_arg, (list, tuple)):
        for elem in at_arg:
            if elem is None:
                continue
            for tok in str(elem).split(","):
                tok = tok.strip()
                if tok:
                    pairs.append(tok)
    else:
        for tok in str(at_arg).split(","):
            tok = tok.strip()
            if tok:
                pairs.append(tok)

    out: Dict[str, float] = {}
    for kv in pairs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except Exception:
            # nie udało się zrzutować → pomijamy
            pass
    return out


# ========= utils: α(rule) =========

def _rule_alpha(kb, rule, xdict: Dict[str, float]) -> float:
    """
    α = T( μ_1, μ_2, ... ) * weight
    Jeśli brakuje wartości wejścia lub MF, zwraca 0.0.
    """
    if not xdict:
        return 0.0
    mus: List[float] = []
    for vname, label in rule.antecedent:
        var = kb.inputs.get(vname)
        if var is None or vname not in xdict:
            return 0.0
        mf = getattr(var, "terms", {}).get(label)
        if mf is None:
            return 0.0
        mus.append(_mf_mu(mf, xdict[vname]))

    try:
        tfn = norms.TNORMS.get(getattr(kb, "tnorm", "min"), norms.TNORMS["min"])
        base = float(tfn(mus)) if mus else 1.0
    except Exception:
        base = min(mus) if mus else 1.0

    w = float(getattr(rule, "weight", 1.0))
    return base * w


# ========= main =========

def cmd_show(args) -> None:
    """
    Flagi wspierane przez parser (dodaj w subkomendzie 'show'):
      --model PATH                 : plik .fz
      --at "x=1,y=2" / --at x=1 ...: punkt do policzenia μ/α (opcjonalnie)
      --include-inactive           : pokaż również reguły inactive (domyślnie: ukryte)
      --fired-only                 : pokaż tylko reguły, które się „odpaliły” dla --at
      --min-alpha FLOAT            : próg α dla --fired-only (domyślnie 0.0)
    """
    kb = parse_fz(args.model)

    xdict = _parse_at(getattr(args, "at", None))
    include_inactive = bool(getattr(args, "include_inactive", False))
    fired_only = bool(getattr(args, "fired_only", False))
    min_alpha = float(getattr(args, "min_alpha", 0.0))

    # --- Inputs ---
    print("Inputs:")
    for name, var in kb.inputs.items():
        terms = list(getattr(var, "terms", {}).items())
        if xdict:
            x = xdict.get(name, None)
            parts: List[str] = []
            for lbl, mf in terms:
                if x is None:
                    parts.append(lbl)
                else:
                    mu = _mf_mu(mf, x)
                    color = _ansi_color(mu)
                    reset = _RESET if color else ""
                    parts.append(f"{color}{lbl}({mu:.2f}){reset}")
            print(f"  {name} [{var.vmin},{var.vmax}] -> " + ", ".join(parts))
        else:
            print(f"  {name} [{var.vmin},{var.vmax}] -> terms: {', '.join(k for k,_ in terms)}")

    # --- Outputs ---
    print("Outputs:")
    for name, var in kb.outputs.items():
        terms = list(getattr(var, "terms", {}).items())
        grid = getattr(var, "grid", None)
        print(f"  {name} [{var.vmin},{var.vmax}] grid={grid} -> terms: {', '.join(k for k,_ in terms)}")

    # --- Rules ---
    print("Rules:")
    print(f"Engine: tnorm={kb.tnorm}, snorm={kb.snorm}, mode={kb.mode}, defuzz={kb.defuzz}")

    shown = 0
    for i, r in enumerate(kb.rules, 1):
        # filtr aktywności
        if not include_inactive and not getattr(r, "active", True):
            continue

        # filtr fired-only (wymaga --at; jeśli brak --at, nie filtrujemy po α)
        alpha_val = None
        if fired_only and xdict:
            alpha_val = _rule_alpha(kb, r, xdict)
            if alpha_val < min_alpha:
                continue

        ants = " AND ".join(f"{v} is {lbl}" for v, lbl in r.antecedent)
        suffix = ""
        if not getattr(r, "active", True):
            suffix += " [inactive]"
        if alpha_val is not None:
            suffix += f"  α={alpha_val:.4f}"

        print(f"  R{i}: IF {ants} THEN {r.consequent[0]} is {r.consequent[1]} (w={r.weight}){suffix}")
        shown += 1

    if shown == 0:
        print("  (brak reguł do wyświetlenia z tymi filtrami)")
