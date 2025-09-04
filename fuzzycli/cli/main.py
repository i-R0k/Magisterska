# Punkt wejścia `mamdani ...`

import argparse
import csv
import sys
from pathlib import Path
from ..fuzzy.io.fz_parser import parse_fz
from ..fuzzy.model.engine import MamdaniEngine
from ..fuzzy.core import norms as _norms

def cmd_validate(args):
    kb = parse_fz(args.model)
    print(f"OK: inputs={len(kb.inputs)}, outputs={len(kb.outputs)}, rules={len(kb.rules)}")
    print(f"tnorm={kb.tnorm}, snorm={kb.snorm}, mode={kb.mode}, defuzz={kb.defuzz}")

def _use_ansi():
    return sys.stdout.isatty()

def _ansi(mu: float) -> str:
    if not _use_ansi():
        # bez kolorów zwróć tylko label (formatowanie wykonane gdzieś indziej)
        if mu >= 0.5: return ""
        if mu >= 0.2: return ""
        return ""
    if mu >= 0.5: return "\x1b[32m"  # green
    if mu >= 0.2: return "\x1b[33m"  # yellow
    return "\x1b[90m"                # grey

def cmd_show(args):
    kb = parse_fz(args.model)
    at = {}
    if args.at:
        for kv in args.at:
            k,v = kv.split("=",1)
            at[k] = float(v)
    print("Inputs:")
    for name, var in kb.inputs.items():
        terms = list(var.terms.items())
        if at:
            parts = []
            x = at.get(name, None)
            for lbl, mf in terms:
                if x is None:
                    parts.append(lbl)
                else:
                    mu = mf.mu(float(x))
                    parts.append(f"{_ansi(mu)}{lbl}({mu:.2f})\x1b[0m")
            print(f"  {name} [{var.vmin},{var.vmax}] -> " + ", ".join(parts))
        else:
            print(f"  {name} [{var.vmin},{var.vmax}] -> terms: {', '.join(k for k,_ in terms)}")
    print("Outputs:")
    for name, var in kb.outputs.items():
        terms = list(var.terms.items())
        print(f"  {name} [{var.vmin},{var.vmax}] grid={var.grid} -> terms: {', '.join(k for k,_ in terms)}")
    print("Rules:")
    for i, r in enumerate(kb.rules, 1):
        ants = " AND ".join(f"{v} is {lbl}" for v,lbl in r.antecedent)
        print(f"  R{i}: IF {ants} THEN {r.consequent[0]} is {r.consequent[1]} (w={r.weight})")

def _parse_keyvals(kvs):
    data = {}
    for kv in kvs:
        k, v = kv.split("=", 1)
        data[k] = float(v)
    return data

def cmd_predict(args):
    kb = parse_fz(args.model)
    eng = MamdaniEngine(kb)
    data = _parse_keyvals(args.kv)
    y = eng.predict(data)
    for oname, val in y.items():
        print(f"{oname}: {val:.6g}")
    if getattr(args, "explain", False):
        # wylicz i wypisz alpha dla reguł (jak w engine)
        tnorm = _norms.TNORMS.get(kb.tnorm, _norms.TNORMS["min"])
        print("\n[explain] rule activations (alpha):")
        for i, rule in enumerate(kb.rules, 1):
            acts = []
            ok = True
            for vname, label in rule.antecedent:
                if vname not in kb.inputs:
                    ok = False; break
                x = float(data.get(vname, 0.0))
                mf = kb.inputs[vname].terms[label]
                acts.append(mf.mu(x))
            if not ok:
                continue
            alpha = tnorm(acts) * float(rule.weight)
            ants = " AND ".join(f"{v} is {lbl}" for v,lbl in rule.antecedent)
            cons = f"{rule.consequent[0]} is {rule.consequent[1]}"
            print(f"  R{i}: alpha={alpha:.4f} :: IF {ants} THEN {cons} (w={rule.weight})")

def _mu_tri(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x <  b: return (x - a) / (b - a if b != a else 1e-12)
    return (c - x) / (c - b if c != b else 1e-12)

def _tri_partition(vmin, vmax, terms):
    """Rownomierny podzial zakresu [vmin,vmax] na 'terms' trojkatow."""
    if terms < 2:  # awaryjnie
        mid = (vmin + vmax) / 2.0
        return [("t1", (vmin, mid, vmax))]
    step = (vmax - vmin) / (terms - 1)
    out = []
    for i in range(terms):
        if i == 0:
            out.append(("t1", (vmin, vmin, vmin + step)))
        elif i == terms - 1:
            out.append((f"t{terms}", (vmax - step, vmax, vmax)))
        else:
            out.append((f"t{i+1}", (vmin + step*(i-1), vmin + step*i, vmin + step*(i+1))))
    return out

def cmd_learn(args):
    """
    Uczenie modelu z CSV (prosty grid + Wang–Mendel) i zapis do .fz.
    CSV: ostatnia kolumna to wyjscie, pozostale to wejscia.
    """
    # 1) wczytaj CSV
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row: 
                continue
            rows.append([float(x) for x in row])

    if len(header) < 2:
        raise SystemExit("CSV musi miec co najmniej 1 wejscie i 1 wyjscie.")

    inputs = header[:-1]
    output = header[-1]
    m = len(inputs)

    # 2) zakresy
    mins = [min(r[i] for r in rows) for i in range(len(header))]
    maxs = [max(r[i] for r in rows) for i in range(len(header))]

    # 3) automatyczne MF (grid/tris)
    mf_defs = {}  # nazwa_zmiennej -> [(label, (a,b,c)), ...]
    for j, name in enumerate(inputs + [output]):
        mf_defs[name] = _tri_partition(mins[j], maxs[j], args.terms)

    # 4) indukcja reguł (Wang–Mendel) z wagami i deduplikacją (max)
    #    Dla kazdego rekordu:
    #      - wybierz najlepszy termin (max mu) dla kazdej zmiennej
    #      - policz mu_ante = [mu_i] i mu_out (dla wyjscia)
    #      - strength = tnorm(mu_ante) * mu_out
    #    Deduplikacja: dla tych samych antecedent+consequent bierzemy max(strength)
    from ..fuzzy.core import norms as _norms  # lokalny import, zeby wykorzystac t-norm
    tnorm_fn = _norms.TNORMS.get(getattr(args, "tnorm", "min"), _norms.TNORMS["min"])

    weighted = {}  # key: (tuple(ante), (output, out_label)) -> strength (float)

    for r in rows:
        ante = []
        mu_ante = []
        # dla kazdego wejscia wybierz termin o maksymalnym mu (z mf_defs)
        for j, name in enumerate(inputs):
            x = r[j]
            best_label, best_params = max(mf_defs[name], key=lambda mf: _mu_tri(x, *mf[1]))
            ante.append((name, best_label))
            mu_ante.append(_mu_tri(x, *best_params))

        # dla wyjscia wybierz najlepszy termin i policz jego mu
        y = r[-1]
        out_label, out_params = max(mf_defs[output], key=lambda mf: _mu_tri(y, *mf[1]))
        mu_out = _mu_tri(y, *out_params)

        # sila reguly: tnorm(mu_ante) * mu_out
        # jesli brak ante (np. zero-wejscie), traktujemy tnorm([])=1.0
        try:
            strength_ante = tnorm_fn(mu_ante) if mu_ante else 1.0
        except Exception:
            # fallback na min
            strength_ante = min(mu_ante) if mu_ante else 1.0

        strength = float(strength_ante) * float(mu_out)

        key = (tuple(ante), (output, out_label))
        prev = weighted.get(key, 0.0)
        if strength > prev:
            weighted[key] = strength

    # opcjonalny prog odrzucajacy bardzo slabe reguly
    min_w = float(getattr(args, "min_weight", 0.0))

    # zmapuj do listy reguł z wagami
    uniq_rules = []
    for (ante_tuple, (ov, olab)), w in weighted.items():
        if w < min_w:
            continue
        uniq_rules.append((list(ante_tuple), (ov, olab), float(w)))


    # 5) zapis .fz
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fz:
        # variables
        for j, name in enumerate(inputs):
            fz.write(f"var input {name} {mins[j]} {maxs[j]}\n")
        fz.write(f"var output {output} {mins[-1]} {maxs[-1]}\n")
        # mfs
        for name, mfs in mf_defs.items():
            for label, (a, b, c) in mfs:
                fz.write(f"mf {name} {label} tri {a} {b} {c}\n")
        # rules
        for ante, (ov, olab), w in uniq_rules:
            conds = " AND ".join(f"{vn} is {lbl}" for vn, lbl in ante)
            if w is not None:
                fz.write(f"rule IF {conds} THEN {ov} is {olab} weight {w}\n")
            else:
                fz.write(f"rule IF {conds} THEN {ov} is {olab}\n")

        # operators & settings
        fz.write(f"tnorm {args.tnorm}\n")
        fz.write(f"snorm {args.snorm}\n")
        fz.write(f"mode {args.mode}\n")
        fz.write("defuzz centroid\n") 

    print(f"Saved model to {out}")

def main():
    ap = argparse.ArgumentParser(prog="mamdani", description="Mamdani fuzzy CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # validate
    sp_v = sub.add_parser("validate", help="Sprawdz i opisz model .fz")
    sp_v.add_argument("--model", required=True)
    sp_v.set_defaults(func=cmd_validate)

    # show
    sp_s = sub.add_parser("show", help="Wypisz szczegoly modelu .fz")
    sp_s.add_argument("--model", required=True)
    sp_s.add_argument("--at", nargs="*", help="opcjonalnie: key=value do pokolorowania dopasowan MF")
    sp_s.set_defaults(func=cmd_show)


    # predict
    sp_p = sub.add_parser("predict", help="Policz wyjscia dla wejsci key=value")
    sp_p.add_argument("--model", required=True)
    sp_p.add_argument("kv", nargs="+", help="key=value, np. temperature=28 humidity=70")
    sp_p.set_defaults(func=cmd_predict)

    # learn (NOWE)
    sp_l = sub.add_parser("learn", help="Ucz system z CSV i zapisz .fz (grid + Wang–Mendel)")
    sp_l.add_argument("--csv", required=True, help="sciezka do pliku CSV (ostatnia kolumna = wyjscie)")
    sp_l.add_argument("--out", required=True, help="sciezka wyjsciowa .fz")
    sp_l.add_argument("--terms", type=int, default=3, help="liczba MF na zmienna (trojkaty)")
    sp_l.add_argument("--partition", choices=["grid"], default="grid", help="rodzaj podzialu (MVP: grid)")
    sp_l.add_argument("--induction", choices=["wm"], default="wm", help="algorytm indukcji (MVP: Wang–Mendel)")
    sp_p.add_argument("--explain", action="store_true", help="wypisz alfy reguł (aktywacje)")
    sp_l.add_argument("--mode", default="FIT")
    sp_l.add_argument("--tnorm", default="min")
    sp_l.add_argument("--snorm", default="max")
    sp_l.add_argument("--defuzz", nargs="+", default=["centroid","0","1","101"],
                      help='np. centroid 0 30 201')
    sp_l.set_defaults(func=cmd_learn)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
