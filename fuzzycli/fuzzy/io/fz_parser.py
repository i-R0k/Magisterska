# Parser naszego pliku `.fz` (mini-DSL)

from __future__ import annotations
from typing import List
from ..core.mfs import Triangular, Trapezoidal, Gaussian
from ..core.rule import Rule
from ..model.variable import InputVariable, OutputVariable
from ..model.knowledge import KnowledgeBase

def parse_fz(path: str) -> KnowledgeBase:
    kb = KnowledgeBase()
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f]
    # temp storage to attach MF to variables later
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        tok = ln.split()
        head = tok[0].lower()

        if head == "var":
            kind = tok[1].lower()
            name = tok[2]
            vmin = float(tok[3]); vmax = float(tok[4])
            if kind == "input":
                kb.add_input(InputVariable(name, vmin, vmax))
            elif kind == "output":
                kb.add_output(OutputVariable(name, vmin, vmax))
            else:
                raise ValueError(f"Unknown var kind: {kind}")

        elif head == "mf":
            vname = tok[1]; label = tok[2]; shape = tok[3].lower()
            if vname in kb.inputs:
                target = kb.inputs[vname]
            elif vname in kb.outputs:
                target = kb.outputs[vname]
            else:
                raise ValueError(f"MF refers to unknown variable: {vname}")
            if shape == "tri":
                a,b,c = map(float, tok[4:7])
                target.add_term(label, Triangular(a,b,c))
            elif shape == "trap":
                a,b,c,d = map(float, tok[4:8])
                target.add_term(label, Trapezoidal(a,b,c,d))
            elif shape == "gauss":
                mu,sigma = map(float, tok[4:6])
                target.add_term(label, Gaussian(mu,sigma))
            else:
                raise ValueError(f"Unknown MF shape: {shape}")

        elif head == "rule":
            # rule IF x is A AND y is B THEN z is C [weight w]
            # simple tokenizer based on keywords
            words = tok[1:]
            # split IF ... THEN ...
            try:
                then_idx = words.index("THEN")
            except ValueError:
                then_idx = words.index("Then") if "Then" in words else -1
            if then_idx == -1:
                raise ValueError("Rule missing THEN")
            if words[0] not in ("IF","If"):
                raise ValueError("Rule must start with IF")
            cond = words[1:then_idx]
            cons = words[then_idx+1:]
            # antecedent pairs (var is label [AND ...])
            ante = []
            i = 0
            while i < len(cond):
                vname = cond[i]; assert cond[i+1].lower()=="is"
                label = cond[i+2]; ante.append((vname,label))
                i += 3
                if i < len(cond) and cond[i].upper() == "AND":
                    i += 1
            # consequent: <ovar> is <olabel> [weight w]
            oname = cons[0]; assert cons[1].lower()=="is"; olabel = cons[2]
            w = 1.0
            if len(cons) > 3:
                if cons[3].lower() == "weight":
                    w = float(cons[4])
            kb.add_rule(Rule(antecedent=ante, consequent=(oname, olabel), weight=w))

        elif head == "tnorm":
            kb.tnorm = tok[1].lower()
        elif head == "snorm":
            kb.snorm = tok[1].lower()
        elif head == "mode":
            kb.mode = tok[1].upper()
        elif head == "defuzz":
            if tok[1].lower() != "centroid":
                raise ValueError("Only centroid supported in MVP")
            if len(tok) >= 3 and tok[2].lower() == "grid":
                # defuzz centroid grid ymin ymax n
                ymin = float(tok[3]); ymax = float(tok[4]); n = int(tok[5])
                for ov in kb.outputs.values():
                    ov.grid = (ymin, ymax, n)
            elif len(tok) >= 3 and tok[2].lower() == "n":
                # defuzz centroid n N  -> tylko liczba punktow siatki,
                # zakres zostanie dobrany automatycznie w engine
                n = int(tok[3])
                for ov in kb.outputs.values():
                    ymin, ymax, _ = ov.grid
                    ov.grid = (ymin, ymax, n)
            else:
                # defuzz centroid -> tryb automatyczny
                # engine sam wybierze zakres [ymin,ymax] z MF wyjscia,
                # a n ustawi na wartosc domyslna (np. 201)
                pass
        elif head == "dtype":
            pass  # placeholder; single-precision handling can be added later
        elif head in ("aggregation", "implication"):
            # acknowledged but not stored separately (Mamdani/min, aggregation=max in MVP)
            pass
        else:
            raise ValueError(f"Unknown directive: {head}")

    return kb
