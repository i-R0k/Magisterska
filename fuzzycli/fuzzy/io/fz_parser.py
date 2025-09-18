"""
Gramatyka (skrót):
  var (input|output) <name> <vmin> <vmax>
  mf  <var> <label> (tri a b c | trap a b c d | gauss mu sigma)
  rule IF <v> is <L> (AND <v> is <L>)* THEN <ovar> is <OL> [weight w]
  tnorm <min|prod>          # rozszerzaj wg potrzeb
  snorm <max|sum|bsum>      # rozszerzaj wg potrzeb
  mode  <FIT|FATI>
  defuzz <centroid|mom|bisector> [grid ymin ymax n | n N]
  dtype <...>               # ignorowane na razie
  aggregation ...           # ignorowane (MVP)
  implication ...           # ignorowane (MVP)
  schema <int>              # opcjonalna wersja formatu

Uwagi:
- Słowa kluczowe bezwzględnie case-insensitive; nazwy zmiennych i etykiet MF – case-sensitive.
- Reguły walidowane PO wczytaniu całej KB (sprawdzamy istnienie zmiennych i etykiet).
- Defuzz grid/n stosowane po wczytaniu wszystkich outputs (niezależnie od kolejności dyrektyw).
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import shlex

from ..core.mfs import Triangular, Trapezoidal, Gaussian
from ..core.rule import Rule
from ..model.variable import InputVariable, OutputVariable
from ..model.knowledge import KnowledgeBase


class FZParseError(Exception):
    def __init__(self, msg: str, line: int, content: str):
        super().__init__(f"[.fz:{line}] {msg}\n  >> {content}")


# mapy do walidacji/normalizacji
_ALLOWED_TNORMS = {"min", "prod", "lukasiewicz", "hamacher"}
_ALLOWED_SNORMS = {"max", "sum", "bsum", "prob", "lukasiewicz", "hamacher"}
_ALLOWED_DEFUZZ = {"centroid", "mom", "bisector", "centroid_adaptive"}
_SHAPES = {"tri", "trap", "gauss"}


def _lex_line(raw: str) -> List[str]:
    """Tokenizuj linię: wspiera komentarze '#' i cudzysłowy."""
    lx = shlex.shlex(raw, posix=True)
    lx.whitespace_split = True
    lx.commenters = "#"
    try:
        return list(lx)
    except ValueError:
        # np. niezamknięty cudzysłów – oddamy puste, wyłapie to logika wyżej
        return []


def parse_fz(path: str) -> KnowledgeBase:
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    return parse_fz_string(src)


def parse_fz_string(source: str) -> KnowledgeBase:
    kb = KnowledgeBase()
    lines = source.splitlines()

    # do walidacji po wszystkim
    pending_rules: List[Tuple[int, str, List[Tuple[str, str]], Tuple[str, str], float]] = []
    # defuzz config (stosujemy na końcu)
    defuzz_method: Optional[str] = None
    defuzz_grid: Optional[Tuple[float, float, int]] = None
    schema_version: int = 1

    for lineno, raw in enumerate(lines, 1):
        tokens = _lex_line(raw)
        if not tokens:
            continue
        head = tokens[0].lower()

        try:
            if head == "schema":
                schema_version = int(tokens[1])

            elif head == "var":
                # var input|output name vmin vmax
                if len(tokens) < 5:
                    raise FZParseError("var: oczekiwano: var (input|output) <name> <vmin> <vmax>", lineno, raw)
                kind = tokens[1].lower()
                name = tokens[2]
                vmin = float(tokens[3]); vmax = float(tokens[4])
                if vmin >= vmax:
                    raise FZParseError(f"var: vmin < vmax wymagane (dostałem {vmin} >= {vmax})", lineno, raw)
                if kind == "input":
                    if name in kb.inputs or name in kb.outputs:
                        raise FZParseError(f"Duplikat zmiennej: {name}", lineno, raw)
                    kb.add_input(InputVariable(name, vmin, vmax))
                elif kind == "output":
                    if name in kb.inputs or name in kb.outputs:
                        raise FZParseError(f"Duplikat zmiennej: {name}", lineno, raw)
                    kb.add_output(OutputVariable(name, vmin, vmax))
                else:
                    raise FZParseError(f"Unknown var kind: {kind}", lineno, raw)

            elif head == "mf":
                # mf vname label shape ...
                if len(tokens) < 5:
                    raise FZParseError("mf: oczekiwano: mf <var> <label> <shape> [params...]", lineno, raw)
                vname, label, shape = tokens[1], tokens[2], tokens[3].lower()
                target = kb.inputs.get(vname) or kb.outputs.get(vname)
                if target is None:
                    raise FZParseError(f"MF dla nieznanej zmiennej: {vname}", lineno, raw)
                if label in getattr(target, "terms", {}):
                    raise FZParseError(f"Duplikat etykiety MF '{label}' w zmiennej '{vname}'", lineno, raw)
                if shape not in _SHAPES:
                    raise FZParseError(f"Unknown MF shape: {shape}", lineno, raw)

                if shape == "tri":
                    if len(tokens) < 7:
                        raise FZParseError("tri: oczekiwano 3 parametrów: a b c", lineno, raw)
                    a, b, c = map(float, tokens[4:7])
                    if not (a <= b <= c):
                        raise FZParseError("tri: wymagane a <= b <= c", lineno, raw)
                    target.add_term(label, Triangular(a, b, c))
                elif shape == "trap":
                    if len(tokens) < 8:
                        raise FZParseError("trap: oczekiwano 4 parametrów: a b c d", lineno, raw)
                    a, b, c, d = map(float, tokens[4:8])
                    if not (a <= b <= c <= d):
                        raise FZParseError("trap: wymagane a <= b <= c <= d", lineno, raw)
                    target.add_term(label, Trapezoidal(a, b, c, d))
                elif shape == "gauss":
                    if len(tokens) < 6:
                        raise FZParseError("gauss: oczekiwano 2 parametrów: mu sigma", lineno, raw)
                    mu, sigma = map(float, tokens[4:6])
                    if sigma <= 0:
                        raise FZParseError("gauss: sigma > 0 wymagane", lineno, raw)
                    target.add_term(label, Gaussian(mu, sigma))

            elif head == "rule":
                # rule IF ... THEN ...
                words = tokens[1:]
                # znajdź THEN (case-insensitive)
                try:
                    then_idx = next(i for i, t in enumerate(words) if t.lower() == "then")
                except StopIteration:
                    raise FZParseError("Rule missing THEN", lineno, raw)
                if not words or words[0].lower() != "if":
                    raise FZParseError("Rule must start with IF", lineno, raw)

                cond = words[1:then_idx]
                cons = words[then_idx + 1:]

                # antecedent: v is L [AND v is L]*
                ante: List[Tuple[str, str]] = []
                i = 0
                while i < len(cond):
                    if i + 2 >= len(cond):
                        raise FZParseError("Antecedent: oczekiwano '<var> is <label>'", lineno, raw)
                    vname = cond[i]; 
                    if cond[i + 1].lower() != "is":
                        raise FZParseError("Antecedent: spodziewano 'is'", lineno, raw)
                    label = cond[i + 2]
                    ante.append((vname, label))
                    i += 3
                    if i < len(cond):
                        if cond[i].lower() == "and":
                            i += 1
                        else:
                            raise FZParseError("Antecedent: spodziewano 'AND' lub koniec", lineno, raw)

                # consequent: <ovar> is <olabel> [weight w] [inactive]
                if len(cons) < 3 or cons[1].lower() != 'is':
                    raise FZParseError("Consequent: oczekiwano '<ovar> is <label>'", lineno, raw)
                oname, olabel = cons[0], cons[2]
                weight = 1.0
                inactive = False
                i = 3
                while i < len(cons):
                    tok = cons[i].lower()
                    if tok == 'weight':
                        if i + 1 >= len(cons):
                            raise FZParseError("Consequent: oczekiwano 'weight <w>'", lineno, raw)
                        weight = float(cons[i+1]); i += 2
                    elif tok == 'inactive':
                        inactive = True; i += 1
                    else:
                        raise FZParseError(f"Consequent: nieznana opcja '{cons[i]}'", lineno, raw)

                pending_rules.append((lineno, raw, ante, (oname, olabel), weight, not inactive))

            elif head == "tnorm":
                if len(tokens) < 2:
                    raise FZParseError("tnorm: podaj nazwę (np. min|prod)", lineno, raw)
                name = tokens[1].lower()
                if name not in _ALLOWED_TNORMS:
                    raise FZParseError(f"tnorm: nieobsługiwane '{name}'", lineno, raw)
                kb.tnorm = name

            elif head == "snorm":
                if len(tokens) < 2:
                    raise FZParseError("snorm: podaj nazwę (np. max|sum|bsum)", lineno, raw)
                name = tokens[1].lower()
                if name not in _ALLOWED_SNORMS:
                    raise FZParseError(f"snorm: nieobsługiwane '{name}'", lineno, raw)
                kb.snorm = name

            elif head == "mode":
                if len(tokens) < 2:
                    raise FZParseError("mode: FIT|FATI", lineno, raw)
                kb.mode = tokens[1].upper()
                if kb.mode not in ("FIT", "FATI"):
                    raise FZParseError("mode: dozwolone FIT|FATI", lineno, raw)

            elif head == "defuzz":
                if len(tokens) < 2:
                    raise FZParseError("defuzz: podaj metodę (centroid|mom|bisector|centroid_adaptive)", lineno, raw)
                method = tokens[1].lower()
                if method not in _ALLOWED_DEFUZZ:
                    raise FZParseError("Supported defuzz: centroid | mom | bisector | centroid_adaptive", lineno, raw)
                defuzz_method = method
                # dodatkowe opcje
                if len(tokens) >= 3 and tokens[2].lower() == "grid":
                    if len(tokens) < 6:
                        raise FZParseError("defuzz grid: oczekiwano 'grid ymin ymax n'", lineno, raw)
                    ymin = float(tokens[3]); ymax = float(tokens[4]); n = int(tokens[5])
                    if ymax <= ymin or n <= 1:
                        raise FZParseError("defuzz grid: wymagane ymin<ymax, n>1", lineno, raw)
                    defuzz_grid = (ymin, ymax, n)
                elif len(tokens) >= 3 and tokens[2].lower() == "n":
                    if len(tokens) < 4:
                        raise FZParseError("defuzz n: oczekiwano 'n N'", lineno, raw)
                    n = int(tokens[3])
                    if n <= 1:
                        raise FZParseError("defuzz n: N>1 wymagane", lineno, raw)
                    # zastosujemy n na końcu – jeśli nie podano ymin/ymax, pozostawiamy jak w zmiennej
                    # (jeśli zmienna nie ma gridu, engine wybierze auto-range)
                    # tu tylko zapamiętujemy „zmień n”
                    # zrobimy to po utworzeniu outputs.
                    defuzz_grid = ("KEEP", "KEEP", n)  # sygnał: zmień tylko n
                # brak dodatkowych – engine może ustawić auto-range

            elif head in ("dtype", "aggregation", "implication"):
                # na razie ignorujemy – zarezerwowane na przyszłość
                continue

            else:
                raise FZParseError(f"Unknown directive: {tokens[0]}", lineno, raw)

        except FZParseError:
            raise
        except Exception as e:
            # opakuj każdy błąd w FZParseError z kontekstem
            raise FZParseError(str(e), lineno, raw) from e

    # Domyślne parametry jeśli nie ustawiono
    if not getattr(kb, "tnorm", None):
        kb.tnorm = "min"
    if not getattr(kb, "snorm", None):
        kb.snorm = "max"
    if not getattr(kb, "mode", None):
        kb.mode = "FIT"
    if not getattr(kb, "defuzz", None):
        kb.defuzz = "centroid"

    # Zastosuj defuzz grid/n do wszystkich zmiennych wyjściowych
    if defuzz_method:
        kb.defuzz = defuzz_method
    if defuzz_grid is not None:
        for ov in kb.outputs.values():
            if defuzz_grid[0] == "KEEP":
                # zmień tylko n, zachowując dotychczasowy zakres
                ymin, ymax, _n = getattr(ov, "grid", (ov.vmin, ov.vmax, 101))
                ov.grid = (ymin, ymax, int(defuzz_grid[2]))
            else:
                ov.grid = (float(defuzz_grid[0]), float(defuzz_grid[1]), int(defuzz_grid[2]))

    # Walidacja reguł i dopisanie do KB
    if not kb.outputs:
        raise FZParseError("Brak zmiennych wyjściowych (var output ...)", lineno=len(lines), content="<eof>")

    for (rlineno, rraw, ante, (oname, olabel), w, active) in pending_rules:
        if oname not in kb.outputs:
            raise FZParseError(f"Rule: nieznana zmienna wyjściowa '{oname}'", rlineno, rraw)
        if olabel not in kb.outputs[oname].terms:
            raise FZParseError(f"Rule: nieznana etykieta wyjścia '{oname}.{olabel}'", rlineno, rraw)
        for v, lab in ante:
            vobj = kb.inputs.get(v) or kb.outputs.get(v)
            if vobj is None:
                raise FZParseError(f"Rule: nieznana zmienna '{v}'", rlineno, rraw)
            if v in kb.outputs and v != oname:
                # dopuszczasz warunki na output? zwykle nie. Tu blokujemy.
                raise FZParseError(f"Rule: antecedent na wyjściu '{v}' nie jest dozwolony", rlineno, rraw)
            if lab not in vobj.terms:
                raise FZParseError(f"Rule: nieznana etykieta '{v}.{lab}'", rlineno, rraw)

        kb.add_rule(Rule(antecedent=ante, consequent=(oname, olabel), weight=w, active=active))
    # (opcjonalnie) zachowaj wersję schematu w KB
    kb.schema_version = schema_version
    return kb
