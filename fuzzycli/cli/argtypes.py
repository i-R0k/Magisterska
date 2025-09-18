import argparse

MODE_CHOICES = ["FIT", "FATI"]
ENC_CHOICES  = ["decimal", "binary", "label"]
PARTITION_CHOICES = ["grid"]
INDUCTION_CHOICES = ["wm"]
TNORM_DEFAULT = "min"
SNORM_DEFAULT = "max"

def parse_cols_list(s: str):
    """'1,2,SepalWidthCm' -> [1,2,'SepalWidthCm'] (int dla cyfr, str dla nazw)."""
    if not s:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok) if tok.isdigit() else tok)
    return out

def parse_var_col_map(s: str):
    """'var=kolumna,...' (kolumna: indeks lub nazwa)."""
    if not s:
        return {}
    m = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"Niepoprawny element: '{pair}' (oczekiwano 'var=kolumna').")
        k, v = (t.strip() for t in pair.split("=", 1))
        if not k or not v:
            raise argparse.ArgumentTypeError(f"Pusty klucz lub wartość w: '{pair}'.")
        m[k] = int(v) if v.isdigit() else v
    return m
