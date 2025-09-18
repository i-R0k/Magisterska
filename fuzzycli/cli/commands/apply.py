import csv
from typing import List, Dict, Union
from ..argtypes import ENC_CHOICES, MODE_CHOICES
from ...fuzzy.io.fz_parser import parse_fz
from ...fuzzy.model.classifier import Classifier

def _is_float_cell(s: str) -> bool:
    try:
        float(s); return True
    except Exception:
        return False

def _parse_mapping_arg(maparg: str) -> Dict[str, Union[int, str]]:
    out = {}
    if not maparg:
        return out
    for part in maparg.split(","):
        if "=" not in part:
            continue
        k, v = (t.strip() for t in part.split("=", 1))
        try:
            out[k] = int(v)
        except ValueError:
            out[k] = v
    return out

def _parse_cols_list(spec: str):
    if not spec: return []
    items = []
    for tok in spec.split(","):
        t = tok.strip()
        if not t: continue
        try:
            items.append(int(t))
        except ValueError:
            items.append(t)
    return items

def _resolve_names_to_indices(names_or_indices, colnames):
    idxs = []
    for spec in names_or_indices:
        if isinstance(spec, int):
            idxs.append(spec)
        else:
            try:
                idxs.append(colnames.index(spec))
            except ValueError:
                raise SystemExit(f"Kolumna '{spec}' nie istnieje w CSV (kolumny: {colnames}).")
    return idxs

def cmd_apply(args):
    """
    Zastosuj model (batch classify) do CSV, z elastycznym mapowaniem kolumn.
    Obsługuje: --map / --in-cols + --ignore-cols, --inputs, --encoding, --mode, --out
    """
    kb = parse_fz(args.model)
    clf = Classifier(kb)

    out_path = getattr(args, "out", None)
    writer = None
    out_f = None

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            raise SystemExit("Pusty plik CSV.")

        # --- wykrycie nagłówka ---
        header_mode = any(not _is_float_cell(c) for c in first)
        if header_mode:
            colnames = [c.strip() for c in first]   # reader stoi już na 1. wierszu danych
        else:
            colnames = [f"c{i}" for i in range(len(first))]
            f.seek(0)                                # czytaj od początku z TEGO SAMEGO pliku
            reader = csv.reader(f)

        # --- mapowanie kolumn ---
        mapping = _parse_mapping_arg(getattr(args, "map", "") or "")
        if mapping and header_mode:
            for vn, spec in list(mapping.items()):
                if isinstance(spec, str):
                    try:
                        mapping[vn] = colnames.index(spec)
                    except ValueError:
                        raise SystemExit(f"Kolumna '{spec}' (dla {vn}) nie istnieje (dostępne: {colnames})")

        in_cols_spec = _parse_cols_list(getattr(args, "in_cols", "") or "")
        ignore_spec  = _parse_cols_list(getattr(args, "ignore_cols", "") or "")
        ignore_idxs: List[int] = _resolve_names_to_indices(ignore_spec, colnames) if ignore_spec else []

        if not mapping and in_cols_spec:
            selected_idxs = _resolve_names_to_indices(in_cols_spec, colnames)
            selected_idxs = [i for i in selected_idxs if i not in ignore_idxs]
            model_inputs = list(kb.inputs.keys())
            if getattr(args, "inputs", None) is not None:
                model_inputs = model_inputs[:int(args.inputs)]
            if len(selected_idxs) != len(model_inputs):
                raise SystemExit(f"--in-cols ma {len(selected_idxs)} kolumn, a model oczekuje {len(model_inputs)} wejść.")
            mapping = {vn: idx for vn, idx in zip(model_inputs, selected_idxs)}

        if not mapping and header_mode:
            auto = {vn: colnames.index(vn) for vn in kb.inputs.keys() if vn in colnames}
            if auto:
                mapping = auto

        if not mapping:
            model_inputs = list(kb.inputs.keys())
            total_cols = len(colnames)
            candidates = [i for i in range(total_cols) if i not in ignore_idxs]
            if len(candidates) < len(model_inputs):
                raise SystemExit(f"Za mało kolumn: dostępne={len(candidates)}, potrzebne={len(model_inputs)}.")
            mapping = {vn: candidates[i] for i, vn in enumerate(model_inputs)}

        print("[apply] Mapowanie var->kolumna:")
        for vn, idx in mapping.items():
            label = colnames[idx] if 0 <= idx < len(colnames) else f"c{idx}"
            print(f"  {vn} <- [{idx}] {label}")
        if ignore_idxs:
            print("[apply] Ignorowane kolumny:", [f"[{i}] {colnames[i]}" for i in ignore_idxs])

        # --- przygotuj writer ---
        if out_path:
            out_f = open(out_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(out_f)

        if not kb.outputs:
            raise SystemExit("Model nie ma zdefiniowanej zmiennej wyjściowej.")
        oname = next(iter(kb.outputs))
        class_labels = list(kb.outputs[oname].terms.keys())  # np. ["setosa","versicolor","virginica"]

        # --- nagłówek wynikowy ---
        encoding = getattr(args, "encoding", "label")
        if encoding == "decimal":
            header_out = ["_pred_decimal"]
        elif encoding == "label":
            header_out = ["_pred_label"]
        else:  # binary one-hot
            header_out = [f"_pred_{lbl}" for lbl in class_labels]
        header_out += [f"_score_{lbl}" for lbl in class_labels]

        if writer:
            writer.writerow(header_out)

        # --- pętla po wierszach wejściowych ---
        for row in reader:
            if not row:
                continue

            # wejścia
            data = {}
            for vn, col_idx in mapping.items():
                val = row[col_idx]
                data[vn] = float(val) if val != "" else 0.0

            # klasyfikacja
            mode = getattr(args, "mode", None) or getattr(kb, "mode", "FIT")
            cres = clf.classify(data, mode=mode)

            # wynik dla bieżącego wyjścia
            cres_on = cres.get(oname, {}) if isinstance(cres, dict) else {}
            chosen = cres_on.get("chosen", None)
            strengths = cres_on.get("strengths", {}) or {}

            # kodowanie predykcji
            if encoding == "decimal":
                outrow_pred = [""] if chosen is None else [int(class_labels.index(chosen))]
            elif encoding == "label":
                outrow_pred = [chosen if chosen is not None else ""]
            else:  # binary one-hot
                outrow_pred = [1 if (lbl == chosen) else 0 for lbl in class_labels]

            # siły klas
            outrow_scores = [float(strengths.get(lbl, 0.0)) for lbl in class_labels]
            outrow = outrow_pred + outrow_scores

            if writer:
                writer.writerow(outrow)
            else:
                print(",".join(str(x) for x in outrow))

    # <- KONIEC with (plik wejściowy zamknięty tutaj)

    if out_f:
        out_f.close()
        print(f"[apply] Wyniki zapisane do {out_path}")

