import os
import csv
import json
from typing import Dict, Sequence, Union, List, Any


def _is_float_cell(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _parse_cols_list(spec: Union[str, Sequence[str], None]) -> List[Union[int, str]]:
    """
    Parsuje listę kolumn z postaci:
      - "A,B,3"
      - ["A", "B", "3"]
      - ["A,B", "3"]
    Zwraca listę, gdzie czysto liczbowe tokeny są zamieniane na int.
    """
    if not spec:
        return []

    tokens: List[str] = []
    if isinstance(spec, (list, tuple)):
        for elem in spec:
            if elem is None:
                continue
            for tok in str(elem).split(","):
                t = tok.strip()
                if t:
                    tokens.append(t)
    else:
        for tok in str(spec).split(","):
            t = tok.strip()
            if t:
                tokens.append(t)

    out: List[Union[int, str]] = []
    for t in tokens:
        if t.lstrip("-").isdigit():
            try:
                out.append(int(t))
                continue
            except ValueError:
                pass
        out.append(t)
    return out


def _resolve_names_to_indices(names_or_indices, colnames: List[str]) -> List[int]:
    idxs: List[int] = []
    for spec in names_or_indices:
        if isinstance(spec, int):
            idxs.append(spec)
        else:
            try:
                idxs.append(colnames.index(spec))
            except ValueError:
                raise SystemExit(f"Kolumna '{spec}' nie istnieje w CSV (kolumny: {colnames}).")
    return idxs


def cmd_prepare(args):
    """
    Przygotowanie danych:
      - --csv           : wejściowy plik CSV
      - --in-cols       : lista kolumn wejściowych (nazwy lub indeksy)
      - --out-col       : kolumna wyjściowa (nazwa lub indeks)
      - --num-cols      : kolumny liczbowe (opcjonalna walidacja)
      - --str-cols      : kolumny tekstowe (label-encoding)
      - --ignore-cols   : kolumny do pominięcia (walidacja konfliktów)
      - --out           : wynikowy numeryczny CSV (wejścia + wyjście)
      - --mapping       : JSON z metadanymi i mapą etykiet
    """
    # --- 1) Wczytaj nagłówek + wszystkie wiersze do pamięci (bez zamykania readera w trakcie) ---
    with open(args.csv, newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        first = next(rdr, None)
        if first is None:
            raise SystemExit("Pusty plik CSV.")

        header_mode = any(not _is_float_cell(c) for c in first)
        if header_mode:
            colnames = [c.strip() for c in first]
            data_rows = list(rdr)  # reszta to dane
        else:
            colnames = [f"c{i}" for i in range(len(first))]
            data_rows = [first] + list(rdr)  # pierwszy wiersz też jest danymi

    # --- 2) Parsowanie i rozwiązywanie kolumn ---
    in_cols_spec = _parse_cols_list(getattr(args, "in_cols", None))
    out_col_spec = getattr(args, "out_col", None)
    num_cols_spec = _parse_cols_list(getattr(args, "num_cols", None))
    str_cols_spec = _parse_cols_list(getattr(args, "str_cols", None))
    ign_cols_spec = _parse_cols_list(getattr(args, "ignore_cols", None))

    if out_col_spec is None:
        raise SystemExit("--out-col jest wymagane.")

    in_idxs = _resolve_names_to_indices(in_cols_spec, colnames)
    out_idx = _resolve_names_to_indices([out_col_spec], colnames)[0]
    num_idxs = _resolve_names_to_indices(num_cols_spec, colnames) if num_cols_spec else []
    str_idxs = _resolve_names_to_indices(str_cols_spec, colnames) if str_cols_spec else []
    ign_idxs = _resolve_names_to_indices(ign_cols_spec, colnames) if ign_cols_spec else []

    # Walidacje ról
    if out_idx in ign_idxs:
        raise SystemExit("Wyjściowa kolumna nie może być w ignore.")
    for i in in_idxs:
        if i in ign_idxs:
            raise SystemExit(f"Kolumna wejściowa [{i}] jest w ignore.")
    if out_idx in in_idxs:
        raise SystemExit("Ta sama kolumna nie może być jednocześnie wejściem i wyjściem.")

    # --- 3) Auto-wykrywanie tekstowych (jeśli nie wskazano) na próbie do 50 rekordów ---
    label_maps: Dict[str, Dict[str, int]] = {}
    auto_str = set(str_idxs)
    if not str_cols_spec:
        probe_rows = data_rows[:50]
        for j in range(len(colnames)):
            if j in ign_idxs:
                continue
            nonnum = 0
            total = 0
            for r in probe_rows:
                if j >= len(r):
                    continue
                total += 1
                if not _is_float_cell(r[j]):
                    nonnum += 1
            if total > 0 and nonnum > total / 2:
                auto_str.add(j)

    # --- 4) Mapa etykiet dla wyjścia (jeśli wyjście jest tekstowe) ---
    if out_idx in auto_str or out_idx in str_idxs:
        label_map: Dict[str, int] = {}
        next_id = 0
        for row in data_rows:
            if out_idx < len(row):
                val = row[out_idx]
                if val not in label_map:
                    label_map[val] = next_id
                    next_id += 1
        label_maps[colnames[out_idx]] = label_map

    # --- 5) Zapis wynikowego CSV ---
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        header_out = [colnames[i] for i in in_idxs] + [colnames[out_idx]]
        w.writerow(header_out)

        for row in data_rows:
            if not row:
                continue
            newrow: List[Any] = []

            # wejścia
            for i in in_idxs:
                cell = row[i] if i < len(row) else ""
                try:
                    newrow.append(float(cell))
                except Exception:
                    newrow.append(float("nan"))

            # wyjście
            if colnames[out_idx] in label_maps:
                val = row[out_idx] if out_idx < len(row) else ""
                # na wszelki wypadek dodaj niewidziane etykiety
                lm = label_maps[colnames[out_idx]]
                if val not in lm:
                    lm[val] = max(lm.values(), default=-1) + 1
                newrow.append(lm[val])
            else:
                cell = row[out_idx] if out_idx < len(row) else ""
                try:
                    newrow.append(float(cell))
                except Exception:
                    newrow.append(float("nan"))

            w.writerow(newrow)

    # --- 6) Zapis metadanych/mappingu ---
    meta = {
        "inputs": [colnames[i] for i in in_idxs],
        "output": colnames[out_idx],
        "ignored": [colnames[i] for i in ign_idxs],
        "label_maps": label_maps,
        "source_csv": args.csv,
        "header_mode": header_mode,
    }

    mapping_dir = os.path.dirname(args.mapping)
    if mapping_dir:
        os.makedirs(mapping_dir, exist_ok=True)

    with open(args.mapping, "w", encoding="utf-8") as jm:
        json.dump(meta, jm, ensure_ascii=False, indent=2)

    print(f"[prepare] Zapisano: {args.out}")
    print(f"[prepare] Mapping:  {args.mapping}")
