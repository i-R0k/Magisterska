from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

from .variable import InputVariable, OutputVariable
from ..core.rule import Rule


@dataclass
class ColumnSpec:
    """
    Opis jednej kolumny danych:
      - dtype: 'num' (liczbowa) | 'str' (tekstowa)
      - role:  'in' (wejście) | 'out' (wyjście) | 'ignore' (pomijana)
      - encode: sposób kodowania tekstu ('label' | 'onehot' | 'binary' | None)
      - scale:  skalowanie liczb ('none' | 'minmax' | 'zscore' | None)
    """
    name: str
    dtype: str                    # 'num' | 'str'
    role: str                     # 'in' | 'out' | 'ignore'
    encode: Optional[str] = None  # 'label' | 'onehot' | 'binary' | None
    scale: Optional[str] = None   # 'none' | 'minmax' | 'zscore' | None


@dataclass
class KnowledgeBase:
    # --- zmienne i reguły ---
    inputs: Dict[str, InputVariable] = field(default_factory=dict)
    outputs: Dict[str, OutputVariable] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)

    # --- ustawienia silnika ---
    tnorm: str = "min"
    snorm: str = "max"
    mode: str = "FIT"             # lub 'FATI'
    defuzz: str = "centroid"      # 'centroid' | 'mom' | 'bisector' | 'centroid_adaptive'

    # --- schema & metadata (pipeline danych) ---
    schema_version: int = 1
    # Słownik po nazwie kolumny → specyfikacja kolumny
    schema: Dict[str, ColumnSpec] = field(default_factory=dict)
    # Mapowania etykiet tekstowych → liczby (np. dla 'label' encoding)
    label_mappings: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Parametry skalerów (np. min/max dla minmax, mean/std dla zscore)
    scaler_params: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ---------- metody pomocnicze (KB) ----------
    def add_input(self, var: InputVariable) -> None:
        self.inputs[var.name] = var

    def add_output(self, var: OutputVariable) -> None:
        self.outputs[var.name] = var

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    # ---------- metody pomocnicze (silnik) ----------
    def set_engine(
        self,
        *,
        tnorm: Optional[str] = None,
        snorm: Optional[str] = None,
        mode: Optional[str] = None,
        defuzz: Optional[str] = None,
    ) -> None:
        if tnorm is not None:
            self.tnorm = tnorm
        if snorm is not None:
            self.snorm = snorm
        if mode is not None:
            self.mode = mode
        if defuzz is not None:
            self.defuzz = defuzz

    # ---------- metody pomocnicze (schema) ----------
    def set_schema_from_dict(self, d: Dict[str, Any]) -> None:
        """
        Ustawia schema KB na podstawie:
          A) {'columns': [{name,dtype,role,encode,scale}, ...]}
          lub
          B) map: 'types' / 'roles' / 'encode' / 'scale'
        """
        cols: Dict[str, ColumnSpec] = {}

        # A) bezpośrednia lista kolumn
        if isinstance(d.get("columns"), list):
            for c in d["columns"]:
                cs = ColumnSpec(
                    name=c["name"],
                    dtype=c.get("dtype", "num"),
                    role=c.get("role", "in"),
                    encode=c.get("encode"),
                    scale=c.get("scale"),
                )
                cols[cs.name] = cs
        else:
            # B) mapy: types/roles/encode/scale
            types = d.get("types", {})
            roles = d.get("roles", {})
            enc = d.get("encode", {})
            scl = d.get("scale", {})
            for name, dtype in types.items():
                cs = ColumnSpec(
                    name=name,
                    dtype=dtype,
                    role=roles.get(name, "in"),
                    encode=enc.get(name),
                    scale=scl.get(name),
                )
                cols[name] = cs

        self.schema = cols

    def export_schema(self) -> Dict[str, Any]:
        """Zwraca schema w formacie JSON-owalnym."""
        return {
            "schema_version": self.schema_version,
            "columns": [asdict(cs) for cs in self.schema.values()],
        }

    # ---------- metadane: etykiety i skalery ----------
    def set_label_mapping(self, col: str, mapping: Dict[str, int]) -> None:
        self.label_mappings[col] = dict(mapping)

    def set_scaler_params(self, col: str, params: Dict[str, float]) -> None:
        self.scaler_params[col] = dict(params)
