# InputVariable/OutputVariable, zakresy, siatki

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple
from ..core.mfs import MembershipFunction
from ..core.types import Float

@dataclass
class InputVariable:
    name: str
    vmin: Float
    vmax: Float
    terms: Dict[str, MembershipFunction] = field(default_factory=dict)

    def add_term(self, label: str, mf: MembershipFunction) -> None:
        self.terms[label] = mf

    def clamp(self, x: Float) -> Float:
        return max(self.vmin, min(self.vmax, x))

@dataclass
class OutputVariable:
    name: str
    vmin: Float
    vmax: Float
    terms: Dict[str, MembershipFunction] = field(default_factory=dict)
    grid: Tuple[Float, Float, int] = (0.0, 1.0, 101)  # ymin, ymax, n

    def add_term(self, label: str, mf: MembershipFunction) -> None:
        self.terms[label] = mf
