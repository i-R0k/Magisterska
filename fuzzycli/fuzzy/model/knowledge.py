# Baza wiedzy (zmienne + reguły + ustawienia)

from dataclasses import dataclass, field
from typing import Dict, List
from .variable import InputVariable, OutputVariable
from ..core.rule import Rule

@dataclass
class KnowledgeBase:
    inputs: Dict[str, InputVariable] = field(default_factory=dict)
    outputs: Dict[str, OutputVariable] = field(default_factory=dict)
    rules: List[Rule] = field(default_factory=list)
    tnorm: str = "min"
    snorm: str = "max"
    mode: str = "FIT"  # or FATI
    defuzz: str = "centroid"  # only centroid supported in MVP

    def add_input(self, var: InputVariable) -> None:
        self.inputs[var.name] = var
    def add_output(self, var: OutputVariable) -> None:
        self.outputs[var.name] = var
    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)
