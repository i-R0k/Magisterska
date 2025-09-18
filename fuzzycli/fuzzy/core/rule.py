from dataclasses import dataclass
from typing import List, Tuple
from ..core.types import Float

Antecedent = List[Tuple[str, str]]

@dataclass
class Rule:
    antecedent: Antecedent
    consequent: Tuple[str, str]  # (output_var, label)
    weight: Float = 1.0
    active: bool = True
