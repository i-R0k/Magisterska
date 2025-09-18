from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List
from .types import Float

def _clamp01(x: Float) -> Float:
    if x <= 0.0:
        return 0.0
    elif x >= 1.0:
        return 1.0
    else:
        return x

class MembershipFunction:
    def mu(self, x: Float) -> Float:
        raise NotImplementedError
    def support(self) -> tuple[Float, Float]:
        raise NotImplementedError

@dataclass(frozen=True)
class Triangular(MembershipFunction):
    a: Float; b: Float; c: Float
    def mu(self, x: Float) -> Float:
        if x <= self.a or x >= self.c: return 0.0
        if x == self.b: return 1.0
        if x < self.b:  return (x - self.a) / (self.b - self.a or 1e-12)
        return (self.c - x) / (self.c - self.b or 1e-12)
    def support(self) -> tuple[Float, Float]:
        return (self.a, self.c)

@dataclass(frozen=True)
class Trapezoidal(MembershipFunction):
    a: Float; b: Float; c: Float; d: Float
    def mu(self, x: Float) -> Float:
        if x <= self.a or x >= self.d: return 0.0
        if self.b <= x <= self.c: return 1.0
        if self.a < x < self.b: return (x - self.a) / (self.b - self.a or 1e-12)
        return (self.d - x) / (self.d - self.c or 1e-12)
    def support(self) -> tuple[Float, Float]:
        return (self.a, self.d)

@dataclass(frozen=True)
class Gaussian(MembershipFunction):
    mu0: Float; sigma: Float
    def mu(self, x: Float) -> Float:
        z = (x - self.mu0) / (self.sigma or 1e-12)
        return float(__import__("math").exp(-0.5 * z * z))
    def support(self) -> tuple[Float, Float]:
        s = 4.0 * self.sigma
        return (self.mu0 - s, self.mu0 + s)
