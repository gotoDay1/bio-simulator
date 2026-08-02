import math
from dataclasses import dataclass

from biosim.models.base import GrowthModel
from biosim.validation import require_positive


@dataclass
class MonodGrowth(GrowthModel):
    """Substrate-limited growth: mu = mu_max * S / (Ks + S).

    mu_max: maximum specific growth rate (1/h), typical range 0.1-1.0
    Ks: half-saturation constant (g/L), typical range 0.01-1.0
    """

    mu_max: float = 0.6
    Ks: float = 0.2
    name = "monod"

    def __post_init__(self) -> None:
        require_positive("mu_max", self.mu_max)
        require_positive("Ks", self.Ks)

    def specific_growth_rate(
        self, X: float, S: float, t: float, C_O2: float | None = None
    ) -> float:
        S = max(S, 0.0)
        return self.mu_max * S / (self.Ks + S)


@dataclass
class LogisticGrowth(GrowthModel):
    """Carrying-capacity limited growth: mu = mu_max * (1 - X / Xmax).

    mu_max: maximum specific growth rate (1/h), typical range 0.1-1.0
    Xmax: carrying capacity / maximum biomass concentration (g/L), typical range 5-50
    """

    mu_max: float = 0.6
    Xmax: float = 10.0
    name = "logistic"

    def __post_init__(self) -> None:
        require_positive("mu_max", self.mu_max)
        require_positive("Xmax", self.Xmax)

    def specific_growth_rate(
        self, X: float, S: float, t: float, C_O2: float | None = None
    ) -> float:
        return self.mu_max * (1.0 - X / self.Xmax)


@dataclass
class GompertzGrowth(GrowthModel):
    """Mechanistic (ODE) Gompertz growth: mu = mu_max * ln(Xmax / X).

    Note: this is the differential/mechanistic form used to plug into an ODE
    mass balance (dX/dt = mu*X = mu_max*X*ln(Xmax/X)), not the closed-form
    regression curve X(t) = Xmax*exp(-exp(...)) sometimes fit to endpoint data.

    mu_max: maximum specific growth rate (1/h), typical range 0.1-1.0
    Xmax: asymptotic maximum biomass concentration (g/L), typical range 5-50
    """

    mu_max: float = 0.6
    Xmax: float = 10.0
    name = "gompertz"

    def __post_init__(self) -> None:
        require_positive("mu_max", self.mu_max)
        require_positive("Xmax", self.Xmax)

    def specific_growth_rate(
        self, X: float, S: float, t: float, C_O2: float | None = None
    ) -> float:
        if X <= 0.0 or X >= self.Xmax:
            return 0.0
        return self.mu_max * math.log(self.Xmax / X)


@dataclass
class MonodGrowthO2(GrowthModel):
    """Dual-substrate Monod growth: mu = mu_max * S/(Ks+S) * C_O2/(Ko2+C_O2).

    Unlike MonodGrowth (substrate-only), this model also gates growth on dissolved
    oxygen availability, so it can represent oxygen-limited growth (e.g. when kLa
    can't keep up with demand). Requires an oxygen model that tracks C_O2 as a state
    (OxygenWithKLa) — BioreactorSimulation rejects pairing this with OxygenDemandOnly.

    mu_max: maximum specific growth rate (1/h), typical range 0.1-1.0
    Ks: half-saturation constant for substrate (g/L), typical range 0.01-1.0
    Ko2: half-saturation constant for dissolved O2 (mg/L), typical range 0.01-0.5
    """

    mu_max: float = 0.6
    Ks: float = 0.2
    Ko2: float = 0.2
    name = "monod_o2"
    requires_oxygen_state = True

    def __post_init__(self) -> None:
        require_positive("mu_max", self.mu_max)
        require_positive("Ks", self.Ks)
        require_positive("Ko2", self.Ko2)

    def specific_growth_rate(
        self, X: float, S: float, t: float, C_O2: float | None = None
    ) -> float:
        S = max(S, 0.0)
        c = max(C_O2, 0.0) if C_O2 is not None else 0.0
        return self.mu_max * S / (self.Ks + S) * c / (self.Ko2 + c)
