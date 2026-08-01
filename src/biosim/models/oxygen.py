from dataclasses import dataclass

from biosim.models.base import OxygenModel
from biosim.validation import require_nonnegative, require_positive


@dataclass
class OxygenDemandOnly(OxygenModel):
    """OUR-only oxygen demand, no dissolved-O2 state tracked (O2 assumed non-limiting).

    Yxo2: biomass yield on oxygen (g biomass / g O2), typical range 0.5-1.5
    mo2: maintenance oxygen coefficient (g O2 / g biomass / h), typical range 0-0.2
    """

    Yxo2: float = 0.9
    mo2: float = 0.05
    name = "demand_only"
    supports_supply_dynamics = False

    def __post_init__(self) -> None:
        require_positive("Yxo2", self.Yxo2)
        require_nonnegative("mo2", self.mo2)

    def demand_rate(self, X: float, dXdt: float, t: float) -> float:
        return -(1.0 / self.Yxo2) * dXdt - self.mo2 * X


@dataclass
class OxygenWithKLa(OxygenModel):
    """OUR demand plus kLa-based OTR supply, dissolved O2 tracked as a state variable.

    Yxo2: biomass yield on oxygen (g biomass / g O2), typical range 0.5-1.5
    mo2: maintenance oxygen coefficient (g O2 / g biomass / h), typical range 0-0.2
    kLa: volumetric mass transfer coefficient (1/h), typical range 10-400
    Cs_star: saturation dissolved O2 concentration (mg/L), typical range 6-8
    """

    Yxo2: float = 0.9
    mo2: float = 0.05
    kLa: float = 100.0
    Cs_star: float = 7.5
    name = "kla_supply"
    supports_supply_dynamics = True

    def __post_init__(self) -> None:
        require_positive("Yxo2", self.Yxo2)
        require_nonnegative("mo2", self.mo2)
        require_positive("kLa", self.kLa)
        require_positive("Cs_star", self.Cs_star)

    def demand_rate(self, X: float, dXdt: float, t: float) -> float:
        return -(1.0 / self.Yxo2) * dXdt - self.mo2 * X

    def supply_rate(self, C_O2: float, t: float) -> float:
        return self.kLa * (self.Cs_star - C_O2)
