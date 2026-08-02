from abc import ABC, abstractmethod
from typing import ClassVar


class GrowthModel(ABC):
    """Returns the specific growth rate mu (1/h); dX/dt = mu * X is computed by the orchestrator.

    requires_oxygen_state indicates whether this model needs the current dissolved-O2
    concentration (C_O2) to compute mu. When True, the orchestrator rejects pairing this
    growth model with an oxygen model that doesn't track C_O2 (supports_supply_dynamics=False).
    """

    name: ClassVar[str]
    requires_oxygen_state: ClassVar[bool] = False

    @abstractmethod
    def specific_growth_rate(
        self, X: float, S: float, t: float, C_O2: float | None = None
    ) -> float:
        ...


class ProductModel(ABC):
    """Returns dP/dt from reaction kinetics only (dilution/feed terms are added by the orchestrator)."""

    name: ClassVar[str]

    @abstractmethod
    def production_rate(self, X: float, P: float, dXdt: float, t: float) -> float:
        ...


class SubstrateModel(ABC):
    """Returns dS/dt from reaction kinetics only (negative = consumption)."""

    name: ClassVar[str]

    @abstractmethod
    def consumption_rate(self, X: float, dXdt: float, dPdt: float, t: float) -> float:
        ...


class OxygenModel(ABC):
    """Returns oxygen uptake rate (OUR, negative) and, optionally, oxygen transfer rate (OTR).

    supports_supply_dynamics indicates whether this model tracks a dissolved
    oxygen concentration state (C_O2) via kLa-based supply. When False, the
    orchestrator omits C_O2 from the state vector entirely and only reports
    cumulative OUR as a diagnostic.
    """

    name: ClassVar[str]
    supports_supply_dynamics: ClassVar[bool] = False

    @abstractmethod
    def demand_rate(self, X: float, dXdt: float, t: float) -> float:
        ...

    def supply_rate(self, C_O2: float, t: float) -> float:
        raise NotImplementedError(
            f"{type(self).__name__} does not support oxygen supply dynamics"
        )
