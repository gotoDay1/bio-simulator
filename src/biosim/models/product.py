from dataclasses import dataclass

from biosim.models.base import ProductModel
from biosim.validation import require_nonnegative


@dataclass
class LuedekingPiretProduct(ProductModel):
    """Growth- and non-growth-associated product formation: dP/dt = alpha*dXdt + beta*X.

    alpha: growth-associated production coefficient (g product / g biomass), typical range 0-10
    beta: non-growth-associated production coefficient (1/h), typical range 0-0.5
    """

    alpha: float = 2.0
    beta: float = 0.05
    name = "luedeking_piret"

    def __post_init__(self) -> None:
        require_nonnegative("alpha", self.alpha)
        require_nonnegative("beta", self.beta)

    def production_rate(self, X: float, P: float, dXdt: float, t: float) -> float:
        return self.alpha * dXdt + self.beta * X


@dataclass
class NoProduct(ProductModel):
    """No product tracking: dP/dt = 0."""

    name = "none"

    def production_rate(self, X: float, P: float, dXdt: float, t: float) -> float:
        return 0.0
