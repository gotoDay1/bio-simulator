import warnings

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from biosim.exceptions import IntegrationError
from biosim.models.base import GrowthModel, OxygenModel, ProductModel, SubstrateModel
from biosim.operation_modes import OperationMode
from biosim.results import SimulationResults
from biosim.state import InitialConditions, StateLayout


class BioreactorSimulation:
    """Composes the four model categories + an operation mode into one ODE system and runs it.

    The RHS is built once per instance from a single general mass-balance form:
        dC/dt = reaction_rate(C) + (F_in/V) * (C_feed - C)
        dV/dt = F_in - F_out
    which collapses to batch/fed-batch/chemostat depending on what the
    OperationMode supplies for F_in(t), F_out(t), and feed concentrations
    (F_out only affects dV/dt - it cancels out of the concentration balance
    under the standard well-mixed-tank assumption).
    """

    def __init__(
        self,
        growth_model: GrowthModel,
        product_model: ProductModel,
        substrate_model: SubstrateModel,
        oxygen_model: OxygenModel,
        operation_mode: OperationMode,
        initial_conditions: InitialConditions,
        t_span: tuple[float, float],
        n_points: int = 200,
        solver_kwargs: dict | None = None,
    ):
        self.growth_model = growth_model
        self.product_model = product_model
        self.substrate_model = substrate_model
        self.oxygen_model = oxygen_model
        self.operation_mode = operation_mode
        self.initial_conditions = initial_conditions
        self.t_span = t_span
        self.n_points = n_points
        self.solver_kwargs = solver_kwargs or {}
        self.layout = StateLayout(oxygen_model.supports_supply_dynamics)

    def _rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        X, S, P, C_O2, V = self.layout.unpack(y)

        mu = self.growth_model.specific_growth_rate(X, S, t)
        dXdt_rxn = mu * X
        dPdt_rxn = self.product_model.production_rate(X, P, dXdt_rxn, t)
        dSdt_rxn = self.substrate_model.consumption_rate(X, dXdt_rxn, dPdt_rxn, t)
        our = self.oxygen_model.demand_rate(X, dXdt_rxn, t)

        F_in = self.operation_mode.inflow_rate(t, V)
        F_out = self.operation_mode.outflow_rate(t, V)
        dilution = F_in / V

        dydt = np.zeros_like(y)
        dydt[self.layout.idx_X] = dXdt_rxn + dilution * (self.operation_mode.X_feed - X)
        dydt[self.layout.idx_S] = dSdt_rxn + dilution * (self.operation_mode.S_feed - S)
        dydt[self.layout.idx_P] = dPdt_rxn + dilution * (self.operation_mode.P_feed - P)
        if self.layout.has_oxygen_state:
            otr = self.oxygen_model.supply_rate(C_O2, t)
            dydt[self.layout.idx_C_O2] = our + otr + dilution * (
                self.operation_mode.C_O2_feed - C_O2
            )
        dydt[self.layout.idx_V] = F_in - F_out
        return dydt

    def run(self) -> SimulationResults:
        y0 = self.layout.build_initial_vector(self.initial_conditions)
        t_eval = np.linspace(self.t_span[0], self.t_span[1], self.n_points)

        solver_kwargs = {"method": "LSODA", "rtol": 1e-6, "atol": 1e-9}
        solver_kwargs.update(self.solver_kwargs)

        sol = solve_ivp(self._rhs, self.t_span, y0, t_eval=t_eval, **solver_kwargs)
        if not sol.success:
            raise IntegrationError(f"ODE integration failed: {sol.message}")

        return self._build_results(sol)

    def _build_results(self, sol) -> SimulationResults:
        data: dict[str, np.ndarray] = {"t": sol.t}
        for i, name in enumerate(self.layout.column_names()):
            data[name] = sol.y[i]

        n = len(sol.t)
        mu_vals = np.zeros(n)
        our_vals = np.zeros(n)
        otr_vals = np.zeros(n) if self.layout.has_oxygen_state else None

        for i in range(n):
            X, S, _P, C_O2, _V = self.layout.unpack(sol.y[:, i])
            t = sol.t[i]
            mu = self.growth_model.specific_growth_rate(X, S, t)
            mu_vals[i] = mu
            our_vals[i] = self.oxygen_model.demand_rate(X, mu * X, t)
            if otr_vals is not None:
                otr_vals[i] = self.oxygen_model.supply_rate(C_O2, t)

        data["mu"] = mu_vals
        data["OUR"] = our_vals
        if otr_vals is not None:
            data["OTR"] = otr_vals

        df = pd.DataFrame(data)
        self._warn_if_negative(df)
        return SimulationResults(df)

    @staticmethod
    def _warn_if_negative(df: pd.DataFrame, tol: float = -1e-3) -> None:
        for col in ("X", "S", "P", "C_O2", "V"):
            if col in df.columns and (df[col] < tol).any():
                warnings.warn(
                    f"Simulated '{col}' went negative (min={df[col].min():.4g}). "
                    "This is a known artifact of maintenance/consumption terms that "
                    "keep acting after a substrate/oxygen is depleted; results after "
                    "depletion should be interpreted with that in mind.",
                    stacklevel=2,
                )
