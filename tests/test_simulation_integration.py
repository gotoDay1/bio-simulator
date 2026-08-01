from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from biosim import (
    Batch,
    BioreactorSimulation,
    Chemostat,
    FedBatch,
    InitialConditions,
    IntegrationError,
    LogisticGrowth,
    MonodGrowth,
    NoProduct,
    OxygenDemandOnly,
    YieldMaintenanceSubstrate,
    constant_feed,
)


def test_monod_growth_matches_pure_exponential_when_substrate_abundant():
    mu_max = 0.5
    X0 = 0.01
    sim = BioreactorSimulation(
        growth_model=MonodGrowth(mu_max=mu_max, Ks=0.01),
        product_model=NoProduct(),
        substrate_model=YieldMaintenanceSubstrate(Yxs=0.5, ms=0.0),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.0),
        operation_mode=Batch(),
        initial_conditions=InitialConditions(X0=X0, S0=1000.0, P0=0.0, V0=1.0),
        t_span=(0.0, 5.0),
        n_points=200,
    )
    results = sim.run()
    X_final_numeric = results.data["X"].iloc[-1]
    X_final_analytic = X0 * np.exp(mu_max * 5.0)
    assert X_final_numeric == pytest.approx(X_final_analytic, rel=1e-2)


def test_logistic_growth_matches_analytic_closed_form():
    mu_max = 0.6
    Xmax = 10.0
    X0 = 0.5
    sim = BioreactorSimulation(
        growth_model=LogisticGrowth(mu_max=mu_max, Xmax=Xmax),
        product_model=NoProduct(),
        substrate_model=YieldMaintenanceSubstrate(Yxs=0.5, ms=0.0),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.0),
        operation_mode=Batch(),
        initial_conditions=InitialConditions(X0=X0, S0=1000.0, P0=0.0, V0=1.0),
        t_span=(0.0, 15.0),
        n_points=200,
    )
    results = sim.run()
    t = results.data["t"].to_numpy()
    X_numeric = results.data["X"].to_numpy()
    X_analytic = Xmax / (1.0 + ((Xmax - X0) / X0) * np.exp(-mu_max * t))
    assert X_numeric == pytest.approx(X_analytic, rel=1e-3)


def test_chemostat_reaches_monod_steady_state():
    mu_max = 0.5
    Ks = 0.2
    D = 0.2
    S_feed = 20.0
    Yxs = 0.5
    sim = BioreactorSimulation(
        growth_model=MonodGrowth(mu_max=mu_max, Ks=Ks),
        product_model=NoProduct(),
        substrate_model=YieldMaintenanceSubstrate(Yxs=Yxs, ms=0.0),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.0),
        operation_mode=Chemostat(D=D, S_feed=S_feed),
        initial_conditions=InitialConditions(X0=0.5, S0=S_feed, P0=0.0, V0=1.0),
        t_span=(0.0, 300.0),
        n_points=500,
    )
    results = sim.run()
    S_final = results.data["S"].iloc[-1]
    X_final = results.data["X"].iloc[-1]
    mu_final = results.data["mu"].iloc[-1]

    S_star_analytic = Ks * D / (mu_max - D)
    X_star_analytic = Yxs * (S_feed - S_star_analytic)

    assert mu_final == pytest.approx(D, rel=1e-2)
    assert S_final == pytest.approx(S_star_analytic, rel=1e-2)
    assert X_final == pytest.approx(X_star_analytic, rel=1e-2)


def test_batch_mass_balance_closure_with_maintenance():
    Yxs = 0.5
    ms = 0.02
    sim = BioreactorSimulation(
        growth_model=MonodGrowth(mu_max=0.6, Ks=0.2),
        product_model=NoProduct(),
        substrate_model=YieldMaintenanceSubstrate(Yxs=Yxs, ms=ms),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.0),
        operation_mode=Batch(),
        initial_conditions=InitialConditions(X0=0.1, S0=20.0, P0=0.0, V0=1.0),
        t_span=(0.0, 8.0),
        n_points=400,
    )
    results = sim.run()
    t = results.data["t"].to_numpy()
    X = results.data["X"].to_numpy()
    S = results.data["S"].to_numpy()

    for i in (100, 200, 399):
        lhs = Yxs * (S[0] - S[i])
        maintenance_integral = np.trapezoid(X[: i + 1], t[: i + 1])
        rhs = (X[i] - X[0]) + Yxs * ms * maintenance_integral
        assert lhs == pytest.approx(rhs, rel=1e-2, abs=1e-3)


def test_fed_batch_volume_matches_constant_feed_integral():
    feed_rate = 0.05
    V0 = 1.0
    sim = BioreactorSimulation(
        growth_model=MonodGrowth(mu_max=0.6, Ks=0.2),
        product_model=NoProduct(),
        substrate_model=YieldMaintenanceSubstrate(Yxs=0.5, ms=0.0),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.0),
        operation_mode=FedBatch(feed_rate_fn=constant_feed(feed_rate), S_feed=50.0),
        initial_conditions=InitialConditions(X0=0.1, S0=10.0, P0=0.0, V0=V0),
        t_span=(0.0, 10.0),
        n_points=100,
    )
    results = sim.run()
    t = results.data["t"].to_numpy()
    V_numeric = results.data["V"].to_numpy()
    V_analytic = V0 + feed_rate * t
    assert V_numeric == pytest.approx(V_analytic, rel=1e-6, abs=1e-6)


def test_integration_error_raised_when_solver_reports_failure(standard_batch_simulation):
    fake_failed_solution = SimpleNamespace(success=False, message="mocked solver failure")
    with (
        patch("biosim.simulation.solve_ivp", return_value=fake_failed_solution),
        pytest.raises(IntegrationError, match="mocked solver failure"),
    ):
        standard_batch_simulation.run()
