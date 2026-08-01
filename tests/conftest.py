import pytest

from biosim import (
    Batch,
    BioreactorSimulation,
    InitialConditions,
    LuedekingPiretProduct,
    MonodGrowth,
    OxygenDemandOnly,
    YieldMaintenanceSubstrate,
)


@pytest.fixture
def standard_batch_simulation() -> BioreactorSimulation:
    return BioreactorSimulation(
        growth_model=MonodGrowth(mu_max=0.6, Ks=0.2),
        product_model=LuedekingPiretProduct(alpha=2.0, beta=0.05),
        substrate_model=YieldMaintenanceSubstrate(Yxs=0.5, ms=0.02),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
        operation_mode=Batch(),
        initial_conditions=InitialConditions(X0=0.1, S0=20.0, P0=0.0, V0=1.0),
        t_span=(0.0, 10.0),
        n_points=100,
    )
