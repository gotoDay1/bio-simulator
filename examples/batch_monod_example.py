"""Minimal usage demo of the biosim library, no Streamlit required.

Run with: python examples/batch_monod_example.py
"""

from biosim import (
    Batch,
    BioreactorSimulation,
    InitialConditions,
    LuedekingPiretProduct,
    MonodGrowth,
    OxygenDemandOnly,
    YieldMaintenanceSubstrate,
)


def main() -> None:
    sim = BioreactorSimulation(
        growth_model=MonodGrowth(mu_max=0.6, Ks=0.2),
        product_model=LuedekingPiretProduct(alpha=2.0, beta=0.05),
        substrate_model=YieldMaintenanceSubstrate(Yxs=0.5, ms=0.02),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
        operation_mode=Batch(),
        initial_conditions=InitialConditions(X0=0.1, S0=20.0, P0=0.0, V0=1.0),
        t_span=(0.0, 7.0),
        n_points=200,
    )
    results = sim.run()

    print(results.data.iloc[[0, -1]].to_string(index=False))
    results.to_csv("batch_monod_example_output.csv")
    print("\nFull time series written to batch_monod_example_output.csv")


if __name__ == "__main__":
    main()
