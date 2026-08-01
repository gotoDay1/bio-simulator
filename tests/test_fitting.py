import pandas as pd
import pytest

from biosim import (
    Batch,
    BioreactorSimulation,
    FedBatch,
    FittingError,
    GompertzGrowth,
    InitialConditions,
    LuedekingPiretProduct,
    ModelSpec,
    MonodGrowth,
    OxygenDemandOnly,
    OxygenWithKLa,
    ParameterSpec,
    YieldMaintenanceSubstrate,
    fit_batch,
    fit_results_to_dataframe,
    stepwise_feed,
)


def _simulate(growth_model, initial_conditions, t_span=(0.0, 20.0), n_points=150, **overrides):
    sim = BioreactorSimulation(
        growth_model=growth_model,
        product_model=overrides.get("product_model", LuedekingPiretProduct(alpha=1.0, beta=0.02)),
        substrate_model=overrides.get(
            "substrate_model", YieldMaintenanceSubstrate(Yxs=0.5, ms=0.01)
        ),
        oxygen_model=overrides.get("oxygen_model", OxygenDemandOnly(Yxo2=0.9, mo2=0.05)),
        operation_mode=overrides.get("operation_mode", Batch()),
        initial_conditions=initial_conditions,
        t_span=t_span,
        n_points=n_points,
    )
    return sim.run().data


def _fixed(name: str, value: float | None) -> ParameterSpec:
    return ParameterSpec(name=name, fixed=True, value=value, is_optional=value is None)


def _free(name: str, guess: float, lower: float = 0.0, upper: float = float("inf")) -> ParameterSpec:
    return ParameterSpec(name=name, fixed=False, value=guess, lower_bound=lower, upper_bound=upper)


def _default_non_growth_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "product",
            LuedekingPiretProduct,
            [_fixed("alpha", 1.0), _fixed("beta", 0.02)],
        ),
        ModelSpec(
            "substrate",
            YieldMaintenanceSubstrate,
            [_fixed("Yxs", 0.5), _fixed("ms", 0.01), _fixed("Yps", None)],
        ),
        ModelSpec(
            "oxygen",
            OxygenDemandOnly,
            [_fixed("Yxo2", 0.9), _fixed("mo2", 0.05)],
        ),
    ]


def test_fit_batch_recovers_growth_params():
    true_growth = MonodGrowth(mu_max=0.6, Ks=0.15)
    ic = InitialConditions(X0=0.05, S0=4.0, P0=0.0, V0=1.0)
    data = _simulate(true_growth, ic)
    experimental_data = data[["t", "X", "S"]]

    model_specs = [
        ModelSpec(
            "growth",
            MonodGrowth,
            [_free("mu_max", guess=0.3, upper=5.0), _free("Ks", guess=0.5, upper=5.0)],
        ),
        *_default_non_growth_specs(),
    ]

    result = fit_batch(
        batch_name="batch1",
        model_specs=model_specs,
        initial_conditions=ic,
        operation_mode=Batch(),
        experimental_data=experimental_data,
    )

    assert result.success
    assert result.param_values["growth"]["mu_max"] == pytest.approx(0.6, rel=0.1)
    assert result.param_values["growth"]["Ks"] == pytest.approx(0.15, rel=0.3)


def test_fit_batch_recovers_gompertz_growth_params():
    true_growth = GompertzGrowth(mu_max=0.4, Xmax=6.0)
    ic = InitialConditions(X0=0.1, S0=50.0, P0=0.0, V0=1.0)
    data = _simulate(true_growth, ic)
    experimental_data = data[["t", "X"]]

    model_specs = [
        ModelSpec(
            "growth",
            GompertzGrowth,
            [
                _free("mu_max", guess=0.25, upper=5.0),
                _free("Xmax", guess=4.0, upper=50.0),
            ],
        ),
        *_default_non_growth_specs(),
    ]

    result = fit_batch(
        batch_name="batch1",
        model_specs=model_specs,
        initial_conditions=ic,
        operation_mode=Batch(),
        experimental_data=experimental_data,
    )

    assert result.success
    assert result.param_values["growth"]["mu_max"] == pytest.approx(0.4, rel=0.15)
    assert result.param_values["growth"]["Xmax"] == pytest.approx(6.0, rel=0.15)


def test_fit_batch_with_fed_batch_stepwise_feed():
    true_growth = MonodGrowth(mu_max=0.5, Ks=0.3)
    operation_mode = FedBatch(
        feed_rate_fn=stepwise_feed(times=[5.0, 10.0], rates=[0.02, 0.06]), S_feed=50.0
    )
    ic = InitialConditions(X0=0.1, S0=10.0, P0=0.0, V0=1.0)
    data = _simulate(true_growth, ic, operation_mode=operation_mode)
    experimental_data = data[["t", "X", "S"]]

    model_specs = [
        ModelSpec(
            "growth",
            MonodGrowth,
            [_free("mu_max", guess=0.3, upper=5.0), _free("Ks", guess=0.5, upper=5.0)],
        ),
        *_default_non_growth_specs(),
    ]

    result = fit_batch(
        batch_name="batch1",
        model_specs=model_specs,
        initial_conditions=ic,
        operation_mode=operation_mode,
        experimental_data=experimental_data,
    )

    assert result.success
    assert result.param_values["growth"]["mu_max"] == pytest.approx(0.5, rel=0.15)


def test_fit_batch_raises_when_no_free_parameters():
    ic = InitialConditions(X0=0.1, S0=4.0, P0=0.0, V0=1.0)
    experimental_data = _simulate(MonodGrowth(), ic)[["t", "X"]]

    model_specs = [
        ModelSpec("growth", MonodGrowth, [_fixed("mu_max", 0.6), _fixed("Ks", 0.2)]),
        *_default_non_growth_specs(),
    ]

    with pytest.raises(FittingError, match="free"):
        fit_batch(
            batch_name="batch1",
            model_specs=model_specs,
            initial_conditions=ic,
            operation_mode=Batch(),
            experimental_data=experimental_data,
        )


def test_fit_batch_raises_when_no_overlapping_columns():
    ic = InitialConditions(X0=0.1, S0=4.0, P0=0.0, V0=1.0)
    experimental_data = pd.DataFrame({"t": [0.0, 1.0, 2.0], "notes": ["a", "b", "c"]})

    model_specs = [
        ModelSpec("growth", MonodGrowth, [_free("mu_max", guess=0.3), _fixed("Ks", 0.2)]),
        *_default_non_growth_specs(),
    ]

    with pytest.raises(FittingError, match="overlapping"):
        fit_batch(
            batch_name="batch1",
            model_specs=model_specs,
            initial_conditions=ic,
            operation_mode=Batch(),
            experimental_data=experimental_data,
        )


def test_fit_batch_raises_when_oxygen_supply_missing_c_o2_0():
    ic = InitialConditions(X0=0.1, S0=4.0, P0=0.0, V0=1.0, C_O2_0=None)
    experimental_data = _simulate(MonodGrowth(), ic)[["t", "X"]]

    model_specs = [
        ModelSpec("growth", MonodGrowth, [_free("mu_max", guess=0.3), _fixed("Ks", 0.2)]),
        ModelSpec(
            "product",
            LuedekingPiretProduct,
            [_fixed("alpha", 1.0), _fixed("beta", 0.02)],
        ),
        ModelSpec(
            "substrate",
            YieldMaintenanceSubstrate,
            [_fixed("Yxs", 0.5), _fixed("ms", 0.01), _fixed("Yps", None)],
        ),
        ModelSpec(
            "oxygen",
            OxygenWithKLa,
            [
                _fixed("Yxo2", 0.9),
                _fixed("mo2", 0.05),
                _fixed("kLa", 100.0),
                _fixed("Cs_star", 7.5),
            ],
        ),
    ]

    with pytest.raises(FittingError, match="C_O2_0"):
        fit_batch(
            batch_name="batch1",
            model_specs=model_specs,
            initial_conditions=ic,
            operation_mode=Batch(),
            experimental_data=experimental_data,
        )


def _two_batch_results():
    ic = InitialConditions(X0=0.05, S0=4.0, P0=0.0, V0=1.0)
    model_specs = [
        ModelSpec("growth", MonodGrowth, [_free("mu_max", guess=0.4), _fixed("Ks", 0.15)]),
        *_default_non_growth_specs(),
    ]

    results = []
    for batch_name, mu_max_true in [("batchA", 0.5), ("batchB", 0.7)]:
        data = _simulate(MonodGrowth(mu_max=mu_max_true, Ks=0.15), ic)
        result = fit_batch(
            batch_name=batch_name,
            model_specs=model_specs,
            initial_conditions=ic,
            operation_mode=Batch(),
            experimental_data=data[["t", "X"]],
        )
        results.append(result)
    return results


def test_fit_results_to_dataframe_shape():
    results = _two_batch_results()
    df = fit_results_to_dataframe(results)

    assert len(df) == 2
    for col in ["batch", "growth_model", "growth_mu_max", "growth_Ks", "cost", "success"]:
        assert col in df.columns


def test_fit_batch_multi_batch_independent_results():
    results = _two_batch_results()

    assert results[0].batch_name == "batchA"
    assert results[1].batch_name == "batchB"
    assert results[0].success
    assert results[1].success
    assert results[0].param_values["growth"]["mu_max"] == pytest.approx(0.5, rel=0.15)
    assert results[1].param_values["growth"]["mu_max"] == pytest.approx(0.7, rel=0.15)
