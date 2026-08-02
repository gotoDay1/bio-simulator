import numpy as np
import pandas as pd
import pytest

from biosim import (
    Batch,
    BioreactorSimulation,
    ExperimentDesignError,
    InitialConditions,
    LuedekingPiretProduct,
    MonodGrowth,
    OxygenDemandOnly,
    ParameterModel,
    ParameterPredictionModel,
    YieldMaintenanceSubstrate,
    compute_historical_objectives,
    evaluate_condition,
    fit_parameter_models,
    resolve_model_classes,
    suggest_next_experiment,
    validate_field_coverage,
)
from biosim.experiment_design import _generate_candidates

_MODEL_CLASSES = {
    "growth": MonodGrowth,
    "product": LuedekingPiretProduct,
    "substrate": YieldMaintenanceSubstrate,
    "oxygen": OxygenDemandOnly,
}
_FIXED_VALUES = {
    "growth": {"Ks": 0.2},
    "product": {"alpha": 1.0, "beta": 0.02},
    "substrate": {"Yxs": 0.5, "ms": 0.01, "Yps": None},
    "oxygen": {"Yxo2": 0.9, "mo2": 0.05},
}
_IC = InitialConditions(X0=0.1, S0=50.0, P0=0.0, V0=1.0)


def _training_row(batch: str, temperature: float, mu_max: float) -> dict:
    return {
        "batch": batch,
        "temperature": temperature,
        "growth_model": "monod",
        "growth_mu_max": mu_max,
        "growth_Ks": 0.2,
        "product_model": "luedeking_piret",
        "product_alpha": 1.0,
        "product_beta": 0.02,
        "substrate_model": "yield_maintenance",
        "substrate_Yxs": 0.5,
        "substrate_ms": 0.01,
        "substrate_Yps": None,
        "oxygen_model": "demand_only",
        "oxygen_Yxo2": 0.9,
        "oxygen_mo2": 0.05,
    }


def _training_data(temperatures, mu_maxes) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _training_row(f"b{i}", t, m)
            for i, (t, m) in enumerate(zip(temperatures, mu_maxes, strict=True))
        ]
    )


def _prediction_model(training_data: pd.DataFrame) -> ParameterPredictionModel:
    return fit_parameter_models(training_data, ["temperature"], ["growth_mu_max"])


class _ConstantGP:
    def __init__(self, mean: float, std: float):
        self._mean = mean
        self._std = std

    def predict(self, X, return_std=False):
        n = X.shape[0]
        means = np.full(n, self._mean)
        if return_std:
            return means, np.full(n, self._std)
        return means


def _stub_prediction_model(mean: float, std: float) -> ParameterPredictionModel:
    pm = ParameterModel(
        parameter_name="growth_mu_max",
        condition_names=["temperature"],
        gp=_ConstantGP(mean, std),
        condition_mean=np.array([30.0]),
        condition_std=np.array([5.0]),
        n_training_rows=10,
    )
    corr = pd.DataFrame([[1.0]], index=["growth_mu_max"], columns=["growth_mu_max"])
    return ParameterPredictionModel(
        condition_names=["temperature"],
        parameter_models={"growth_mu_max": pm},
        skipped_parameters={},
        residual_correlation=corr,
        correlation_fallback_pairs={},
    )


def test_compute_historical_objectives_matches_hand_computed_value():
    temperatures = [20.0, 30.0, 40.0]
    mu_maxes = [0.2, 0.3, 0.4]
    training_data = _training_data(temperatures, mu_maxes)
    prediction_model = _prediction_model(training_data)

    expected_sim = BioreactorSimulation(
        growth_model=MonodGrowth(mu_max=0.3, Ks=0.2),
        product_model=LuedekingPiretProduct(alpha=1.0, beta=0.02),
        substrate_model=YieldMaintenanceSubstrate(Yxs=0.5, ms=0.01, Yps=None),
        oxygen_model=OxygenDemandOnly(Yxo2=0.9, mo2=0.05),
        operation_mode=Batch(),
        initial_conditions=_IC,
        t_span=(0.0, 20.0),
        n_points=200,
    ).run()
    expected = float(expected_sim.data.iloc[-1]["P"] * expected_sim.data.iloc[-1]["V"])

    results, skipped = compute_historical_objectives(
        training_data, prediction_model, _MODEL_CLASSES, _FIXED_VALUES, _IC, Batch(), t_end=20.0
    )

    assert skipped == {}
    matching = [r for r in results if r.batch_name == "b1"]
    assert len(matching) == 1
    assert matching[0].objective == pytest.approx(expected, rel=1e-6)


def test_compute_historical_objectives_skips_incomplete_and_invalid_rows():
    temperatures = [20.0, 30.0, 40.0]
    mu_maxes = [0.2, 0.3, 0.4]
    training_data = _training_data(temperatures, mu_maxes)
    prediction_model = _prediction_model(training_data)

    # b3: missing predicted parameter value.
    missing_row = _training_row("b3", 25.0, mu_max=float("nan"))
    # b4: invalid parameter value (mu_max must be > 0).
    invalid_row = _training_row("b4", 35.0, mu_max=-0.5)
    training_data = pd.concat(
        [training_data, pd.DataFrame([missing_row, invalid_row])], ignore_index=True
    )

    results, skipped = compute_historical_objectives(
        training_data, prediction_model, _MODEL_CLASSES, _FIXED_VALUES, _IC, Batch(), t_end=20.0
    )

    assert "b3" in skipped
    assert "b4" in skipped
    assert {r.batch_name for r in results} == {"b0", "b1", "b2"}


def test_resolve_model_classes_raises_on_inconsistent_model_column():
    training_data = _training_data([20.0, 30.0, 40.0], [0.2, 0.3, 0.4])
    training_data.loc[1, "growth_model"] = "logistic"

    with pytest.raises(ExperimentDesignError, match="inconsistent"):
        resolve_model_classes(training_data)


def test_validate_field_coverage_raises_on_uncovered_field():
    fixed_values = {k: dict(v) for k, v in _FIXED_VALUES.items()}
    del fixed_values["growth"]["Ks"]

    with pytest.raises(ExperimentDesignError, match="neither"):
        validate_field_coverage(_MODEL_CLASSES, {"growth_mu_max"}, fixed_values)


def test_validate_field_coverage_raises_on_unexpected_predicted_column():
    with pytest.raises(ExperimentDesignError, match="not used by any selected model"):
        validate_field_coverage(
            _MODEL_CLASSES, {"growth_mu_max", "growth_bogus"}, _FIXED_VALUES
        )


def test_evaluate_condition_drops_failing_draws_without_crashing():
    model = _stub_prediction_model(mean=0.05, std=0.1)

    evaluation = evaluate_condition(
        model, _MODEL_CLASSES, _FIXED_VALUES, _IC, Batch(), 20.0,
        {"temperature": 30.0}, n_mc_samples=40, rng=np.random.default_rng(5),
    )

    assert evaluation.n_valid_draws + evaluation.n_dropped_draws == 40
    assert evaluation.n_dropped_draws > 0
    assert evaluation.n_valid_draws > 0
    assert np.isfinite(evaluation.mean)
    assert np.isfinite(evaluation.std)


def test_evaluate_condition_returns_nan_when_all_draws_fail():
    model = _stub_prediction_model(mean=-5.0, std=0.01)

    evaluation = evaluate_condition(
        model, _MODEL_CLASSES, _FIXED_VALUES, _IC, Batch(), 20.0,
        {"temperature": 30.0}, n_mc_samples=10, rng=np.random.default_rng(1),
    )

    assert evaluation.n_valid_draws == 0
    assert evaluation.n_dropped_draws == 10
    assert np.isnan(evaluation.mean)
    assert np.isnan(evaluation.std)


def test_suggest_next_experiment_picks_correct_direction_for_monotonic_objective():
    temperatures = [20.0, 25.0, 30.0, 35.0, 40.0]
    mu_maxes = [0.10, 0.15, 0.20, 0.25, 0.30]
    training_data = _training_data(temperatures, mu_maxes)
    prediction_model = _prediction_model(training_data)

    suggestion = suggest_next_experiment(
        prediction_model, training_data, _FIXED_VALUES, _IC, Batch(), t_end=20.0,
        bounds={"temperature": (20.0, 42.0)},
        n_mc_samples=5, grid_resolution=6, min_valid_draws=1,
        rng=np.random.default_rng(7),
    )

    assert suggestion.best_condition["temperature"] >= 33.0


def test_suggest_next_experiment_raises_when_no_candidate_is_evaluable():
    temperatures = [20.0, 30.0, 40.0]
    mu_maxes = [0.2, 0.3, 0.4]
    training_data = _training_data(temperatures, mu_maxes)
    prediction_model = _prediction_model(training_data)

    with pytest.raises(ExperimentDesignError, match="min_valid_draws"):
        suggest_next_experiment(
            prediction_model, training_data, _FIXED_VALUES, _IC, Batch(), t_end=20.0,
            bounds={"temperature": (20.0, 42.0)},
            n_mc_samples=5, grid_resolution=4, min_valid_draws=100,
            rng=np.random.default_rng(2),
        )


def test_suggest_next_experiment_raises_on_nonpositive_t_end():
    training_data = _training_data([20.0, 30.0, 40.0], [0.2, 0.3, 0.4])
    prediction_model = _prediction_model(training_data)

    with pytest.raises(ExperimentDesignError, match="t_end"):
        suggest_next_experiment(
            prediction_model, training_data, _FIXED_VALUES, _IC, Batch(), t_end=0.0,
            bounds={"temperature": (20.0, 42.0)},
        )


def test_suggest_next_experiment_raises_on_invalid_bounds():
    training_data = _training_data([20.0, 30.0, 40.0], [0.2, 0.3, 0.4])
    prediction_model = _prediction_model(training_data)

    with pytest.raises(ExperimentDesignError, match="lower"):
        suggest_next_experiment(
            prediction_model, training_data, _FIXED_VALUES, _IC, Batch(), t_end=20.0,
            bounds={"temperature": (30.0, 20.0)},
            n_mc_samples=5, grid_resolution=4,
        )


def test_generate_candidates_uses_grid_for_two_dims_and_lhs_beyond():
    rng = np.random.default_rng(0)
    two_dim = _generate_candidates(
        ["temperature", "pH"], {"temperature": (20.0, 40.0), "pH": (5.0, 7.0)},
        grid_resolution=5, n_candidates=200, rng=rng,
    )
    assert len(two_dim) == 5 * 5

    three_dim = _generate_candidates(
        ["temperature", "pH", "DO"],
        {"temperature": (20.0, 40.0), "pH": (5.0, 7.0), "DO": (0.0, 100.0)},
        grid_resolution=5, n_candidates=37, rng=rng,
    )
    assert len(three_dim) == 37
