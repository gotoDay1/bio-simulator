"""Simulation-based next-experiment suggestion built on top of parameter_prediction.py.

The objective is NOT a predicted parameter itself (e.g. mu_max) but a downstream
quantity obtained by actually running the ODE simulator with (predicted) parameters:
the absolute product amount P(t_end) * V(t_end) (mass, not concentration) at a fixed
simulated end time t_end. Parameter uncertainty (including cross-parameter
correlation, via parameter_prediction.sample_parameters) is propagated through the
nonlinear simulator via Monte Carlo, giving a (mean, std) of the objective at each
candidate condition. Candidate conditions are then ranked by Expected Improvement
against the best objective actually achieved among the historical training batches
(using their own real fitted parameter values, not GP predictions). This is a
single-objective Bayesian optimization loop over conditions - not a Pareto-front
multi-objective search.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from biosim.exceptions import (
    ExperimentDesignError,
    IntegrationError,
    InvalidParameterError,
)
from biosim.fitting import CATEGORY_ORDER
from biosim.models.registry import (
    GROWTH_MODELS,
    OXYGEN_MODELS,
    PRODUCT_MODELS,
    SUBSTRATE_MODELS,
)
from biosim.operation_modes import OperationMode
from biosim.parameter_prediction import ParameterPredictionModel, sample_parameters
from biosim.results import SimulationResults
from biosim.simulation import BioreactorSimulation
from biosim.state import InitialConditions

_REGISTRIES = {
    "growth": GROWTH_MODELS,
    "product": PRODUCT_MODELS,
    "substrate": SUBSTRATE_MODELS,
    "oxygen": OXYGEN_MODELS,
}
_DEFAULT_N_MC_SAMPLES = 15
_DEFAULT_GRID_RESOLUTION = 10
_DEFAULT_N_CANDIDATES = 150
_DEFAULT_XI = 0.01
_DEFAULT_MIN_VALID_DRAWS = 5
_DEFAULT_N_POINTS = 200


@dataclass
class HistoricalObjective:
    batch_name: str
    objective: float
    parameter_values: dict[str, float]


@dataclass
class CandidateEvaluation:
    mean: float
    std: float
    n_valid_draws: int
    n_dropped_draws: int


@dataclass
class ExperimentSuggestion:
    best_condition: dict[str, float]
    best_mean: float
    best_std: float
    best_ei: float
    f_best: float
    f_best_batch: str
    historical_objectives: list[HistoricalObjective]
    skipped_historical_batches: dict[str, str]
    evaluated_candidates: pd.DataFrame


def resolve_model_classes(training_data: pd.DataFrame) -> dict[str, type]:
    """Resolve the model class registered for each category from the {category}_model
    columns of a wide training DataFrame; all rows must agree on the model used per
    category (the shared-model-per-run assumption already made by the Fitting page)."""
    model_classes: dict[str, type] = {}
    for category in CATEGORY_ORDER:
        col = f"{category}_model"
        if col not in training_data.columns:
            raise ExperimentDesignError(f"training_data is missing required column '{col}'.")
        unique = training_data[col].dropna().unique()
        if len(unique) == 0:
            raise ExperimentDesignError(f"training_data column '{col}' has no non-null values.")
        if len(unique) > 1:
            raise ExperimentDesignError(
                f"training_data has inconsistent '{col}' values across batches: "
                f"{sorted(unique)} (all rows must share the same model)."
            )
        name = unique[0]
        registry = _REGISTRIES[category]
        if name not in registry:
            raise ExperimentDesignError(
                f"Unknown {category} model '{name}' in training_data "
                f"(known: {sorted(registry)})."
            )
        model_classes[category] = registry[name]
    return model_classes


def validate_field_coverage(
    model_classes: dict[str, type],
    predicted_columns: set[str],
    fixed_values: dict[str, dict[str, float | None]],
) -> None:
    """Every dataclass field (except feed_rate_fn) of every selected model class must be
    covered by exactly one of: a predicted parameter column ('{category}_{field}'), or a
    caller-supplied fixed value. Predicted columns not used by any selected model's fields
    are also rejected."""
    expected_columns: set[str] = set()
    for category, cls in model_classes.items():
        fixed_for_category = set(fixed_values.get(category, {}))
        for f in dataclasses.fields(cls):
            if f.name == "feed_rate_fn":
                continue
            col = f"{category}_{f.name}"
            expected_columns.add(col)
            if col not in predicted_columns and f.name not in fixed_for_category:
                raise ExperimentDesignError(
                    f"{category} model {cls.__name__} field '{f.name}' is neither a "
                    f"predicted parameter column ('{col}') nor supplied in "
                    f"fixed_values['{category}']."
                )
    unexpected = predicted_columns - expected_columns
    if unexpected:
        raise ExperimentDesignError(
            f"prediction_model has parameter columns not used by any selected model: "
            f"{sorted(unexpected)}"
        )


def _build_model_kwargs(
    category: str,
    cls: type,
    parameter_row: dict[str, float],
    fixed_values: dict[str, dict[str, float | None]],
) -> dict:
    kwargs = dict(fixed_values.get(category, {}))
    for f in dataclasses.fields(cls):
        if f.name == "feed_rate_fn":
            continue
        col = f"{category}_{f.name}"
        value = parameter_row.get(col)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            kwargs[f.name] = float(value)
    return kwargs


def _build_simulation(
    model_classes: dict[str, type],
    fixed_values: dict[str, dict[str, float | None]],
    parameter_row: dict[str, float],
    initial_conditions: InitialConditions,
    operation_mode: OperationMode,
    t_end: float,
    n_points: int,
) -> BioreactorSimulation:
    kwargs_by_category = {
        category: _build_model_kwargs(category, cls, parameter_row, fixed_values)
        for category, cls in model_classes.items()
    }
    return BioreactorSimulation(
        growth_model=model_classes["growth"](**kwargs_by_category["growth"]),
        product_model=model_classes["product"](**kwargs_by_category["product"]),
        substrate_model=model_classes["substrate"](**kwargs_by_category["substrate"]),
        oxygen_model=model_classes["oxygen"](**kwargs_by_category["oxygen"]),
        operation_mode=operation_mode,
        initial_conditions=initial_conditions,
        t_span=(0.0, t_end),
        n_points=n_points,
    )


def _objective_from_simulation(results: SimulationResults) -> float:
    last = results.data.iloc[-1]
    return float(last["P"] * last["V"])


def compute_historical_objectives(
    training_data: pd.DataFrame,
    prediction_model: ParameterPredictionModel,
    model_classes: dict[str, type],
    fixed_values: dict[str, dict[str, float | None]],
    initial_conditions: InitialConditions,
    operation_mode: OperationMode,
    t_end: float,
    n_points: int = _DEFAULT_N_POINTS,
) -> tuple[list[HistoricalObjective], dict[str, str]]:
    """Simulate each training batch's own (actually fitted, not GP-predicted) parameter
    values to get its realized absolute product amount at t_end. Rows missing any
    predicted-parameter value, or whose parameters fail to simulate, are skipped."""
    if t_end <= 0:
        raise ExperimentDesignError(f"t_end must be > 0, got {t_end}")
    validate_field_coverage(model_classes, set(prediction_model.parameter_models), fixed_values)

    parameter_columns = list(prediction_model.parameter_models)
    results: list[HistoricalObjective] = []
    skipped: dict[str, str] = {}
    for idx, row in training_data.iterrows():
        batch_name = str(row["batch"]) if "batch" in training_data.columns else f"row{idx}"
        missing = [c for c in parameter_columns if pd.isna(row.get(c))]
        if missing:
            skipped[batch_name] = f"missing values in columns: {missing}"
            continue
        parameter_row = {c: float(row[c]) for c in parameter_columns}
        try:
            sim = _build_simulation(
                model_classes, fixed_values, parameter_row, initial_conditions,
                operation_mode, t_end, n_points,
            )
            sim_results = sim.run()
        except (InvalidParameterError, IntegrationError) as exc:
            skipped[batch_name] = str(exc)
            continue
        results.append(
            HistoricalObjective(
                batch_name=batch_name,
                objective=_objective_from_simulation(sim_results),
                parameter_values=parameter_row,
            )
        )
    return results, skipped


def evaluate_condition(
    prediction_model: ParameterPredictionModel,
    model_classes: dict[str, type],
    fixed_values: dict[str, dict[str, float | None]],
    initial_conditions: InitialConditions,
    operation_mode: OperationMode,
    t_end: float,
    condition: dict[str, float],
    n_mc_samples: int,
    *,
    n_points: int = _DEFAULT_N_POINTS,
    rng: np.random.Generator | None = None,
) -> CandidateEvaluation:
    """Draw n_mc_samples joint parameter vectors (via sample_parameters, respecting
    cross-parameter correlation) at `condition`, simulate each, and summarize the
    resulting objective distribution. A draw whose parameters fail to simulate is
    dropped (not the whole candidate)."""
    if t_end <= 0:
        raise ExperimentDesignError(f"t_end must be > 0, got {t_end}")

    samples = sample_parameters(prediction_model, condition, n_mc_samples, rng=rng)
    objectives: list[float] = []
    for _, sample_row in samples.iterrows():
        try:
            sim = _build_simulation(
                model_classes, fixed_values, sample_row.to_dict(), initial_conditions,
                operation_mode, t_end, n_points,
            )
            sim_results = sim.run()
        except (InvalidParameterError, IntegrationError):
            continue
        objectives.append(_objective_from_simulation(sim_results))

    n_valid = len(objectives)
    n_dropped = n_mc_samples - n_valid
    if n_valid == 0:
        return CandidateEvaluation(mean=float("nan"), std=float("nan"), n_valid_draws=0,
                                    n_dropped_draws=n_dropped)
    arr = np.array(objectives, dtype=float)
    std = float(arr.std(ddof=1)) if n_valid > 1 else 0.0
    return CandidateEvaluation(mean=float(arr.mean()), std=std, n_valid_draws=n_valid,
                                n_dropped_draws=n_dropped)


def _expected_improvement(mean: float, std: float, f_best: float, xi_abs: float) -> float:
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0.0:
        return 0.0
    z = (mean - f_best - xi_abs) / std
    return float((mean - f_best - xi_abs) * norm.cdf(z) + std * norm.pdf(z))


def _generate_candidates(
    condition_names: list[str],
    bounds: dict[str, tuple[float, float]],
    grid_resolution: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    for name in condition_names:
        if name not in bounds:
            raise ExperimentDesignError(f"Missing search bounds for condition '{name}'.")
        lo, hi = bounds[name]
        if lo >= hi:
            raise ExperimentDesignError(f"bounds['{name}'] lower >= upper ({lo} >= {hi}).")

    if len(condition_names) <= 2:
        axes = [np.linspace(bounds[n][0], bounds[n][1], grid_resolution) for n in condition_names]
        mesh = [m.ravel() for m in np.meshgrid(*axes, indexing="ij")]
        return [
            {n: float(mesh[i][k]) for i, n in enumerate(condition_names)}
            for k in range(mesh[0].size)
        ]

    sampler = qmc.LatinHypercube(d=len(condition_names), seed=rng)
    unit = sampler.random(n=n_candidates)
    lows = [bounds[n][0] for n in condition_names]
    highs = [bounds[n][1] for n in condition_names]
    scaled = qmc.scale(unit, lows, highs)
    return [
        {n: float(scaled[k, i]) for i, n in enumerate(condition_names)}
        for k in range(scaled.shape[0])
    ]


def suggest_next_experiment(
    prediction_model: ParameterPredictionModel,
    training_data: pd.DataFrame,
    fixed_values: dict[str, dict[str, float | None]],
    initial_conditions: InitialConditions,
    operation_mode: OperationMode,
    t_end: float,
    bounds: dict[str, tuple[float, float]],
    *,
    n_mc_samples: int = _DEFAULT_N_MC_SAMPLES,
    grid_resolution: int = _DEFAULT_GRID_RESOLUTION,
    n_candidates: int = _DEFAULT_N_CANDIDATES,
    xi: float = _DEFAULT_XI,
    min_valid_draws: int = _DEFAULT_MIN_VALID_DRAWS,
    n_points: int = _DEFAULT_N_POINTS,
    rng: np.random.Generator | None = None,
) -> ExperimentSuggestion:
    """Propose the single next condition to try, maximizing Expected Improvement of the
    absolute product amount P(t_end)*V(t_end) over `bounds`, where the objective at each
    candidate condition is estimated via Monte Carlo simulation of correlation-aware
    parameter draws from `prediction_model`. Not a Pareto-front multi-objective search:
    a single scalar objective, single best candidate returned."""
    if t_end <= 0:
        raise ExperimentDesignError(f"t_end must be > 0, got {t_end}")

    model_classes = resolve_model_classes(training_data)
    validate_field_coverage(model_classes, set(prediction_model.parameter_models), fixed_values)
    rng = rng if rng is not None else np.random.default_rng()

    historical, skipped = compute_historical_objectives(
        training_data, prediction_model, model_classes, fixed_values,
        initial_conditions, operation_mode, t_end, n_points=n_points,
    )
    if not historical:
        raise ExperimentDesignError(
            "No training batch produced a valid simulation; cannot determine f_best."
        )
    best_historical = max(historical, key=lambda h: h.objective)
    f_best = best_historical.objective
    xi_abs = xi * max(abs(f_best), 1e-9)

    candidates = _generate_candidates(
        prediction_model.condition_names, bounds, grid_resolution, n_candidates, rng
    )

    rows = []
    for condition in candidates:
        evaluation = evaluate_condition(
            prediction_model, model_classes, fixed_values, initial_conditions,
            operation_mode, t_end, condition, n_mc_samples, n_points=n_points, rng=rng,
        )
        ei = (
            _expected_improvement(evaluation.mean, evaluation.std, f_best, xi_abs)
            if evaluation.n_valid_draws >= min_valid_draws
            else float("nan")
        )
        rows.append(
            {
                **condition,
                "mean": evaluation.mean,
                "std": evaluation.std,
                "ei": ei,
                "n_valid_draws": evaluation.n_valid_draws,
                "n_dropped_draws": evaluation.n_dropped_draws,
            }
        )
    evaluated_candidates = pd.DataFrame(rows)

    eligible = evaluated_candidates[evaluated_candidates["n_valid_draws"] >= min_valid_draws]
    if eligible.empty:
        raise ExperimentDesignError(
            f"No candidate condition produced >= min_valid_draws={min_valid_draws} valid "
            "simulations across the given bounds; widen bounds or check fixed parameter "
            "values."
        )
    best_row = eligible.loc[eligible["ei"].idxmax()]
    best_condition = {name: float(best_row[name]) for name in prediction_model.condition_names}

    return ExperimentSuggestion(
        best_condition=best_condition,
        best_mean=float(best_row["mean"]),
        best_std=float(best_row["std"]),
        best_ei=float(best_row["ei"]),
        f_best=f_best,
        f_best_batch=best_historical.batch_name,
        historical_objectives=historical,
        skipped_historical_batches=skipped,
        evaluated_candidates=evaluated_candidates,
    )
