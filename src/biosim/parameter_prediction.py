"""Condition -> fitted-parameter regression prediction.

Trains one independent Gaussian Process per parameter column (e.g. growth_mu_max,
substrate_Yxs) mapping numeric experimental condition columns (e.g. temperature, pH)
to that parameter's fitted value, using the wide-format table produced by
fit_results_to_dataframe (optionally augmented with condition columns) or any
equivalent wide CSV. This is regression/interpolation over conditions the user has
already run+fit, not a search/optimization loop and not a replacement for fit_batch's
least_squares fitting (see experiment_design.py for the downstream next-experiment
search built on top of this module).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from biosim.exceptions import PredictionError

RESERVED_COLUMNS = {"batch", "cost", "success", "message"}
_MODEL_COLUMN_SUFFIX = "_model"
_DEFAULT_MIN_ROWS = 3
_DEFAULT_MIN_CORRELATION_ROWS = 3
_DEFAULT_Z_SCORE = 1.96
_MIN_RELATIVE_STD = 1e-8


def _check_reserved(columns: list[str]) -> None:
    for col in columns:
        if col in RESERVED_COLUMNS or col.endswith(_MODEL_COLUMN_SUFFIX):
            raise PredictionError(
                f"Column '{col}' is reserved and cannot be used as a condition or "
                "parameter column."
            )


@dataclass
class ParameterModel:
    """A trained GP for one parameter column plus what's needed to standardize new inputs."""

    parameter_name: str
    condition_names: list[str]
    gp: GaussianProcessRegressor
    condition_mean: np.ndarray
    condition_std: np.ndarray
    n_training_rows: int


@dataclass
class ParameterPredictionModel:
    """Collection of independently-trained per-parameter GPs sharing the same condition
    columns, plus a residual correlation matrix across those parameters."""

    condition_names: list[str]
    parameter_models: dict[str, ParameterModel]
    skipped_parameters: dict[str, str]
    residual_correlation: pd.DataFrame
    correlation_fallback_pairs: dict[tuple[str, str], str]


@dataclass
class ParameterPrediction:
    parameter_name: str
    mean: float
    std: float
    lower: float
    upper: float


def _make_kernel() -> ConstantKernel:
    return ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
    ) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1))


def _fit_single_parameter(
    data: pd.DataFrame,
    condition_columns: list[str],
    condition_matrix_full: np.ndarray,
    parameter_name: str,
    min_rows: int,
) -> ParameterModel:
    y_all = pd.to_numeric(data[parameter_name], errors="coerce").to_numpy(dtype=float)
    row_mask = ~np.isnan(y_all)
    n = int(row_mask.sum())
    if n < min_rows:
        raise PredictionError(
            f"Parameter '{parameter_name}' has only {n} non-null rows "
            f"(< min_rows={min_rows})."
        )

    y = y_all[row_mask]
    X = condition_matrix_full[row_mask]

    scale = max(float(np.abs(y).max()), 1e-8)
    if float(np.std(y)) < _MIN_RELATIVE_STD * scale:
        raise PredictionError(
            f"Parameter '{parameter_name}' has near-zero variation across training rows; "
            "cannot fit a Gaussian Process meaningfully."
        )

    condition_mean = X.mean(axis=0)
    condition_std = X.std(axis=0)
    condition_std = np.where(condition_std == 0.0, 1.0, condition_std)
    X_scaled = (X - condition_mean) / condition_std

    gp = GaussianProcessRegressor(
        kernel=_make_kernel(),
        alpha=1e-10,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=0,
    )
    gp.fit(X_scaled, y)

    return ParameterModel(
        parameter_name=parameter_name,
        condition_names=list(condition_columns),
        gp=gp,
        condition_mean=condition_mean,
        condition_std=condition_std,
        n_training_rows=n,
    )


def _compute_residual_correlation(
    data: pd.DataFrame,
    parameter_models: dict[str, ParameterModel],
    min_correlation_rows: int,
) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    names = list(parameter_models.keys())
    corr = np.eye(len(names))
    fallback_pairs: dict[tuple[str, str], str] = {}

    residuals: dict[str, pd.Series] = {}
    for name, pm in parameter_models.items():
        sub = data.dropna(subset=[name])
        X = (sub[pm.condition_names].to_numpy(dtype=float) - pm.condition_mean) / pm.condition_std
        resid = sub[name].to_numpy(dtype=float) - pm.gp.predict(X)
        residuals[name] = pd.Series(resid, index=sub.index)

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j <= i:
                continue
            paired = pd.concat([residuals[a], residuals[b]], axis=1, keys=["a", "b"]).dropna()
            n = len(paired)
            reason = None
            if n < min_correlation_rows:
                r = 0.0
                reason = (
                    f"only {n} rows with both '{a}' and '{b}' present "
                    f"(< min_correlation_rows={min_correlation_rows}); treated as uncorrelated"
                )
            else:
                std_a = float(paired["a"].std(ddof=1))
                std_b = float(paired["b"].std(ddof=1))
                if std_a == 0.0 or std_b == 0.0:
                    r = 0.0
                    reason = f"zero residual variance for '{a}' or '{b}'; treated as uncorrelated"
                else:
                    r = float(np.corrcoef(paired["a"], paired["b"])[0, 1])
                    if not np.isfinite(r):
                        r = 0.0
                        reason = (
                            f"non-finite correlation for ('{a}', '{b}'); treated as uncorrelated"
                        )
            if reason is not None:
                fallback_pairs[(a, b)] = reason
            corr[i, j] = corr[j, i] = r

    return pd.DataFrame(corr, index=names, columns=names), fallback_pairs


def fit_parameter_models(
    data: pd.DataFrame,
    condition_columns: list[str],
    parameter_columns: list[str],
    *,
    min_rows: int = _DEFAULT_MIN_ROWS,
    min_correlation_rows: int = _DEFAULT_MIN_CORRELATION_ROWS,
) -> ParameterPredictionModel:
    """Fit one independent GP per parameter column against condition_columns.

    Structural problems (empty column lists, name collisions, reserved names, missing
    columns, non-numeric conditions, too few rows overall) raise PredictionError
    immediately. Per-parameter problems (too few non-null rows, near-zero variation)
    only skip that parameter (recorded in the result's skipped_parameters), unless
    every requested parameter ends up skipped.
    """
    if not condition_columns:
        raise PredictionError("condition_columns must not be empty.")
    if not parameter_columns:
        raise PredictionError("parameter_columns must not be empty.")

    overlap = set(condition_columns) & set(parameter_columns)
    if overlap:
        raise PredictionError(
            f"condition_columns and parameter_columns overlap: {sorted(overlap)}"
        )

    _check_reserved(list(condition_columns) + list(parameter_columns))

    missing_columns = [
        c for c in list(condition_columns) + list(parameter_columns) if c not in data.columns
    ]
    if missing_columns:
        raise PredictionError(f"Columns not found in data: {missing_columns}")

    try:
        condition_all = data[condition_columns].astype(float)
    except (ValueError, TypeError) as exc:
        raise PredictionError(
            f"condition_columns {condition_columns} must be numeric: {exc}"
        ) from exc

    valid_mask = condition_all.notna().all(axis=1)
    if int(valid_mask.sum()) < min_rows:
        raise PredictionError(
            f"Only {int(valid_mask.sum())} rows have complete values for all condition "
            f"columns {condition_columns} (< min_rows={min_rows})."
        )

    data_valid = data.loc[valid_mask].reset_index(drop=True)
    condition_matrix_full = condition_all.loc[valid_mask].to_numpy(dtype=float)

    parameter_models: dict[str, ParameterModel] = {}
    skipped_parameters: dict[str, str] = {}
    for parameter_name in parameter_columns:
        try:
            parameter_models[parameter_name] = _fit_single_parameter(
                data_valid, condition_columns, condition_matrix_full, parameter_name, min_rows
            )
        except PredictionError as exc:
            skipped_parameters[parameter_name] = str(exc)

    if not parameter_models:
        raise PredictionError(
            "All requested parameter columns were skipped: "
            f"{skipped_parameters}"
        )

    residual_correlation, correlation_fallback_pairs = _compute_residual_correlation(
        data_valid, parameter_models, min_correlation_rows
    )

    return ParameterPredictionModel(
        condition_names=list(condition_columns),
        parameter_models=parameter_models,
        skipped_parameters=skipped_parameters,
        residual_correlation=residual_correlation,
        correlation_fallback_pairs=correlation_fallback_pairs,
    )


def _check_conditions_match(model: ParameterPredictionModel, conditions: dict[str, float]) -> None:
    expected = set(model.condition_names)
    actual = set(conditions.keys())
    if expected != actual:
        raise PredictionError(
            f"conditions keys {sorted(actual)} do not match the trained condition "
            f"names {sorted(expected)}."
        )


def predict_parameters(
    model: ParameterPredictionModel,
    conditions: dict[str, float],
    *,
    z_score: float = _DEFAULT_Z_SCORE,
) -> list[ParameterPrediction]:
    """Marginal (independent per-parameter) prediction at a single queried condition."""
    _check_conditions_match(model, conditions)

    predictions: list[ParameterPrediction] = []
    for parameter_name, pm in model.parameter_models.items():
        x = np.array([[conditions[name] for name in pm.condition_names]], dtype=float)
        x_scaled = (x - pm.condition_mean) / pm.condition_std
        mean, std = pm.gp.predict(x_scaled, return_std=True)
        m = float(mean[0])
        s = float(std[0])
        predictions.append(
            ParameterPrediction(
                parameter_name=parameter_name,
                mean=m,
                std=s,
                lower=m - z_score * s,
                upper=m + z_score * s,
            )
        )
    return predictions


def _safe_cholesky(sigma: np.ndarray, rel_jitter: float = 1e-10, max_tries: int = 6) -> np.ndarray:
    """Cholesky-decompose sigma, adding diagonal jitter if it isn't (numerically)
    positive-definite. Pairwise-estimated correlation matrices aren't guaranteed PSD, so
    the jitter is sized from the actual most-negative eigenvalue (not guessed and grown
    geometrically from a tiny seed) to reliably fix both mild and more substantial
    violations in one or two tries."""
    n = sigma.shape[0]
    if n == 0:
        return sigma
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        pass

    scale = max(float(np.trace(sigma)) / n, 1e-12)
    min_eig = float(np.linalg.eigvalsh(sigma).min())
    jitter = max(-min_eig, 0.0) + rel_jitter * scale
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(sigma + jitter * np.eye(n))
        except np.linalg.LinAlgError:
            jitter *= 10
    raise PredictionError(
        f"Covariance matrix is not positive-definite even after jitter up to {jitter:.2e}; "
        "check residual_correlation for near-collinear parameters."
    )


def sample_parameters(
    model: ParameterPredictionModel,
    conditions: dict[str, float],
    n_samples: int,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw n_samples joint parameter vectors at conditions, respecting residual_correlation.

    Reuses predict_parameters for the marginal (mean, std) per parameter at the queried
    condition, then samples from N(mean_vector, D @ R @ D) where D = diag(std) and R is
    residual_correlation restricted to the parameters actually predicted.
    """
    if n_samples <= 0:
        raise PredictionError(f"n_samples must be > 0, got {n_samples}")

    predictions = predict_parameters(model, conditions)
    names = [p.parameter_name for p in predictions]
    mean_vec = np.array([p.mean for p in predictions], dtype=float)
    std_vec = np.array([p.std for p in predictions], dtype=float)

    R = model.residual_correlation.loc[names, names].to_numpy(dtype=float)
    sigma = (std_vec[:, None] * R) * std_vec[None, :]
    L = _safe_cholesky(sigma)

    rng = rng if rng is not None else np.random.default_rng()
    z = rng.standard_normal((n_samples, len(names)))
    samples = mean_vec + z @ L.T
    return pd.DataFrame(samples, columns=names)


def predictions_to_dataframe(predictions: list[ParameterPrediction]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "parameter": [p.parameter_name for p in predictions],
            "mean": [p.mean for p in predictions],
            "std": [p.std for p in predictions],
            "lower": [p.lower for p in predictions],
            "upper": [p.upper for p in predictions],
        }
    )
