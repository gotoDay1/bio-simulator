import numpy as np
import pandas as pd
import pytest

from biosim import (
    ParameterModel,
    ParameterPredictionModel,
    PredictionError,
    fit_parameter_models,
    predict_parameters,
    predictions_to_dataframe,
    sample_parameters,
)


def _noisy_training_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    temperature = np.array([20.0, 22.0, 25.0, 28.0, 30.0, 33.0, 35.0, 37.0, 40.0])
    noise = rng.normal(scale=0.005, size=temperature.shape)
    mu_max = 0.01 * temperature + 0.1 + noise
    return pd.DataFrame(
        {
            "batch": [f"b{i}" for i in range(len(temperature))],
            "temperature": temperature,
            "growth_mu_max": mu_max,
        }
    )


def test_fit_parameter_models_recovers_relationship_and_predicts_interpolated_point():
    df = _noisy_training_df()
    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max"])

    assert model.skipped_parameters == {}
    preds = predict_parameters(model, {"temperature": 27.5})
    assert len(preds) == 1
    pred = preds[0]

    true_value = 0.01 * 27.5 + 0.1
    assert pred.mean == pytest.approx(true_value, abs=0.03)
    assert pred.std > 0
    assert pred.lower < pred.mean < pred.upper


def test_predict_uncertainty_grows_away_from_training_data():
    df = _noisy_training_df()
    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max"])

    near = predict_parameters(model, {"temperature": 27.5})[0]
    far = predict_parameters(model, {"temperature": 500.0})[0]
    assert far.std >= near.std


def test_fit_parameter_models_raises_on_too_few_rows():
    df = _noisy_training_df().iloc[:2]
    with pytest.raises(PredictionError, match="min_rows"):
        fit_parameter_models(df, ["temperature"], ["growth_mu_max"])


def test_fit_parameter_models_raises_on_reserved_column_name():
    df = _noisy_training_df()
    with pytest.raises(PredictionError, match="batch"):
        fit_parameter_models(df, ["batch"], ["growth_mu_max"])


def test_fit_parameter_models_raises_on_condition_parameter_overlap():
    df = _noisy_training_df()
    with pytest.raises(PredictionError, match="overlap"):
        fit_parameter_models(df, ["temperature"], ["temperature"])


def test_fit_parameter_models_raises_on_non_numeric_condition_column():
    df = _noisy_training_df()
    df["temperature"] = ["hot", "cold", "warm"] * 3
    with pytest.raises(PredictionError, match="numeric"):
        fit_parameter_models(df, ["temperature"], ["growth_mu_max"])


def test_fit_parameter_models_skips_constant_parameter_column():
    df = _noisy_training_df()
    df["growth_Ks"] = 0.2  # identical across all rows

    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max", "growth_Ks"])

    assert "growth_Ks" in model.skipped_parameters
    assert "growth_Ks" not in model.parameter_models
    assert "growth_mu_max" in model.parameter_models


def test_fit_parameter_models_skips_parameter_with_too_few_non_null_values():
    df = _noisy_training_df()
    df["growth_Ks"] = [0.2, 0.25, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max", "growth_Ks"])

    assert "growth_Ks" in model.skipped_parameters
    assert "growth_mu_max" in model.parameter_models


def test_fit_parameter_models_raises_when_all_parameters_skipped():
    df = _noisy_training_df()
    df["growth_Ks"] = np.nan

    with pytest.raises(PredictionError):
        fit_parameter_models(df, ["temperature"], ["growth_Ks"])


def test_predict_parameters_raises_on_unknown_condition_name():
    df = _noisy_training_df()
    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max"])
    with pytest.raises(PredictionError):
        predict_parameters(model, {"pH": 7.0})


def test_predict_parameters_raises_on_missing_condition_name():
    df = _noisy_training_df()
    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max"])
    with pytest.raises(PredictionError):
        predict_parameters(model, {})


def test_predictions_to_dataframe_shape():
    df = _noisy_training_df()
    df["growth_Ks"] = 0.2 + 0.001 * df["temperature"]
    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max", "growth_Ks"])
    preds = predict_parameters(model, {"temperature": 27.5})

    out = predictions_to_dataframe(preds)
    assert list(out.columns) == ["parameter", "mean", "std", "lower", "upper"]
    assert len(out) == len(model.parameter_models)


def test_fit_parameter_models_two_condition_columns_smoke():
    rng = np.random.default_rng(1)
    n = 8
    temperature = rng.uniform(20, 40, size=n)
    pH = rng.uniform(5.5, 7.5, size=n)
    df = pd.DataFrame(
        {
            "batch": [f"b{i}" for i in range(n)],
            "temperature": temperature,
            "pH": pH,
            "growth_mu_max": 0.01 * temperature + 0.02 * pH + 0.05,
        }
    )
    model = fit_parameter_models(df, ["temperature", "pH"], ["growth_mu_max"])
    preds = predict_parameters(model, {"temperature": 30.0, "pH": 6.5})
    assert np.isfinite(preds[0].mean)
    assert np.isfinite(preds[0].std)


def test_compute_residual_correlation_matches_injected_sign_and_magnitude():
    rng = np.random.default_rng(2)
    n = 12
    temperature = np.linspace(20, 40, n)
    common = rng.normal(size=n)
    param_a = 0.01 * temperature + 0.5 * common + rng.normal(scale=0.01, size=n)
    param_b = 0.02 * temperature + 0.5 * common + rng.normal(scale=0.01, size=n)
    df = pd.DataFrame(
        {
            "batch": [f"b{i}" for i in range(n)],
            "temperature": temperature,
            "growth_mu_max": param_a,
            "growth_Ks": param_b,
        }
    )
    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max", "growth_Ks"])
    assert model.residual_correlation.loc["growth_mu_max", "growth_Ks"] > 0.3


def test_residual_correlation_falls_back_to_identity_with_insufficient_common_rows():
    rng = np.random.default_rng(3)
    n = 10
    temperature = np.linspace(20.0, 40.0, n)
    df = pd.DataFrame(
        {
            "batch": [f"b{i}" for i in range(n)],
            "temperature": temperature,
            "growth_mu_max": 0.01 * temperature + 0.1 + rng.normal(scale=0.005, size=n),
            "growth_Ks": 0.2 + 0.001 * temperature + rng.normal(scale=0.005, size=n),
        }
    )
    # growth_mu_max valid only for rows 0-6, growth_Ks valid only for rows 6-9:
    # each column individually has >= min_rows non-null values, but they share
    # only row 6 in common - too few to estimate a correlation from.
    df.loc[df.index[7:], "growth_mu_max"] = np.nan
    df.loc[df.index[:6], "growth_Ks"] = np.nan

    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max", "growth_Ks"])

    assert "growth_mu_max" in model.parameter_models
    assert "growth_Ks" in model.parameter_models

    assert model.residual_correlation.loc["growth_mu_max", "growth_Ks"] == 0.0
    key = ("growth_Ks", "growth_mu_max")
    assert key in model.correlation_fallback_pairs or (
        "growth_mu_max",
        "growth_Ks",
    ) in model.correlation_fallback_pairs

    samples = sample_parameters(model, {"temperature": 27.5}, 10, rng=np.random.default_rng(0))
    assert samples.shape == (10, 2)


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


def _stub_parameter_model(name: str, mean: float, std: float) -> ParameterModel:
    return ParameterModel(
        parameter_name=name,
        condition_names=["x"],
        gp=_ConstantGP(mean, std),
        condition_mean=np.array([0.0]),
        condition_std=np.array([1.0]),
        n_training_rows=10,
    )


def test_sample_parameters_respects_given_correlation_matrix():
    pm_a = _stub_parameter_model("a", 0.0, 1.0)
    pm_b = _stub_parameter_model("b", 0.0, 1.0)
    corr = pd.DataFrame([[1.0, 0.7], [0.7, 1.0]], index=["a", "b"], columns=["a", "b"])
    model = ParameterPredictionModel(
        condition_names=["x"],
        parameter_models={"a": pm_a, "b": pm_b},
        skipped_parameters={},
        residual_correlation=corr,
        correlation_fallback_pairs={},
    )

    samples = sample_parameters(model, {"x": 0.0}, 5000, rng=np.random.default_rng(0))
    observed_corr = np.corrcoef(samples["a"], samples["b"])[0, 1]
    assert observed_corr == pytest.approx(0.7, abs=0.1)


def test_sample_parameters_cholesky_jitter_handles_non_psd_matrix():
    pm_a = _stub_parameter_model("a", 0.0, 1.0)
    pm_b = _stub_parameter_model("b", 0.0, 1.0)
    pm_c = _stub_parameter_model("c", 0.0, 1.0)
    corr = pd.DataFrame(
        [[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    model = ParameterPredictionModel(
        condition_names=["x"],
        parameter_models={"a": pm_a, "b": pm_b, "c": pm_c},
        skipped_parameters={},
        residual_correlation=corr,
        correlation_fallback_pairs={},
    )

    samples = sample_parameters(model, {"x": 0.0}, 100, rng=np.random.default_rng(0))
    assert np.isfinite(samples.to_numpy()).all()


def test_sample_parameters_raises_on_nonpositive_n_samples():
    df = _noisy_training_df()
    model = fit_parameter_models(df, ["temperature"], ["growth_mu_max"])
    with pytest.raises(PredictionError):
        sample_parameters(model, {"temperature": 27.5}, 0)
