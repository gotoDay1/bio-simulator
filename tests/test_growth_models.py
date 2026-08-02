import math

import pytest

from biosim import (
    GompertzGrowth,
    InvalidParameterError,
    LogisticGrowth,
    MonodGrowth,
    MonodGrowthO2,
)


class TestMonodGrowth:
    def test_saturates_near_mu_max_when_substrate_abundant(self):
        model = MonodGrowth(mu_max=0.6, Ks=0.2)
        mu = model.specific_growth_rate(X=1.0, S=1000.0, t=0.0)
        assert mu == pytest.approx(0.6, rel=1e-3)

    def test_zero_at_zero_substrate(self):
        model = MonodGrowth(mu_max=0.6, Ks=0.2)
        assert model.specific_growth_rate(X=1.0, S=0.0, t=0.0) == 0.0

    def test_monotonically_increasing_in_substrate(self):
        model = MonodGrowth(mu_max=0.6, Ks=0.2)
        mus = [model.specific_growth_rate(X=1.0, S=s, t=0.0) for s in (0.0, 0.1, 1.0, 10.0)]
        assert mus == sorted(mus)

    def test_exact_formula(self):
        model = MonodGrowth(mu_max=0.5, Ks=1.0)
        assert model.specific_growth_rate(X=1.0, S=1.0, t=0.0) == pytest.approx(0.25)

    def test_rejects_invalid_params(self):
        with pytest.raises(InvalidParameterError):
            MonodGrowth(mu_max=0.0, Ks=0.2)
        with pytest.raises(InvalidParameterError):
            MonodGrowth(mu_max=0.6, Ks=-0.1)


class TestMonodGrowthO2:
    def test_exact_formula(self):
        model = MonodGrowthO2(mu_max=0.6, Ks=0.2, Ko2=0.2)
        mu = model.specific_growth_rate(X=1.0, S=1.0, t=0.0, C_O2=1.0)
        assert mu == pytest.approx(0.6 * (1.0 / 1.2) * (1.0 / 1.2))

    def test_zero_at_zero_substrate(self):
        model = MonodGrowthO2(mu_max=0.6, Ks=0.2, Ko2=0.2)
        assert model.specific_growth_rate(X=1.0, S=0.0, t=0.0, C_O2=5.0) == 0.0

    def test_zero_at_zero_oxygen(self):
        model = MonodGrowthO2(mu_max=0.6, Ks=0.2, Ko2=0.2)
        assert model.specific_growth_rate(X=1.0, S=1000.0, t=0.0, C_O2=0.0) == 0.0

    def test_none_oxygen_treated_as_depleted(self):
        model = MonodGrowthO2(mu_max=0.6, Ks=0.2, Ko2=0.2)
        assert model.specific_growth_rate(X=1.0, S=1000.0, t=0.0, C_O2=None) == 0.0

    def test_approaches_plain_monod_when_oxygen_abundant(self):
        model = MonodGrowthO2(mu_max=0.6, Ks=0.2, Ko2=0.2)
        plain = MonodGrowth(mu_max=0.6, Ks=0.2)
        mu_o2 = model.specific_growth_rate(X=1.0, S=1.0, t=0.0, C_O2=1000.0)
        mu_plain = plain.specific_growth_rate(X=1.0, S=1.0, t=0.0)
        assert mu_o2 == pytest.approx(mu_plain, rel=1e-3)

    def test_monotonically_increasing_in_oxygen(self):
        model = MonodGrowthO2(mu_max=0.6, Ks=0.2, Ko2=0.2)
        mus = [
            model.specific_growth_rate(X=1.0, S=10.0, t=0.0, C_O2=c)
            for c in (0.0, 0.1, 1.0, 10.0)
        ]
        assert mus == sorted(mus)

    def test_rejects_invalid_params(self):
        with pytest.raises(InvalidParameterError):
            MonodGrowthO2(mu_max=0.0, Ks=0.2, Ko2=0.2)
        with pytest.raises(InvalidParameterError):
            MonodGrowthO2(mu_max=0.6, Ks=-0.1, Ko2=0.2)
        with pytest.raises(InvalidParameterError):
            MonodGrowthO2(mu_max=0.6, Ks=0.2, Ko2=0.0)


class TestLogisticGrowth:
    def test_near_mu_max_when_x_small(self):
        model = LogisticGrowth(mu_max=0.6, Xmax=10.0)
        mu = model.specific_growth_rate(X=1e-6, S=0.0, t=0.0)
        assert mu == pytest.approx(0.6, rel=1e-3)

    def test_zero_at_carrying_capacity(self):
        model = LogisticGrowth(mu_max=0.6, Xmax=10.0)
        assert model.specific_growth_rate(X=10.0, S=0.0, t=0.0) == pytest.approx(0.0)

    def test_negative_above_carrying_capacity(self):
        model = LogisticGrowth(mu_max=0.6, Xmax=10.0)
        assert model.specific_growth_rate(X=12.0, S=0.0, t=0.0) < 0.0

    def test_rejects_invalid_params(self):
        with pytest.raises(InvalidParameterError):
            LogisticGrowth(mu_max=0.6, Xmax=0.0)


class TestGompertzGrowth:
    def test_exact_formula(self):
        model = GompertzGrowth(mu_max=0.5, Xmax=10.0)
        X = 2.0
        expected = 0.5 * math.log(10.0 / X)
        assert model.specific_growth_rate(X=X, S=0.0, t=0.0) == pytest.approx(expected)

    def test_clips_to_zero_at_or_above_xmax(self):
        model = GompertzGrowth(mu_max=0.5, Xmax=10.0)
        assert model.specific_growth_rate(X=10.0, S=0.0, t=0.0) == 0.0
        assert model.specific_growth_rate(X=15.0, S=0.0, t=0.0) == 0.0

    def test_clips_to_zero_at_or_below_zero_biomass(self):
        model = GompertzGrowth(mu_max=0.5, Xmax=10.0)
        assert model.specific_growth_rate(X=0.0, S=0.0, t=0.0) == 0.0
        assert model.specific_growth_rate(X=-1.0, S=0.0, t=0.0) == 0.0

    def test_rejects_invalid_params(self):
        with pytest.raises(InvalidParameterError):
            GompertzGrowth(mu_max=0.5, Xmax=-1.0)
