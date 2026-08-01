import pytest

from biosim import InvalidParameterError, YieldMaintenanceSubstrate


class TestYieldMaintenanceSubstrate:
    def test_exact_formula_without_product_term(self):
        model = YieldMaintenanceSubstrate(Yxs=0.5, ms=0.02)
        dSdt = model.consumption_rate(X=3.0, dXdt=1.5, dPdt=0.4, t=0.0)
        assert dSdt == pytest.approx(-(1.0 / 0.5) * 1.5 - 0.02 * 3.0)

    def test_product_term_ignored_when_yps_none(self):
        model = YieldMaintenanceSubstrate(Yxs=0.5, ms=0.0, Yps=None)
        dSdt = model.consumption_rate(X=0.0, dXdt=0.0, dPdt=100.0, t=0.0)
        assert dSdt == 0.0

    def test_product_term_included_when_yps_set(self):
        model = YieldMaintenanceSubstrate(Yxs=0.5, ms=0.0, Yps=0.2)
        dSdt = model.consumption_rate(X=0.0, dXdt=0.0, dPdt=1.0, t=0.0)
        assert dSdt == pytest.approx(-(1.0 / 0.2) * 1.0)

    def test_is_negative_consumption(self):
        model = YieldMaintenanceSubstrate(Yxs=0.5, ms=0.02)
        dSdt = model.consumption_rate(X=1.0, dXdt=0.5, dPdt=0.0, t=0.0)
        assert dSdt < 0.0

    def test_rejects_invalid_params(self):
        with pytest.raises(InvalidParameterError):
            YieldMaintenanceSubstrate(Yxs=0.0, ms=0.02)
        with pytest.raises(InvalidParameterError):
            YieldMaintenanceSubstrate(Yxs=0.5, ms=-0.01)
        with pytest.raises(InvalidParameterError):
            YieldMaintenanceSubstrate(Yxs=0.5, ms=0.02, Yps=0.0)
