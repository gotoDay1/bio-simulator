import pytest

from biosim import InvalidParameterError, OxygenDemandOnly, OxygenWithKLa


class TestOxygenDemandOnly:
    def test_exact_formula(self):
        model = OxygenDemandOnly(Yxo2=0.9, mo2=0.05)
        our = model.demand_rate(X=3.0, dXdt=1.5, t=0.0)
        assert our == pytest.approx(-(1.0 / 0.9) * 1.5 - 0.05 * 3.0)

    def test_does_not_support_supply_dynamics(self):
        model = OxygenDemandOnly()
        assert model.supports_supply_dynamics is False
        with pytest.raises(NotImplementedError):
            model.supply_rate(C_O2=5.0, t=0.0)

    def test_rejects_invalid_params(self):
        with pytest.raises(InvalidParameterError):
            OxygenDemandOnly(Yxo2=0.0, mo2=0.05)
        with pytest.raises(InvalidParameterError):
            OxygenDemandOnly(Yxo2=0.9, mo2=-0.01)


class TestOxygenWithKLa:
    def test_demand_formula(self):
        model = OxygenWithKLa(Yxo2=0.9, mo2=0.05, kLa=100.0, Cs_star=7.5)
        our = model.demand_rate(X=3.0, dXdt=1.5, t=0.0)
        assert our == pytest.approx(-(1.0 / 0.9) * 1.5 - 0.05 * 3.0)

    def test_supports_supply_dynamics(self):
        model = OxygenWithKLa()
        assert model.supports_supply_dynamics is True

    def test_supply_positive_when_undersaturated(self):
        model = OxygenWithKLa(kLa=100.0, Cs_star=7.5)
        otr = model.supply_rate(C_O2=5.0, t=0.0)
        assert otr == pytest.approx(100.0 * (7.5 - 5.0))
        assert otr > 0.0

    def test_supply_negative_when_supersaturated(self):
        model = OxygenWithKLa(kLa=100.0, Cs_star=7.5)
        otr = model.supply_rate(C_O2=9.0, t=0.0)
        assert otr < 0.0

    def test_rejects_invalid_params(self):
        with pytest.raises(InvalidParameterError):
            OxygenWithKLa(kLa=0.0)
        with pytest.raises(InvalidParameterError):
            OxygenWithKLa(Cs_star=0.0)
