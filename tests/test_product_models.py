import pytest

from biosim import InvalidParameterError, LuedekingPiretProduct, NoProduct


class TestLuedekingPiretProduct:
    def test_exact_formula(self):
        model = LuedekingPiretProduct(alpha=2.0, beta=0.05)
        dPdt = model.production_rate(X=3.0, P=0.0, dXdt=1.5, t=0.0)
        assert dPdt == pytest.approx(2.0 * 1.5 + 0.05 * 3.0)

    def test_pure_growth_associated_when_beta_zero(self):
        model = LuedekingPiretProduct(alpha=2.0, beta=0.0)
        assert model.production_rate(X=100.0, P=0.0, dXdt=0.0, t=0.0) == 0.0
        assert model.production_rate(X=0.0, P=0.0, dXdt=1.0, t=0.0) == pytest.approx(2.0)

    def test_pure_non_growth_associated_when_alpha_zero(self):
        model = LuedekingPiretProduct(alpha=0.0, beta=0.05)
        assert model.production_rate(X=0.0, P=0.0, dXdt=5.0, t=0.0) == 0.0
        assert model.production_rate(X=2.0, P=0.0, dXdt=0.0, t=0.0) == pytest.approx(0.1)

    def test_rejects_negative_params(self):
        with pytest.raises(InvalidParameterError):
            LuedekingPiretProduct(alpha=-1.0, beta=0.05)
        with pytest.raises(InvalidParameterError):
            LuedekingPiretProduct(alpha=2.0, beta=-0.1)


class TestNoProduct:
    def test_always_zero(self):
        model = NoProduct()
        assert model.production_rate(X=5.0, P=10.0, dXdt=2.0, t=1.0) == 0.0
