import pytest

from biosim import (
    Batch,
    Chemostat,
    FedBatch,
    InvalidParameterError,
    constant_feed,
    step_feed,
    stepwise_feed,
)


class TestBatch:
    def test_no_flow(self):
        mode = Batch()
        assert mode.inflow_rate(t=5.0, V=2.0) == 0.0
        assert mode.outflow_rate(t=5.0, V=2.0) == 0.0


class TestFedBatch:
    def test_uses_feed_rate_fn_for_inflow_only(self):
        mode = FedBatch(feed_rate_fn=constant_feed(0.5), S_feed=100.0)
        assert mode.inflow_rate(t=3.0, V=2.0) == pytest.approx(0.5)
        assert mode.outflow_rate(t=3.0, V=2.0) == 0.0

    def test_step_feed_profile(self):
        mode = FedBatch(feed_rate_fn=step_feed(t_start=2.0, rate=1.0), S_feed=100.0)
        assert mode.inflow_rate(t=1.0, V=1.0) == 0.0
        assert mode.inflow_rate(t=2.0, V=1.0) == pytest.approx(1.0)
        assert mode.inflow_rate(t=5.0, V=1.0) == pytest.approx(1.0)

    def test_rejects_negative_feed_concentration(self):
        with pytest.raises(InvalidParameterError):
            FedBatch(feed_rate_fn=constant_feed(0.5), S_feed=-1.0)

    def test_stepwise_feed_profile(self):
        mode = FedBatch(feed_rate_fn=stepwise_feed(times=[1.0, 3.0], rates=[0.1, 0.4]), S_feed=50.0)
        assert mode.inflow_rate(t=0.0, V=1.0) == pytest.approx(0.1)
        assert mode.inflow_rate(t=3.0, V=1.0) == pytest.approx(0.4)


class TestStepwiseFeed:
    def test_holds_first_rate_before_first_breakpoint(self):
        fn = stepwise_feed(times=[2.0, 5.0, 8.0], rates=[0.02, 0.05, 0.03])
        assert fn(0.0) == pytest.approx(0.02)
        assert fn(1.999) == pytest.approx(0.02)

    def test_steps_at_each_breakpoint_inclusive(self):
        fn = stepwise_feed(times=[2.0, 5.0, 8.0], rates=[0.02, 0.05, 0.03])
        assert fn(2.0) == pytest.approx(0.02)
        assert fn(4.999) == pytest.approx(0.02)
        assert fn(5.0) == pytest.approx(0.05)
        assert fn(7.999) == pytest.approx(0.05)
        assert fn(8.0) == pytest.approx(0.03)

    def test_holds_last_rate_after_last_breakpoint(self):
        fn = stepwise_feed(times=[2.0, 5.0], rates=[0.02, 0.05])
        assert fn(100.0) == pytest.approx(0.05)

    def test_single_breakpoint_acts_like_constant(self):
        fn = stepwise_feed(times=[3.0], rates=[0.1])
        assert fn(0.0) == pytest.approx(0.1)
        assert fn(3.0) == pytest.approx(0.1)
        assert fn(50.0) == pytest.approx(0.1)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(InvalidParameterError):
            stepwise_feed(times=[1.0, 2.0], rates=[0.1])

    def test_rejects_empty(self):
        with pytest.raises(InvalidParameterError):
            stepwise_feed(times=[], rates=[])

    def test_rejects_non_increasing_times(self):
        with pytest.raises(InvalidParameterError):
            stepwise_feed(times=[2.0, 2.0], rates=[0.1, 0.2])
        with pytest.raises(InvalidParameterError):
            stepwise_feed(times=[3.0, 1.0], rates=[0.1, 0.2])

    def test_rejects_negative_rate(self):
        with pytest.raises(InvalidParameterError):
            stepwise_feed(times=[1.0, 2.0], rates=[0.1, -0.1])


class TestChemostat:
    def test_inflow_equals_outflow_holds_volume_constant(self):
        mode = Chemostat(D=0.1, S_feed=20.0)
        V = 3.0
        assert mode.inflow_rate(t=0.0, V=V) == pytest.approx(0.1 * V)
        assert mode.outflow_rate(t=0.0, V=V) == pytest.approx(0.1 * V)
        assert mode.inflow_rate(t=0.0, V=V) == mode.outflow_rate(t=0.0, V=V)

    def test_rejects_invalid_params(self):
        with pytest.raises(InvalidParameterError):
            Chemostat(D=0.0, S_feed=20.0)
        with pytest.raises(InvalidParameterError):
            Chemostat(D=0.1, S_feed=-5.0)
