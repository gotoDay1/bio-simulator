import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from biosim.exceptions import InvalidParameterError
from biosim.validation import require_nonnegative, require_positive


class OperationMode(ABC):
    """Supplies inflow/outflow rates and feed-stream concentrations for the mass balance.

    Contains no reaction kinetics - the orchestrator's RHS combines these terms
    with the model reaction rates uniformly across batch/fed-batch/chemostat.
    """

    name: ClassVar[str]
    S_feed: float = 0.0
    X_feed: float = 0.0
    P_feed: float = 0.0
    C_O2_feed: float = 0.0

    @abstractmethod
    def inflow_rate(self, t: float, V: float) -> float:
        """F_in(t) in L/h."""

    @abstractmethod
    def outflow_rate(self, t: float, V: float) -> float:
        """F_out(t) in L/h."""


@dataclass
class Batch(OperationMode):
    """No inflow or outflow: F_in = F_out = 0."""

    name = "batch"

    def inflow_rate(self, t: float, V: float) -> float:
        return 0.0

    def outflow_rate(self, t: float, V: float) -> float:
        return 0.0


@dataclass
class FedBatch(OperationMode):
    """Feed added via a user-supplied flow-rate profile; volume grows, no outflow.

    feed_rate_fn: F_in(t) in L/h (see constant_feed/step_feed/exponential_feed/stepwise_feed
        helpers)
    S_feed: substrate concentration in the feed stream (g/L)
    """

    feed_rate_fn: Callable[[float], float]
    S_feed: float = 0.0
    X_feed: float = 0.0
    P_feed: float = 0.0
    C_O2_feed: float = 0.0
    name = "fed_batch"

    def __post_init__(self) -> None:
        require_nonnegative("S_feed", self.S_feed)

    def inflow_rate(self, t: float, V: float) -> float:
        return self.feed_rate_fn(t)

    def outflow_rate(self, t: float, V: float) -> float:
        return 0.0


@dataclass
class Chemostat(OperationMode):
    """Constant dilution rate D; F_in = F_out = D*V holds volume constant.

    D: dilution rate (1/h), typical range 0.05-0.5 (must be < mu_max to avoid washout)
    S_feed: substrate concentration in the feed stream (g/L)
    """

    D: float
    S_feed: float = 0.0
    X_feed: float = 0.0
    P_feed: float = 0.0
    C_O2_feed: float = 0.0
    name = "chemostat"

    def __post_init__(self) -> None:
        require_positive("D", self.D)
        require_nonnegative("S_feed", self.S_feed)

    def inflow_rate(self, t: float, V: float) -> float:
        return self.D * V

    def outflow_rate(self, t: float, V: float) -> float:
        return self.D * V


def constant_feed(rate: float) -> Callable[[float], float]:
    """F_in(t) = rate (L/h), constant for all t."""

    def _feed(t: float) -> float:
        return rate

    return _feed


def step_feed(t_start: float, rate: float) -> Callable[[float], float]:
    """F_in(t) = 0 for t < t_start, rate for t >= t_start."""

    def _feed(t: float) -> float:
        return rate if t >= t_start else 0.0

    return _feed


def exponential_feed(F0: float, mu_set: float) -> Callable[[float], float]:
    """F_in(t) = F0 * exp(mu_set * t), a common feed-forward profile to hold mu ~ mu_set."""

    def _feed(t: float) -> float:
        return F0 * math.exp(mu_set * t)

    return _feed


def stepwise_feed(times: Sequence[float], rates: Sequence[float]) -> Callable[[float], float]:
    """F_in(t): piecewise-constant (step-hold) profile through multiple (time, rate) breakpoints.

    For t <= times[0], F_in = rates[0] (holds the first rate, unlike step_feed which
    defaults to 0 before t_start). For times[i] <= t < times[i+1], F_in = rates[i] (a
    new rate takes effect at its breakpoint, inclusive - matching step_feed's t >= t_start
    convention). For t >= times[-1], F_in = rates[-1] (flat extrapolation). A single
    breakpoint is allowed and behaves like constant_feed.
    """
    if len(times) != len(rates):
        raise InvalidParameterError(
            f"times and rates must have equal length, got {len(times)} and {len(rates)}"
        )
    if len(times) == 0:
        raise InvalidParameterError("stepwise_feed requires at least one (time, rate) breakpoint")

    times_arr = np.asarray(times, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)
    if np.any(np.diff(times_arr) <= 0):
        raise InvalidParameterError(
            "times must be strictly increasing (no duplicate or decreasing timestamps)"
        )
    if np.any(rates_arr < 0):
        raise InvalidParameterError("feed rates must be >= 0")

    def _feed(t: float) -> float:
        idx = np.searchsorted(times_arr, t, side="right") - 1
        return float(rates_arr[max(idx, 0)])

    return _feed
