class BiosimError(Exception):
    """Base class for all biosim errors."""


class InvalidParameterError(BiosimError):
    """Raised when a model or simulation parameter is physically invalid."""


class IntegrationError(BiosimError):
    """Raised when the ODE solver fails to integrate the system."""


class ExperimentalDataError(BiosimError):
    """Raised when an experimental-data CSV does not match its required fixed column schema."""


class FeedProfileError(BiosimError):
    """Raised when a feed-rate profile CSV does not match its required column schema."""


class FittingError(BiosimError):
    """Raised when a fitting configuration or run is invalid."""
