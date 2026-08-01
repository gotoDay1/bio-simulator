class BiosimError(Exception):
    """Base class for all biosim errors."""


class InvalidParameterError(BiosimError):
    """Raised when a model or simulation parameter is physically invalid."""


class IntegrationError(BiosimError):
    """Raised when the ODE solver fails to integrate the system."""
