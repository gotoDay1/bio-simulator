from biosim.exceptions import InvalidParameterError


def require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise InvalidParameterError(f"{name} must be > 0, got {value}")


def require_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise InvalidParameterError(f"{name} must be >= 0, got {value}")
