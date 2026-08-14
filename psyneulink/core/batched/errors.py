"""Public runtime errors for batched simulation."""


class BatchedNumericalError(RuntimeError):
    """Raised when a batched simulation produces a NaN or infinite outcome."""
