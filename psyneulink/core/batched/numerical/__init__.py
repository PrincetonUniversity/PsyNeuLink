"""Reusable numerical likelihood backends, separate from graph admission.

Importing this module does not build an extension. These low-level plans accept
conditioned coefficients; they do not certify a Composition's reduction to them.
"""

from importlib import import_module

__all__ = [
    "FirstPassageProblem", "FirstPassageMesh", "FirstPassagePlan",
    "compile_first_passage",
    "CompiledContinuousPhase", "ContinuousPhaseResult", "compile_continuous_phase",
]


def __getattr__(name):
    # Torch is optional. Importing this package (including for test/lint
    # collection) must not require it until a numerical API is requested.
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = "dynamics" if name in {
        "CompiledContinuousPhase", "ContinuousPhaseResult", "compile_continuous_phase",
    } else "planning"
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value
