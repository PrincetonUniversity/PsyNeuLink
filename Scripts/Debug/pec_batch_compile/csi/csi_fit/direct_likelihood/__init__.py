"""Continuous-time direct-likelihood prototype for the CSI research model.

This package is intentionally research-local.  It does not change the public
PsyNeuLink parameter-estimation API.
"""

from .likelihood import ContinuousCSILikelihood, LikelihoodResult, SolverConfig
from .model import CONDITIONS, CSITrialData, ContinuousCSIParameters
from .solver import MovingBoundaryDDMSolver, wiener_choice_density
from .discrete_solver import EndpointCrossingDDMSolver
from .recovery import RecoverySimulationResult, simulate_sequential_trials
from .generator_validation import (
    FirstPassageSamples,
    PrescribedDDMCase,
    generator_validation_report,
    simulate_prescribed_first_passage,
)

__all__ = [
    "CONDITIONS",
    "CSITrialData",
    "ContinuousCSIParameters",
    "ContinuousCSILikelihood",
    "LikelihoodResult",
    "MovingBoundaryDDMSolver",
    "EndpointCrossingDDMSolver",
    "SolverConfig",
    "wiener_choice_density",
    "RecoverySimulationResult",
    "simulate_sequential_trials",
    "FirstPassageSamples",
    "PrescribedDDMCase",
    "generator_validation_report",
    "simulate_prescribed_first_passage",
]
