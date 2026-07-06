"""Adapter exposing a per-subject ``log_likelihood_s`` over real PEC objects (M2).

Wraps one PEC per subject and presents the ``log_likelihood_s(theta, subject)`` interface the
Laplace EM driver expects. All composition complexity is hidden behind ``pec.log_likelihood``,
so the EM machinery is unchanged regardless of how complex each subject's model is.
"""

import numpy as np


class PECGroupLikelihood:
    """Per-subject likelihood backed by one PEC per subject."""

    def __init__(self, subjects, transform):
        self.subjects = subjects
        self.transform = transform
        self.n_subjects = len(subjects)
        first = subjects[0]["pec"]
        self.fit_param_names = list(first.controller.function.fit_param_names)
        self.n_params = len(self.fit_param_names)

    def log_likelihood_s(self, theta, s):
        sub = self.subjects[s]
        return float(sub["pec"].log_likelihood(*np.asarray(theta, float), inputs=sub["inputs"]))

    @classmethod
    def bounds_from_pec(cls, pec):
        """(lower, upper) arrays in fit-parameter order from a PEC's search ranges."""
        b = pec.controller.function.fit_param_bounds
        names = pec.controller.function.fit_param_names
        lower = np.array([b[n][0] for n in names])
        upper = np.array([b[n][1] for n in names])
        return lower, upper
