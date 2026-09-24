"""Positive Gaussian-transition quadrature for two competing response LCAs.

Survivor mass lives on a tensor product Gauss-Legendre grid in pre-logistic
integrator coordinates. The upper edge follows the activity threshold and LC
gain. Conditional Gaussian transitions are independent across the two units;
their means depend on BOTH previous states through recurrent inhibition.

The one-dimensional transition quadratures are normalized to their analytic
in-domain Gaussian probabilities. This preserves probability while leaving
spatial approximation error to be assessed by grid refinement and the reported
quadrature defect. Exit probabilities include both accumulators crossing in
one step and select the larger final activity, matching PNL's argmax rule.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from .model import DT


@dataclass(frozen=True)
class SolverConfig:
    points: int = 97
    lower_bound: float = -.5
    noise: float = .1
    winner_points: int = 24
    checkpoint_steps: int = 8

    def __post_init__(self):
        if self.points < 8 or self.winner_points < 4 or self.checkpoint_steps < 0:
            raise ValueError("Invalid spatial/winner quadrature or checkpoint size.")
        if not math.isfinite(self.lower_bound) or not math.isfinite(self.noise) or self.noise <= 0:
            raise ValueError("A finite lower bound and positive response noise are required.")


@dataclass(frozen=True)
class ResponseDistribution:
    choice_step: torch.Tensor
    survival: torch.Tensor
    lower_loss: torch.Tensor
    mass_error: torch.Tensor
    quadrature_defect: torch.Tensor


def normal_interval(low, high):
    """Standard-normal interval probability without positive-tail cancellation."""
    return torch.where(low > 0, torch.special.ndtr(-low) - torch.special.ndtr(-high),
                       torch.special.ndtr(high) - torch.special.ndtr(low))


class ResponseSolver:
    def __init__(self, config=None):
        self.config = config or SolverConfig()
        self._grid = np.polynomial.legendre.leggauss(self.config.points)
        self._winner_grid = np.polynomial.legendre.leggauss(self.config.winner_points)

    def _nodes(self, boundary, grid, weights):
        width = boundary - self.config.lower_bound
        return self.config.lower_bound + .5 * width * (grid + 1.), .5 * width * weights

    def _exit_probabilities(self, means, boundary, winner_grid, winner_weights):
        sd = self.config.noise * math.sqrt(DT)
        tails = torch.special.ndtr((means - boundary) / sd)
        total = tails[:, 0] + tails[:, 1] - tails[:, 0] * tails[:, 1]
        # Integrate P(Y_other < y) in the winner's upper-tail CDF coordinate.
        u = (tails[:, :, None] * winner_grid).clamp_min(1.e-15)
        difference = torch.stack((means[:, 0] - means[:, 1], means[:, 1] - means[:, 0]), dim=1) / sd
        win = tails * (torch.special.ndtr(difference[:, :, None] - torch.special.ndtri(u))
                       * winner_weights).sum(dim=2)
        return win * (total / win.sum(dim=1).clamp_min(torch.finfo(means.dtype).tiny))[:, None]

    def _transition(self, mass, means, boundary, grid, weights, winner_grid, winner_weights):
        nodes, quadrature = self._nodes(boundary, grid, weights)
        sd = self.config.noise * math.sqrt(DT)
        z = (nodes[None, None, :] - means[:, :, None]) / sd
        kernel = torch.exp(-.5 * z.square()) * (quadrature / (sd * math.sqrt(2. * math.pi)))
        in_domain = normal_interval((self.config.lower_bound - means) / sd, (boundary - means) / sd)
        numerical = kernel.sum(dim=2)
        defect = (mass[:, None] * (numerical - in_domain).abs()).sum()
        kernel = kernel * (in_domain / numerical.clamp_min(torch.finfo(means.dtype).tiny))[:, :, None]
        next_mass = kernel[:, 0].T @ (mass[:, None] * kernel[:, 1])
        exited = (mass[:, None] * self._exit_probabilities(means, boundary, winner_grid, winner_weights)).sum(dim=0)
        below = torch.special.ndtr((boundary - means) / sd)
        lost = (mass * (below.prod(dim=1) - in_domain.prod(dim=1)).clamp_min(0.)).sum()
        return next_mass, exited, lost, defect

    def _block(self, mass, previous_gain, inputs, gains, threshold, bias, grid, weights, wg, ww):
        probabilities, losses, defects = [], [], []
        for drive, gain in zip(inputs, gains):
            previous_boundary = torch.logit(threshold) / previous_gain - bias
            previous_nodes, _ = self._nodes(previous_boundary, grid, weights)
            x, y = torch.meshgrid(previous_nodes, previous_nodes, indexing="ij")
            means = torch.stack(((1. - 8. * DT) * x + DT * (drive[0] - 8. * torch.sigmoid(previous_gain * (y + bias))),
                                 (1. - 8. * DT) * y + DT * (drive[1] - 8. * torch.sigmoid(previous_gain * (x + bias)))), dim=-1)
            mass, exited, loss, defect = self._transition(mass.reshape(-1), means.reshape(-1, 2),
                                                         torch.logit(threshold) / gain - bias,
                                                         grid, weights, wg, ww)
            probabilities.append(exited)
            losses.append(loss)
            defects.append(defect)
            previous_gain = gain
        return mass, torch.stack(probabilities), torch.stack(losses), torch.stack(defects)

    def solve(self, path, threshold, bias):
        if getattr(path, "time_step", DT) != DT:
            raise ValueError("The discrete likelihood requires a 10 ms path; refined paths are for convergence audits.")
        if path.inputs.shape != (len(path.gain), 2) or not bool(torch.isfinite(path.inputs).all()):
            raise ValueError("Invalid deterministic response path.")
        bounds = torch.logit(threshold) / path.gain - bias
        if not bool(torch.isfinite(bounds).all()) or bool((path.gain <= 0).any()) or bool((bounds <= self.config.lower_bound).any()):
            raise ValueError("Invalid gain/threshold or lower grid bound above the absorbing boundary.")
        grid, weights = (torch.as_tensor(a, dtype=path.gain.dtype, device=path.gain.device) for a in self._grid)
        wg, ww = (torch.as_tensor((a + 1.) / 2. if i == 0 else a / 2., dtype=path.gain.dtype,
                                 device=path.gain.device) for i, a in enumerate(self._winner_grid))
        means = DT * (path.inputs[0] - 8. * path.initial_activity.flip(0))
        mass, first, loss, defect = self._transition(path.gain.new_ones(1), means[None, :], bounds[0], grid, weights, wg, ww)
        probabilities, losses, defects = [first[None]], [loss[None]], [defect[None]]
        block_size = self.config.checkpoint_steps or len(path.gain)
        for start in range(1, len(path.gain), block_size):
            args = (mass, path.gain[start - 1], path.inputs[start:start + block_size],
                    path.gain[start:start + block_size], threshold, bias, grid, weights, wg, ww)
            if self.config.checkpoint_steps and torch.is_grad_enabled() and any(a.requires_grad for a in args):
                mass, pmf, lost, error = checkpoint(self._block, *args, use_reentrant=False)
            else:
                mass, pmf, lost, error = self._block(*args)
            probabilities.append(pmf)
            losses.append(lost)
            defects.append(error)
        pmf = torch.cat(probabilities)
        lower_loss = torch.cat(losses).sum()
        survival = mass.sum()
        return ResponseDistribution(pmf, survival, lower_loss, (pmf.sum() + survival + lower_loss - 1.).abs(),
                                    torch.cat(defects).max())
