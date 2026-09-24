"""Conservative 2D Fokker-Planck solver with moving absorbing upper edges.

The transformed coordinates z=(x-lower)/(boundary(t)-lower) fix the domain to
[0,1]^2. Drift includes the mesh velocity -z*boundary_rate. Exponentially
fitted finite-volume fluxes form a positive Markov generator. SSP-RK2 advances
survivor mass and absorbing flux together, with CFL-controlled substeps.
There are no Gaussian endpoint-crossing tests or RT observation noise here.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ContinuousConfig:
    points: int = 65
    time_step: float = .001
    ode_step: float = .0005
    lower_bound: float = -.25
    noise: float = .1
    leak: float = 8.
    competition: float = 8.
    lc_clock_ratio: float = 20.
    cfl: float = .8
    checkpoint_steps: int = 16
    recompute_rates: bool = True
    ode_backend: str = "torch"
    flux_backend: str = "torch"
    cpu_threads: int = 1
    gpu_graphs: bool = True

    def __post_init__(self):
        if self.ode_backend not in ("torch", "generated"):
            raise ValueError("ODE backend must be 'torch' or 'generated'.")
        if self.flux_backend not in ("torch", "native", "triton"):
            raise ValueError("Flux backend must be 'torch', 'native', or 'triton'.")
        if not isinstance(self.cpu_threads, int) or self.cpu_threads < 1:
            raise ValueError("cpu_threads must be a positive integer.")
        if self.points < 8 or self.checkpoint_steps < 0:
            raise ValueError("Invalid grid/checkpoint size.")
        values = (self.time_step, self.ode_step, self.noise, self.lc_clock_ratio, self.cfl)
        if any(not math.isfinite(x) or x <= 0 for x in values) or self.cfl > 1:
            raise ValueError("Positive finite steps, noise, clock ratio, and CFL <= 1 are required.")
        if not math.isfinite(self.lower_bound) or self.lower_bound >= 0:
            raise ValueError("The lower truncation bound must be finite and below zero.")
        if any(not math.isfinite(x) or x < 0 for x in (self.leak, self.competition)):
            raise ValueError("Leak and competition must be finite and nonnegative.")


@dataclass(frozen=True)
class ContinuousDistribution:
    choice_mass: torch.Tensor
    survival: torch.Tensor
    lower_loss: torch.Tensor
    mass_error: torch.Tensor
    minimum_mass: torch.Tensor
    time_step: float
    substeps: int
    maximum_cfl: float

    def interval_probability(self, choice, low, high):
        if float(choice) not in (0., 1.):
            raise ValueError("Choice must be 0 or 1.")
        low = torch.as_tensor(low, dtype=self.choice_mass.dtype, device=self.choice_mass.device)
        high = torch.as_tensor(high, dtype=self.choice_mass.dtype, device=self.choice_mass.device)
        if not bool(torch.isfinite(low) & torch.isfinite(high) & (high > low)):
            raise ValueError("Expected a finite RT interval with high > low.")
        mass = self.choice_mass[:, int(choice)]
        edges = torch.arange(len(mass), dtype=mass.dtype, device=mass.device) * self.time_step
        left = ((low - edges) / self.time_step).clamp(0., 1.)
        right = ((high - edges) / self.time_step).clamp(0., 1.)
        # Reconstruct a linear density inside each bin, preserving its integral
        # and positivity. Constant density would give an artificial zero shift
        # derivative whenever a narrow RT interval lies inside one time cell.
        if len(mass) > 1:
            slope = torch.cat(((mass[1] - mass[0])[None], .5 * (mass[2:] - mass[:-2]),
                               (mass[-1] - mass[-2])[None]))
            slope = torch.maximum(torch.minimum(slope, 2. * mass), -2. * mass)
        else:
            slope = torch.zeros_like(mass)
        mean_density = mass + slope * (.5 * (left + right) - .5)
        return ((right - left) * mean_density).sum()


def _bernoulli(z):
    """Stable z/expm1(z), with finite derivatives at zero and large Peclet."""
    small = z.abs() < 1.e-4
    a = torch.where(small, torch.ones_like(z), z.abs())
    positive = a * torch.exp(-a) / (-torch.expm1(-a))
    exact = torch.where(z >= 0, positive, positive - z)
    return torch.where(small, 1. - z / 2. + z.square() / 12. - z.pow(4) / 720., exact)


def _euler(mass, rates, dt):
    flows = mass.unsqueeze(-3) * rates
    incoming = (F.pad(flows[1, :-1, :], (0, 0, 1, 0)) + F.pad(flows[0, 1:, :], (0, 0, 0, 1))
                + F.pad(flows[3, :, :-1], (1, 0, 0, 0)) + F.pad(flows[2, :, 1:], (0, 1, 0, 0)))
    updated = (1. - dt * rates.sum(dim=0)) * mass + dt * incoming
    flux = torch.stack((flows[1, -1, :].sum(), flows[3, :, -1].sum(),
                        flows[0, 0, :].sum() + flows[2, :, 0].sum()))
    return updated, flux


def _heun(mass, rates0, rates1, dt):
    predicted, flux0 = _euler(mass, rates0, dt)
    advanced, flux1 = _euler(predicted, rates1, dt)
    return .5 * (mass + advanced), .5 * dt * (flux0 + flux1)


class ContinuousResponseSolver:
    def __init__(self, config=None):
        self.config = config or ContinuousConfig()

    def _rates(self, inputs, gain, bias, boundary, boundary_rate):
        if self.config.flux_backend == "native":
            from .continuous_flux import native_rates
            return native_rates(inputs, gain, bias, boundary, boundary_rate, self.config)
        if self.config.flux_backend == "triton":
            from .continuous_flux_gpu import gpu_rates
            return gpu_rates(inputs, gain, bias, boundary, boundary_rate, self.config)
        n, lower = self.config.points, self.config.lower_bound
        h = 1. / n
        faces = torch.linspace(0., 1., n + 1, dtype=gain.dtype, device=gain.device)
        centers = (torch.arange(n, dtype=gain.dtype, device=gain.device) + .5) / n
        width = boundary - lower
        length = width[..., None, None]
        own = lower + length * faces[:, None]
        other = lower + length * centers[None, :]
        shared = (-self.config.leak * own - self.config.competition * torch.sigmoid(gain[..., None, None] * (other + bias))
                  - faces[:, None] * boundary_rate[..., None, None])
        diffusion = .5 * self.config.noise**2 / length.square()
        result = []
        for axis in (0, 1):
            velocity = (inputs[..., axis, None, None] + shared) / length
            peclet = velocity * h / diffusion
            plus = diffusion / h**2 * _bernoulli(-peclet[..., 1:-1, :])
            minus = diffusion / h**2 * _bernoulli(peclet[..., 1:-1, :])
            down = torch.cat((2. * diffusion / h**2 * _bernoulli(.5 * peclet[..., :1, :]), minus), dim=-2)
            up = torch.cat((plus, 2. * diffusion / h**2 * _bernoulli(-.5 * peclet[..., -1:, :])), dim=-2)
            result.extend((down, up) if axis == 0 else (down.transpose(-1, -2), up.transpose(-1, -2)))
        return torch.stack(result, dim=-3)

    def _block(self, mass, inputs, gain, bias, boundary, boundary_rate, substeps, dt):
        rates = self._rates(inputs, gain, bias, boundary, boundary_rate)
        return self._propagate(mass, rates, substeps, dt)

    def _propagate(self, mass, rates, substeps, dt):
        if self.config.flux_backend == "native":
            from .continuous_flux import native_flux_block
            return native_flux_block(mass, rates, substeps, dt, self.config.cpu_threads)
        if self.config.flux_backend == "triton":
            from .continuous_flux_gpu import gpu_flux_block
            return gpu_flux_block(mass, rates, substeps, dt, graphs=self.config.gpu_graphs)
        exits, minimum = [], mass.min()
        for k in range(len(rates) - 1):
            flux = mass.new_zeros(3)
            for j in range(substeps):
                r0 = rates[k] + (j / substeps) * (rates[k + 1] - rates[k])
                r1 = rates[k] + ((j + 1) / substeps) * (rates[k + 1] - rates[k])
                mass, increment = _heun(mass, r0, r1, dt / substeps)
                flux = flux + increment
            exits.append(flux)
            minimum = torch.minimum(minimum, mass.min())
        return mass, torch.stack(exits), minimum

    def solve_coefficients(self, inputs, gain, bias, boundary, boundary_rate, *, time_step=None):
        dt = self.config.time_step if time_step is None else time_step
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("The time step must be finite and positive.")
        length = len(gain)
        if length < 2 or inputs.shape != (length, 2) or boundary.shape != (length,) or boundary_rate.shape != (length,):
            raise ValueError("Coefficient arrays must contain matching time endpoints.")
        if not all(bool(torch.isfinite(v).all()) for v in (inputs, gain, boundary, boundary_rate)):
            raise FloatingPointError("Nonfinite continuous model coefficients.")
        if bool((gain <= 0).any()) or bool((boundary <= 0).any()):
            raise FloatingPointError("Positive gain and a boundary above the reset state are required.")
        n, low = self.config.points, self.config.lower_bound
        z0 = -low / (boundary[0] - low)
        index = z0 * n - .5
        left = torch.floor(index).to(torch.long)
        fraction = index - left
        weights = gain.new_zeros(n).scatter_add(0, torch.stack((left, left + 1)).clamp(0, n - 1),
                                               torch.stack((1. - fraction, fraction)))
        mass = weights[:, None] * weights[None, :]
        # A parameter-dependent integer stability count is numerical topology;
        # values/gradients retain the physical dt and all operator coefficients.
        maximum = 0.
        block = self.config.checkpoint_steps or (length - 1)
        retained = []
        if not self.config.recompute_rates:
            # The retention option already saves these tensors for the adjoint.
            # Reuse them for the CFL scan instead of evaluating every nonlinear
            # coefficient twice. Inference also opts into this memory tradeoff.
            for start in range(0, length - 1, block):
                sl = slice(start, min(start + block + 1, length))
                rate = self._rates(inputs[sl], gain[sl], bias, boundary[sl], boundary_rate[sl])
                retained.append(rate)
                with torch.no_grad():
                    maximum = max(maximum, float(rate.sum(dim=-3).max()))
        else:
            with torch.no_grad():
                for start in range(0, length, 64):
                    sl = slice(start, start + 64)
                    rate = self._rates(inputs[sl], gain[sl], bias, boundary[sl], boundary_rate[sl])
                    maximum = max(maximum, float(rate.sum(dim=-3).max()))
        substeps = max(1, math.ceil(dt * maximum / self.config.cfl))
        parts, minimum = [], mass.min()
        for index, start in enumerate(range(0, length - 1, block)):
            stop = min(start + block + 1, length)
            args = (mass, inputs[start:stop], gain[start:stop], bias,
                    boundary[start:stop], boundary_rate[start:stop], substeps, dt)
            if retained:
                mass, flux, local_min = self._propagate(mass, retained[index], substeps, dt)
                retained[index] = None  # Release inference coefficients after use.
            elif self.config.checkpoint_steps and self.config.recompute_rates and torch.is_grad_enabled() and any(v.requires_grad for v in args[:6]):
                mass, flux, local_min = checkpoint(self._block, *args, use_reentrant=False)
            else:
                mass, flux, local_min = self._block(*args)
            parts.append(flux)
            minimum = torch.minimum(minimum, local_min)
        flux = torch.cat(parts)
        survival, lost = mass.sum(), flux[:, 2].sum()
        return ContinuousDistribution(flux[:, :2], survival, lost, (flux.sum() + survival - 1.).abs(), minimum,
                                      dt, substeps, dt * maximum / substeps)

    def solve(self, path, parameters):
        boundary = torch.logit(parameters[0]) / path.gain - parameters[2]
        boundary_rate = -torch.logit(parameters[0]) * path.gain_rate / path.gain.square()
        return self.solve_coefficients(path.inputs, path.gain, parameters[2], boundary, boundary_rate,
                                       time_step=path.time_step)
