"""Float64 Wiener first-passage log densities for constant drift/fixed bounds.

Complementary image and eigenfunction expansions (Navarro & Fuss, 2009):
https://papers.djnavarro.net/2009_firstpassage.pdf
Pairing, leading-exponential scaling and fixed truncations here are our own
evaluation scheme, not an implementation of that paper's adaptive error rule.
"""

import math

from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError


def _small_time_log_density(tau, w, drift=0.):
    """Lower-bound density; tau<=.2, 0<w<1. Eight image pairs.

    Near w=0 pair +/-k corrections to the leading image. Near w=1 pair
    adjacent reflected images, using expm1 to retain the small boundary gap.
    All exponents are scaled by the leading image, so short-RT densities do
    not underflow before taking the logarithm. Optional dimensionless drift
    is combined with the leading-image exponent (zero gives the standard law).
    """
    import torch

    close_lower = w <= .5
    result = torch.empty_like(tau)
    if torch.any(close_lower):
        t, z = tau[close_lower, None], w[close_lower, None]
        k = torch.arange(1, 9, dtype=t.dtype, device=t.device)
        distance = 2 * k - z
        log_ratio = torch.log1p(2 * z / distance) - 4 * k * z / t
        correction = ((distance / z) * torch.exp(-2 * k * (k - z) / t)
                      * (-torch.expm1(log_ratio))).sum(-1)
        result[close_lower] = (w[close_lower].log() + torch.log1p(-correction))
    if torch.any(~close_lower):
        t, z = tau[~close_lower, None], w[~close_lower, None]
        k = torch.arange(8, dtype=t.dtype, device=t.device)
        distance = 2 * k + z
        gap = 2 * (1 - z)
        log_ratio = torch.log1p(gap / distance) - gap * (2 * distance + gap) / (2 * t)
        scaled = distance * torch.exp(-2 * k * (k + z) / t) * (-torch.expm1(log_ratio))
        result[~close_lower] = scaled.sum(-1).log()
    # Combine the leading-image and drift exponents before evaluating them.
    # This avoids subtraction of huge logs near a strong-drift density peak.
    return result - .5 * math.log(2 * math.pi) - 1.5 * tau.log() - (w + drift * tau).square() / (2 * tau)


def _large_time_log_density(tau, w):
    """Standard lower-bound density; tau>=.2. Sixteen eigenmodes.

    Scale out the first exponential and sin(pi*w). Reflection plus sinc
    ratios keeps coefficients accurate near either starting-point boundary.
    """
    import torch

    k = torch.arange(1, 17, dtype=tau.dtype, device=tau.device)
    # Select one smooth representation at w=.5. minimum's averaged derivative
    # there would spuriously erase the even modes' starting-point derivative.
    u = torch.where(w <= .5, w, 1 - w)
    ratio = k.square() * torch.sinc(u[..., None] * k) / torch.sinc(u[..., None])
    signs = torch.where((k.to(torch.int64) % 2) == 1, 1., -1.)
    ratio = torch.where(w[..., None] > .5, ratio * signs, ratio)
    scaled = (ratio * torch.exp(-.5 * math.pi**2 * tau[..., None] * (k.square() - 1))).sum(-1)
    return (math.log(math.pi) + torch.sin(math.pi * u).log()
            - .5 * math.pi**2 * tau + scaled.log())


def wiener_log_density(time, drift, boundary, noise, starting_value, choice):
    """Joint choice/time density, not RT density conditional on choice.

    Broadcastable CPU float64 tensors. Bounds are +/-boundary, noise is the
    diffusion standard deviation, and choice=1 denotes the upper boundary.
    Outside time support (time<=0) returns -inf; no likelihood floor is used.
    Invalid parameters raise rather than changing the modeled process.
    """
    import torch

    values = (time, drift, boundary, noise, starting_value, choice)
    if any(not isinstance(v, torch.Tensor) or v.dtype != torch.float64 or v.device.type != "cpu" for v in values):
        raise ValueError("Wiener inputs must be CPU float64 tensors.")
    time, drift, boundary, noise, starting_value, choice = torch.broadcast_tensors(*values)
    if any(not torch.isfinite(v).all() for v in (time, drift, boundary, noise, starting_value, choice)):
        raise ValueError("Wiener inputs must be finite.")
    if torch.any(boundary <= 0) or torch.any(noise <= 0) or torch.any(starting_value.abs() >= boundary):
        raise ValueError("Wiener parameters require positive noise/boundary and an interior starting value.")
    if torch.any((choice != 0) & (choice != 1)):
        raise ValueError("Wiener choices must be exactly 0 (lower) or 1 (upper).")
    shape = time.shape
    time, drift, boundary, noise, starting_value, choice = (v.reshape(-1) for v in (time, drift, boundary, noise, starting_value, choice))
    supported = time > 0
    # The zero term keeps the empty-support result connected to parameters.
    result = drift * 0 + float("-inf")
    if torch.any(supported):
        t, mu, a, sigma, x, upper = (v[supported] for v in (time, drift, boundary, noise, starting_value, choice))
        width = 2 * a
        scale = (sigma / width).square()
        tau = t * scale
        w = torch.where(upper == 1, a - x, a + x) / width
        rho = torch.where(upper == 1, -mu, mu) * width / sigma.square()
        if (torch.any((w <= 0) | (w >= 1)) or torch.any(tau <= 0)
                or not torch.isfinite(tau).all() or not torch.isfinite(rho).all()):
            raise LikelihoodPlanningError("wiener.numeric_range", "Wiener nondimensionalization exceeded the supported float64 range.")
        small = tau <= .2
        standard = torch.empty_like(tau)
        if torch.any(small):
            standard[small] = _small_time_log_density(tau[small], w[small], rho[small])
        if torch.any(~small):
            standard[~small] = (_large_time_log_density(tau[~small], w[~small])
                                - rho[~small] * w[~small] - .5 * rho[~small].square() * tau[~small])
        log_density = scale.log() + standard
        if not torch.isfinite(log_density).all():
            raise LikelihoodPlanningError("wiener.numeric_range", "Wiener series evaluation failed; no floor or renormalization was applied.")
        result = result.index_put((supported,), log_density)
    return result.reshape(shape)
