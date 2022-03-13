import numpy as np
import math
import logging
from numpy import exp, log, sin, sqrt, pi, tanh, cosh, sinh

logger = logging.getLogger(__name__)


def coth(x):
    """
    Hyperbolic cotangent, coth(x) = cosh(x) / sinh(x)
    """
    return cosh(x) / sinh(x)


def __stdWfptLargeTime(t, w, nterms):
    # large time expansion from navarro & fuss
    logger.debug("Large time expansion, %i terms" % nterms)
    piSqOv2 = 4.93480220054

    terms = pi * np.fromiter((k * exp(-k ** 2 * t * piSqOv2) * sin(k * pi * w) for k in range(1, nterms)), np.float64)

    return np.sum(terms)


def __stdWfptSmallTime(t, w, nterms):
    # small time expansion navarro & fuss (todo: gondan et al. improvement)
    logger.debug("Small time expansion, %i terms" % nterms)

    fr = -math.floor((nterms - 1) / 2)
    to = math.ceil((nterms - 1) / 2)

    terms = 1 / sqrt(2 * pi * t ** 3) * np.fromiter(
        ((w + 2 * k) * exp(-((w + 2 * k) ** 2) / (2 * t)) for k in range(fr, to)), np.float64)
    return np.sum(terms)


def wfpt_logp(t, c, x0, t0, a, z, eps=1e-10):
    """
        Log probability of first passage time of double-threshold wiener process
        (aka "pure DDM" of Bogacz et al.). Uses series truncation of Navarro & Fuss 2009
    """

    # boundary sep is 2 * thresh
    boundarySep = 2 * z
    # initial condition is absolute, so to make it relative divide by boundarySep
    relativeInitCond = ((x0 + z) / boundarySep)
    # normalize time
    normT = (t - t0) / (boundarySep ** 2)

    # if t is below NDT, or x0 is outside bounds, probability of any rt is 0
    if normT < 0 or x0 > z or x0 < -z:
        return -np.inf

    if c == 1:  # by default return hit of lower bound, so if resp is correct flip
        a = -a
        relativeInitCond = 1 - relativeInitCond

    largeK = np.int(sqrt((-2 * log(pi * normT * eps)) / (pi ** 2 * normT)))
    smallK = 2 + sqrt(-2 * normT * log(2 * eps * sqrt(2 * pi * normT)))
    if largeK < smallK:
        # make sure eps is small enough for for bound to be valid
        if eps > (1 / (2 * sqrt(2 * pi * normT))):
            smallK = 2
        p = __stdWfptLargeTime(normT, relativeInitCond, math.ceil(largeK))
    else:
        # make sure eps is small enough for for bound to be valid
        if eps > (1 / (pi * sqrt(normT))):
            smallK = math.ceil((1 / (pi * sqrt(t))))
        p = __stdWfptSmallTime(normT, relativeInitCond, math.ceil(smallK))

    # scale from the std case to whatever is our actual
    scaler = (1 / boundarySep ** 2) * exp(-a * boundarySep * relativeInitCond - (a ** 2 * t / 2))

    return scaler * p
    #return log(scaler * p)


def __standardize_bogacz(x0, a, z, s):
    """
        Standardize the way Bogacz et al. 2006 do:
        threshold/drift ratio, signal to noise ratio, x0 to drift ratio
    """
    ztilde = z / a
    atilde = (a / s) ** 2
    x0tilde = x0 / a
    return ztilde, atilde, x0tilde


def __standardize_srivastava(x0, a, z, s):
    """
        Standardize the way Srivastava et al. (submitted) do:
        normalized threshold, normalized starting point
    """
    kz = (a * z) / (s * s)
    kx = (a * x0) / (s * s)
    return kz, kx


def __simulate_wfpt_single(x0, t0, a, z, dt):
    particle = x0
    t = 0;
    while abs(particle) < z:
        particle = particle + np.random.normal(loc=a * dt, scale=sqrt(dt))
        t = t + 1
    return t0 + t * dt if particle > z else -t0 - t * dt


def simulate_wfpt(x0, t0, a, z, dt=0.01):
    """
        Draws from the Wiener first passage time. Slow and imprecise,
        used primarily for testing. Production usage of this function
        is not recommended.
    """
    # promote if we get scalars (probably not the best way to do this)
    x0 = np.atleast_1d(x0)
    t0 = np.atleast_1d(t0)
    a = np.atleast_1d(a)
    z = np.atleast_1d(z)

    return np.fromiter((__simulate_wfpt_single(_x0, _t0, _a, _z, dt) for _x0, _t0, _a, _z in zip(x0, t0, a, z)),
                       np.float64)


def wfpt_rt(x0, t0, a, z, s=1):
    """
    Expected first passage time of a two-boundary wiener process.
    Uses Bogacz et al. 2006 expression for nonzero drift,
    Srivastava et al. expression for zero-drift.
    """
    if abs(a) < 1e-8:  # a close to 0 (avoid float comparison)
        # use expression for limit a->0 from Srivastava et al. 2016
        return t0 + (z ** 2 - x0 ** 2) / (s ** 2)
    # expression from Bogacz et al. 2006 for nonzero drift
    else:
        ztilde, atilde, x0tilde = __standardize_bogacz(x0, a, z, s)
        return ztilde * tanh(ztilde * atilde) + ((2 * ztilde * (1 - exp(-2 * x0tilde * atilde))) / (
                    exp(2 * ztilde * atilde) - exp(-2 * ztilde * atilde)) - x0tilde) + t0


def wfpt_er(x0, t0, a, z, s=1):
    """
    Crossing probability in the -drift direction (aka "Error rate") for wiener process.
    Uses Bogacz et al. 2006 expression for nonzero drift, Srivastava et al.
    expression for zero-drift.
    """
    if abs(a) < 1e-8:  # a close to 0 (avoid float comparison)
        # use expression for limit a->0 from Srivastava et al. 2016
        return (z - x0) / (2 * z)
    # expression from Bogacz et al. 2006 for nonzero drift
    else:
        ztilde, atilde, x0tilde = __standardize_bogacz(x0, a, z, s)
        return 1 / (1 + exp(2 * ztilde * atilde)) - (
                    (1 - exp(-2 * x0 * atilde)) / (exp(2 * ztilde * atilde) - exp(-2 * ztilde * atilde)))


def wfpt_dt_upper(x0, t0, a, z, s=1):
    """
    Expected conditional first passage time, conditioned on crossing threshold
    in the +drift direction (aka "upper" or "correct")
    """

    if abs(a) < 1e-8:  # a close to 0 (avoid float comparison)
        return (4 * x ** 2 - (z + x0) ** 2) / (3 * s ** 2)
    kz, kx = __standardize_srivastava(x0, a, z, s)
    return (s ** 2) / (a ** 2) * (2 * kz * coth(2 * kz) - (kx + kz) * coth(kx + kz))


def wfpt_dt_lower(x0, t0, a, z, s=1):
    """
    Expected conditional first passage time, conditioned on crossing threshold
    in the -drift direction (aka "lower" or "error")
    """
    return wfpt_dt_upper(-x0, t0, a, z, s)