'''
utility functions
'''

import numpy as np
from scipy.fft import fht as _fht, fhtoffset as _fhtoffset

G = 4.30091727e-9 # km^2 * Mpc / s^2 / Msun
C = 2.99792458e6  # km/s
LN10 = np.log(10)


def a2z(a: float) -> float:
    if a <= 0 or a > 1:
        return float('inf')
    return 1.0 / a - 1.0

def z2a(z: float) -> float:
    return 1.0 / (1.0 + z)


def pk_to_xi(k, pk):
    """
    3D P(k) → ξ(r) via FFTLog (Hankel transform of order 1/2).
    ξ(r) = (1/2π²) ∫ P(k) j_0(kr) k² dk
    k must be log-spaced.
    """
    N = len(k)
    dln = np.log(k[-1] / k[0]) / (N - 1)
    offset = _fhtoffset(dln, mu=0.5, initial=0.0)

    A = _fht(pk * k**0.5, dln, mu=0.5, offset=offset)
    r = np.exp(offset) / k[-1] * np.exp(np.arange(N) * dln)
    xi = np.sqrt(np.pi / 2) / (2 * np.pi**2) * A / r**0.5

    return r, xi


def cl_to_wtheta(ls, cl):
    """
    C_l → w(θ) via FFTLog (Hankel transform of order 0).
    w(θ) = (1/2π) ∫ C_l J_0(lθ) l dl   (θ in radians)
    ls must be log-spaced.
    """
    N = len(ls)
    dln = np.log(ls[-1] / ls[0]) / (N - 1)
    offset = _fhtoffset(dln, mu=0, initial=0.0)

    wtheta = _fht(cl / (2 * np.pi), dln, mu=0, offset=offset)
    theta = np.exp(offset) / ls[-1] * np.exp(np.arange(N) * dln)

    return theta, wtheta
