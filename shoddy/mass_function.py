
from .utils import *

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


def dM2dlogM(val, M):
    return val * M * LN10

def dlogM2dM(val, M):
    return val / (LN10 * M)


def log_trapz_weights(M):
    """Trapezoid node weights w_i such that ``sum(f * w) == integral(f dM)``,
    with the integration carried out in ln(M) (``dM = M dlnM``).  Far more
    accurate than trapezoid directly in M for log-spaced mass grids.
    """
    lnM = np.log(M)
    d = np.diff(lnM)
    w = np.empty_like(lnM)
    w[0], w[-1] = d[0] / 2, d[-1] / 2
    w[1:-1] = (d[:-1] + d[1:]) / 2
    return w * M


class MassFunction(ABC):

    def __init__(self, config):
        self.config = config
        self._hmf_cache = None
        self._bias_cache = None
        self._weights_cache = None

    def _on_grid(self, M_halo, cache_attr, func):
        """
        Evaluate func(M_halo), caching the result for the config's
        default mass grid. Off-grid calls (a new M_halo) are
        computed fresh and not cached.
        """
        if M_halo is self.config.mass_grid or np.array_equal(M_halo, self.config.mass_grid):
            if getattr(self, cache_attr) is None:
                setattr(self, cache_attr, func(self.config.mass_grid))
            return getattr(self, cache_attr)
        return func(M_halo)

    def hmf(self, M_halo):
        """Halo mass function dn/dM, cached for the config's default grid."""
        return self._on_grid(M_halo, '_hmf_cache', self._hmf)

    def bias(self, M_halo):
        """Halo bias, cached for the config's default grid."""
        return self._on_grid(M_halo, '_bias_cache', self._bias)

    @abstractmethod
    def _hmf(self, M_halo) -> NDArray:
        """Compute the halo mass function on M_halo (uncached)."""

    @abstractmethod
    def _bias(self, M_halo) -> NDArray:
        """Compute the halo bias on M_halo (uncached)."""

    def nu(self, M):
        return self.config.crit / self.config.sigma_m(M)

    def integration_weights(self, M_halo):
        """Weights w_i such that ``quant @ w == integral(hmf * quant dM)``,
        cached for the config's default grid.

        Combines the halo mass function with log-space trapezoid node weights,
        so callers can integrate over the grid with a single dot product.
        """
        return self._on_grid(M_halo, '_weights_cache', self._integration_weights)

    def _integration_weights(self, M_halo):
        return self.hmf(M_halo) * log_trapz_weights(np.asarray(M_halo))

    def halo_integral(self, M_halo, quant, axis=0):
        if len(M_halo) != quant.shape[axis]:
            raise ValueError("Mass array length does not match integrand shape on axis")

        w = self.integration_weights(M_halo)
        newshape = [1] * quant.ndim
        newshape[axis] = len(M_halo)
        return (quant * w.reshape(newshape)).sum(axis=axis)


class Tinker(MassFunction):

    def _hmf(self, M_halo):
        sig_m = self.config.sigma_m(M_halo)
        # dlnsig/dM = (dlnsig/dlnM) / M
        dlnsig_dM = self.config.dlnsig_dlnm(M_halo) / M_halo
        return self.fsig(sig_m) * self.config.rho_m * (-dlnsig_dM) / M_halo

    def fsig(self, sig, A=0.186, a=1.47, b=2.57, c=1.19):
        return A * ((sig/b)**(-a) + 1) * np.exp(-c / sig**2)

    def _bias(self, M_halo):
        v = self.nu(M_halo)
        y = np.log10(self.config.delta)
        A = 1 + 0.24*y*np.exp(-(4/y)**4)
        a = 0.44*y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107*y + 0.19*np.exp(-(4/y)**4)
        c = 2.4
        return 1 - A * v**a / (v**a + self.config.crit**a) + B * v**b + C * v**c


class Behroozi13(Tinker):

    def _hmf(self, M_halo):
        a = z2a(self.config.z)
        tink = dM2dlogM(super()._hmf(M_halo), M_halo)
        a_correction = 0.144 / (1 + np.exp(14.79*(a - 0.213)))
        m_correction = (M_halo / 10**11.5)**(0.5 / (1+np.exp(6.5*a)))
        log_hmf = a_correction * m_correction + np.log10(tink)
        return dlogM2dM(10**log_hmf, M_halo)

