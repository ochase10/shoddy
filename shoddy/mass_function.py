
from .utils import *
from .utils import _trapz

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


def dM2dlogM(val, M):
    return val * M * LN10

def dlogM2dM(val, M):
    return val / (LN10 * M)


class MassFunction(ABC):

    def __init__(self, config):
        self.config = config
        self._hmf_cache = None
        self._bias_cache = None

    @abstractmethod
    def hmf(self, M_halo, **kwargs) -> NDArray:
        pass

    @abstractmethod
    def bias(self, M_halo, **kwargs) -> NDArray:
        pass

    def nu(self, M):
        return self.config.crit / self.config.sigma_m(M)

    def halo_integral(self, M_halo, quant, axis=0, hmf_arr=None):
        if len(M_halo) != quant.shape[axis]:
            raise ValueError("Mass array length does not match integrand shape on axis")

        newshape = [1] * quant.ndim
        newshape[axis] = len(M_halo)

        if hmf_arr is None:
            hmf_arr = self.hmf(M_halo)
        return _trapz(hmf_arr.reshape(newshape) * quant, M_halo, axis=axis)


class Tinker(MassFunction):

    def hmf(self, M_halo):
        sig_m = self.config.sigma_m(M_halo)
        # dlnsig/dM = (dlnsig/dlnM) / M
        dlnsig_dM = self.config.dlnsig_dlnm(M_halo) / M_halo
        return self.fsig(sig_m) * self.config.rho_m * (-dlnsig_dM) / M_halo

    def fsig(self, sig, A=0.186, a=1.47, b=2.57, c=1.19):
        return A * ((sig/b)**(-a) + 1) * np.exp(-c / sig**2)

    def bias(self, M_halo):
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

    def hmf(self, M_halo):
        a = z2a(self.config.z)
        tink = dM2dlogM(super().hmf(M_halo), M_halo)
        a_correction = 0.144 / (1 + np.exp(14.79*(a - 0.213)))
        m_correction = (M_halo / 10**11.5)**(0.5 / (1+np.exp(6.5*a)))
        log_hmf = a_correction * m_correction + np.log10(tink)
        return dlogM2dM(10**log_hmf, M_halo)

