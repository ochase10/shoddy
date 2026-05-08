
from .utils import *
from .halo_config import HaloConfig

from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import quad


def dM2dlogM(val, M):
    return val * M * LN10

def dlogM2dM(val, M):
    return val / (LN10 * M)


class MassFunction(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def hmf(self):
        pass


    def nu(self, M):
        return self.config.crit/self.config.sigma_m(M)


    def dlnsig_dlogM(self, M_halo, dlogM=0.1):
        sig1 = self.config.sigma_m(M_halo*10**(-dlogM))
        sig2 = self.config.sigma_m(M_halo*10**(dlogM))
        return np.log(sig2/sig1) / (2*dlogM)


    def dlnsig_dM(self, M_halo, dlogM=0.1):
        M1 = M_halo*10**(-dlogM)
        M2 = M_halo*10**(dlogM)
        sig1 = self.config.sigma_m(M1)
        sig2 = self.config.sigma_m(M2)
        return np.log(sig2/sig1) / (M2 - M1)
    

    def halo_integral(self, M_halo, quant):

        # if quant is a function then we can use quad
        # if quant is an array then we'd want to see which 

        return 1. #TODO define this integral using quad or trapz
    

class Tinker(MassFunction):

    def hmf(self, M_halo, dlogm=0.1):
        sig_m = self.config.sigma_m(M_halo)
        return self.fsig(sig_m) * self.config.rho_m * -self.dlnsig_dM(M_halo, dlogm) / M_halo


    def fsig(self, sig, A=0.186, a=1.47, b=2.57, c=1.19):
        return A * ((sig/b)**(-a) + 1) * np.exp(-c / sig**2)


    def bias(self, M_halo):
        v = self.nu(M_halo)
        y = np.log10(self.config.delta)
        A = 1+0.24*y*np.exp(-(4/y)**4)
        a = 0.44*y-0.88
        B = 0.183
        b = 1.5
        C = 0.019+0.107 * y + 0.19*np.exp(-(4/y)**4)
        c = 2.4
        return 1 - A * v**a / (v**a + self.config.crit**a) + B * v**b + C * v**c


class Behroozi13(Tinker):

    def hmf(self, M_halo, dlogm=0.1):
        a = z2a(self.config.z)
        tink = dM2dlogM(super().hmf(M_halo, dlogm), M_halo)
        a_correction = 0.144 / (1 + np.exp(14.79*(a - 0.213)))
        m_correction = (M_halo / 10**11.5)**(0.5 / (1+np.exp(6.5*a)))
        log_hmf = a_correction * m_correction + np.log10(tink)
        return dlogM2dM(10**log_hmf, M_halo)

