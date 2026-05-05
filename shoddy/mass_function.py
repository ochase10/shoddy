
from .utils import *

from abc import ABC, abstractmethod
import numpy as np


def dM2dlogM(val, M):
    return val * M * LN10

def dlogM2dM(val, M):
    return val / (LN10 * M)

def get_rhocrit(cosmo, z):
        return (3*cosmo.hubble_parameter(z)**2/(8*np.pi*G))

def get_rho_m(cosmo, z):
    return get_rhocrit(cosmo, z) * (cosmo.get_Omega('cdm', z) + cosmo.get_Omega('baryon', z) + cosmo.get_Omega("nu", z))

def lagrangian_radius(M, cosmo, z):
    return (3 * M / (4 * np.pi * get_rho_m(cosmo, z)))**(1/3)

def sigma_m(M, cosmo, z):
    return cosmo.get_sigmaR(lagrangian_radius(M, cosmo, z), z_indices=-1, hubble_units=False)


class MassFunction(ABC):

    def __init__(self,
                 cosmology,
                 z,
                 dens_crit=1.686,
                 delta=200.,
                 **kwargs):
        
        self.cosmo = cosmology
        self.z = z
        self.crit = dens_crit
        self.delta = delta


    @abstractmethod
    def hmf(self):
        pass


    def nu(self, M):
        return self.crit/sigma_m(M, self.cosmo, self.z)


    def dlnsig_dlogM(self, M_halo, dlogM=0.1):
        sig1 = sigma_m(M_halo*10**(-dlogM), self.cosmo, self.z)
        sig2 = sigma_m(M_halo*10**(dlogM), self.cosmo, self.z)
        return np.log(sig2/sig1) / (2*dlogM)


    def dlnsig_dM(self, M_halo, dlogM=0.1):
        M1 = M_halo*10**(-dlogM)
        M2 = M_halo*10**(dlogM)
        sig1 = sigma_m(M1, self.cosmo, self.z)
        sig2 = sigma_m(M2, self.cosmo, self.z)
        return np.log(sig2/sig1) / (M2 - M1)
    

    def set_delta(self, new_delta):
        self.delta = new_delta

    def set_crit(self, new_crit):
        self.crit = new_crit
    

class Tinker(MassFunction):

    def __init__(self, cosmology, z, dens_crit=1.686, delta=200, **kwargs):
        super().__init__(cosmology, z, dens_crit, delta, **kwargs)


    def hmf(self, M_halo, dlogm=0.1):
        sig_m = sigma_m(M_halo, self.cosmo, self.z)
        return self.fsig(sig_m) * get_rho_m(self.cosmo, self.z) * -self.dlnsig_dM(M_halo, dlogm) / M_halo


    def fsig(self, sig, A=0.186, a=1.47, b=2.57, c=1.19):
        return A * ((sig/b)**(-a) + 1) * np.exp(-c / sig**2)
    

    def bias(self, M_halo):
        v = self.nu(M_halo)
        y = np.log10(self.delta)
        A = 1+0.24*y*np.exp(-(4/y)**4)
        a = 0.44*y-0.88
        B = 0.183
        b = 1.5
        C = 0.019+0.107 * y + 0.19*np.exp(-(4/y)**4)
        c = 2.4
        return 1 - A * v**a / (v**a + self.crit**a) + B * v**b + C * v**c


class Behroozi13(Tinker):

    def __init__(self, cosmology, z, dens_crit=1.686, delta=200, **kwargs):
        super().__init__(cosmology, z, dens_crit, delta, **kwargs)


    def hmf(self, M_halo, dlogm=0.1):
        a = z2a(self.z)
        tink = dM2dlogM(super().hmf(M_halo, dlogm), M_halo)
        a_correction = 0.144 / (1 + np.exp(14.79*(a - 0.213)))
        m_correction = (M_halo / 10**11.5)**(0.5 / (1+np.exp(6.5*a)))
        log_hmf = a_correction * m_correction + np.log10(tink)
        return dlogM2dM(10**log_hmf, M_halo)



