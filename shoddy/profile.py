
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from scipy.special import sici



class HaloProfile(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def u(self, ks, M) -> NDArray[np.floating]:
        pass

    def Ac(self, c):
        return np.log(1+c) - c/(1+c)


class NFW(HaloProfile):

    def u(self, ks, M):
        if not isinstance(M, np.ndarray):
            M = np.array(M)

        # rvir is a physical radius (M200c definition); k is a comoving
        # wavenumber.  Convert to comoving so k*r is dimensionally consistent.
        r_vir = self.config.rvir(M) * (1 + self.config.z)
        con = self.conc(M)
        r_s = r_vir/con

        krv = np.outer(ks, r_vir)
        krs = np.outer(ks, r_s)

        Si_sv, Ci_sv = sici(krv+krs)
        Si_s, Ci_s = sici(krs)
        
        cterm = np.cos(krs) * (Ci_sv - Ci_s)
        sterm = np.sin(krs) * (Si_sv - Si_s)
        exterm = np.sin(krv)/(krv+krs)
        norm = np.log(1+con) - con/(1+con)

        result = (cterm + sterm - exterm) / norm
        # u(k→0) = 1 analytically; sici terms diverge and cancel at k=0 producing NaN
        result[~np.isfinite(result)] = 1.0
        return result

    def conc(self, M, cnorm=7.85, alpha=0.71, beta=-0.081, m0=2e12):
            return cnorm/(1+self.config.z)**(alpha) * (M / m0)**(beta)


MODELS = {'nfw': NFW}