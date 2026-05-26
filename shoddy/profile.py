
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from scipy.special import sici



class HaloProfile(ABC):

    def __init__(self, config):
        self.config = config
        self._u_cache = None
        self._cache_key = None

    @staticmethod
    def _ks_fingerprint(ks):
        """O(1) fingerprint: catches same-values-different-object across calls."""
        n = len(ks)
        return (n, ks.dtype.str, float(ks[0]), float(ks[n // 2]), float(ks[-1]))

    def k_profile(self, ks, M, recompute=False) -> NDArray[np.floating]:
        key = (self._ks_fingerprint(ks), id(M))
        if not recompute and self._u_cache is not None and key == self._cache_key:
            return self._u_cache
        result = self._compute_profile(ks, M)
        self._u_cache = result
        self._cache_key = key
        return result

    @abstractmethod
    def _compute_profile(self, ks, M) -> NDArray[np.floating]:
        pass

    def conc(self, M, cnorm=7.85, alpha=0.71, beta=-0.081, m0=2e12):
        return cnorm/(1+self.config.z)**(alpha) * (M / m0)**(beta)

    def Ac(self, c):
        return np.log(1+c) - c/(1+c)


class NFW(HaloProfile):

    def _compute_profile(self, ks, M):
        if not isinstance(M, np.ndarray):
            M = np.array(M)

        r_vir = self.config.rvir(M)
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

        return (cterm + sterm - exterm) / norm