
import copy, warnings
from numbers import Real

import numpy as np
from . import mass_function, occupation, profile
from .utils import C, _trapz, lookup
from .config import HaloConfig, _EXTRAP_K_MAX, _INTERP_K_MAX


def _check_k_grid(ks, warn_k=1e4):
    if np.max(ks) < warn_k:
        warnings.warn("k grid may not extend high enough to allow for accurate \
                       correlation function computations.")


class Model():

    def __init__(
            self,
            z=0,
            cosmo_pars=None,
            hmf='behroozi',
            halo_prof='nfw',
            hod=None,
            hod_pars=None,
            halo_mass_grid=None,
            k_grid=None,
            z_range=None,
            camb_kmax=_INTERP_K_MAX):

        if k_grid is not None:
            self.ks = np.asarray(k_grid).copy()
            _check_k_grid(self.ks)

        else:
            self.ks = np.logspace(np.log10(1e-4), np.log10(_EXTRAP_K_MAX), 1501)

        self.config = HaloConfig(z,
                                 cosmo_pars,
                                 mass_grid=halo_mass_grid,
                                 z_range=z_range,
                                 kmax=camb_kmax)

        self.profile = lookup(profile.MODELS, halo_prof, "Profile")(self.config)
        self.u_grid = self.profile.u(self.ks, self.ms)

        self.hmf = lookup(mass_function.MODELS, hmf, "HMF")(self.config)
        self.dndm = self.hmf.dndm(self.ms)

        if hod is not None:
            self.set_hod(hod, hod_pars)
        else:
            self._hod = None

        sig_d2 = _trapz(self.Pk_m(self.ks), self.ks) / (6 * np.pi**2)
        self.k_star = 0.584 / np.sqrt(sig_d2)


    @property
    def z(self):
        return self.config.z

    @property
    def ms(self):
        return self.config.mass_grid

    @property
    def hod(self):
        if self._hod is None:
            raise ValueError("HOD not defined")
        return self._hod


    def Pk_m(self, ks, z=None):
        return self.config.Pk_m(ks, z)


    def set_hod(self, new_hod, pars=None):
        if pars is None:
            pars = {}
        if not isinstance(pars, dict):
            raise TypeError("HOD parameters argument should be a dict")
        self._hod = lookup(occupation.MODELS, new_hod, "HOD")(**pars)


    def update_hod_pars(self, **new_pars):
        self.hod.update_pars(**new_pars)


    def with_hod(self, hod_pars, new_hod=None):
        m = copy.copy(self)

        if new_hod is None:
            m._hod = self.hod.with_pars(hod_pars)
        else:
            m.set_hod(new_hod, hod_pars)

        return m


    def n_gal(self, Ms=None):
        if Ms is None:
            Ms = self.ms
        return float(self.hmf.halo_integral(Ms, self.hod.N_hod(Ms)))


    def galaxy_bias(self, Ms=None):
        if Ms is None:
            Ms = self.ms

        ng = self.n_gal(Ms)
        n_avg = self.hod.N_hod(Ms)
        b_halo = self.hmf.bias(Ms)

        return self.hmf.halo_integral(Ms, n_avg*b_halo) / ng

    
    def Pk_cs(self, Ms, u, ng):
        igrand = self.hod.avg_NcNs(Ms)[None, :] * u
        res = self.hmf.halo_integral(Ms, igrand, axis=1)
        return 2.0 * res / ng**2


    def Pk_ss(self, Ms, u, ng):

        igrand = self.hod.avg_Ns2(Ms)[None, :] * u**2
        res = self.hmf.halo_integral(Ms, igrand, axis=1)
        return res / ng**2


    def _k_m_profile(self, ks, Ms):
        use_cache = ks is None and Ms is None
        Ms = self.ms if Ms is None else Ms
        ks = self.ks if ks is None else ks

        u = self.u_grid if use_cache else self.profile.u(ks, Ms)

        return ks, Ms, u


    def Pk_1h(self, ks=None, Ms=None, damp=None):
        ks, Ms, u = self._k_m_profile(ks, Ms)

        ng = self.n_gal(Ms)

        p_1h = self.Pk_cs(Ms, u, ng) + self.Pk_ss(Ms, u, ng)

        if damp is None or damp is False:
            return p_1h
        elif damp is True:
            damp = self.k_star
        elif not isinstance(damp, Real):
            raise TypeError("damp must be None, bool, or a real number")
        
        return p_1h * -np.expm1(-(np.asarray(ks) / damp)**2)


    def F_k(self, ks=None, Ms=None):
        ks, Ms, u = self._k_m_profile(ks, Ms)
        
        ng = self.n_gal(Ms)
        igrand = self.hod.N_hod(Ms)[None, :] * self.hmf.bias(Ms)[None, :] * u
        
        return self.hmf.halo_integral(Ms, igrand, axis=1) / ng


    def Pk_2h(self, ks=None, z=None, Ms=None):
        Fk = self.F_k(ks, Ms)
        ks = self.ks if ks is None else ks
        return self.Pk_m(ks, z) * Fk**2


    def P_gal(self, terms=None, ks=None, z=None, Ms=None, damp_1h=None):

        z_ind, z_dep = self._resolve_terms(terms, ks=ks, Ms=Ms, damp=damp_1h)
        z_ind = 0 if z_ind is None else z_ind
        z_dep = 0 if z_dep is None else z_dep
        
        ks = self.ks if ks is None else ks
        return z_ind + self.Pk_m(ks, z) * z_dep


    def _resolve_terms(self, terms, **kwargs):
        if isinstance(terms, str):
            terms = [terms]
        if not terms:
            terms = _TERMS.keys()
        A = None
        B = None
        for term in terms:
            fn, which = _TERMS[term]
            amp = fn(self, **kwargs)
            if which == 'k':
                if A is None:
                    A = amp
                else:
                    A = A + amp
            elif which == 'pkm':
                if B is None:
                    B = amp
                else:
                    B = B + amp
            else:
                raise ValueError("Unrecognized term type")

        return (A, B)

    
_TERMS = {'1h': (lambda m, ks=None, Ms=None, damp=None, **kw: m.Pk_1h(ks=ks, Ms=Ms, damp=damp), 'k'),
          '2h': (lambda m, ks=None, Ms=None, **kw: m.F_k(ks=ks, Ms=Ms)**2, 'pkm')}
        