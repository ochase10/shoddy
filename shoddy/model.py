
import copy, warnings
from numbers import Real

import numpy as np
from . import mass_function, occupation, profile, subgalaxy
from .utils import _trapz, lookup
from .config import HaloConfig


class Model():

    def __init__(
            self,
            z=0,
            cosmo_pars=None,
            hmf='behroozi',
            halo_prof='nfw',
            hod=None,
            pars={},
            gal_prof=None,
            halo_mass_grid=None,
            k_grid=None,
            z_range=None,
            camb_kmax=None):

        self.config = HaloConfig(z=z,
                                 cosmo_pars=cosmo_pars,
                                 mass_grid=halo_mass_grid,
                                 z_range=z_range,
                                 k_grid=k_grid,
                                 kmax=camb_kmax)



        self.halo_profile = lookup(profile.MODELS, halo_prof, "Profile")(self.config)
        self.hmf = lookup(mass_function.MODELS, hmf, "HMF")(self.config)

        if hod is not None:
            self.set_hod(hod)
        else:
            self._hod = None

        if gal_prof is not None:
            self.set_gal_occ(gal_prof)
        else:
            self._gal_occ = None

        if pars:
            self.update_pars(**pars)



        self.dndm = self.hmf.dndm(self.ms)

        sig_d2 = _trapz(self.Pk_m(self.ks), self.ks) / (6 * np.pi**2)
        self.k_star = 0.584 / np.sqrt(sig_d2)


    @property
    def z(self):
        return self.config.z

    @property
    def ms(self):
        return self.config.mass_grid

    @property
    def ks(self):
        return self.config.ks

    @property
    def hod(self):
        if self._hod is None:
            raise ValueError("HOD not defined")
        return self._hod

    @property
    def gal_occ(self):
        if self._gal_occ is None:
            raise ValueError("Galaxy occupation not defined")
        return self._gal_occ


    def Pk_m(self, ks, z=None):
        return self.config.Pk_m(ks, z)


    def set_hod(self, new_hod, pars=None):
        if pars is None:
            pars = {}
        if not isinstance(pars, dict):
            raise TypeError("HOD parameters argument should be a dict")
        self._hod = lookup(occupation.MODELS, new_hod, "HOD")(**pars)


    def set_gal_occ(self, new_occ, pars=None):
        if pars is None:
            pars = {}
        if not isinstance(pars, dict):
            raise TypeError("HOD parameters argument should be a dict")
        
        self._gal_occ = lookup(subgalaxy.MODELS, new_occ, "Galaxy Occupation")(**pars)


    def _separate_pars(self, pars_dict):
        hod_pars, gal_pars = {}, {}

        if self._hod is None and self._gal_occ is None:
            raise ValueError("No occupation prescription defined to assign parameters to.")

        for key, value in pars_dict.items():
            if self._hod is not None and key in self.hod.PARS:
                hod_pars[key] = value
            elif self._gal_occ is not None and key in self.gal_occ.PARS:
                gal_pars[key] = value
            elif self._hod is not None or self._gal_occ is not None:
                raise ValueError(f"Unrecognized parameter key `{key}`")

        return hod_pars, gal_pars


    def update_pars(self, **new_pars):
        hod_pars, gal_pars = self._separate_pars(new_pars)
        self.hod.update_pars(**hod_pars)
        self.gal_occ.update_pars(**gal_pars)


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


    def _grids(self, ks=None, Ms=None):
        ks = self.ks if ks is None else ks
        Ms = self.ms if Ms is None else Ms
        return ks, Ms


    def Pk_1h(self, ks=None, Ms=None, damp=None):
        u = self.halo_profile.u(ks, Ms)
        ks, Ms = self._grids(ks, Ms)

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
        u = self.halo_profile.u(ks, Ms)
        ks, Ms = self._grids(ks, Ms)
        
        ng = self.n_gal(Ms)
        igrand = self.hod.N_hod(Ms)[None, :] * self.hmf.bias(Ms)[None, :] * u
        
        return self.hmf.halo_integral(Ms, igrand, axis=1) / ng


    def Pk_2h(self, ks=None, z=None, Ms=None):
        Fk = self.F_k(ks, Ms)
        ks, _ = self._grids(ks)
        return self.Pk_m(ks, z) * Fk**2


    def P_gal(self, terms=None, ks=None, z=None, Ms=None, damp_1h=None):

        z_ind, z_dep = self._resolve_terms(terms, ks=ks, Ms=Ms, damp=damp_1h)
        z_ind = 0 if z_ind is None else z_ind
        z_dep = 0 if z_dep is None else z_dep
        
        ks, _ = self._grids(ks)
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
        