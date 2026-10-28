
import copy
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
            pars=None,
            gal_prof='gauss',
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
        
        
        hod_cls, gal_cls, hod_pars, gal_pars = None, None, None, None
        if hod is not None:
            hod_cls = lookup(occupation.MODELS, hod, "HOD")

        if gal_prof is not None:
            gal_cls = lookup(subgalaxy.MODELS, gal_prof, "Galaxy Occupation")

        if pars:
            hod_pars, gal_pars = self._separate_pars(pars, hod_cls, gal_cls)

        if hod_cls is not None:
            self.set_hod(hod_cls, hod_pars)
        else:
            self._hod = None

        self.set_gal_occ(gal_cls, gal_pars)

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
            raise TypeError("Parameters argument should be a dict")
        self._hod = lookup(occupation.MODELS, new_hod, "HOD")(**pars)


    def set_gal_occ(self, new_occ, pars=None):
        if pars is None:
            pars = {}
        if not isinstance(pars, dict):
            raise TypeError("Parameters argument should be a dict")
        self._gal_occ = lookup(subgalaxy.MODELS, new_occ, "Galaxy Occupation")(**pars)


    def _separate_pars(self, pars_dict, hod=None, gal=None):
        hod_pars, gal_pars = {}, {}

        if hod is None and gal is None:
            raise ValueError("No occupation prescription defined to assign parameters to.")

        for key, value in pars_dict.items():
            if hod is not None and key in hod.PARS:
                hod_pars[key] = value
            elif gal is not None and key in gal.PARS:
                gal_pars[key] = value
            else:
                raise ValueError(f"Unrecognized parameter key `{key}`")

        return hod_pars, gal_pars


    def update_pars(self, **new_pars):
        hod_pars, gal_pars = self._separate_pars(new_pars, self._hod, self._gal_occ)
        if hod_pars:
            self.hod.update_pars(**hod_pars)
        if gal_pars:
            self.gal_occ.update_pars(**gal_pars)


    def with_pars(self, new_pars):
        m = copy.copy(self)

        hod_pars, gal_pars = self._separate_pars(new_pars, self._hod, self._gal_occ)

        if self._hod is not None:
            m._hod = self.hod.with_pars(hod_pars)
        if self._gal_occ is not None:
            m._gal_occ = self.gal_occ.with_pars(gal_pars)

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

    
    def Pk_cs(self, Ms, u_h, ng, u_g):
        igrand = self.hod.avg_NcNs(Ms)[None, :] * u_h
        res = self.hmf.halo_integral(Ms, igrand, axis=1)
        return 2.0 * res * u_g**2 / ng**2


    def Pk_ss(self, Ms, u_h, ng, u_g):
        igrand = self.hod.avg_Ns2(Ms)[None, :] * u_h**2
        res = self.hmf.halo_integral(Ms, igrand, axis=1)
        return res * u_g**2 / ng**2


    def _grids(self, ks=None, Ms=None):
        ks = self.ks if ks is None else ks
        Ms = self.ms if Ms is None else Ms
        return ks, Ms


    def Pk_1h(self, ks=None, Ms=None, damp=None):
        u_h = self.halo_profile.u(ks, Ms)
        ks, Ms = self._grids(ks, Ms)

        u_g = self.gal_occ.u(self._physical_k_kpc(ks))
        ng = self.n_gal(Ms)

        p_1h = self.Pk_cs(Ms, u_h, ng, u_g) + self.Pk_ss(Ms, u_h, ng, u_g)

        if damp is None or damp is False:
            return p_1h
        elif damp is True:
            damp = self.k_star
        elif not isinstance(damp, Real):
            raise TypeError("damp must be None, bool, or a real number")
        
        return p_1h * -np.expm1(-(np.asarray(ks) / damp)**2)


    def F_k(self, ks=None, Ms=None):
        u_h = self.halo_profile.u(ks, Ms)
        ks, Ms = self._grids(ks, Ms)

        ng = self.n_gal(Ms)
        igrand = self.hod.N_hod(Ms)[None, :] * self.hmf.bias(Ms)[None, :] * u_h

        u_g = self.gal_occ.u(self._physical_k_kpc(ks))
        return self.hmf.halo_integral(Ms, igrand, axis=1) * u_g / ng


    def Pk_2h(self, ks=None, z=None, Ms=None):
        Fk = self.F_k(ks, Ms)
        ks, _ = self._grids(ks)
        return self.Pk_m(ks, z) * Fk**2


    def _physical_k_kpc(self, k):
        return k * (1+self.z) / 1000 


    def Pk_1g(self, ks=None, Ms=None):
        ks, Ms = self._grids(ks, Ms)
        
        u_g = self.gal_occ.u(self._physical_k_kpc(ks))
        ng = self.n_gal(Ms)

        # No integral here because it reduces to n_gal times factors.
        # That cancels with 1 ng in the denominator. 
        # DC enters here so that the 1g term is indpendent of it since it 
        #  enters into ng linearly (ng = DC * n_host)
        return u_g**2 * (self.gal_occ.pars['corr']+1) * self.hod.pars['dc'] / ng


    def P_tot(self, terms=None, ks=None, z=None, Ms=None, damp_1h=None):

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
          '2h': (lambda m, ks=None, Ms=None, **kw: m.F_k(ks=ks, Ms=Ms)**2, 'pkm'),
          '1g': (lambda m, ks=None, Ms=None, **kw: m.Pk_1g(ks=ks,Ms=Ms), 'k')}
        