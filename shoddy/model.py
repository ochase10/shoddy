
from . import mass_function, profile, hod
from .utils import *
from .halo_config import HaloConfig

import camb
import numpy as np


class Model:

    _default_cosmo_pars = {
        'H0': 70.,
        'omch2': 0.25 * 0.7**2,
        'ombh2': 0.05 * 0.7**2,
        'omk': 0.0,
        'As': 2e-9,
        'ns': 0.96,
        'mnu': 0.0
    }

    def __init__(
            self,
            z=0,
            cosmo_pars=None,
            hmf='tinker',
            halo_prof='nfw',
            hod=None,
            hod_pars={},
            halo_mass_grid=None,
            k_grid=None,
            **kwargs
    ):

        if halo_mass_grid is not None:
            self.ms = halo_mass_grid
        else:
            self.ms = np.logspace(np.log10(1e9), np.log10(1e17), 256)

        self.log_ms = np.log10(self.ms)

        if k_grid is not None:
            self.ks = k_grid
        else:
            self.ks = np.logspace(np.log10(1e-4), np.log10(1e2), 1001)

        #### Set up cosmology ###
        self.cosmo_pars = self._default_cosmo_pars.copy()
        if cosmo_pars is not None:
            for key, val in cosmo_pars.items():
                self.cosmo_pars[key] = val

        self.rhocrit0 = (3*self.cosmo_pars['H0']**2/(8*np.pi*G)) # Msun / Mpc^3
    
        self.init_cosmo(cosmo_pars)
        ###

        self.halo_data = HaloConfig(self.cosmo, z, **kwargs)

        self.set_hmf(hmf, self.halo_data)
        self.set_halo_profile(halo_prof, self.halo_data)

        if hod is not None:
            self.set_hod(hod, hod_pars)
        else:
            self.hod = None


    def init_cosmo(self, pars):

        cambpars = camb.set_params(**pars, 
                                   lmax=2000,
                                   WantTransfer=True,
                                   WantCls=False)
        cambpars.set_matter_power(redshifts=[self.halo_data.z], kmax=max(self.ks)*2)

        self.cosmo = camb.get_results(cambpars)
    

    def matter_power_spectrum(self, ks):

        if self.pkm_interp is None:
            self.pkm_interp = lambda k: self.cosmo.get_matter_power_interpolator(nonlinear=False, hubble_units=False, k_hunit=False).P(self.halo_data.z, k)
        
        return self.pkm_interp(ks)


    def set_hmf(self, new_hmf, config, **kwargs):
        if type(new_hmf) is str:
            new_hmf = new_hmf.lower()

            if new_hmf == 'tinker':
                self.HMF = mass_function.Tinker(config, **kwargs)

            elif new_hmf == 'behroozi':
                self.HMF = mass_function.Behroozi13(config, **kwargs)
            
            else:
                raise Exception("HMF not recognized")
    
        elif isinstance(new_hmf, mass_function.MassFunction):
                self.HMF = new_hmf

        else:
            raise Exception("hmf argument must be string or MassFunction object")
        

    def set_halo_profile(self, new_prof, config, **kwargs):
        if type(new_prof) is str:
            new_prof = new_prof.lower()

            if new_prof == 'nfw':
                self.prof = profile.NFW(config, **kwargs)
            
            else:
                raise Exception("Halo profile not recognized")
    
        elif isinstance(new_prof, profile.HaloProfile):
                self.prof = new_prof

        else:
            raise Exception("halo_prof argument must be string or callable")
        
    
    def set_hod(self, new_hod, pars={}):
        
        if type(pars) is not dict:
            raise Exception("HOD parameters argument should be type dict")
        
        elif type(new_hod) is str:
            new_hod = new_hod.lower()

            if new_hod == 'zheng07':
                self.hod = hod.Zheng07(**pars)
            
            else:
                raise Exception("Halo profile not recognized")
    
        elif isinstance(new_hod, hod.HOD):
                self.hod = new_hod

        else:
            raise Exception("halo_prof argument must be string or HOD object")
        

    def check_HOD_defined(self):
        if self.hod is None:
            raise Exception("HOD must be defined to get galaxy density")


    def galaxy_density(self, Ms=None, recompute=False, **kwargs):
        self.check_HOD_defined()
        assert self.hod is not None

        if self.n_gal is None or Ms is not None or recompute:
                    
            if Ms is None:
                Ms = self.ms

            self.n_gal = self.HMF.halo_integral(Ms, self.hod.N_hod(Ms), **kwargs)

        return self.n_gal
    

    def Pk_cs(self, ks=None, Ms=None, **kwargs):
        self.check_HOD_defined()
        assert self.hod is not None
        
        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        ng = self.galaxy_density(Ms, **kwargs)

        ave_ncns = self.hod.avg_NcNs(Ms)[None,:]
        
        u = self.prof.k_profile(ks, Ms) # k x m

        igrand = ave_ncns * u
        res = self.HMF.halo_integral(Ms, igrand, axis=1, **kwargs)

        return 2.0 * res / ng**2



    def Pk_ss(self, ks=None, Ms=None, **kwargs):

        self.check_HOD_defined()
        assert self.hod is not None
        
        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        ng = self.galaxy_density(Ms, **kwargs)

        ave_ns2 = self.hod.avg_Ns2(Ms)[None,:]
        
        u = self.prof.k_profile(ks, Ms) # k x m

        igrand = ave_ns2 * u**2
        res = self.HMF.halo_integral(Ms, igrand, axis=1, **kwargs)

        return res / ng**2


    def Pk_1h(self, ks=None, Ms=None, **kwargs):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        p_cs = self.Pk_cs(Ms, ks, **kwargs)
        p_ss = self.Pk_ss(Ms, ks, **kwargs)

        return p_cs + p_ss

    def Pk_2h(self, ks=None, Ms=None, **kwargs):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks
        ng = self.galaxy_density(Ms, **kwargs)

        N_of_M = self.hod.N_hod(Ms)[None,:]

        b_h = self.HMF.bias(Ms)[None,:]
        
        u = self.prof.k_profile(ks, Ms) # k x m

        igrand = N_of_M * b_h * u
        res = (self.HMF.halo_integral(Ms, igrand, axis=1, **kwargs) / ng)**2

        return self.matter_power_spectrum(ks) * res

    def P_gal(self, Ms=None, ks=None, trunc_1h_k=None, **kwargs):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        p_1h = self.Pk_1h(ks=ks, Ms=Ms, **kwargs) 
        if trunc_1h_k is not None:
            p_1h *= (1 - np.exp(-ks/trunc_1h_k))
        p_2h = self.Pk_2h(ks=ks, Ms=Ms, **kwargs)

        return p_1h + p_2h