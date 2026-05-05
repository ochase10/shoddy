
from . import mass_function, profile, hod
from .utils import *

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
            halo_bias='tinker10',
            halo_prof='NFW',
            halo_prof_u=None,
            hod=None,
            hod_pars={},
            halo_mass_grid=None,
            k_grid=None,
            **kwargs
    ):
        self.z = z

        if halo_mass_grid is not None:
            self.ms = halo_mass_grid
        else:
            self.ms = np.logspace(np.log10(1e9), np.log10(1e17), 256)

        self.log_ms = np.log10(self.ms)

        if k_grid is not None:
            self.ks = k_grid
        else:
            self.ks = np.logspace(np.log10(1e-4), np.log10(1e2), 1001)

        self.set_hmf(hmf)
        self.set_halo_profile(halo_prof, halo_prof_u)


        if hod is not None and type(hod_pars) is dict:
            self.set_hod(hod, hod_pars)
        elif type(hod_pars) is not dict:
            raise Exception("hod_pars argument should be type dict")
        else:
            self.hod = None

        #### Set up cosmology ###
        self.cosmo_pars = self._default_cosmo_pars.copy()
        if cosmo_pars is not None:
            for key, val in cosmo_pars.items():
                self.cosmo_pars[key] = val

        self.rhocrit0 = (3*self.cosmo_pars['H0']**2/(8*np.pi*G)) # Msun / Mpc^3
    
        self.init_cosmo(cosmo_pars)
        ###

    def init_cosmo(self, pars):

        cambpars = camb.set_params(**pars, 
                                   lmax=2000,
                                   WantTransfer=True,
                                   WantCls=False)
        cambpars.set_matter_power(redshifts=[self.z], kmax=max(self.ks)*2)

        self.cosmo = camb.get_results(cambpars)
    

    def matter_power_spectrum(self, ks):

        if self.pkm_interp is None:
            self.pkm_interp = lambda k: self.cosmo.get_matter_power_interpolator(nonlinear=False, hubble_units=False, k_hunit=False).P(self.z, k)
        
        return self.pkm_interp(ks)


    def set_hmf(self, new_hmf, **kwargs):
        if type(new_hmf) is str:
            new_hmf = new_hmf.lower()

            if new_hmf == 'tinker':
                self.HMF = mass_function.Tinker(self.cosmo, self.z, **kwargs)

            elif new_hmf == 'behroozi':
                self.hmf = mass_function.Behroozi13(self.cosmo, self.z, **kwargs)
            
            else:
                raise Exception("HMF not recognized")
    
        elif isinstance(new_hmf, hod.HOD):
                self.hod = new_hmf

        else:
            raise Exception("hmf argument must be string or MassFunction object")
        

    def set_halo_profile(self, new_prof, new_prof_u=None):
        if type(new_prof) is str:
            new_prof = new_prof.lower()

            if new_prof == 'NFW':
                self.prof = profile.nfw
                self.prof_u = profile.u_nfw
            
            else:
                raise Exception("Halo profile not recognized")
    
        elif callable(new_prof):
                self.prof = new_prof
                
                if callable(new_prof_u):
                    self.prof_u = new_prof_u
                else:
                    self.prof_u = profile.u_func

        else:
            raise Exception("halo_prof argument must be string or callable")
        
    
    def set_hod(self, new_hod, pars):
        if type(new_hod) is str:
            new_hod = new_hod.lower()

            if new_hod == 'Zheng07':
                self.hod = hod.Zheng07(**pars)
            
            else:
                raise Exception("Halo profile not recognized")
    
        elif isinstance(new_hod, hod.HOD):
                self.hod = new_hod

        else:
            raise Exception("halo_prof argument must be string or HOD object")


    def halo_integral(self, quant):
        return 1. #TODO define this integral using quad or trapz


    def galaxy_density(self, recompute=False):
        if self.hod is None:
            raise Exception("HOD must be defined to get galaxy density")

        if self.n_gal is None or recompute:
            self.n_gal = self.halo_integral(self.hod.N_hod(self.ms))

        return self.n_gal
    
    def Pk_cs(M_min, sig_logM, M0, M1, alpha):

        ng = n_gal(M_min, sig_logM, M0, M1, alpha, z=z)

        ave_ncns = avg_NcNs(Ms, M_min, sig_logM, M0, M1, alpha)[None,:]
        hmf = hmf(Ms, z=z)[None,:]
        
        u = u_nfw(ks, Ms, z=z) # k x m

        igrand = ave_ncns * hmf * u
        res = np.trapz(igrand, Ms, axis=1)

        return 2.0 * res / ng**2

    def Pk_ss(M_min, sig_logM, M0, M1, alpha, z=z0, ks=ks, Ms=Ms, hmf=behroozi_hmf):

        ng = n_gal(M_min, sig_logM, M0, M1, alpha, z=z)

        ave_ns2 = avg_Ns2(Ms, M_min, sig_logM, M0, M1, alpha)[None,:]
        hmf = hmf(Ms, z=z)[None,:]
        
        u = u_nfw(ks, Ms, z=z) # k x m

        igrand = ave_ns2 * hmf * u**2
        res = np.trapz(igrand, Ms, axis=1)

        return res / ng**2


    def Pk_1h(M_min, sig_logM, M0, M1, alpha, z=z0, ks=ks, Ms=Ms, hmf=behroozi_hmf):

        p_cs = Pk_cs(M_min, sig_logM, M0, M1, alpha, z=z, ks=ks, Ms=Ms, hmf=hmf)
        p_ss = Pk_ss(M_min, sig_logM, M0, M1, alpha, z=z, ks=ks, Ms=Ms, hmf=hmf)

        return p_cs + p_ss

    def Pk_2h(M_min, sig_logM, M0, M1, alpha, ks=ks, m_pk=Pk_z, z=z0, Ms=Ms, hmf=behroozi_hmf):

        ng = n_gal(M_min, sig_logM, M0, M1, alpha, z=z)

        N_of_M = N_hod(Ms, M_min, sig_logM, M0, M1, alpha)[None,:]
        hmf = hmf(Ms, z=z)[None,:]
        b_h = halo_bias(Ms, z)[None,:]
        
        u = u_nfw(ks, Ms, z=z) # k x m

        igrand = N_of_M * hmf * b_h * u
        res = (np.trapz(igrand, Ms, axis=1) / ng)**2

        return m_pk * res

    def P_gal(M_min, sig_logM, M0, M1, alpha, ks=ks, m_pk=Pk_z, z=z0, Ms=Ms, hmf=behroozi_hmf, trunc_1h=None):
        
        p_1h = Pk_1h(M_min, sig_logM, M0, M1, alpha, z=z, ks=ks, Ms=Ms, hmf=hmf) 
        if trunc_1h is not None:
            p_1h *= (1 - np.exp(-ks/trunc_1h))
        p_2h = Pk_2h(M_min, sig_logM, M0, M1, alpha, z=z, ks=ks, Ms=Ms, hmf=hmf, m_pk=m_pk)

        return p_1h + p_2h