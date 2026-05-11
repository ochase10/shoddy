
from .utils import *

class HaloConfig:

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
    
        self.rhocrit = (3*self.cosmo.hubble_parameter(self.z)**2/(8*np.pi*G))
        self.rho_m = self.rhocrit * (self.cosmo.get_Omega('cdm', self.z) + self.cosmo.get_Omega('baryon', self.z) + self.cosmo.get_Omega("nu", self.z))

    def lagrangian_radius(self, M):
        return (3 * M / (4 * np.pi * self.rho_m))**(1/3)

    def sigma_m(self, M):
        return self.cosmo.get_sigmaR(self.lagrangian_radius(M), z_indices=-1, hubble_units=False)
    
    def rvir(self, M):
        return self.lagrangian_radius(M) / self.delta**(1/3)
    
    def set_delta(self, new_delta):
        self.delta = new_delta

    def set_crit(self, new_crit):
        self.crit = new_crit

