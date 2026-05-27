
from .utils import *
from scipy.interpolate import make_interp_spline


class HaloConfig:

    def __init__(self,
                 cosmology,
                 z,
                 dens_crit=1.686,
                 delta=200.,
                 mass_grid=None,
                 z_sigma_idx=-1,
                 **kwargs):

        self.cosmo = cosmology
        self.z = z
        self.crit = dens_crit
        self.delta = delta

        self.rhocrit = (3*self.cosmo.hubble_parameter(self.z)**2/(8*np.pi*G))
        self.rho_m = self.rhocrit * (self.cosmo.get_Omega('cdm', self.z) + self.cosmo.get_Omega('baryon', self.z) + self.cosmo.get_Omega("nu", self.z))

        if mass_grid is None:
            mass_grid = np.logspace(9, 17, 256)
        self._build_sigma_interp(mass_grid, z_sigma_idx)

    def _build_sigma_interp(self, mass_grid, z_sigma_idx=-1):
        log_m = np.log10(mass_grid)
        r_lag = self.lagrangian_radius(mass_grid)
        sig = self.cosmo.get_sigmaR(r_lag, z_indices=z_sigma_idx, hubble_units=False)
        log_sig = np.log10(sig)
        self._sigma_interp = make_interp_spline(log_m, log_sig, k=3)
        # pre-compute dlnsig/dlnM on the grid for use in hmf
        dlogsig_dlogm = self._sigma_interp.derivative()(log_m)
        self._dlnsig_dlnm_interp = make_interp_spline(log_m, dlogsig_dlogm, k=3)

    def lagrangian_radius(self, M):
        return (3 * M / (4 * np.pi * self.rho_m))**(1/3)

    def sigma_m(self, M):
        return 10**self._sigma_interp(np.log10(M))

    def dlnsig_dlnm(self, M):
        return self._dlnsig_dlnm_interp(np.log10(M))

    def rvir(self, M):
        return (3 * M / (4 * np.pi * self.rhocrit * self.delta))**(1/3)

    def set_delta(self, new_delta):
        self.delta = new_delta

    def set_crit(self, new_crit):
        self.crit = new_crit

