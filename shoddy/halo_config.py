
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
        # rho_m must be the comoving (present-day) mean matter density, not the
        # physical density at z.  The Lagrangian radius R_L satisfies
        # M = (4π/3) rho_m R_L^3 where rho_m is constant in comoving coords.
        # Using rho_m(z) = rho_m0*(1+z)^3 would give a physical R_L that is
        # (1+z) times too small, causing sigma(M) and dn/dM to be wrong at z>0.
        H0 = self.cosmo.hubble_parameter(0)
        rhocrit0 = 3 * H0**2 / (8 * np.pi * G)
        self.rho_m = rhocrit0 * (self.cosmo.get_Omega('cdm', 0) + self.cosmo.get_Omega('baryon', 0) + self.cosmo.get_Omega("nu", 0))

        if mass_grid is None:
            mass_grid = np.logspace(9, 17, 256)
        self._build_sigma_interp(mass_grid, z_sigma_idx)

    def _build_sigma_interp(self, mass_grid, z_sigma_idx=-1):
        ln_m = np.log(mass_grid)
        r_lag = self.lagrangian_radius(mass_grid)
        sig = self.cosmo.get_sigmaR(r_lag, z_indices=z_sigma_idx, hubble_units=False)
        lnsig = np.log(sig)
        self._sigma_interp = make_interp_spline(ln_m, lnsig, k=3)
        # pre-compute dlnsig/dlnM on the grid for use in hmf
        dlnsig_dlnm = self._sigma_interp.derivative()(ln_m)
        self._dlnsig_dlnm_interp = make_interp_spline(ln_m, dlnsig_dlnm, k=3)

    def lagrangian_radius(self, M):
        return (3 * M / (4 * np.pi * self.rho_m))**(1/3)

    def sigma_m(self, M):
        return np.exp(self._sigma_interp(np.log(M)))

    def dlnsig_dlnm(self, M):
        return self._dlnsig_dlnm_interp(np.log(M))

    def rvir(self, M):
        return (3 * M / (4 * np.pi * self.rhocrit * self.delta))**(1/3)