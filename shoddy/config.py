
from .utils import *
from . import mass_function, occupation, profile

import camb
from scipy.interpolate import make_interp_spline


class HaloConfig:

    _default_cosmo_pars = {
        'H0': 70.,
        'omch2': 0.25 * 0.7**2,
        'ombh2': 0.05 * 0.7**2,
        'omk': 0.0,
        'As': 2e-9,
        'ns': 0.96,
        'mnu': 0.0,
        'lmax': 2000,
        'WantTransfer': True,
        'WantCls': False
    }

    def __init__(self,
                 z,
                 cosmo_pars=None,
                 mass_grid=None,
                 z_range=None,
                 kmax=200.,
                 dens_crit=1.686,
                 delta=200.):

        self.z = z
        self.crit = dens_crit
        self.delta = delta

        self.cosmo_pars = self._default_cosmo_pars.copy()
        if cosmo_pars is not None:
            self.cosmo_pars.update(cosmo_pars)
        self.init_cosmo(self.cosmo_pars, kmax, z_range=z_range)

        self.rhocrit = (3*self.cosmo.hubble_parameter(self.z)**2/(8*np.pi*G))

        H0 = self.cosmo.hubble_parameter(0)
        rhocrit0 = 3 * H0**2 / (8 * np.pi * G)
        self.rho_m = rhocrit0 * (self.cosmo.get_Omega('cdm', 0) + self.cosmo.get_Omega('baryon', 0) + self.cosmo.get_Omega("nu", 0))

        if mass_grid is None:
            mass_grid = np.logspace(9, 17, 256)
        elif len(mass_grid) <= 2:
            raise ValueError("Halo mass grid must contain more than 2 values")
        self.mass_grid = np.asarray(mass_grid).copy()
        self._build_sigma_interp(self.mass_grid, self._z_sigma_idx)





    def init_cosmo(self, pars, kmax, z_range=None,
                   z_pad = 3.,
                   z_step = 0.1):
        """
        Run CAMB and cache the results.
        """

        # CAMB max z grid size is 256
        Z_MAX_PTS = 256

        cambpars = camb.set_params(**pars)

        if z_range is None:
            lo, hi = self.z - z_pad, self.z + z_pad
        else:
            lo, hi = min(z_range), max(z_range)

        lo = max(0., min(lo, self.z))
        hi = max(hi, self.z)

        npts = int(np.clip(round((hi - lo) / z_step) + 1, 2, Z_MAX_PTS - 1))
        usezs = np.linspace(lo, hi, npts)
        if not np.any(np.isclose(usezs, self.z)):
            usezs = np.append(usezs, self.z)
        usezs = np.sort(usezs)[::-1]

        cambpars.set_matter_power(redshifts=usezs, kmax=kmax, nonlinear=False)

        self.cosmo = camb.get_results(cambpars)
        self._z_sigma_idx = int(np.argmin(np.abs(usezs - self.z)))

    
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
