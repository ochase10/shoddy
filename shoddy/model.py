
from . import mass_function, profile, hod
from .utils import *
from .utils import _trapz
from .halo_config import HaloConfig

import camb
import numpy as np
from scipy.interpolate import make_interp_spline
from mcfit import P2xi, Hankel


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

        #### Set up cosmology ###
        self.cosmo_pars = self._default_cosmo_pars.copy()
        if cosmo_pars is not None:
            self.cosmo_pars.update(cosmo_pars)

        self.rhocrit0 = (3*self.cosmo_pars['H0']**2/(8*np.pi*G)) # Msun / Mpc^3
    
        self.init_cosmo(self.cosmo_pars)
        ###

        self.halo_data = HaloConfig(self.cosmo, z, mass_grid=self.ms, z_sigma_idx=self._z_sigma_idx, **kwargs)

        self.set_hmf(hmf, self.halo_data)
        self.set_halo_profile(halo_prof, self.halo_data)
        self._precompute_halo_arrays()

        if hod is not None:
            self.set_hod(hod, hod_pars)
        else:
            self.hod = None


    @staticmethod
    def _ms_fingerprint(ms):
        """O(1) fingerprint matching the pattern used in HaloProfile._ks_fingerprint."""
        n = len(ms)
        return (n, ms.dtype.str, float(ms[0]), float(ms[n // 2]), float(ms[-1]))

    def _is_default_grid(self, Ms):
        return self._ms_fingerprint(Ms) == self._ms_key

    def _precompute_halo_arrays(self):
        self._hmf_arr = self.HMF.hmf(self.ms)
        self._bias_arr = self.HMF.bias(self.ms)
        self._ms_key = self._ms_fingerprint(self.ms)

    def init_cosmo(self, pars):

        cambpars = camb.set_params(**pars, 
                                   lmax=2000,
                                   WantTransfer=True,
                                   WantCls=False)
        
        usezs = np.linspace(self.z - 1.5, self.z + 1.5, 20)[::-1]
        usezs = usezs[usezs >= 0]
        if not np.any(np.isclose(usezs, self.z)):
            usezs = np.sort(np.append(usezs, self.z))[::-1]
        cambpars.set_matter_power(redshifts=usezs, kmax=max(self.ks)*2)

        self.cosmo = camb.get_results(cambpars)
        self.pkm_interp = None
        self._z_sigma_idx = int(np.argmin(np.abs(usezs - self.z)))
    

    def _ensure_pk_interp(self):
        if self.pkm_interp is None:
            self.pkm_interp = self.cosmo.get_matter_power_interpolator(
                nonlinear=False, hubble_units=False, k_hunit=False
            )

    def matter_power_spectrum(self, ks, z=None):
        """
        Linear matter power spectrum.

        Parameters
        ----------
        ks : array-like
            Wavenumbers [1/Mpc].
        z : float or array-like, optional
            Redshift(s).  Scalar (or None → self.z) evaluates at a single
            redshift and returns a 1-D array.  A 1-D array is treated as
            element-wise pairs ``(ks[i], z[i])`` and also returns 1-D —
            this avoids the Cartesian-product overhead of CAMB's default
            grid evaluation and is the path used by ``limber_cl``.
        """
        if z is None:
            z = self.z
        self._ensure_pk_interp()
        assert self.pkm_interp is not None
        # grid=False for array z: evaluate at (k_i, z_i) pairs, not all combos
        grid = np.ndim(z) == 0
        return self.pkm_interp.P(z, ks, grid=grid).ravel()


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
            raise Exception("halo_prof argument must be string or HaloProfile object")
        
    
    def set_hod(self, new_hod, pars={}):
        
        if type(pars) is not dict:
            raise Exception("HOD parameters argument should be type dict")
        
        elif type(new_hod) is str:
            new_hod = new_hod.lower()

            if new_hod == 'zheng07':
                self.hod = hod.Zheng07(**pars)
            
            else:
                raise Exception("HOD not recognized")

        elif isinstance(new_hod, hod.HOD):
                self.hod = new_hod

        else:
            raise Exception("halo_prof argument must be string or HOD object")
        
        self.n_gal = None


    def update_hod_pars(self, **new_pars):
        """Update HOD parameters and invalidate the cached galaxy density."""
        self.check_HOD_defined()
        assert self.hod is not None
        self.hod.update_pars(**new_pars)
        self.n_gal = None


    def check_HOD_defined(self):
        if self.hod is None:
            raise Exception("HOD must be defined to get galaxy density")


    def galaxy_density(self, Ms=None, recompute=False):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            if self.n_gal is not None and not recompute:
                return self.n_gal
            self.n_gal = self.HMF.halo_integral(self.ms, self.hod.N_hod(self.ms), hmf_arr=self._hmf_arr)
            return self.n_gal

        hmf_arr = self._hmf_arr if self._is_default_grid(Ms) else None
        return self.HMF.halo_integral(Ms, self.hod.N_hod(Ms), hmf_arr=hmf_arr)
    

    def Pk_cs(self, Ms, u, ng):
        self.check_HOD_defined()
        assert self.hod is not None

        hmf_arr = self._hmf_arr if self._is_default_grid(Ms) else None
        ave_ncns = self.hod.avg_NcNs(Ms)[None, :]
        igrand = ave_ncns * u
        res = self.HMF.halo_integral(Ms, igrand, axis=1, hmf_arr=hmf_arr)
        return 2.0 * res / ng**2

    def Pk_ss(self, Ms, u, ng):
        self.check_HOD_defined()
        assert self.hod is not None

        hmf_arr = self._hmf_arr if self._is_default_grid(Ms) else None
        ave_ns2 = self.hod.avg_Ns2(Ms)[None, :]
        igrand = ave_ns2 * u**2
        res = self.HMF.halo_integral(Ms, igrand, axis=1, hmf_arr=hmf_arr)
        return res / ng**2

    def Pk_1h(self, ks=None, Ms=None):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        ng = self.galaxy_density(Ms)
        u = self.prof.k_profile(ks, Ms)

        return self.Pk_cs(Ms, u, ng) + self.Pk_ss(Ms, u, ng)

    def Pk_2h(self, ks=None, z=None, Ms=None):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        ng = self.galaxy_density(Ms)
        N_of_M = self.hod.N_hod(Ms)[None, :]
        if self._is_default_grid(Ms):
            b_h = self._bias_arr[None, :]
            hmf_arr = self._hmf_arr
        else:
            b_h = self.HMF.bias(Ms)[None, :]
            hmf_arr = None
        u = self.prof.k_profile(ks, Ms)

        igrand = N_of_M * b_h * u
        res = (self.HMF.halo_integral(Ms, igrand, axis=1, hmf_arr=hmf_arr) / ng)**2

        # matter_power_spectrum handles scalar and array z uniformly (grid=False for arrays)
        return self.matter_power_spectrum(ks, z) * res


    def P_gal(self, ks=None, z=None, Ms=None, trunc_1h_k=1e-2):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        p_1h = self.Pk_1h(ks=ks, Ms=Ms)
        if trunc_1h_k is not None:
            p_1h *= (1 - np.exp(-ks/trunc_1h_k))

        return p_1h + self.Pk_2h(ks=ks, Ms=Ms, z=z)


    def limber_cl(self, power_func, z_arr=None, nz=None, ls=None):
        """
        Generic Limber projection of an angular power spectrum.

        Parameters
        ----------
        power_func : callable
            Power spectrum with signature ``(ks, z) -> P_arr`` where both
            ``ks`` and ``z`` are flat 1-D arrays of the same length
            (one entry per Limber pair) and the return value is also 1-D.
            For z-independent spectra (e.g. ``Pk_1h``) the ``z`` argument
            can simply be ignored.
        z_arr : array-like, optional
            Redshift grid for the integral.  Defaults to 51 points centred
            on the model redshift.
        nz : array-like or callable, optional
            Redshift distribution n(z).  A callable is evaluated on
            ``z_arr`` and normalised; an array is used directly (must align
            with ``z_arr``); ``None`` gives a top-hat over ``z_arr``.
        ls : array-like, optional
            Multipoles at which to evaluate C_l.  Defaults to 1001
            log-spaced points from 1 to 10^6.

        Returns
        -------
        cl : ndarray
            Angular power spectrum C_l.
        ls : ndarray
            Corresponding multipoles.

        Examples
        --------
        Full galaxy power (default in ``cf_ang``)::

            model.limber_cl(lambda ks, z: model.P_gal(ks=ks, z=z))

        1-halo term only (z-independent, ignore z)::

            model.limber_cl(lambda ks, z: model.Pk_1h(ks=ks))

        2-halo term only::

            model.limber_cl(lambda ks, z: model.Pk_2h(ks=ks, z=z))

        Matter power spectrum (positional args already match)::

            model.limber_cl(model.matter_power_spectrum)
        """

        if power_func is None:
            self.check_HOD_defined()
            power_func = lambda ks, z: self.P_gal(ks=ks, z=z, Ms=self.ms)
        
        if nz is not None and z_arr is None and not callable(nz):
            raise ValueError("z_arr must be provided when nz is an array")

        if z_arr is None:
            z_arr = np.linspace(self.halo_data.z - 0.5, self.halo_data.z + 0.5, 51)
        z_arr = np.asarray(z_arr)
        mask = z_arr > 0
        if isinstance(nz, np.ndarray):
            nz = nz[mask]
        z_arr = z_arr[mask]

        if callable(nz):
            nz = np.asarray(nz(z_arr))
            nz /= _trapz(nz, z_arr)
        elif nz is None:
            nz = np.ones_like(z_arr) / (z_arr[-1] - z_arr[0])

        h_z = self.halo_data.cosmo.hubble_parameter(z_arr)
        chi_z = self.halo_data.cosmo.comoving_radial_distance(z_arr)

        if ls is None:
            ls = np.logspace(0, 6, 1001)

        n_z, n_l = len(z_arr), len(ls)
        ks_2d = (ls[None, :] + 0.5) / chi_z[:, None]  # (n_z, n_l)

        # Flatten to 1-D pairs: each (z_i, l_j) maps to one (k, z) entry
        ks_flat = ks_2d.ravel()
        z_flat  = np.repeat(z_arr, n_l)

        power_2d = power_func(ks_flat, z_flat).reshape(n_z, n_l)

        cl = _trapz(
            h_z[:, None] * nz[:, None]**2 / C / chi_z[:, None]**2 * power_2d,
            z_arr, axis=0
        )

        return cl, ls
    

    def cf_3d(self, rs=None, Ms=None, ks=None, power=None):

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        if power is not None and len(power) != len(ks):
            raise ValueError("Power spectrum array length must match k array")

        if power is None:
            power = self.P_gal(ks=ks, Ms=Ms)

        r, xi = P2xi(ks, l=0, q=1.5, lowring=True)(power, extrap=True)

        if rs is not None:
            xi = make_interp_spline(r, xi)(rs)
            r = rs

        return xi, r

    def cf_ang(self, power_func=None, theta=None, nz=None, z_arr=None, Ms=None, ls=None, trunc_1h_k=1e-2):

        if power_func is None:
            self.check_HOD_defined()
            power_func = lambda ks, z: self.P_gal(ks=ks, z=z, Ms=Ms, trunc_1h_k=trunc_1h_k)

        cl, ls = self.limber_cl(power_func, z_arr=z_arr, nz=nz, ls=ls)

        # Hankel computes ∫ a(l) J_0(θl) l dl; cl/(2π) gives w(θ) directly
        theta_out, wtheta = Hankel(ls, nu=0, q=1, lowring=True)(cl / (2 * np.pi), extrap=True)
        theta_out = np.rad2deg(theta_out)

        if theta is not None:
            wtheta = make_interp_spline(theta_out, wtheta)(theta)
            theta_out = theta

        return wtheta, theta_out