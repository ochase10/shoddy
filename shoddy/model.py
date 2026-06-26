
import copy

import camb
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import norm
from mcfit import P2xi, Hankel

from . import mass_function, profile, hod
from .utils import G, C, _trapz
from .halo_config import HaloConfig
from .caching import Cached, cached_quantity


class Model(Cached):

    # Registries mapping the string names accepted by the constructor and the
    # set_* methods to their component classes.  Adding a new model is a single
    # entry here rather than another branch in a dispatch chain.
    _hmf_registry = {'tinker': mass_function.Tinker,
                     'behroozi': mass_function.Behroozi13}
    _profile_registry = {'nfw': profile.NFW}
    _hod_registry = {'zheng07': hod.Zheng07}

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

    def __init__(
            self,
            z=0,
            cosmo_pars=None,
            hmf='behroozi',
            halo_prof='nfw',
            hod=None,
            hod_pars={},
            halo_mass_grid=None,
            k_grid=None,
            **kwargs
    ):
        
        self.z = z

        if halo_mass_grid is not None:
            if len(halo_mass_grid) > 2:
                self.ms = np.asarray(halo_mass_grid).copy()
            else:
                raise ValueError("Halo mass grid must contain more than 2 values")
        else:
            self.ms = np.logspace(9, 16, 256)

        self.log_ms = np.log10(self.ms)

        if k_grid is not None:
            self.ks = np.asarray(k_grid).copy()
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
        # Warm the NFW profile cache on self.ks so the first cf_ang/Pk call
        # doesn't pay the sici computation cost.
        self.prof.k_profile(self.ks, self.ms)

        if hod is not None:
            self.set_hod(hod, hod_pars)
        else:
            self.hod = None


    def init_cosmo(self, pars):

        cambpars = camb.set_params(**pars)
        
        usezs = np.concatenate((np.arange(self.z - 1.5, self.z, 0.1),np.arange(self.z, self.z+1.5, 0.1)))[::-1]
        usezs = usezs[usezs >= 0]
        if not np.any(np.isclose(usezs, self.z)):
            usezs = np.sort(np.append(usezs, self.z))[::-1]
        cambpars.set_matter_power(redshifts=usezs, kmax=max(self.ks)*2, nonlinear=False)

        self.cosmo = camb.get_results(cambpars)
        self._invalidate('pkm_interp')
        self._z_sigma_idx = int(np.argmin(np.abs(usezs - self.z)))


    @cached_quantity
    def pkm_interp(self):
        return self.cosmo.get_matter_power_interpolator(
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
        # grid=False for array z: evaluate at (k_i, z_i) pairs, not all combos
        grid = np.ndim(z) == 0
        return self.pkm_interp.P(z, ks, grid=grid).ravel()


    @staticmethod
    def _resolve_component(spec, registry, base, label, *args, **kwargs):
        """Turn a name or instance into a component object.

        A string is looked up (case-insensitively) in ``registry`` and the
        matching class is instantiated with ``*args, **kwargs``; an existing
        instance of ``base`` is returned unchanged.
        """
        if isinstance(spec, str):
            try:
                cls = registry[spec.lower()]
            except KeyError:
                raise ValueError(
                    f"Unknown {label} '{spec}'. Options: {sorted(registry)}"
                ) from None
            return cls(*args, **kwargs)
        if isinstance(spec, base):
            return spec
        raise TypeError(
            f"{label} must be a str or {base.__name__} instance, got {type(spec).__name__}"
        )


    def set_hmf(self, new_hmf, config, **kwargs):
        self.HMF = self._resolve_component(
            new_hmf, self._hmf_registry, mass_function.MassFunction, 'HMF', config, **kwargs
        )
        # The HMF feeds every halo-grid quantity and the galaxy density.
        self._invalidate('_n_gal')


    def set_halo_profile(self, new_prof, config, **kwargs):
        self.prof = self._resolve_component(
            new_prof, self._profile_registry, profile.HaloProfile, 'halo profile', config, **kwargs
        )


    def set_hod(self, new_hod, pars={}):
        if not isinstance(pars, dict):
            raise TypeError("HOD parameters argument should be a dict")
        self.hod = self._resolve_component(
            new_hod, self._hod_registry, hod.HOD, 'HOD', **pars
        )
        self._invalidate('_n_gal')


    def update_hod_pars(self, **new_pars):
        """Update HOD parameters and invalidate the cached galaxy density."""
        self.check_HOD_defined()
        assert self.hod is not None
        self.hod.update_pars(**new_pars)
        self._invalidate('_n_gal')

    def with_hod(self, hod_pars):
        """
        Return a shallow copy of the model with updated HOD parameters, without
        mutating self. All expensive objects (cosmology, HMF, profile arrays) are
        shared by reference — only the HOD and its cached n_gal differ.

        Intended for stateless likelihood evaluation in MCMC:

            def log_prob(params):
                m = model.with_hod({'M_min': params[0], 'sig_logM': params[1], ...})
                return -0.5 * chi2(m.cf_ang(theta=theta_data)[0], data)

        Parameters
        ----------
        hod_pars : dict
            HOD parameters to override.  Keys must match the constructor
            arguments of the current HOD class (e.g. Zheng07 expects
            M_min, sig_logM, M0, M1, alpha).  Unspecified parameters
            are inherited from the current HOD.
        """
        self.check_HOD_defined()
        assert self.hod is not None
        # Prime the power spectrum interpolator before copying so all copies
        # share the same object rather than each recreating it.
        _ = self.pkm_interp
        m = copy.copy(self)
        m.hod = type(self.hod)(**{**self.hod.pars, **hod_pars})
        # copy.copy shallow-copies __dict__, carrying over any cached n_gal from
        # self; drop it so the new HOD's density is recomputed.  The shared HMF
        # arrays and pk interpolator are intentionally kept by reference.
        m._invalidate('_n_gal')
        return m


    def check_HOD_defined(self):
        if self.hod is None:
            raise Exception("HOD must be defined to get galaxy density")


    @cached_quantity
    def _n_gal(self):
        # Galaxy density on the default grid; invalidated when the HOD changes
        # (see set_hod / update_hod_pars / with_hod).
        return float(self.HMF.halo_integral(self.ms, self.hod.N_hod(self.ms)))

    def n_gal(self, Ms=None, recompute=False):
        """Mean galaxy number density.  The default mass grid uses the cached
        value; an explicit off-grid ``Ms`` is integrated fresh."""
        self.check_HOD_defined()
        assert self.hod is not None
        
        if Ms is None or np.array_equal(Ms, self.ms):
            if recompute:
                self._invalidate('_n_gal')
            return self._n_gal
        return float(self.HMF.halo_integral(Ms, self.hod.N_hod(Ms)))
    

    def Pk_cs(self, Ms, u, ng):
        self.check_HOD_defined()
        assert self.hod is not None

        igrand = self.hod.avg_NcNs(Ms)[None, :] * u
        res = self.HMF.halo_integral(Ms, igrand, axis=1)
        return 2.0 * res / ng**2

    def Pk_ss(self, Ms, u, ng):
        self.check_HOD_defined()
        assert self.hod is not None

        igrand = self.hod.avg_Ns2(Ms)[None, :] * u**2
        res = self.HMF.halo_integral(Ms, igrand, axis=1)
        return res / ng**2

    def Pk_1h(self, ks=None, Ms=None):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        ng = self.n_gal(Ms)
        u = self.prof.k_profile(ks, Ms)

        return self.Pk_cs(Ms, u, ng) + self.Pk_ss(Ms, u, ng)

    def Pk_2h(self, ks=None, z=None, Ms=None):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        ng = self.n_gal(Ms)
        u = self.prof.k_profile(ks, Ms)
        igrand = self.hod.N_hod(Ms)[None, :] * self.HMF.bias(Ms)[None, :] * u
        res = (self.HMF.halo_integral(Ms, igrand, axis=1) / ng) ** 2

        # matter_power_spectrum handles scalar and array z uniformly (grid=False for arrays)
        return self.matter_power_spectrum(ks, z) * res


    def P_gal(self, ks=None, z=None, Ms=None, trunc_1h_k=1e-2):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks
        else:
            ks = np.asarray(ks)

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
        z_arr = z_arr[mask]
        if len(z_arr) < 2:
            raise ValueError("z_arr must contain at least 2 positive redshift samples")

        if nz is None:
            nz = norm.pdf(z_arr, self.z, 0.25)
        elif not callable(nz):
            nz = np.asarray(nz)
            if len(nz) != len(z_arr):
                raise ValueError("nz array must match length of z_arr")
        else:
            nz = np.asarray(nz(z_arr))
        
        nz /= _trapz(nz, z_arr)
        
        h_z = self.halo_data.cosmo.hubble_parameter(z_arr)
        chi_z = self.halo_data.cosmo.comoving_radial_distance(z_arr)

        if ls is None:
            ls = np.logspace(0, 6, 1001)
        else:
            ls = np.asarray(ls)

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

    def _build_fast_power_func(self, trunc_1h_k):
        """
        Return a fast Limber power function by exploiting that P_gal(k, z) splits as

            P_1h(k)  +  F(k)^2 * Pmm(k, z)

        where P_1h and F are z-independent.  Both are precomputed on an extended
        k-grid [self.ks, 10^5 Mpc^-1] and interpolated, so the
        Limber integrand only evaluates the cheap CAMB Pmm call at the full set of
        k-z pairs.  The grid is extended beyond self.ks so the spline captures the
        natural NFW decay (P → 0) rather than terminating at a non-zero boundary
        value, which would otherwise produce spurious power at high Limber l.
        """
        assert self.hod is not None
        ng       = self.n_gal()
        p1h_grid = self.Pk_1h()                         # (n_k,) on self.ks via dot products
        u_grid   = self.prof.k_profile(self.ks, self.ms)  # (n_k, n_M) — cache hit
        N_hod    = self.hod.N_hod(self.ms)              # (n_M,)
        bias     = self.HMF.bias(self.ms)                  # (n_M,) — cached on the HMF
        weights  = self.HMF.integration_weights(self.ms)   # (n_M,) — cached on the HMF
        F_grid   = (N_hod * bias * u_grid @ weights) / ng  # (n_k,)

        # Extend precomputed grid into high-k regime where u(k,M) → 0.
        # 12 log-spaced points from 2*k_max to 10^5 Mpc^-1 capture the NFW
        # tail so the spline decays smoothly instead of hitting a sharp boundary.
        if self.ks[-1] < 1e4:
            ks_hi  = np.geomspace(self.ks[-1] * 2, 1e5, 12)
            u_hi   = self.prof._compute_profile(ks_hi, self.ms)         # (12, n_M)
            p1h_hi = (2.0 * (self.hod.avg_NcNs(self.ms) * u_hi    @ weights)
                    +      (self.hod.avg_Ns2(self.ms)  * u_hi**2 @ weights)) / ng**2
            F_hi   = (N_hod * bias * u_hi @ weights) / ng
            ks_full      = np.concatenate([self.ks, ks_hi])
            
            p1h_full = np.concatenate([p1h_grid, p1h_hi])
            F_full   = np.concatenate([F_grid,   F_hi])
        else:
            ks_full = self.ks
            p1h_full = p1h_grid
            F_full = F_grid
        
        log_ks_full  = np.log(ks_full)
        ks_min, ks_max = ks_full[0], ks_full[-1]

        def _power(ks, z):
            # np.interp is ~9x faster than a scipy B-spline at 51k evaluation
            # points and accurate to < 0.01% on this 1013-point log-k grid.
            log_k = np.log(np.clip(ks, ks_min, ks_max))
            p1h = np.interp(log_k, log_ks_full, p1h_full)
            if trunc_1h_k is not None:
                p1h *= (1 - np.exp(-ks / trunc_1h_k))
            F = np.interp(log_k, log_ks_full, F_full)
            result = p1h + F**2 * self.matter_power_spectrum(ks, z)
            return np.maximum(0.0, result)   # guard against rounding below 0 at extremes

        return _power

    def cf_ang(self, power_func=None, theta=None, nz=None, z_arr=None, Ms=None, ls=None, trunc_1h_k=1e-2):

        if power_func is None:
            self.check_HOD_defined()
            # Fast path: precompute z-independent quantities on self.ks and
            # interpolate, avoiding large halo integrals at every Limber k-value.
            # Falls back to the full P_gal when a custom Ms grid is requested.
            if Ms is None:
                power_func = self._build_fast_power_func(trunc_1h_k)
            else:
                assert self.hod is not None
                power_func = lambda ks, z: self.P_gal(ks=ks, z=z, Ms=Ms, trunc_1h_k=trunc_1h_k)

        cl, ls = self.limber_cl(power_func, z_arr=z_arr, nz=nz, ls=ls)

        # Hankel computes ∫ a(l) J_0(θl) l dl; cl/(2π) gives w(θ) directly
        theta_out, wtheta = Hankel(ls, nu=0, q=1, lowring=True)(cl / (2 * np.pi), extrap=True)
        theta_out = np.rad2deg(theta_out)

        if theta is not None:
            wtheta = make_interp_spline(theta_out, wtheta)(theta)
            theta_out = theta

        return wtheta, theta_out
    
    def galaxy_bias(self, Ms=None):
        self.check_HOD_defined()
        assert self.hod is not None

        if Ms is None:
            Ms = self.ms

        ng = self.n_gal(Ms)
        n_avg = self.hod.N_hod(Ms)
        b_halo = self.HMF.bias(Ms)

        return self.HMF.halo_integral(Ms, n_avg*b_halo) / ng
        