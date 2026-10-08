
import copy
import warnings

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import norm
from mcfit import P2xi, Hankel

from . import mass_function, occupation, profile
from .utils import C, _trapz, lookup
from .config import HaloConfig

class Model():

    def __init__(
            self,
            z=0,
            cosmo_pars=None,
            hmf='behroozi',
            halo_prof='nfw',
            hod=None,
            hod_pars=None,
            halo_mass_grid=None,
            k_grid=None,
            z_range=None):

        if k_grid is not None:
            self.ks = np.asarray(k_grid).copy()
        else:
            self.ks = np.logspace(np.log10(1e-4), np.log10(1e2), 1001)

        self.config = HaloConfig(z,
                                 cosmo_pars,
                                 mass_grid=halo_mass_grid,
                                 z_range=z_range,
                                 kmax=max(self.ks)*2)

        self.profile = lookup(profile.MODELS, halo_prof, "Profile")(self.config)
        self.u_grid = self.profile.u(self.ks, self.ms)

        self.hmf = lookup(mass_function.MODELS, hmf, "HMF")(self.config)
        self.dndm = self.hmf.dndm(self.ms)

        if hod is not None:
            self.set_hod(hod, hod_pars)
        else:
            self._hod = None

        sig_d2 = _trapz(self.Pk_m(self.ks), self.ks) / (6 * np.pi**2)
        self.k_star = 0.584 / np.sqrt(sig_d2)


    @property
    def z(self):
        return self.config.z

    @property
    def ms(self):
        return self.config.mass_grid

    @property
    def hod(self):
        if self._hod is None:
            raise ValueError("HOD not defined")
        return self._hod

    def Pk_m(self, ks, z=None):
        return self.config.Pk_m(ks, z)


    def set_hod(self, new_hod, pars=None):
        if pars is None:
            pars = {}
        if not isinstance(pars, dict):
            raise TypeError("HOD parameters argument should be a dict")
        self._hod = lookup(occupation.MODELS, new_hod, "HOD")(**pars)


    def update_hod_pars(self, **new_pars):
        self.hod.update_pars(**new_pars)


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


    def Pk_1h(self, ks=None, Ms=None, damp=None):

        if Ms is None and ks is None:
            u = self.u_grid
        else:
            Ms = self.ms if Ms is None else Ms
            ks = self.ks if ks is None else ks
            u = self.profile.u(ks, Ms)
            
        ng = self.n_gal(Ms)

        p_1h = self.Pk_cs(Ms, u, ng) + self.Pk_ss(Ms, u, ng)

        if damp is None or damp is False:
            return p_1h
        elif type(damp) is float:
            return p_1h * -np.expm1(-(np.asarray(ks) / damp)**2)
        
        return p_1h * -np.expm1(-(np.asarray(ks) / self.k_star)**2)

    def Pk_2h(self, ks=None, z=None, Ms=None):
        if Ms is None and ks is None:
            u = self.u_grid
        else:
            Ms = self.ms if Ms is None else Ms
            ks = self.ks if ks is None else ks
            u = self.profile.u(ks, Ms)

        ng = self.n_gal(Ms)
        
        igrand = self.hod.N_hod(Ms)[None, :] * self.hmf.bias(Ms)[None, :] * u
        res = (self.hmf.halo_integral(Ms, igrand, axis=1) / ng) ** 2

        return self.Pk_m(ks, z) * res


    def P_gal(self, ks=None, z=None, Ms=None, damp_1h=None):
        Ms = self.ms if Ms is None else Ms
        ks = self.ks if ks is None else ks

        return self.Pk_1h(ks=ks, Ms=Ms, damp=damp_1h) + self.Pk_2h(ks=ks, Ms=Ms, z=z)


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
            with ``z_arr``); Default is Gaussian with width 0.25.
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

            model.limber_cl(model.Pk_m)
        """
        
        if nz is not None and z_arr is None and not callable(nz):
            raise ValueError("z_arr must be provided when nz is an array")

        if z_arr is None:
            z_arr = np.linspace(self.z - 0.5, self.z + 0.5, 51)
        z_arr = np.asarray(z_arr)
        mask = z_arr > 0
        z_arr = z_arr[mask]
        if len(z_arr) < 2:
            raise ValueError("z_arr must contain at least 2 positive redshift samples")

        pk_zmin, pk_zmax = self.config.pkm_interp.zmin, self.config.pkm_interp.zmax
        if z_arr.min() < pk_zmin or z_arr.max() > pk_zmax:
            warnings.warn(
                f"z_arr spans [{z_arr.min():.2f}, {z_arr.max():.2f}] but the matter "
                f"power interpolator covers only [{pk_zmin:.2f}, {pk_zmax:.2f}]. CAMB "
                f"clamps outside this range rather than extrapolating. Pass "
                f"z_range=(zmin, zmax) to Model() to widen it.")

        if nz is None:
            nz = norm.pdf(z_arr, self.z, 0.25)
        elif not callable(nz):
            nz = np.asarray(nz)
            if len(nz) != len(z_arr):
                raise ValueError("nz array must match length of z_arr")
        else:
            nz = np.asarray(nz(z_arr))
        
        nz /= _trapz(nz, z_arr)
        
        h_z = self.config.cosmo.hubble_parameter(z_arr)
        chi_z = self.config.cosmo.comoving_radial_distance(z_arr)

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
    

    # Comoving k [1/Mpc] by which the profile of any occupied halo has decayed;
    # 1-halo grids are extended to here so FFTLog sees the full 1-halo falloff.
    _K_1H_MAX = 1e5

    def _pk_1h_extended(self, ks, Ms):
        """P_1h on a log-uniform k-grid reaching ``_K_1H_MAX``.

        At z ≳ 2 the small halos that dominate the 1-halo term have not decayed
        by the default k_max = 100/Mpc, and FFTLog-transforming the chopped
        function rings at all r beyond the halo scale.  P_1h needs no CAMB
        call, so it can be evaluated well past the CAMB grid for the cost of a
        few sici calls.  FFTLog needs uniform log spacing, so rather than
        appending points the grid is rebuilt at the input's density per decade.
        """
        if ks[-1] >= self._K_1H_MAX / 10:
            return ks, self.Pk_1h(ks=ks, Ms=Ms)
        n = int(np.ceil(len(ks) * np.log10(self._K_1H_MAX / ks[0])
                        / np.log10(ks[-1] / ks[0])))
        ks_ext = np.geomspace(ks[0], self._K_1H_MAX, n)
        ng = self.n_gal(Ms)
        # _compute_profile bypasses the profile's single-slot cache so the
        # warm self.ks entry survives for later P_gal / cf_ang calls.
        u = self.prof._compute_profile(ks_ext, Ms)
        return ks_ext, self.Pk_cs(Ms, u, ng) + self.Pk_ss(Ms, u, ng)

    def cf_3d(self, rs=None, Ms=None, ks=None, power=None, damp_1h_k=None):
        """
        3-D galaxy correlation function via FFTLog.

        When ``power`` is None the 1- and 2-halo terms are transformed
        separately: the 2-halo term on ``ks`` (bounded by the CAMB k range)
        and the 1-halo term on the extended grid from ``_pk_1h_extended``.
        An explicit ``power`` array is transformed on ``ks`` as-is.

        ``damp_1h_k`` defaults to None (no damping), unlike ``P_gal``: the
        1-halo plateau only contributes to xi at zero lag, while damping it
        necessarily drives xi_1h negative around the 1-to-2-halo transition
        (see ``_apply_1h_damping``).
        """
        if Ms is None:
            Ms = self.ms
        if ks is None:
            ks = self.ks

        if power is not None:
            if len(power) != len(ks):
                raise ValueError("Power spectrum array length must match k array")
            r, xi = P2xi(ks, l=0, q=1.5, lowring=True)(power, extrap=True)
        else:
            r, xi = P2xi(ks, l=0, q=1.5, lowring=True)(
                self.Pk_2h(ks=ks, Ms=Ms), extrap=True
            )
            ks_1h, p_1h = self._pk_1h_extended(ks, Ms)
            p_1h = self._apply_1h_damping(p_1h, ks_1h, self._resolve_k_damp(damp_1h_k))
            r_1h, xi_1h = P2xi(ks_1h, l=0, q=1.5, lowring=True)(p_1h, extrap=True)
            # The two transforms return different r grids; xi_1h is smooth and
            # ~0 at both ends of the overlap, so linear interpolation is safe.
            xi = xi + np.interp(np.log(r), np.log(r_1h), xi_1h)

        if rs is not None:
            xi = make_interp_spline(r, xi)(rs)
            r = rs

        return xi, r

    def _build_fast_power_func(self, k_star):
        """
        Return a fast Limber power function by exploiting that P_gal(k, z) splits as

            P_1h(k)  +  F(k)^2 * Pmm(k, z)

        where P_1h and F are z-independent.  Both are precomputed on an extended
        k-grid [self.ks, 10^5 Mpc^-1] and interpolated, so the
        Limber integrand only evaluates the cheap CAMB Pmm call at the full set of
        k-z pairs.  The grid is extended beyond self.ks so the spline captures the
        natural NFW decay (P → 0) rather than terminating at a non-zero boundary
        value, which would otherwise produce spurious power at high Limber l.
        The 1-halo damping (``k_star``, see ``_apply_1h_damping``) is baked into
        the precomputed P_1h.
        """
        assert self.hod is not None
        ng       = self.n_gal()
        p1h_grid = self.Pk_1h()                         # (n_k,) on self.ks via dot products
        u_grid   = self.prof.k_profile(self.ks, self.ms)  # (n_k, n_M) — cache hit
        N_hod    = self.hod.N_hod(self.ms)              # (n_M,)
        bias     = self.hmf.bias(self.ms)                  # (n_M,) — cached on the HMF
        weights  = self.hmf.integration_weights(self.ms)   # (n_M,) — cached on the HMF
        F_grid   = (N_hod * bias * u_grid @ weights) / ng  # (n_k,)

        # Extend precomputed grid into high-k regime where u(k,M) → 0, at
        # ~10 points per decade, so the interpolant decays smoothly instead of
        # hitting a sharp boundary.
        if self.ks[-1] < self._K_1H_MAX / 10:
            n_hi   = int(np.ceil(10 * np.log10(self._K_1H_MAX / (2 * self.ks[-1]))))
            ks_hi  = np.geomspace(self.ks[-1] * 2, self._K_1H_MAX, n_hi)
            u_hi   = self.prof._compute_profile(ks_hi, self.ms)         # (n_hi, n_M)
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

        p1h_full = self._apply_1h_damping(p1h_full, ks_full, k_star)

        log_ks_full  = np.log(ks_full)
        ks_min, ks_max = ks_full[0], ks_full[-1]
        # Interpolate in log-log: P_1h and F fall by many decades across the
        # sparse high-k extension, where linear-in-P segments put kinks into
        # C_l at high Limber l.  Power-law stretches (including the baked-in
        # low-k damping) are exact in log-log.  The floor keeps log() finite
        # if a tail value underflows to zero.
        log_p1h = np.log(np.clip(p1h_full, 1e-300, None))
        log_F   = np.log(np.clip(F_full,   1e-300, None))

        def _power(ks, z):
            # np.interp is ~9x faster than a scipy B-spline at 51k evaluation
            # points and accurate to < 0.01% on this ~1030-point log-k grid.
            log_k = np.log(np.clip(ks, ks_min, ks_max))
            p1h = np.exp(np.interp(log_k, log_ks_full, log_p1h))
            F   = np.exp(np.interp(log_k, log_ks_full, log_F))
            result = p1h + F**2 * self.Pk_m(ks, z)
            return np.maximum(0.0, result)   # guard against Pmm rounding below 0 at extremes

        return _power

    def cf_ang(self, power_func=None, theta=None, nz=None, z_arr=None, Ms=None, ls=None, damp_1h_k=None):
        """
        Angular galaxy correlation function w(theta) via Limber + Hankel.

        ``damp_1h_k`` defaults to None (no damping), unlike ``P_gal``: the
        1-halo plateau contributes a flat, shot-noise-like C_l whose Hankel
        transform lives at theta = 0 only, while damping it necessarily
        drives w_1h negative around the 1-to-2-halo transition (see
        ``_apply_1h_damping``).
        """
        if power_func is None:
            self.check_HOD_defined()
            k_star = self._resolve_k_damp(damp_1h_k)
            # Fast path: precompute z-independent quantities on self.ks and
            # interpolate, avoiding large halo integrals at every Limber k-value.
            # Falls back to the full P_gal when a custom Ms grid is requested.
            if Ms is None:
                power_func = self._build_fast_power_func(k_star)
            else:
                assert self.hod is not None
                power_func = lambda ks, z: self.P_gal(ks=ks, z=z, Ms=Ms, damp_1h_k=k_star)

        cl, ls = self.limber_cl(power_func, z_arr=z_arr, nz=nz, ls=ls)

        # Hankel computes ∫ a(l) J_0(θl) l dl; cl/(2π) gives w(θ) directly
        theta_out, wtheta = Hankel(ls, nu=0, q=1, lowring=True)(cl / (2 * np.pi), extrap=True)
        theta_out = np.rad2deg(theta_out)

        if theta is not None:
            wtheta = make_interp_spline(theta_out, wtheta)(theta)
            theta_out = theta

        return wtheta, theta_out
    
    
        