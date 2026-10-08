
import copy
from numbers import Real

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import norm
from mcfit import P2xi, Hankel

from . import mass_function, occupation, profile
from .utils import C, _trapz, lookup
from .config import HaloConfig



def _resolve_nz(z, z_arr, nz):

    if nz is None and z_arr is None:
        z_arr = np.linspace(max(z - 0.5, 0.01), z + 0.5, 51)
        nz = norm.pdf(z_arr, z, 0.25)
    elif nz is None or z_arr is None:
        raise ValueError("nz and z_arr arguments cannot be passed alone")
    
    nz = np.asarray(nz)
    z_arr = np.asarray(z_arr)

    if len(nz) != len(z_arr):
        raise ValueError("nz and z_arr must be the same length")

    mask = z_arr > 0
    z_arr = z_arr[mask]
    nz = nz[mask]

    if len(z_arr) < 2:
        raise ValueError("z_arr must contain at least 2 positive redshift samples")

    nz_norm = nz / _trapz(nz, z_arr)

    return z_arr, nz_norm


def _extend_k_grid(ks, k_max=1e5):

    dlnk = np.diff(np.log(ks))
    if not np.allclose(dlnk, dlnk[0]):
        raise ValueError("k grid must be evenly spaced in log")
    if dlnk[0] <= 0:
        raise ValueError("k grid must be strictly increasing")

    if ks[-1] < k_max / 10:
        k_ext = np.arange(np.log(ks[-1]), np.log(k_max), dlnk[0])
        return np.exp(k_ext[1:])
    return np.array([])


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


    def _k_m_profile(self, ks, Ms):
        use_cache = ks is None and Ms is None
        Ms = self.ms if Ms is None else Ms
        ks = self.ks if ks is None else ks
        u = self.u_grid if use_cache else self.profile.u(ks, Ms)

        return ks, Ms, u


    def Pk_1h(self, ks=None, Ms=None, damp=None):
        ks, Ms, u = self._k_m_profile(ks, Ms)

        ng = self.n_gal(Ms)

        p_1h = self.Pk_cs(Ms, u, ng) + self.Pk_ss(Ms, u, ng)

        if damp is None or damp is False:
            return p_1h
        elif damp is True:
            damp = self.k_star
        elif not isinstance(damp, Real):
            raise TypeError("damp must be None, bool, or a real number")
        
        return p_1h * -np.expm1(-(np.asarray(ks) / damp)**2)


    def Pk_2h(self, ks=None, z=None, Ms=None):
        ks, Ms, u = self._k_m_profile(ks, Ms)

        ng = self.n_gal(Ms)
        
        igrand = self.hod.N_hod(Ms)[None, :] * self.hmf.bias(Ms)[None, :] * u
        res = (self.hmf.halo_integral(Ms, igrand, axis=1) / ng) ** 2

        return self.Pk_m(ks, z) * res


    def P_gal(self, ks=None, z=None, Ms=None, damp_1h=None):
        return self.Pk_1h(ks=ks, Ms=Ms, damp=damp_1h) + self.Pk_2h(ks=ks, Ms=Ms, z=z)

    
    def limber_cl(self, power_func, z_arr=None, nz=None, ls=None):

        z_arr, nz = _resolve_nz(self.z, z_arr, nz)
    
        h_z = self.config.cosmo.hubble_parameter(z_arr)
        chi_z = self.config.cosmo.comoving_radial_distance(z_arr)

        ls = np.logspace(0, 6, 1001) if ls is None else np.asarray(ls)

        ks_2d = (ls[None, :] + 0.5) / chi_z[:, None]  # (n_z, n_l)

        z_2d = np.broadcast_to(z_arr[:,None], ks_2d.shape)

        ks_flat, z_flat = ks_2d.ravel(), z_2d.ravel()

        power_2d = power_func(ks_flat, z_flat).reshape(ks_2d.shape)
        kernel = h_z * nz**2 / C / chi_z**2

        cl = _trapz(kernel[:,None] * power_2d, z_arr, axis=0)

        return cl, ls


    def cf_3d(self, rs=None, Ms=None, ks=None, damp_1h=None):
        """
        3-D galaxy correlation function via FFTLog.
        """
 
        kgrid = self.ks if ks is None else ks
        kext = _extend_k_grid(kgrid)
        all_ks = np.concatenate((kgrid, kext))

        p1h = np.concatenate((self.Pk_1h(ks=ks, Ms=Ms, damp=damp_1h), 
                              self.Pk_1h(ks=kext, Ms=Ms, damp=damp_1h)))
        p2h = np.concatenate((self.Pk_2h(ks=ks, Ms=Ms), np.zeros_like(kext)))
        
        r, xi = P2xi(all_ks, l=0, q=1.5, lowring=True)(p1h+p2h, extrap=True)

        if rs is not None:
            xi = make_interp_spline(np.log(r), xi)(np.log(rs))
            r = rs

        return xi, r


    def cf_ang(self, power_func=None, theta=None, nz=None, z_arr=None, Ms=None, ls=None, damp_1h=None):
        """
        Angular galaxy correlation function w(theta) via Limber + Hankel.

        """
        if power_func is None:

            if Ms is None:
                power_func = self._build_fast_power_func(k_star)
            else:
                power_func = lambda ks, z: self.P_gal(ks=ks, z=z, Ms=Ms, damp_1h=damp_1h)

        cl, ls = self.limber_cl(power_func, z_arr=z_arr, nz=nz, ls=ls)

        # Hankel computes ∫ a(l) J_0(θl) l dl; cl/(2π) gives w(θ) directly
        theta_out, wtheta = Hankel(ls, nu=0, q=1, lowring=True)(cl / (2 * np.pi), extrap=True)
        theta_out = np.rad2deg(theta_out)

        if theta is not None:
            wtheta = make_interp_spline(theta_out, wtheta)(theta)
            theta_out = theta

        return wtheta, theta_out


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


    
    
        