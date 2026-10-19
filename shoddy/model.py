
import copy, warnings
from numbers import Real

import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.stats import norm
from mcfit import P2xi, Hankel

from . import mass_function, occupation, profile
from .utils import C, _trapz, lookup
from .config import HaloConfig, _EXTRAP_K_MAX, _INTERP_K_MAX



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


def _check_k_grid(ks, warn_k=1e4):
    if np.max(ks) < warn_k:
        warnings.warn("k grid may not extend high enough to allow for accurate \
                       correlation function computations.")


def _log_interp(new_x, old_x, arr):

    if arr is None:
        return 0
            
    if np.any(arr <= 0):
        arr = np.clip(arr, 1e-300, None)
        warnings.warn("<=0 values encountered when interpolating")

    return np.exp(np.interp(np.log(new_x), np.log(old_x), np.log(arr)))


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
            z_range=None,
            camb_kmax=_INTERP_K_MAX):

        if k_grid is not None:
            self.ks = np.asarray(k_grid).copy()
            _check_k_grid(self.ks)

        else:
            self.ks = np.logspace(np.log10(1e-4), np.log10(_EXTRAP_K_MAX), 1501)

        self.config = HaloConfig(z,
                                 cosmo_pars,
                                 mass_grid=halo_mass_grid,
                                 z_range=z_range,
                                 kmax=camb_kmax)

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


    def F_k(self, ks=None, Ms=None):
        ks, Ms, u = self._k_m_profile(ks, Ms)
        
        ng = self.n_gal(Ms)
        igrand = self.hod.N_hod(Ms)[None, :] * self.hmf.bias(Ms)[None, :] * u
        
        return self.hmf.halo_integral(Ms, igrand, axis=1) / ng


    def Pk_2h(self, ks=None, z=None, Ms=None):
        Fk = self.F_k(ks, Ms)
        ks = self.ks if ks is None else ks
        return self.Pk_m(ks, z) * Fk**2


    def P_gal(self, ks=None, z=None, Ms=None, damp_1h=None):
        return self.Pk_1h(ks=ks, Ms=Ms, damp=damp_1h) + self.Pk_2h(ks=ks, Ms=Ms, z=z)


    def _limber_setup(self, zs, nz, ls=None):

        z_arr, nz = _resolve_nz(self.z, zs, nz)
        h_z = self.config.cosmo.hubble_parameter(z_arr)
        chi_z = self.config.cosmo.comoving_radial_distance(z_arr)
        kernel = h_z * nz**2 / C / chi_z**2

        ls = np.logspace(0, 6, 1001) if ls is None else np.asarray(ls)
        
        ks_2d = (ls[None, :] + 0.5) / chi_z[:, None]  # (n_z, n_l)

        z_2d = np.broadcast_to(z_arr[:, None], ks_2d.shape)

        pkm = self.Pk_m(ks=ks_2d.ravel(), z=z_2d.ravel()).reshape(ks_2d.shape)
        

        return ls, kernel, ks_2d, z_arr, pkm


    
    def _limber_compute(self, terms, kernel, k2d, z_arr, Ms, damp_1h, pkm):

        z_ind, z_dep = self._resolve_terms(terms, ks=None, Ms=Ms, damp=damp_1h)
        
        ind_interp = _log_interp(k2d, self.ks, z_ind)
        dep_interp = _log_interp(k2d, self.ks, z_dep)
        
        power = ind_interp
        if z_dep is not None:
            power += pkm * dep_interp

        
        cl = _trapz(kernel[:,None] * power, z_arr, axis=0)

        return cl


    def cf_3d(self, rs=None, Ms=None, ks=None, damp_1h=None):
        """
        3-D galaxy correlation function via FFTLog.
        """
        if ks is not None:
            _check_k_grid(ks)

        p_gal = self.P_gal(ks, Ms=Ms, damp_1h=damp_1h)

        ks = self.ks if ks is None else ks
        
        r, xi = P2xi(ks, l=0, q=1.5, lowring=True)(p_gal, extrap=True)

        if rs is not None:
            xi = make_interp_spline(np.log(r), xi)(np.log(rs))
            r = rs

        return xi, r


    def cf_ang(self, terms=None, theta=None, nz=None, z_arr=None, Ms=None, ls=None, damp_1h=None):
        """
        Angular galaxy correlation function w(theta) via Limber + Hankel.
        """

        ls, kernel, ks_2d, zs, pkm = self._limber_setup(z_arr, nz, ls)
        cl = self._limber_compute(terms, kernel, ks_2d, zs, Ms, damp_1h, pkm)

        # Hankel computes ∫ a(l) J_0(θl) l dl; cl/(2π) gives w(θ) directly
        theta_out, wtheta = Hankel(ls, nu=0, q=1, lowring=True)(cl / (2 * np.pi), extrap=True)
        theta_out = np.rad2deg(theta_out)

        if theta is not None:
            wtheta = make_interp_spline(np.log(theta_out), wtheta)(np.log(theta))
            theta_out = theta

        return wtheta, theta_out


    def _resolve_terms(self, terms, **kwargs):
        if terms is None:
            terms = _TERMS.keys()
        A = None
        B = None
        for term in terms:
            fn, which = _TERMS[term]
            amp = fn(self, **kwargs)
            if which == 'k':
                if A is None:
                    A = amp
                else:
                    A = A + amp
            elif which == 'pkm':
                if B is None:
                    B = amp
                else:
                    B = B + amp
            else:
                raise ValueError("Unrecognized term type")

        return (A, B)

    
_TERMS = {'1h': (lambda m, ks=None, Ms=None, damp=None, **kw: m.Pk_1h(ks=ks, Ms=Ms, damp=damp), 'k'),
          '2h': (lambda m, ks=None, Ms=None, **kw: m.F_k(ks=ks, Ms=Ms)**2, 'pkm')}
        