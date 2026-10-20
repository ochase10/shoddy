
from abc import ABC, abstractmethod

import warnings
import numpy as np
from scipy.stats import norm
from scipy.interpolate import make_interp_spline
from mcfit import P2xi, Hankel

from .utils import C, _trapz


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


def _log_interp(new_x, old_x, arr):

    if arr is None:
        return 0
            
    if np.any(arr <= 0):
        arr = np.clip(arr, 1e-300, None)
        warnings.warn("<=0 values encountered when interpolating")

    return np.exp(np.interp(np.log(new_x), np.log(old_x), np.log(arr)))


class Observable(ABC):

    def __init__(self, model):
        model.hod
        self.model=model

    def __call__(self, hod_pars, **kwargs):
        return self._evaluate(self.model.with_hod(hod_pars), **kwargs)

    @abstractmethod
    def _evaluate(self, evalmodel, **kwargs):
        '''
        '''


class ACF(Observable):
    def __init__(self, model, zs=None, nz=None, ls=None):

        super().__init__(model)

        self.ls = np.logspace(0, 6, 1001) if ls is None else np.asarray(ls)
        self.z_arr, nz = _resolve_nz(model.z, zs, nz)

        h_z = model.config.cosmo.hubble_parameter(self.z_arr)
        chi_z = model.config.cosmo.comoving_radial_distance(self.z_arr)
        self.ks_2d = (self.ls[None, :] + 0.5) / chi_z[:, None]  # (n_z, n_l)
        z_2d = np.broadcast_to(self.z_arr[:, None], self.ks_2d.shape)

        self.kernel = h_z * nz**2 / C / chi_z**2
        self.pkm = model.Pk_m(ks=self.ks_2d.ravel(), z=z_2d.ravel()).reshape(self.ks_2d.shape)

        self.transform = Hankel(self.ls, nu=0, q=1, lowring=True)


    def _evaluate(self, newmod, theta=None, terms=None, Ms=None, damp_1h=None):

        z_ind, z_dep = newmod._resolve_terms(terms, ks=None, Ms=Ms, damp=damp_1h)
        
        ind_interp = _log_interp(self.ks_2d, newmod.ks, z_ind)
        dep_interp = _log_interp(self.ks_2d, newmod.ks, z_dep)
        
        power = ind_interp
        if z_dep is not None:
            power += self.pkm * dep_interp

        cl = _trapz(self.kernel[:,None] * power, self.z_arr, axis=0)

        # Hankel computes ∫ a(l) J_0(θl) l dl; cl/(2π) gives w(θ) directly
        theta_out, wtheta = self.transform(cl / (2 * np.pi), extrap=True)
        theta_out = np.rad2deg(theta_out)

        if theta is not None:
            wtheta = make_interp_spline(np.log(theta_out), wtheta)(np.log(theta))
            theta_out = theta

        return wtheta, theta_out


class CF3D(Observable):

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