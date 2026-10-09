"""Model-level tests: construction, dispatch, power spectra, projection, and
numerical regression snapshots.

The REGRESSION values were captured from the implementation prior to the cache
refactor.  They are not analytic truths but guard against any change silently
altering the outputs; update them deliberately if the physics changes.
"""

import numpy as np
import pytest

from mcfit import P2xi

from shoddy import Model
from shoddy.mass_function import Tinker


# --- Construction & validation -------------------------------------------------


def test_default_construction(model):
    assert model.ms.shape == (256,)
    assert model.ks.shape == (1501,)
    with pytest.raises(ValueError):
        model.hod


def test_rejects_tiny_mass_grid():
    with pytest.raises(ValueError):
        Model(halo_mass_grid=[1e12, 1e13])


def test_custom_grids_are_copied():
    ms = np.logspace(10, 15, 64)
    ks = np.logspace(-3, 1, 100)
    with pytest.warns(UserWarning):  # short k grid
        m = Model(halo_mass_grid=ms, k_grid=ks)
    ms[0] = -1.0  # mutating the caller's array must not affect the model
    assert m.ms[0] != -1.0
    assert m.ks.shape == (100,)


def test_n_gal_requires_hod(model):
    with pytest.raises(Exception):
        model.n_gal()


# --- Component dispatch (registry) ---------------------------------------------


def test_hmf_by_string():
    m = Model(hmf="tinker")
    assert isinstance(m.hmf, Tinker)


def test_unknown_hmf_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown HMF"):
        Model(hmf="does-not-exist")


def test_unknown_profile_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown Profile"):
        Model(halo_prof="does-not-exist")


def test_unknown_hod_raises_valueerror(model):
    with pytest.raises(ValueError, match="Unknown HOD"):
        model.set_hod("does-not-exist")


def test_hod_pars_must_be_dict(model):
    with pytest.raises(TypeError):
        model.set_hod("zheng07", pars=[1, 2, 3])


# --- Power spectra structure ---------------------------------------------------


def test_pk_shapes(model_with_hod):
    m = model_with_hod
    assert m.Pk_1h().shape == m.ks.shape
    assert m.Pk_2h().shape == m.ks.shape
    assert m.P_gal().shape == m.ks.shape


def test_pgal_is_sum_of_terms_without_damping(model_with_hod):
    m = model_with_hod
    total = m.P_gal(damp_1h=None)
    assert np.allclose(total, m.Pk_1h() + m.Pk_2h())


def test_pgal_positive(model_with_hod):
    assert np.all(model_with_hod.P_gal() > 0)


def test_custom_mass_grid_matches_default(model_with_hod):
    # Passing the default grid explicitly must match the cached fast path.
    m = model_with_hod
    assert np.allclose(m.P_gal(), m.P_gal(Ms=m.ms))
    assert np.isclose(m.n_gal(), m.n_gal(Ms=m.ms))


def test_pgal_2d_matches_direct(model_with_hod):
    # The interpolated fast path must agree with direct evaluation off-grid.
    # Worst agreement (~3e-3) is at k ~ 1e3 where u(k, M) oscillates.
    m = model_with_hod
    ks = np.logspace(-3, 3, 50)
    zs = np.linspace(0.05, 0.5, 50)
    assert np.allclose(m.P_gal_2d(ks, zs), m.P_gal(ks=ks, z=zs), rtol=5e-3)


def test_galaxy_bias_above_one(model_with_hod):
    # For a M_min ~ 1e12 sample at z=0 the linear bias is > 1.
    assert model_with_hod.galaxy_bias() > 1.0


# --- Projection / correlation functions ----------------------------------------


def test_cf_3d_at_requested_radii(model_with_hod):
    xi, r = model_with_hod.cf_3d(rs=[1.0, 10.0])
    assert np.allclose(r, [1.0, 10.0])
    assert xi.shape == (2,)
    assert xi[0] > xi[1]  # correlation falls with separation


def test_cf_ang_at_requested_theta(model_with_hod):
    w, theta = model_with_hod.cf_ang(theta=[0.01, 0.1])
    assert np.allclose(theta, [0.01, 0.1])
    assert np.all(w > 0)
    assert w[0] > w[1]


def test_cf_3d_total_not_below_2h(model_with_hod):
    # xi_1h is a pair count and hence non-negative, so the total must not dip
    # below the 2-halo term.  The k-space Gaussian damping violates this by
    # construction (its compensation drives xi_1h negative near the 1-to-2-halo
    # transition), which is why cf_3d defaults to damp_1h=None; transform
    # artifacts are allowed at the few-per-mille level.
    m = model_with_hod
    xi, r = m.cf_3d()
    _, xi2 = P2xi(m.ks, l=0, q=1.5, lowring=True)(m.Pk_2h(), extrap=True)
    sel = (r > 0.1) & (r < 100) & (xi2 > 0)
    assert np.min((xi[sel] - xi2[sel]) / xi2[sel]) > -5e-3


def test_limber_cl_array_nz_requires_zarr(model_with_hod):
    with pytest.raises(ValueError):
        model_with_hod.limber_cl(None, nz=np.ones(5))


# --- with_hod consistency ------------------------------------------------------


def test_with_hod_matches_fresh_set(model_with_hod):
    pars = {"M_min": 5e12, "sig_logM": 0.4, "M0": 2e12, "M1": 2e13, "alpha": 1.1}
    child = model_with_hod.with_hod(pars)
    reference = Model()
    reference.set_hod("zheng07", pars)
    assert np.isclose(child.n_gal(), reference.n_gal(), rtol=1e-10)
    assert np.allclose(child.P_gal(), reference.P_gal(), rtol=1e-10)


# --- Numerical regression snapshots --------------------------------------------

# Captured in the hod env with undamped defaults.  Power spectra are evaluated
# at fixed k so the snapshot does not depend on the default k grid.
REGRESSION = {
    "n_gal": 0.003573437968350265,
    "galaxy_bias": 1.2035170371749628,
    "Pk_1h_0": 1569.3435897667737,
    "Pk_2h_0": 2656.2543139295067,
    "P_gal_0": 4225.59790369628,
    "P_gal_k01": 15900.66071409825,
}


def test_regression_snapshots(model_with_hod):
    m = model_with_hod
    ks = np.array([1e-4, 1e-1])
    p_gal = m.P_gal(ks=ks)
    got = {
        "n_gal": m.n_gal(),
        "galaxy_bias": m.galaxy_bias(),
        "Pk_1h_0": m.Pk_1h(ks=ks)[0],
        "Pk_2h_0": m.Pk_2h(ks=ks)[0],
        "P_gal_0": p_gal[0],
        "P_gal_k01": p_gal[1],
    }
    for key, expected in REGRESSION.items():
        assert np.isclose(got[key], expected, rtol=1e-8), (key, got[key], expected)


def test_regression_cf_3d(model_with_hod):
    xi, _ = model_with_hod.cf_3d(rs=[1.0, 10.0])
    assert np.allclose(xi, [75.18974835, 0.87250518], rtol=1e-6)


def test_regression_cf_ang(hod_pars):
    # Setup (defaults where not passed):
    #   model:  z=1, default cosmology, HMF ('behroozi'), NFW profile and grids
    #   HOD:    zheng07 with HOD_PARS (conftest), dc=1
    #   n(z):   Gaussian, mean z=1, sigma=0.25, sampled on 51 points over [0.5, 1.5]
    #   ells:   1001 log-spaced points over [1, 1e6]
    #   power:  P_gal_2d fast path, undamped 1-halo term (damp_1h=None)
    m = Model(z=1)
    m.set_hod("zheng07", hod_pars)
    w, _ = m.cf_ang(theta=[0.01, 0.1])
    assert np.allclose(w, [0.07145347, 0.0155232], rtol=1e-6)
