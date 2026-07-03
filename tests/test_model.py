"""Model-level tests: construction, dispatch, power spectra, projection, and
numerical regression snapshots.

The REGRESSION values were captured from the implementation prior to the cache
refactor.  They are not analytic truths but guard against any change silently
altering the outputs; update them deliberately if the physics changes.
"""

import numpy as np
import pytest

from shoddy import Model
from shoddy.mass_function import Tinker
from shoddy.hod import Zheng07


# --- Construction & validation -------------------------------------------------


def test_default_construction(model):
    assert model.ms.shape == (256,)
    assert model.ks.shape == (1001,)
    assert model.hod is None


def test_rejects_tiny_mass_grid():
    with pytest.raises(ValueError):
        Model(halo_mass_grid=[1e12, 1e13])


def test_custom_grids_are_copied():
    ms = np.logspace(10, 15, 64)
    ks = np.logspace(-3, 1, 100)
    m = Model(halo_mass_grid=ms, k_grid=ks)
    ms[0] = -1.0  # mutating the caller's array must not affect the model
    assert m.ms[0] != -1.0
    assert m.ks.shape == (100,)


def test_n_gal_requires_hod(model):
    with pytest.raises(Exception):
        model.n_gal()


# --- Component dispatch (registry) ---------------------------------------------


def test_set_hmf_by_string(model):
    model.set_hmf("tinker", model.halo_data)
    assert isinstance(model.HMF, Tinker)


def test_set_hmf_by_instance(model):
    inst = Tinker(model.halo_data)
    model.set_hmf(inst, model.halo_data)
    assert model.HMF is inst


def test_unknown_hmf_raises_valueerror(model):
    with pytest.raises(ValueError, match="Unknown HMF"):
        model.set_hmf("does-not-exist", model.halo_data)


def test_unknown_profile_raises_valueerror(model):
    with pytest.raises(ValueError, match="Unknown halo profile"):
        model.set_halo_profile("does-not-exist", model.halo_data)


def test_unknown_hod_raises_valueerror(model):
    with pytest.raises(ValueError, match="Unknown HOD"):
        model.set_hod("does-not-exist")


def test_wrong_component_type_raises_typeerror(model):
    with pytest.raises(TypeError):
        model.set_hmf(12345, model.halo_data)


def test_hod_pars_must_be_dict(model):
    with pytest.raises(TypeError):
        model.set_hod("zheng07", pars=[1, 2, 3])


def test_set_hod_by_instance(model):
    h = Zheng07(M_min=1e12, sig_logM=0.3, M0=1e12, M1=1e13, alpha=1.0)
    model.set_hod(h)
    assert model.hod is h


# --- Power spectra structure ---------------------------------------------------


def test_pk_shapes(model_with_hod):
    m = model_with_hod
    assert m.Pk_1h().shape == m.ks.shape
    assert m.Pk_2h().shape == m.ks.shape
    assert m.P_gal().shape == m.ks.shape


def test_pgal_is_sum_of_terms_without_damping(model_with_hod):
    m = model_with_hod
    total = m.P_gal(damp_1h_k=None)
    assert np.allclose(total, m.Pk_1h() + m.Pk_2h())


def test_pgal_positive(model_with_hod):
    assert np.all(model_with_hod.P_gal() > 0)


def test_custom_mass_grid_matches_default(model_with_hod):
    # Passing the default grid explicitly must match the cached fast path.
    m = model_with_hod
    assert np.allclose(m.P_gal(), m.P_gal(Ms=m.ms))
    assert np.isclose(m.n_gal(), m.n_gal(Ms=m.ms))


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
    # transition), which is why cf_3d defaults to damp_1h_k=None; transform
    # artifacts are allowed at the few-per-mille level.
    m = model_with_hod
    xi, r = m.cf_3d()
    xi2, _ = m.cf_3d(power=m.Pk_2h())
    sel = (r > 0.1) & (r < 100) & (xi2 > 0)
    assert np.min((xi[sel] - xi2[sel]) / xi2[sel]) > -5e-3


def test_limber_cl_array_nz_requires_zarr(model_with_hod):
    with pytest.raises(ValueError):
        model_with_hod.limber_cl(None, nz=np.ones(5))


def test_cf_3d_power_length_mismatch_raises(model_with_hod):
    with pytest.raises(ValueError):
        model_with_hod.cf_3d(power=np.ones(3))


# --- with_hod consistency ------------------------------------------------------


def test_with_hod_matches_fresh_set(model_with_hod):
    pars = {"M_min": 5e12, "sig_logM": 0.4, "M0": 2e12, "M1": 2e13, "alpha": 1.1}
    child = model_with_hod.with_hod(pars)
    reference = Model()
    reference.set_hod("zheng07", pars)
    assert np.isclose(child.n_gal(), reference.n_gal(), rtol=1e-10)
    assert np.allclose(child.P_gal(), reference.P_gal(), rtol=1e-10)


# --- Numerical regression snapshots --------------------------------------------

# Captured after the 1-halo low-k damping changed from a fixed exponential
# truncation to the z-aware Gaussian form (P_gal_0/P_gal_500 shifted), in the
# c3d environment (CAMB-dependent values drift at the ~1e-9 level between
# environments).
REGRESSION = {
    "n_gal": 0.0035734281857500074,
    "galaxy_bias": 1.2035126140927694,
    "Pk_1h_0": 1569.2686435588319,
    "Pk_2h_0": 2656.234789803066,
    "P_gal_0": 2656.2380117366606,
    "P_gal_500": 15699.660299045367,
}


def test_regression_snapshots(model_with_hod):
    m = model_with_hod
    got = {
        "n_gal": m.n_gal(),
        "galaxy_bias": m.galaxy_bias(),
        "Pk_1h_0": m.Pk_1h()[0],
        "Pk_2h_0": m.Pk_2h()[0],
        "P_gal_0": m.P_gal()[0],
        "P_gal_500": m.P_gal()[500],
    }
    for key, expected in REGRESSION.items():
        assert np.isclose(got[key], expected, rtol=1e-10), (key, got[key], expected)


def test_regression_cf_3d(model_with_hod):
    # Captured with the undamped default (damp_1h_k=None) in the c3d env.
    xi, _ = model_with_hod.cf_3d(rs=[1.0, 10.0])
    assert np.allclose(xi, [75.21341329, 0.87249875], rtol=1e-6)


def test_regression_cf_ang(model_with_hod):
    # Captured with the undamped default (damp_1h_k=None) in the c3d env.
    w, _ = model_with_hod.cf_ang(theta=[0.01, 0.1])
    assert np.allclose(w, [1.52010388, 0.20330296], rtol=1e-6)
