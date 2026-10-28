"""Model-level tests: construction, dispatch, power spectra, projection, and
numerical regression snapshots.

The REGRESSION values were captured from the implementation prior to the cache
refactor.  They are not analytic truths but guard against any change silently
altering the outputs; update them deliberately if the physics changes.
"""

import numpy as np
import pytest

from mcfit import P2xi
from scipy.stats import norm

from shoddy import Model
from shoddy.mass_function import Tinker
from shoddy.observables import ACF
from shoddy.subgalaxy import r2sig

from conftest import HOD_PARS


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
    assert m.P_tot().shape == m.ks.shape


def test_ptot_is_sum_of_terms_without_damping(model_with_hod):
    m = model_with_hod
    total = m.P_tot(damp_1h=None)
    assert np.allclose(total, m.Pk_1h() + m.Pk_2h())


def test_ptot_positive(model_with_hod):
    assert np.all(model_with_hod.P_tot() > 0)


def test_custom_mass_grid_matches_default(model_with_hod):
    # Passing the default grid explicitly must match the cached fast path.
    m = model_with_hod
    assert np.allclose(m.P_tot(), m.P_tot(Ms=m.ms))
    assert np.isclose(m.n_gal(), m.n_gal(Ms=m.ms))


def test_interpolated_terms_match_direct(model_with_hod):
    # Grid amplitudes log-interpolated off-grid (the Limber path) must agree with
    # direct evaluation. Worst agreement (~3e-3) is at k ~ 1e3 where u(k, M) oscillates.
    from shoddy.observables import _log_interp

    m = model_with_hod
    ks = np.logspace(-3, 3, 50)
    zs = np.linspace(0.05, 0.5, 50)
    A, B = m._resolve_terms(None)
    p_interp = _log_interp(ks, m.ks, A) + _log_interp(ks, m.ks, B) * m.Pk_m(ks, zs)
    assert np.allclose(p_interp, m.P_tot(ks=ks, z=zs), rtol=5e-3)


def test_resolve_terms_buckets(model_with_hod):
    m = model_with_hod
    A, B = m._resolve_terms(["1h"])
    assert B is None and np.allclose(A, m.Pk_1h())
    A, B = m._resolve_terms(["2h"])
    assert A is None and np.allclose(B, m.F_k() ** 2)


def test_resolve_terms_unknown_raises(model_with_hod):
    with pytest.raises(KeyError):
        model_with_hod._resolve_terms(["bogus"])


def test_galaxy_bias_above_one(model_with_hod):
    # For a M_min ~ 1e12 sample at z=0 the linear bias is > 1.
    assert model_with_hod.galaxy_bias() > 1.0


# --- Projection / correlation functions ----------------------------------------


@pytest.mark.skip(reason="CF3D not yet converted to an Observable")
def test_cf_3d_at_requested_radii(model_with_hod):
    xi, r = model_with_hod.cf_3d(rs=[1.0, 10.0])
    assert np.allclose(r, [1.0, 10.0])
    assert xi.shape == (2,)
    assert xi[0] > xi[1]  # correlation falls with separation


def test_cf_ang_at_requested_theta(model_with_hod):
    w, theta = ACF(model_with_hod)({}, theta=[0.01, 0.1])
    assert np.allclose(theta, [0.01, 0.1])
    assert np.all(w > 0)
    assert w[0] > w[1]


@pytest.mark.skip(reason="CF3D not yet converted to an Observable")
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


def test_acf_array_nz_requires_zarr(model_with_hod):
    with pytest.raises(ValueError):
        ACF(model_with_hod, nz=np.ones(5))


def test_acf_requires_hod(model):
    with pytest.raises(ValueError):
        ACF(model)


def test_cf_ang_terms_sum_to_total(hod_pars):
    # w(theta) is linear in P, so per-term results must sum to the total.
    m = Model(z=1)
    m.set_hod("zheng07", hod_pars)
    acf = ACF(m)
    theta = np.logspace(-3, 0, 10)
    w_tot, _ = acf({}, theta=theta)
    w_1h, _ = acf({}, terms="1h", theta=theta)
    w_2h, _ = acf({}, terms=["2h"], theta=theta)
    assert np.allclose(w_1h + w_2h, w_tot, rtol=1e-10)


# --- with_pars consistency ----------------------------------------------------


def test_with_pars_matches_fresh_set(model_with_hod):
    pars = {"M_min": 5e12, "sig_logM": 0.4, "M0": 2e12, "M1": 2e13, "alpha": 1.1}
    child = model_with_hod.with_pars(pars)
    reference = Model()
    reference.set_hod("zheng07", pars)
    assert np.isclose(child.n_gal(), reference.n_gal(), rtol=1e-10)
    assert np.allclose(child.P_tot(), reference.P_tot(), rtol=1e-10)


def test_with_pars_routes_to_components(model_with_hod):
    m = model_with_hod
    hod_only = m.with_pars({"M_min": 5e12})
    gal_only = m.with_pars({"corr": 0.0})
    mixed = m.with_pars({"M_min": 5e12, "R": 1.0})

    assert hod_only.hod.pars["M_min"] == 5e12 and hod_only.gal_occ.pars["corr"] == -1
    assert gal_only.gal_occ.pars["corr"] == 0.0 and gal_only.hod.pars["M_min"] == HOD_PARS["M_min"]
    assert mixed.hod.pars["M_min"] == 5e12 and mixed.gal_occ.pars["R"] == 1.0


def test_with_pars_leaves_parent_unchanged(model_with_hod):
    m = model_with_hod
    hod, gal = m.hod, m.gal_occ
    m.with_pars({"M_min": 5e12, "corr": 0.0, "R": 1.0})
    assert m.hod is hod and m.gal_occ is gal
    assert m.hod.pars["M_min"] == HOD_PARS["M_min"]
    assert m.gal_occ.pars == {"corr": -1, "R": 0}


def test_with_pars_unknown_key_raises(model_with_hod):
    with pytest.raises(ValueError, match="Unrecognized parameter"):
        model_with_hod.with_pars({"bogus": 1.0})


def test_update_pars_routes_to_components(fresh_model_with_hod):
    m = fresh_model_with_hod
    m.update_pars(M_min=5e12, corr=0.0, R=1.0)
    assert m.hod.pars["M_min"] == 5e12
    assert m.gal_occ.pars == {"corr": 0.0, "R": 1.0}
    with pytest.raises(ValueError):
        m.update_pars(bogus=1.0)


def test_constructor_routes_pars():
    m = Model(hod="zheng07", pars={**HOD_PARS, "corr": 0.0, "R": 1.0})
    assert m.hod.pars["M_min"] == HOD_PARS["M_min"]
    assert m.gal_occ.pars == {"corr": 0.0, "R": 1.0}
    with pytest.raises(ValueError, match="Unrecognized parameter"):
        Model(hod="zheng07", pars={**HOD_PARS, "bogus": 1.0})


# --- One-galaxy term -----------------------------------------------------------


def test_default_galaxy_component_is_off(model):
    assert model.gal_occ.pars == {"corr": -1, "R": 0}


def test_pk_1g_zero_by_default(model_with_hod):
    m = model_with_hod
    assert np.all(m.Pk_1g() == 0)
    assert np.allclose(m.P_tot(), m.Pk_1h() + m.Pk_2h())


def test_pk_1g_amplitude_is_corr_plus_one_times_dc_over_ngal(model_with_hod):
    m = model_with_hod.with_pars({"corr": 1.0, "dc": 0.5, "R": 1.0})
    assert np.isclose(m.Pk_1g(ks=np.array([1e-6]))[0] * m.n_gal(), 2.0 * 0.5)


def test_pk_1g_linear_in_corr_plus_one(model_with_hod):
    p1 = model_with_hod.with_pars({"corr": 0.0, "R": 1.0}).Pk_1g()
    p3 = model_with_hod.with_pars({"corr": 2.0, "R": 1.0}).Pk_1g()
    assert np.allclose(p3, 3 * p1)  # (corr + 1) = 1 -> 3


def test_ptot_includes_1g(model_with_hod):
    m = model_with_hod.with_pars({"corr": 0.0, "R": 1.0})
    assert np.allclose(m.P_tot(), m.Pk_1h() + m.Pk_2h() + m.Pk_1g())


def test_resolve_terms_1g_bucket(model_with_hod):
    m = model_with_hod.with_pars({"corr": 0.0, "R": 1.0})
    A, B = m._resolve_terms(["1g"])
    assert B is None and np.allclose(A, m.Pk_1g())


def test_u_g_damps_1h_and_2h(model_with_hod):
    # u_g multiplies each object's offset, so 1h ~ u_g^2 and F_k ~ u_g.
    base = model_with_hod
    m = base.with_pars({"R": 1.0})
    u_g = m.gal_occ.u(m._physical_k_kpc(m.ks))
    assert np.any(u_g < 0.99)
    assert np.allclose(m.Pk_1h(), base.Pk_1h() * u_g**2)
    assert np.allclose(m.F_k(), base.F_k() * u_g)


def test_R_converted_from_physical_kpc(hod_pars):
    # R in physical kpc -> comoving Mpc is R (1 + z) / 1000.
    z, R = 1.0, 2.0
    m = Model(z=z, hod="zheng07", pars={**hod_pars, "R": R})
    sig_c = r2sig(R) * (1 + z) / 1000
    assert np.allclose(m.gal_occ.u(m._physical_k_kpc(m.ks)),
                       np.exp(-(m.ks * sig_c) ** 2 / 2))


def test_acf_routes_galaxy_pars(model_with_hod):
    acf = ACF(model_with_hod)
    theta = np.logspace(-4, -1, 5)
    w_off, _ = acf({}, theta=theta)
    w_on, _ = acf({"corr": 0.0, "R": 1.0}, theta=theta)
    assert np.all(w_on[:2] > w_off[:2])  # 1g adds small-scale power
    with pytest.raises(ValueError):
        acf({"bogus": 1.0})


def test_cf_ang_terms_sum_to_total_with_1g(hod_pars):
    m = Model(z=1, hod="zheng07", pars={**hod_pars, "corr": 0.0, "R": 1.0})
    acf = ACF(m)
    theta = np.logspace(-3, 0, 10)
    w_tot, _ = acf({}, theta=theta)
    w_terms = [acf({}, terms=t, theta=theta)[0] for t in ("1h", "2h", "1g")]
    # 1h+1g share a log-interp, so splitting them is only exact to interpolation error
    assert np.allclose(sum(w_terms), w_tot, rtol=1e-3)


# --- Numerical regression snapshots --------------------------------------------

# Regression inputs are pinned explicitly so changing a library default does not
# move the snapshots.  The grids are the pre-Oct-2026 defaults.
REG_MODEL = {
    "cosmo_pars": {"H0": 70.0, "omch2": 0.25 * 0.7**2, "ombh2": 0.05 * 0.7**2,
                   "omk": 0.0, "As": 2e-9, "ns": 0.96, "mnu": 0.0},
    "hmf": "behroozi",
    "halo_prof": "nfw",
    "halo_mass_grid": np.logspace(9, 17, 256),
    "k_grid": np.logspace(-4, 5, 1501),
    "camb_kmax": 200.0,
}
REG_HOD = {**HOD_PARS, "dc": 1.0}


@pytest.fixture(scope="module")
def reg_model():
    return Model(z=0, hod="zheng07", pars=dict(REG_HOD), **REG_MODEL)


@pytest.fixture(scope="module")
def reg_model_z1():
    return Model(z=1, hod="zheng07", pars=dict(REG_HOD), **REG_MODEL)


# Power spectra are evaluated at fixed k, undamped.
REGRESSION = {
    "n_gal": 0.003573437968350265,
    "galaxy_bias": 1.2035170371749628,
    "Pk_1h_0": 1569.3435897667737,
    "Pk_2h_0": 2656.2543139295067,
    "P_gal_0": 4225.59790369628,
    "P_gal_k01": 15900.66071409825,
}


def test_regression_snapshots(reg_model):
    m = reg_model
    ks = np.array([1e-4, 1e-1])
    p_gal = m.P_tot(ks=ks)
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


@pytest.mark.skip(reason="CF3D not yet converted to an Observable")
def test_regression_cf_3d(reg_model):
    xi, _ = reg_model.cf_3d(rs=[1.0, 10.0])
    assert np.allclose(xi, [75.18974835, 0.87250518], rtol=1e-6)


def test_regression_cf_ang(reg_model_z1):
    # Gaussian n(z) (mean 1, sigma 0.25), all terms, undamped 1-halo term.
    zs = np.linspace(0.5, 1.5, 51)
    acf = ACF(reg_model_z1, zs=zs, nz=norm.pdf(zs, 1.0, 0.25),
              ls=np.logspace(0, 6, 1001))
    w, _ = acf({}, theta=[0.01, 0.1])
    assert np.allclose(w, [0.07565478, 0.01535156], rtol=1e-6)
