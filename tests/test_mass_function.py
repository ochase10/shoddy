"""Tests for halo mass functions, bias, and the HaloConfig backing them."""

import numpy as np
import pytest

from shoddy import MassFunction
from shoddy.mass_function import Tinker, Behroozi13, dM2dlogM, dlogM2dM
from shoddy.utils import _trapz


def test_jacobian_roundtrip():
    M = np.logspace(10, 15, 50)
    val = np.random.default_rng(0).random(50)
    assert np.allclose(dlogM2dM(dM2dlogM(val, M), M), val)


def test_hmf_positive(model):
    assert np.all(model.hmf.dndm(model.ms) > 0)


def test_hmf_decreases_with_mass(model):
    # dn/dM falls steeply toward high mass.
    hmf = model.hmf.dndm(model.ms)
    assert np.all(np.diff(hmf) < 0)


def test_bias_increases_with_mass(model):
    b = model.hmf.bias(model.ms)
    assert np.all(np.diff(b) > 0)


def test_nu_increases_with_mass(model):
    nu = model.hmf.nu(model.ms)
    assert np.all(np.diff(nu) > 0)


def test_sigma_decreases_with_mass(model):
    sig = model.config.sigma_m(model.ms)
    assert np.all(np.diff(sig) < 0)


def test_halo_integral_matches_manual_trapz(model):
    # halo_integral integrates in ln(M): dM = M dlnM.
    f = np.ones_like(model.ms)
    hmf = model.hmf.dndm(model.ms)
    expected = _trapz(hmf * model.ms, np.log(model.ms))
    assert np.isclose(model.hmf.halo_integral(model.ms, f), expected, rtol=1e-12)


def test_halo_integral_axis(model):
    # Integrating a (n_k, n_M) array along the mass axis.
    arr = np.ones((5, len(model.ms)))
    out = model.hmf.halo_integral(model.ms, arr, axis=1)
    assert out.shape == (5,)
    assert np.allclose(out, out[0])


def test_halo_integral_shape_mismatch_raises(model):
    with pytest.raises(ValueError):
        model.hmf.halo_integral(model.ms, np.ones(3))


def test_behroozi_applies_correction_to_tinker(model):
    # Behroozi13 is a small multiplicative correction on Tinker; at z=0 the
    # correction is tiny (~1e-6) but nonzero, so the arrays must not be identical.
    behroozi = Behroozi13(model.config).dndm(model.ms)
    tinker = Tinker(model.config).dndm(model.ms)
    assert not np.array_equal(behroozi, tinker)
    assert np.all(behroozi > 0)


def test_tinker_is_mass_function(model):
    assert isinstance(Tinker(model.config), MassFunction)


class _StubConfig:
    def __init__(self, z, delta):
        self.z, self.delta = z, delta


def test_tinker_hyperpars_match_table_at_z0():
    t = Tinker(_StubConfig(z=0, delta=200))
    assert np.allclose([t.A, t.a, t.b, t.c], Tinker._hyper_z0[0, 1:])


def test_tinker_hyperpars_z_evolution():
    t0 = Tinker(_StubConfig(z=0, delta=200))
    t1 = Tinker(_StubConfig(z=1, delta=200))
    assert np.isclose(t1.A, t0.A * 2**-0.14)
    assert np.isclose(t1.a, t0.a * 2**-0.06)
    assert t1.b < t0.b
    assert t1.c == t0.c


def test_tinker_hyperpars_do_not_mutate_table():
    table = Tinker._hyper_z0.copy()
    Tinker(_StubConfig(z=1, delta=500))
    assert np.array_equal(Tinker._hyper_z0, table)
