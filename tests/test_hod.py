"""Tests for HOD occupation models."""

import numpy as np
import pytest

from shoddy import HOD
from shoddy.hod import Zheng07


@pytest.fixture
def zheng():
    return Zheng07(M_min=1e12, sig_logM=0.3, M0=1e12, M1=1e13, alpha=1.0)


def test_centrals_saturate_to_dc(zheng):
    # Far above M_min the erf -> 1, so centrals -> dc (=1 by default).
    high = zheng.centrals(np.array([1e16]))
    assert np.isclose(high[0], 1.0, atol=1e-6)


def test_centrals_half_at_mmin(zheng):
    # At M = M_min the erf argument is 0, so centrals = 0.5 * dc.
    val = zheng.centrals(np.array([1e12]))
    assert np.isclose(val[0], 0.5)


def test_centrals_monotonic(zheng):
    M = np.logspace(10, 15, 50)
    c = zheng.centrals(M)
    assert np.all(np.diff(c) >= 0)


def test_satellites_zero_below_M0(zheng):
    # The (M - M0) term is clamped to zero, so no satellites below M0.
    assert zheng.satellites(np.array([1e11]))[0] == 0.0


def test_satellites_nonnegative(zheng):
    M = np.logspace(10, 15, 50)
    assert np.all(zheng.satellites(M) >= 0)


def test_N_hod_is_centrals_plus_satellites(zheng):
    M = np.logspace(11, 15, 20)
    assert np.allclose(zheng.N_hod(M), zheng.centrals(M) + zheng.satellites(M))


def test_avg_quantities(zheng):
    M = np.logspace(11, 15, 20)
    assert np.allclose(zheng.avg_NcNs(M), zheng.centrals(M) * zheng.satellites(M))
    assert np.allclose(zheng.avg_Ns2(M), zheng.satellites(M) ** 2)


def test_update_pars(zheng):
    zheng.update_pars(M_min=1e13)
    assert zheng.pars["M_min"] == 1e13
    # half-occupation point should now be at the new M_min
    assert np.isclose(zheng.centrals(np.array([1e13]))[0], 0.5)


def test_zheng07_is_hod_subclass(zheng):
    assert isinstance(zheng, HOD)
