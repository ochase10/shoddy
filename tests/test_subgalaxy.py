"""Tests for within-galaxy occupation/profile components."""

import numpy as np
import pytest

from shoddy.subgalaxy import Gaussian, GalaxyOccupation, r2sig
from shoddy.utils import _trapz


@pytest.fixture
def gauss():
    return Gaussian(R=1.0, corr=1.0)


def test_defaults_are_off():
    # corr=-1 (no 1g term) and R=0 (u_g = 1) reproduce the plain HOD.
    g = Gaussian()
    assert g.pars == {"corr": -1, "R": 0}
    k = np.logspace(-4, 4, 50)
    assert np.all(g.u(k) == 1.0)


def test_pars_inherit_from_base():
    assert "corr" in Gaussian.PARS and "R" in Gaussian.PARS
    assert isinstance(Gaussian(), GalaxyOccupation)


def test_u_unity_at_k0(gauss):
    assert np.isclose(gauss.u(np.array([0.0]))[0], 1.0)


def test_u_matches_analytic(gauss):
    k = np.logspace(-2, 1, 20)
    sig = 1.0 / np.sqrt(2 * np.log(2))
    assert np.allclose(gauss.u(k), np.exp(-(k * sig) ** 2 / 2))


def test_R_is_half_width_half_max(gauss):
    assert np.isclose(gauss.rho(1.0) / gauss.rho(0.0), 0.5)


def test_rho_normalised(gauss):
    r = np.linspace(0, 20 * r2sig(1.0), 20001)
    assert np.isclose(_trapz(4 * np.pi * r**2 * gauss.rho(r), r), 1.0, rtol=1e-6)


def test_update_pars_reruns_setup(gauss):
    gauss.update_pars(R=2.0)
    assert np.isclose(gauss.sig2, r2sig(2.0) ** 2)


def test_with_pars_leaves_parent_unchanged(gauss):
    child = gauss.with_pars({"R": 3.0})
    assert child is not gauss
    assert child.pars == {"corr": 1.0, "R": 3.0}
    assert gauss.pars == {"corr": 1.0, "R": 1.0}
    assert np.isclose(gauss.sig2, r2sig(1.0) ** 2)


def test_unknown_par_raises():
    with pytest.raises(ValueError):
        Gaussian(r_d=1.0)
    with pytest.raises(ValueError):
        Gaussian().update_pars(r_d=1.0)
