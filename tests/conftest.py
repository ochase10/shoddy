"""Shared fixtures for the shoddy test suite.

Building a Model runs CAMB (~0.5 s), so the base models are session-scoped and
reused.  Tests that mutate a model must request a fresh (function-scoped) copy
via ``fresh_model_with_hod`` rather than the shared instances.
"""

import numpy as np
import pytest

from shoddy import Model


# A fixed, physically reasonable Zheng07 parameter set used throughout the suite.
HOD_PARS = {
    "M_min": 1e12,
    "sig_logM": 0.3,
    "M0": 1e12,
    "M1": 1e13,
    "alpha": 1.0,
}


@pytest.fixture(scope="session")
def hod_pars():
    return dict(HOD_PARS)


@pytest.fixture(scope="session")
def model():
    """A bare model (no HOD set) at z=0 with default cosmology/grids."""
    return Model()


@pytest.fixture(scope="session")
def model_with_hod():
    """Session-scoped model with a Zheng07 HOD. Do not mutate."""
    m = Model()
    m.set_hod("zheng07", dict(HOD_PARS))
    return m


@pytest.fixture
def fresh_model_with_hod():
    """Function-scoped model with a Zheng07 HOD, safe to mutate."""
    m = Model()
    m.set_hod("zheng07", dict(HOD_PARS))
    return m
