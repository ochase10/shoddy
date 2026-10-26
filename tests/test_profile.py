"""Tests for halo density profiles (NFW)."""

import numpy as np


def test_profile_unity_at_k0(model):
    # u(k -> 0, M) = 1 by construction (the Fourier transform of a normalised
    # profile).  The sici terms diverge and are explicitly patched to 1.0.
    u = model.halo_profile.u(np.array([1e-8]), model.ms)
    assert np.allclose(u[0], 1.0)


def test_profile_decays_at_high_k(model):
    # At large k the profile is well below unity (structure is smoothed out).
    u = model.halo_profile.u(np.array([1e3]), model.ms)
    assert np.all(u[0] < 1.0)
    assert np.all(np.isfinite(u[0]))


def test_profile_finite_everywhere(model):
    u = model.halo_profile.u(model.ks, model.ms)
    assert np.all(np.isfinite(u))
    assert u.shape == (len(model.ks), len(model.ms))


def test_u_cache_matches_profile(model):
    # The no-argument (cached) path must equal a direct evaluation on the grids.
    prof = model.halo_profile
    assert prof.u() is prof.u_cache
    assert np.allclose(prof.u(), prof.u(model.ks, model.ms))


def test_concentration_decreases_with_mass(model):
    M = np.logspace(11, 15, 20)
    c = model.halo_profile.conc(M)
    assert np.all(np.diff(c) < 0)  # beta < 0 -> concentration falls with mass
