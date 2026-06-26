"""Tests for the caching framework and Model's use of it.

These pin the behaviour that the cache refactor is responsible for: lazy
computation, automatic invalidation when inputs change, and correct sharing /
non-mutation semantics in ``with_hod``.
"""

import numpy as np
import pytest

from shoddy.caching import Cached, cached_quantity


class _Counter(Cached):
    def __init__(self):
        self.calls = 0

    @cached_quantity
    def value(self):
        self.calls += 1
        return 42


def test_cached_quantity_computed_once():
    c = _Counter()
    assert c.value == 42
    assert c.value == 42
    assert c.calls == 1  # second access is a cache hit


def test_invalidate_named_recomputes():
    c = _Counter()
    _ = c.value
    c._invalidate("value")
    _ = c.value
    assert c.calls == 2


def test_invalidate_all_clears_everything():
    c = _Counter()
    _ = c.value
    assert "value" in c.__dict__
    c._invalidate()  # no args -> clear all cached quantities
    assert "value" not in c.__dict__


def test_invalidate_missing_is_noop():
    c = _Counter()
    c._invalidate("value")  # never computed; must not raise


def test_cached_names_discovers_quantities():
    assert "value" in _Counter._cached_names()


# --- Model-level cache behaviour -----------------------------------------------


def test_n_gal_is_cached(model_with_hod):
    first = model_with_hod.n_gal()
    assert model_with_hod.n_gal() == first
    assert "_n_gal" in model_with_hod.__dict__


def test_update_hod_pars_invalidates_density(fresh_model_with_hod):
    m = fresh_model_with_hod
    ng0 = m.n_gal()
    m.update_hod_pars(M_min=1e13)
    ng1 = m.n_gal()
    assert not np.isclose(ng0, ng1)


def test_recompute_flag_forces_recompute(fresh_model_with_hod):
    m = fresh_model_with_hod
    _ = m.n_gal()
    # Mutate the HOD behind the cache's back, then force a recompute.
    m.hod.pars["M_min"] = 1e13
    forced = m.n_gal(recompute=True)
    assert np.isclose(forced, m.n_gal())


def test_set_hmf_invalidates_halo_arrays(fresh_model_with_hod):
    m = fresh_model_with_hod
    _ = m.HMF.integration_weights(m.ms), m.n_gal()
    old_hmf = m.HMF
    m.set_hmf("tinker", m.halo_data)
    assert "_n_gal" not in m.__dict__
    # The hmf, bias and integration-weight arrays are cached on the HMF instance
    # itself; set_hmf builds a fresh instance, so those caches reset automatically.
    assert m.HMF is not old_hmf
    assert m.HMF._hmf_cache is None
    assert m.HMF._bias_cache is None
    assert m.HMF._weights_cache is None


def test_with_hod_does_not_mutate_parent(model_with_hod):
    ng0 = model_with_hod.n_gal()
    child = model_with_hod.with_hod({"M_min": 1e13})
    assert not np.isclose(child.n_gal(), ng0)
    # parent density unchanged
    assert np.isclose(model_with_hod.n_gal(), ng0)


def test_with_hod_shares_expensive_arrays(model_with_hod):
    # with_hod shares whatever is already cached at copy time, so warm the
    # parent's caches first; uncached quantities would be recomputed (identically)
    # in the child rather than shared.
    _ = (model_with_hod.HMF.integration_weights(model_with_hod.ms),
         model_with_hod.pkm_interp)
    _ = model_with_hod.HMF.bias(model_with_hod.ms)  # warm the HMF bias cache
    child = model_with_hod.with_hod({"M_min": 1e13})
    # The hmf and bias arrays now live on the (shared) HMF instance.
    assert child.HMF is model_with_hod.HMF
    assert child.HMF._hmf_cache is model_with_hod.HMF._hmf_cache
    assert child.HMF._hmf_cache is not None
    assert child.HMF._bias_cache is model_with_hod.HMF._bias_cache
    assert child.HMF._bias_cache is not None
    assert child.HMF._weights_cache is model_with_hod.HMF._weights_cache
    assert child.HMF._weights_cache is not None
    assert child.pkm_interp is model_with_hod.pkm_interp


def test_halo_weights_match_trapz_integration(model_with_hod):
    m = model_with_hod
    f = m.hod.N_hod(m.ms)
    direct = m.HMF.halo_integral(m.ms, f)
    via_weights = f @ m.HMF.integration_weights(m.ms)
    assert np.isclose(direct, via_weights, rtol=1e-12)
