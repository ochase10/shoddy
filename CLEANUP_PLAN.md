# Cleanup Plan

Ordered so earlier phases unblock later ones. Run `pytest` (env `hod`) after each phase.
Items marked **Decide** need a design choice before implementation.

## Phase 0 — Quick fixes (independent, low risk)

- [x] Replace mutable defaults `hod_pars={}` / `pars={}` with `None` (`Model.__init__`, `Model.set_hod`)
- [x] `check_HOD_defined`: raise a specific exception type instead of bare `Exception`
- [x] `limber_cl`: fix docstring (default `nz` is Gaussian σ=0.25, not top-hat) — or change the default to match the doc
- [x] `limber_cl`: either drop the dead `power_func is None` branch or give `power_func` a default of `None`
- [x] Zheng07 duty cycle: check whether `dc` should also scale `avg_NcNs` / `avg_Ns2` (and whether it belongs in `centrals`/`satellites` instead of an `N_hod` override); add a test for the intended scaling
- [ ] Replace `from .utils import *` with explicit imports (`halo_config.py`, `mass_function.py`)
- [x] Remove unused `a2z` (or keep if planned) and unused `Model.rhocrit0`
- [x] Revisit uncommitted `init_cosmo` change: uppercase kwargs `Z_PAD`/`Z_STEP` → either module constants or lowercase params

## Phase 1 — State ownership (Model vs HaloConfig)

- [x] **Decide:** which object owns `z`, `cosmo`, and the mass grid. Options:
  - HaloConfig owns them; Model reads through it
  - Model owns them; HaloConfig becomes a thin derived-quantities helper (σ(M), r_vir, ρ_m)
  - Merge HaloConfig into Model (or a new `Cosmology` object)
- [x] Single default mass grid (currently `1e9–1e16` in Model vs `1e9–1e17` in HaloConfig)
- [x] Single name for the object: pick one of `HaloConfig` / `halo_data` / `config`
- [x] Make `limber_cl` use one consistent source for `z` and `cosmo`
- [x] Stop silently swallowing `**kwargs` (`Model`, `HaloConfig`, `HOD`) — pass through explicitly or remove
- [x] Update tests referencing `model.halo_data`

## Phase 2 — Unify caching

- [x] **Decide:** one caching mechanism for the package. Likely `caching.py` (`Cached` / `cached_quantity`), with grid-dependent values cached only on the default grid
- [x] Port `MassFunction._on_grid` caches (`hmf`, `bias`, `integration_weights`) to it
- [x] Port `HaloProfile` single-slot cache to it; design so Model no longer calls `prof._compute_profile` directly
- [x] Remove `recompute=` flags (`n_gal`, `k_profile`) if invalidation makes them redundant
- [x] Update `test_caching.py` / `test_profile.py` accordingly

## Phase 3 — Component wiring

- [x] **Decide:** whether HOD takes the config like HMF/profile, or stays config-free (and document why)
- [x] `set_hmf` / `set_halo_profile`: drop the `config` argument; use the Model's own
- [x] `HOD` base: remove `**kwargs` and the `pars = {}` placeholder overwritten by subclasses
- [ ] Move `conc` to `NFW` (Duffy-type parameters are NFW-specific); reuse `Ac` inside `NFW._compute_profile`

## Phase 4 — Deduplicate model.py

- [x] **Decide:** how to handle repeated `check_HOD_defined(); assert hod is not None` and `Ms`/`ks` defaulting — helper method, property that raises, or decorator
- [ ] One 1-halo implementation (`Pk_cs` + `Pk_ss`) reused by `Pk_1h`, `_pk_1h_extended`, `_build_fast_power_func`
- [x] One high-k grid extension helper (pick one point-density rule)
- [ ] Build `P2xi` once per call in `cf_3d`

## Phase 5 — Split model.py

- [ ] **Decide:** module layout, e.g.
  - `cosmology.py` — CAMB setup, `matter_power_spectrum`, `k_damp_1h`
  - `power.py` (or stay on Model) — `Pk_*`, `P_gal`, `n_gal`, `galaxy_bias`
  - `projection.py` — `limber_cl`, `cf_3d`, `cf_ang`, fast power func
- [ ] Keep `Model` as the user-facing entry point

## Phase 6 — Naming and docs

- [ ] **Decide:** naming convention for spectra/correlation methods (`Pk_1h`, `P_gal`, `matter_power_spectrum`, `cf_3d`, `cf_ang`, `galaxy_bias`)
- [ ] Consistent attribute names (`HMF`, `prof`, `hod`)
- [ ] Trim docstrings/comments that explain past bug fixes (e.g. `halo_config.py` ρ_m comment, `_build_fast_power_func`, `caching.py` module doc incl. nonexistent `_invalidate_all`)
- [ ] Real package docstring in `__init__.py`; implement or drop `Zheng07.__str__` TODO
- [ ] Update `README.md` and `demo.ipynb` for any API changes
