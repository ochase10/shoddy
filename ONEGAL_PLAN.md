# One-Galaxy (1g) Term Plan

Add a third halo-model level (halos → host galaxies → objects) so that point-like tracers
(LRDs, BHs, GCs) can have more than one object per galaxy. Physics follows `owen_note.pdf`.
Run `pytest` (env `hod`) after each phase. Items marked **Decide** need a design choice
before implementation.

## Physics summary

- The tracer is always *objects*. The existing HOD gives host-galaxy occupation
  ⟨G⟩ = Nc + Ns. Every galaxy hosts objects with mean λ and
  ⟨N(N−1)⟩_gal = m λ², where m = 1 + δs is the **multiplicity**.
  - m = 0: at most one per galaxy (nucleus-like) → no 1g term
  - m = 1: Poisson
  - m > 1: synchronized
  - Only the second moment enters P(k), so the underlying distribution (binomial / Poisson /
    negative binomial) does not matter here. No "mode" switch needed — m is a continuous
    parameter ≥ 0.
- n̄ = λ n_host, with n_host = ∫ dn/dM ⟨G⟩.
- Terms (u_h = halo profile, u_g = object-within-galaxy profile):
  - 1g: (1/n̄²) ∫ dn/dM m λ² ⟨G⟩ |u_g|²
  - 1h: (1/n̄²) ∫ dn/dM λ² (2NcNs u_h + Ns² u_h²) |u_g|²
  - 2h: P_m · (F_k u_g)²
- **λ cancels from every term when λs = λc (rs = 1).** λ is a derived parameter
  (λ = n_obj / n_host); it only matters if rs ≠ 1 or if a likelihood needs n̄ itself
  (e.g. Poisson pair counts μ = RR(1 + w)).
- u_g enters 1h and 2h because each object is independently displaced from its galaxy
  centre (independent offsets convolve → profiles multiply). With r_d ~ kpc, u_g ≈ 1 wherever
  1h/2h matter.
- Galaxy profile (note's default): exponential, u_g(k) = (1 + k² r_d²)⁻², r_d ∈ [0.2, 2]
  physical kpc. Analytic and M-independent.

## Phase 1 — Design decisions

- [ ] **Decide:** where multiplicity lives
  - Bundled with the galaxy profile in one "galaxy" component with its own `pars` dict — *suggested*;
    parameter routing then has two destinations (HOD, galaxy)
  - Loose attribute on `Model` — three routing destinations
  - On the `HOD` base class — couples within-galaxy physics to halo occupation; `with_pars`
    rebuilds via `type(self)(**pars)` so every subclass would need to accept it
- [ ] **Decide:** whether 1h/2h carry u_g factors
  - Keep (formally consistent, ~free) — *suggested*
  - Drop (u_g ≈ 1 on 1h/2h scales)
- [ ] **Decide:** whether to support rs = λs/λc ≠ 1 (breaks λ cancellation) — suggest deferring
- [ ] **Decide:** whether the galaxy component is swappable after construction (analogue of
  `set_hod(new_hod, ...)`) or fixed at `Model.__init__`
- [ ] **Decide:** redshift used for r_d physical → comoving conversion. `config.z` is a single
  value, but `ACF` integrates over an n(z) range — accept `config.z` or make u_g z-dependent

## Phase 2 — Galaxy component

- [ ] New component class(es) (e.g. `galaxy.py` with an ABC + `Exponential`), mirroring the
  `HaloProfile` / `HOD` pattern; owns `pars` (multiplicity, r_d, ...)
- [ ] `u_g(ks)` computed on the fly — no `u_cache` (r_d is sampled; analytic, k-only, cheap)
- [ ] Immutable update: `with_pars` returns a new object, never mutates (shallow `copy.copy` of
  `Model` must not share a mutated component)
- [ ] Units: code k is comoving 1/Mpc (`hubble_units=False`); convert r_d from physical kpc
  (× (1 + z) / 1000)
- [ ] Default state is "off": m = 0 and u_g ≡ 1 (r_d = 0), so a `Model` without a galaxy
  component reproduces the current HOD exactly

## Phase 3 — Parameter routing

- [ ] Rename `with_hod` / `update_hod_pars` → `with_pars` / `update_pars` that update either or
  both components
- [ ] Each component declares the keys it owns (HOD already does via `pars.keys()`);
  `with_pars` splits the incoming dict by membership
- [ ] Unknown keys raise (a typo must not silently become a dead sampling dimension)
- [ ] Check for key collisions between components at construction
- [ ] **Decide:** whether a mutating `update_pars` should exist for the galaxy component at all
- [ ] Propagate the rename: `Observable.__call__`, tests, notebooks

## Phase 4 — Terms

- [ ] `Pk_1g(ks=None, Ms=None)`; register in `_TERMS` under the `'k'` bucket
- [ ] Apply u_g factors to `Pk_1h` and `F_k` (if kept, per Phase 1)
- [ ] Normalisation: confirm 1/n̄² with λ cancelled reduces to 1/n_host² (1h, 2h) and
  m / n_host (1g amplitude, matches note's A = (1 + δs)/n_host)
- [ ] Check `_resolve_terms` behaviour when 1g is off (m = 0): zero array vs skipping the term
- [ ] `ACF` should need no changes beyond the rename — verify

## Phase 5 — Tests

- [ ] Galaxy component at defaults reproduces all existing regression values unchanged
- [ ] m = 0 → `Pk_1g` is zero; total equals 1h + 2h
- [ ] u_g(k → 0) = 1; u_g matches analytic form
- [ ] 1g amplitude at k → 0 equals m / n_host
- [ ] `with_pars` routing: HOD-only, galaxy-only, mixed, unknown key raises, parent model
  unchanged after `with_pars`
- [ ] Per-term `ACF` results (1h + 2h + 1g) sum to the total
- [ ] New regression snapshot with pinned inputs and a non-trivial galaxy component

## Future

- [ ] Projected 1g correlation analytically (Matérn kernel K(R) = (R/r_d)³ K₃(R/r_d) / (96π r_d²)) —
  avoids FFT ringing; would bypass the Hankel path for this term
- [ ] Alternative galaxy profiles (Gaussian, e^{−r}/r, power-law tails) — the note finds results
  insensitive to shape
- [ ] `CF3D` conversion (see `OBSERVABLES_PLAN.md`) before the 1g term is used in 3D
