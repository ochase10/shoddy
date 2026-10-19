# Observables Plan

Fast, repeated evaluation of observables (e.g. inside an MCMC) without reintroducing a
general caching API. Items marked **Decide** need a design choice before implementation.

## Motivation

Timing of `base.with_hod(pars).cf_ang(theta)` (z=1, default 51-pt n(z), 1001 ells,
50 thetas, ~16 ms total):

| stage | ms | depends on HOD? |
|---|---|---|
| `with_hod` | 0.003 | yes |
| `Pk_m(ks, zs)` on 51k scattered (k, z) points | 9.5 | **no** |
| `Pk_1h` grid | 2.2 | yes |
| `F_k` grid | 1.1 | yes |
| 2x `np.interp` onto 51k points | 1.0 | values yes, target ks no |
| H(z), chi(z) | 0.2 | no |
| Hankel construct | 0.37 | no |
| Hankel apply | 0.06 | yes |
| spline to theta | 0.19 | yes |

- `Pk_m` cost is spline *evaluation*, not construction. CAMB's `RectBivariateSpline` with
  `grid=False` evaluates each (k, z) independently (~180 ns/pt); `grid=True` on a 51x1001
  tensor grid is ~0.5 ms. Limber ks aren't a tensor grid (k = (l+1/2)/chi(z)).
- ~60% of each call is HOD-independent work that is recomputed every time.
- Swapping spline -> `np.interp` for the final theta/r interpolation saves only ~1-5% of a call;
  accuracy loss < 1e-3 relative (cf_3d ~6e-4, cf_ang ~1e-4). Not worth prioritizing.

## Idea

Frozen observable objects: built once from a `Model` plus a fixed observable configuration
(n(z), ells, theta, ...). `__init__` precomputes everything HOD-independent; `__call__` does
only HOD-dependent work. No invalidation logic — different configuration means a new object.

```python
wt = WTheta(model, theta, z_arr=..., nz=..., ls=...)
w = wt(model.with_hod(hod_pars))
```

This is an additional path, not a replacement for `Model` / `with_hod`.

## Design decisions

- [x] **Decide:** how the target observable is specified
  - One class per observable (`WTheta`, `Xi3D`, `NGal`, ...) with a small shared base — *suggested*
  - Single class with string dispatch (`Predictor(model, "cf_ang", ...)`) — drifts toward the old caching API
- [ ] **Decide:** `__call__` signature
  - Takes an HOD'd `Model` — *suggested*; makes joint fits trivial (call `with_hod` once, pass to each observable)
  - Takes `hod_pars` dict
- [ ] **Decide:** where observable logic lives
  - Observables own projection/transform logic; `Model.cf_ang` / `Model.cf_3d` become thin wrappers
    (`return WTheta(self, theta, ...)(self)`) — *suggested*; single source of truth
  - Keep `Model` methods; observables reimplement assembly — duplication, needs agreement test
  - Remove observables from `Model` entirely — cleanest split, breaks API
- [x] **Decide:** fate of generic `power_func` in `cf_ang` / `limber_cl`
  - Fast path needs P_gal = HOD terms x `Pk_m` to precompute `Pk_m`; an arbitrary `power_func` breaks that
  - Options: keep `limber_cl(power_func, ...)` as a standalone utility the fast class doesn't use, or drop the argument
- [ ] **Decide:** joint fits — thin composite wrapper holding several observables, or leave to user code

Suggested split of responsibility:
- `Model`: halo-model physics — `n_gal`, `galaxy_bias`, `Pk_1h`, `F_k`, `P_gal`
- Observables: projection/transform — Limber, Hankel/FFTLog, interpolation to theta/r

## Implementation notes (WTheta)

- [ ] Sort each stage in the table above into `__init__` vs `__call__`
- [ ] Precompute: `ks_2d`, `Pk_m(ks_2d, z)`, Limber kernel, Hankel object
- [ ] Target ks for the `p1h` / `Fk` interpolation are fixed — interpolation indices/weights could be precomputed
- [ ] Explore: rearrange the Limber z-sum so per-call work stays on 1D arrays over `model.ks`
      instead of 51k-point arrays
- [ ] Reuse `Model.Pk_1h` / `Model.F_k` in `__call__` rather than duplicating physics
- [ ] Rough target: ~4-5 ms per call (vs ~16 ms), less if the 51k-point arrays are avoided

## Testing

- [ ] `test_regression_cf_ang` / `test_regression_cf_3d` must pass unchanged after moving logic
- [ ] New test: observable object output matches `Model` wrapper for several HOD parameter sets
- [ ] Benchmark script for per-call time (see timing table above)
