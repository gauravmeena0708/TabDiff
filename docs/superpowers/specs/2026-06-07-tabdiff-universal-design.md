# tabdiff-universal — Design Spec

**Date:** 2026-06-07
**Status:** Approved (brainstorming), ready for implementation planning
**Target repo:** `sub/TabDiff` (this repo)
**Source of the idea:** `sub/diffutabgen/methods/universal_guider.py`

## 1. Motivation

diffutabgen exposes a training-free constraint sampler, `universal_guider.py`, that
enforces equalities, inequalities, and aggregates (`age>30`, `mean(balance)=1000`,
`education=Bachelors`) by injecting a gradient/guidance signal at each reverse
diffusion step. Its output quality is bounded by the underlying diffuser, and the
diffuser available to it in diffutabgen (trained via the RelDDPM/core DDPM path) is
**weak**.

TabDiff trains a **strong** tabular diffusion model. Its current conditional path
(`sample_impute`, REPAINT-style imputation, exposed through `sdcontract` as the
eq-only `tabdiff` method) only supports **exact equality** — it cannot express
inequalities or aggregates.

**Goal:** bring `universal_guider`'s full *native* constraint dialect to TabDiff's
strong model as a new method, `tabdiff-universal`, exposed through TabDiff's
`sdcontract` plugin — without degrading the existing `tabdiff` paths.

## 2. Scope

### In scope (native constraint dialect, both numeric and categorical)
- `col=val`, `col!=val`
- `col>val`, `col>=val`, `col<val`, `col<=val` (numeric)
- `mean(col)=val` (numeric aggregate)
- `col<val~p` / `col>val~p` fraction targets (numeric distribution-matching)
- Per-spec `@scale=N` override and the shared `~p` / `@scale` suffix grammar
- A single new variant: `tabdiff-universal`

### Out of scope (with rationale)
- **Proxies** (`core/proxies/*`): a workaround for diffutabgen's *entangled* wrapper
  latents, where a column cannot be read directly off the latent. TabDiff's numeric
  channel is per-column normalized (`denoised[:, idx]` is the column's x0 estimate)
  and categoricals carry per-column logits, so constraints read columns directly.
  The problem proxies solve does not exist here.
- **Wrapper variants** (`#vae`, `#cttvae`, `#tokens`, …): diffutabgen has a pluggable
  wrapper registry needing one variant per embedding. TabDiff has exactly one fixed
  representation (EDM numeric + MDLM categorical). Nothing to vary.
- **MMD / distribution matching**: a genuine separate capability, not a representation
  workaround. Excluded from this scope. *Note:* unlike proxies/wrappers it would port
  cleanly later (operate on `denoised[:, idx]` against a reference sample) if wanted.
- **Any training change**, and **any change to TabDiff's core model/sampler beyond two
  appended methods** (see §5).

## 3. Background: why a direct port is impossible

`universal_guider.universal_cond_fn` is bound to diffutabgen's DDPM API:
`eps`-parametrization, `_predict_xstart_from_eps`, a single concatenated continuous
latent with `global_bit_indices`, the `posterior_variance`/`alphas` schedule, and a
`sample(control_tools={'cond_fn': ...})` plumbing path.

TabDiff (`UnifiedCtimeDiffusion`) is structurally different:
- **Numeric** columns: EDM continuous diffusion. `_denoise_fn` returns `denoised`
  (an **x0 estimate**, per-column normalized). Differentiable w.r.t. `x_num`.
- **Categorical** columns: MDLM / absorbing-state **discrete** diffusion
  (`_subs_parameterization`, `_mdlm_update`, `_sample_categorical`). No continuous
  latent; the only lever is the per-step unmasking **logits**.

So `universal_cond_fn` cannot be reused. What *is* reusable is representation-independent:
the **spec-dialect parser** (`_split_constraint`, `@scale`/`~p` parsing) and the
**constraint→loss math**. The sampler integration is reimplemented natively (Approach C).

## 4. Architecture

Three units with clear boundaries:

### 4.1 `tabdiff/guidance.py` (new, self-contained)
- `parse_constraint_spec(spec, info, num_transform, int_transform) -> Constraint`
  - Reuses `universal_guider._split_constraint` and the `@scale=` / `~p` suffix
    parsing **verbatim** (representation-independent).
  - Resolves `col` against TabDiff `info` (`column_names`, `num_col_idx`,
    `cat_col_idx`, `target_col_idx`), reusing the existing column-alias fuzzy match
    from `_engine.generate_multi_conditional`.
  - Numeric targets are normalized through `int_transform`/`num_transform` **exactly**
    as `_engine.generate_multi_conditional` does today (the mean-row substitution
    trick, `_engine.py` lines 237–246), so the target lands in `denoised`'s space.
  - Categorical targets resolve to `(cat_column_position, class_id)` via
    `get_category_encoding`, mapped to the logit slice via
    `slices_for_classes_with_mask`.
- Constraint classes, each exposing `.loss(x)` and `.scale`:
  - **Numeric** (operate on `denoised[:, idx]`): `Equality`, `GreaterThan`,
    `LessThan`, `NotEqual`, `Mean`, `Fraction`. Loss math mirrors the corresponding
    classes in diffutabgen `core/constraints.py`, specialized to a single column index
    (no `global_bit_indices`).
  - **Categorical** (operate on a column's logit slice): `CatEquality`, `CatNotEqual`.
- No torch model import; unit-testable against a fake `info`.

### 4.2 `UnifiedCtimeDiffusion.sample_guided(...)` + `_edm_update_guided(...)` (new methods)
Two **appended** methods on the existing class. `edm_update`, `sample_impute`,
`sample`, `sample_all` are left **byte-for-byte identical** (max-isolation decision —
verifiable via `git diff`). `_edm_update_guided` is a copy of `edm_update` with the two
intervention points below; `sample_guided` is a copy of `sample_impute`'s reverse loop
that calls `_edm_update_guided` and threads the constraint list + guidance knobs.

### 4.3 Engine + contract wiring (`sdcontract/`)
- `_engine.generate_guided(req, native_constraints, ...)` — new function parallel to
  `generate_multi_conditional`; `generate()` dispatches to it when `variant ==
  "universal"`. Existing engine functions untouched.
- `sdcontract/generate` — add `"universal"` to a `_PASSTHROUGH_MODES` set so the
  universal dialect passes through raw (mirrors diffutabgen's `generate` endpoint).
- `meta.json` / `capabilities.json` — register
  `"tabdiff-universal": { "variant": "universal", "enabled": true }`.

## 5. Guidance mechanics

### 5.1 Numeric — guidance on the x0 estimate (intervention point A)
Inside `_edm_update_guided`, after `denoised` is computed and **before** the Euler step
(`d_cur = (x_num_hat - denoised) / sigma_num_hat`):
1. Optimize a correction `Δ` (init 0) over `m` inner steps minimizing
   `Σ scaleᵢ · lossᵢ(denoised + Δ)`. This is the direct translation of
   `universal_guider._compute_backward_delta` (paper Eq. 7), operating on TabDiff's
   **native x0 estimate**. For pointwise losses (`>`, `<`, `=`, `mean`, `fraction`) the
   gradient is closed-form ⇒ cheap, **no extra denoiser calls**.
2. Use `denoised + Δ` in the Euler step; the constraint propagates through the
   unchanged EDM update.
3. Strength annealed by the noise level (guide harder near clean), reusing the
   `guidance_schedule` idea from `universal_guider`.
4. **Optional** `exact_gradient` mode: backprop through `_denoise_fn` w.r.t.
   `x_num_hat` (DPS) under a local `torch.enable_grad()`. Default **off** (the engine
   samples under `no_grad`; x0-space guidance is the cheaper default).

### 5.2 Categorical — guidance on the unmasking logits (intervention point B)
Inside `_edm_update_guided`, after `logits = self._subs_parameterization(raw_logits,
x_cat_hat)` and **before** `_mdlm_update`:
- `col = target`: add `+β·w(t)` to the target class's logit within that column's
  `slices_for_classes_with_mask` slice.
- `col != target`: add `−β·w(t)` to the excluded class's logit.
- `β` = categorical scale; `w(t)` anneals over t.
- **Soft by construction** (accepted tradeoff): high CSR with the strong model but not
  guaranteed-exact like REPAINT. `cat_snap_final` flag optionally hard-sets the
  unmasking to the target class on the final step(s) for equality — an exactness
  escape hatch.

## 6. Request / knob surface

`generate_guided` reads these optional request fields (defaults mirror
`universal_guider`); a bare constraint list works with all defaults:

| field | meaning | default |
|---|---|---|
| `num_scale` | numeric constraint strength | as `universal_guider` (0.01-ish, normalized) |
| `cat_scale` | categorical logit β | analog of 0.2 |
| `mean_scale` | `mean(...)` strength | as `universal_guider` |
| `backward_steps` | numeric inner-loop `m` | small (e.g. 5) |
| `guidance_schedule` | t-annealing of strength | `none`/`sqrt_alpha`-style |
| `cat_snap_final` | hard-snap categorical equality on final step | `false` |
| `exact_gradient` | DPS through denoiser for numerics | `false` |

Per-spec `@scale=N` and `~p` are honored by the shared parser and override the
blanket scales.

## 7. Error handling

- Unknown column / unknown class → `ValueError` listing valid names (reuse
  `universal_guider` messages + engine alias match).
- Numeric op on a categorical column, or `mean(catcol)` → explicit `ValueError`.
- Empty constraint list under `variant == "universal"` → fall back to unconditional
  `sample_all` (matches `generate()`'s existing "nothing to condition on" behavior).
- Contract discipline unchanged: one JSON object on stdout, logs on stderr.

## 8. Testing & success criteria

- **Original-paths-intact regression (primary safety goal):** with a fixed seed,
  unconditional and `none`-variant TabDiff output is identical before vs. after this
  change. Proves the appended methods do not perturb existing behavior.
- **Unit (`tabdiff/guidance.py`, no model):** every operator parses; `@scale`, `~p`,
  `mean()` parse; numeric target normalization against a fake `info`; categorical
  class→logit-slice mapping. Fast.
- **Numeric guidance (tiny synthetic checkpoint):** CSR rises markedly with guidance
  vs. without for `age>30` and `mean(col)=v`.
- **Categorical guidance:** `col=val` lands high CSR (and `cat_snap_final` ⇒ exact);
  `col!=val` avoids the class.
- **Contract smoke:** JSON-in/JSON-out through `sdcontract/generate` for a mixed
  constraint set on `adult`; assert `ok:true` and output columns match the training
  schema.

## 9. File-change footprint

**New**
- `tabdiff/guidance.py`
- `tests/...` for the above (location per repo convention)
- this spec

**Edited (additive only; existing code paths byte-for-byte unchanged)**
- `tabdiff/models/unified_ctime_diffusion.py` — append `_edm_update_guided` +
  `sample_guided`. `edm_update`/`sample_impute`/`sample`/`sample_all` untouched.
- `sdcontract/_engine.py` — append `generate_guided`; add one dispatch branch in
  `generate()`.
- `sdcontract/generate` — add `"universal"` to a passthrough-modes set.
- `sdcontract/meta.json`, `sdcontract/capabilities.json` — register the variant.

## 10. Open items for planning

- Exact default values for `num_scale`/`cat_scale`/`mean_scale` in TabDiff's
  normalized space (calibrate against one `adult` sweep during implementation).
- Whether `_edm_update_guided` should also support the CFG (`y_only_model`) path or
  assert it off for v1 (CFG + guidance interaction is untested).
- Test harness location and the tiny-checkpoint fixture (reuse any existing TabDiff
  test fixtures if present).
