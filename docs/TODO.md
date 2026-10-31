# CLARA — open work

Current state is in the [README](../README.md) status table. This list is the
remaining work, highest priority first.

## P1 — blocking for real time

- [ ] **Shrink the model to fit a CPU budget.** Measured cost is ~1,250% of one
      core per 48 kHz mono stream (0.262 ms/sample); real time needs < 100% for
      stereo, so ~12–24x. `in_proj` (1612×128) is 61% of the MACs per sample.
      Levers: `d_model` 128 → 32–48, `in_proj` width, MIMO rank 4 → 1–2, 6 → 4
      layers. Kernel tuning is exhausted (1.45x total, and the GEMM is at the
      CPU's issue limit) — this is a modelling decision, not a code one.
- [ ] **Fix the float32 drift.** 2.0% of peak divergence vs a float64 oracle within
      12 ms, 7.3% within 93 ms; peak `|ssm_real|` saturates the 4096-sample clamp.
      Options: double accumulation in the recurrence, or fix the dynamics so the
      clamp never fires. Add a drift assertion once it is fixed.
- [ ] **Trained model.** `models/model.bin` is an untrained fixture. Retrain and
      regenerate the reference fixtures (`python/tools/dump_engine_reference.py`).

## P1.5 — packaging and CI

- [ ] **CI is not running** (account-level Actions condition; last executed run
      2026-07-10). CI claims in `CHANGELOG.md` are verified locally only.
- [ ] **`-ffast-math` and NaN/Inf checks.** Release builds use `-ffast-math`, so
      `std::isfinite`/`std::isnan` fold to constants; non-finite guards must use
      the bit-level `mambaIsFinite` (`src/core/Platform.hpp`).
- [ ] **Native GUI** (`CLAP_EXT_GUI`) — deferred; the plugin is headless today.

## P2 — product evidence

- [ ] One flagship effect, trained, with a same-loudness A/B in a real DAW
- [ ] DAW load test (never performed on this codebase)
- [ ] CPU% and latency numbers printed by the plugin itself
- [ ] Presets, installer, manual

## P3 — after P1

- [ ] **H3:** rewrite the C++ engine for the R=8 PerTrackProcessor
      (`SsmInferenceCore` implements R=4 today). Reconcile with P1 first: at 4.26M
      params it is further from real time than the current 1.66M model.
- [ ] Vectorize the SSM recurrence by reordering state to `[h][s][d]` (~34% of
      per-sample cost)
- [ ] Per-effect variants, CLAP text/audio encoders
- [ ] macOS/Windows verification
