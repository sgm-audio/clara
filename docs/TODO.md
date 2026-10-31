# CLARA — open work

Current state is in the [README](../README.md). This list is the
remaining work, highest priority first.

## P1 — blocking for real time

- [ ] **Shrink the model to fit a CPU budget.** Measured 2026-10-31: 0.19–0.21
      ms/sample (Release, zig clang 22.1.8, shared 8-vCPU Xeon VM), about 1,000%
      of one core per 48 kHz mono stream. Real time needs stereo under 100%, so
      roughly 20x. `in_proj` (1612×128) is 61% of the MACs per sample.
      Levers: `d_model` 128 → 32–48, `in_proj` width, MIMO rank 4 → 1–2, 6 → 4
      layers. The recurrence also repeats work across the 64 `d` rows
      (every row holds the same values); fixing that is a model decision.
- [ ] **Fix the float32 drift.** 2.0% of peak divergence vs a float64 oracle within
      12 ms, 7.3% within 93 ms; peak `|ssm_real|` saturates the 4096-sample clamp.
      Options: double accumulation in the recurrence, or fix the dynamics so the
      clamp never fires. Add a drift assertion once it is fixed.
- [ ] **Trained model.** `models/model.bin` is an untrained fixture. Retrain and
      regenerate the reference fixtures (`python/tools/dump_engine_reference.py`).

## P1.5 — packaging and CI

- [ ] **CI has not completed a run since 2026-07-10.** Actions is blocked by an
      account billing lock. CI claims in `CHANGELOG.md` are verified locally only.
- [x] **`-ffast-math` removed** (2026-10-31). Release now uses
      `-fno-math-errno -fno-trapping-math`. Keep using the bit-level
      `mambaIsFinite` for guards anyway.
- [ ] **`test_perf_budget` needs a new baseline.** Without `-ffast-math` the
      calibration loop is about 2x slower and the engine about 4.5x faster, so
      the ratio went from ~3.4 to ~0.38 against a budget of 3.0. The guard is
      now loose; pick a budget from a few machines, or make it a median-of-N.
- [ ] **clap-validator full run:** 5 failures (state-reproducibility ×3,
      param-fuzz-basic/bounds time out at 45 s). The ctest subset excludes them.
- [ ] **Native GUI** (`CLAP_EXT_GUI`, X11/Win32 framebuffer, plus webview) exists
      but hasn't been opened in a host.

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
