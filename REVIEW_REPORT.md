# CLARA — Mamba Audio Suite: end-to-end repository review

**Repository:** `sgm-audio/clara` (private) · **Reviewed revision:** `d6c64fe` + branch `arena/01a105e2-clara` (10 commits, see *Changes Applied* and *Post-review Follow-up*)
**Review date:** 2026-10-04 · **Scope:** repository hygiene, documentation accuracy, project scaffolding, code quality, bugs & correctness
**Method:** read the tree; ran the build, the full test suite and the project's pinned linters before and after every change; reproduced every behavior claim with a minimal harness before writing it down.

---

## Summary

CLARA is a C++17 audio-plugin repository (CLAP plugin + ARA2 library + clap-wrapper VST3 target) around a custom Mamba-3 SSM
inference engine, plus a Python side for training and weight export. The engine is real, tested, and honest about its limits:
the build is warning-free, all four CTest targets pass, the Python↔C++ agreement test is meaningful, and the maintainers have
already documented that the model is ~12× too slow for real time and untrained.

The review found **no secrets, no committed build artifacts, no history-rewrite payloads**, and no critical-severity defect in
the shipping code. What it did find is a repository whose *outside* is less healthy than its *inside*:

1. **CI has not run since 2026-10-04.** Every workflow run (pushes, `pull_request`, scheduled CodeQL) ends in
   `startup_failure` with zero jobs. The workflow file parses, is registered `active`, and is byte-identical to `main`; this is
   account/repo-level (Actions billing/spending limit or Actions disabled) and needs a repository admin, not a code change.
2. **The VST3 path is broken as built.** clap-wrapper looks for `MambaSuite.clap` while the build produces `MambaSuiteClap.clap`,
   so `GetPluginFactory()` returns `nullptr`. Reproduced with a minimal `dlopen` host.
3. **Three real correctness bugs** in the plugin surface, all fixed here with tests: unclamped/NaN parameter values reaching the
   model from host events, the webview JSON, and project state; a NaN/Inf guard that `-ffast-math` silently compiled away; and an
   ARA model path that resolved against the host's working directory instead of the ARA library.
4. **Documentation drift** away from the code in the README, AGENTS.md and the training docs — including a broken
   one-command setup that referenced a script that does not exist.

Ten commits on this branch fix what is safe to fix (docs, formatters, `.gitignore`, `.editorconfig`, scripts, three localized
bugs with tests, lint-scope bugs), and record what must be decided by a human (CI root cause, `-ffast-math` policy, repository
layout, the canonical project URL). Build and tests were green at every commit; no behavior was changed silently.
The VST3 name mismatch and the broken install components — originally escalated rather than applied — were subsequently fixed at
the maintainer's request and are documented in *Post-review Follow-up*, including the one user-visible rename
(`MambaSuiteClap.clap` → `MambaSuite.clap`) and the new install-layout regression test.

---

## Critical Findings

**None.** No credential or secret is present in the tree or in the 45-commit history (scanned tracked files by pattern for
keys/tokens; no hits). No auth, payment, or data-migration surface exists in this repository. The two findings a reader might
expect at this severity — the CI outage and the VST3 wiring — are availability/packaging defects, not security incidents, and
are listed as High below.

---

## Changes Applied

Seven review commits on `arena/01a105e2-clara` (the report and its corrections follow them; the three follow-up commits are listed in *Post-review Follow-up*). Each was verified with a build
(`cmake --build`, no warnings) and the full suite (`ctest`, 4/4 PASS), and the branch tip was re-verified after the report's
own corrections: build 0 warnings/errors, `ctest` 4/4 PASS.

| Commit | Theme | Content |
|---|---|---|
| `8f00cfd` | style | clang-format 19.1.7 on 3 C++ files; black 25.1.0 + ruff 0.9.9 on 3 Python files. Cosmetic; build/tests re-verified. |
| `ecbe032` | tests | `SKIP_RETURN_CODE 77` + `TIMEOUT 300` for `test_mamba3`; removed an unused-variable warning in `test_perf_budget.cpp`. |
| `b7572f8` | docs | README: CI status, C++17 (not C++20), tests-opt-in, install-component breakage, VST3 name mismatch, `models/`-next-to-`.clap`, stale `CMAKE_POLICY_VERSION_MINIMUM` note. AGENTS.md platform/CI/`MAX_CHUNK_SIZE` corrections. Historical banners on `docs/BLUEPRINT_COMPARISON.md` and `docs/CHANGELOG.md`; ADR-001 link fix. `.gitignore`: `build-*/`, `venv_rocm/`, `get-pip.py`. |
| `201fe94` | chore | `ruff.toml` `force-exclude` (pre-commit invocation bypassed `archive/` exclusion); notebook import fixes; `scripts/{bootstrap_local,launch_train,run_test,run_train}.sh` path fixes; `docs/TRAINING_QUICKSTART.md` rewritten against the real flags; new `docs/TODO.md` §P1.5; `.editorconfig`. |
| `4d1f2a3` | fix | The three correctness fixes below, with tests. |
| `8a57d95` | docs | `validate_export.py` docstring said "Version 3"; the check is version 2. |
| `3ccaa74` | docs | Fixed the last four broken relative links in `archive/mamba1-onnx-era/README.md` (now zero broken among all 55 tracked markdown files). |

**Bug fixes in `4d1f2a3` (each with a test that fails before / passes after):**

1. **Parameter values were not clamped** (`src/clap/MambaClapPlugin.cpp`, `src/gui/MambaWebviewGUI.hpp`). All three params are
   declared over `[0, 1]` and the mix/conditioning math assumes it, but host param events, the webview's JSON (the page can post
   anything) and a saved project's state wrote straight into the atomics. One out-of-range or NaN value then reached every
   sample of the dry path. Values are now clamped at all three entry points; the duplicated state/state-context loaders were
   merged into `mambaReadPresetState()`.
2. **`-ffast-math` silently defeated `std::isfinite`** (`src/core/Platform.hpp` and call sites). Release builds compile with
   `-ffast-math` (`-ffinite-math-only`), so `std::isfinite`/`std::isnan` may be folded to constants — the NaN/Inf guard added
   in fix 1 was a no-op in Release, which the new test caught by failing. Added `mambaIsFinite()` (IEEE-754 exponent-bit
   inspection) and used it in the guards *and* in the tests' finite-output assertions, which were themselves vacuous in Release.
3. **ARA engine loaded weights from the wrong directory** (`src/ara/MambaAraController.cpp`). The playback renderer passed
   `nullptr` to `mambaResolveModulePath()`, which skips platform path resolution and returns `./models/model.bin` — resolved
   against the *host's* working directory. Now resolves relative to the ARA library, as the CLAP plugin does. The helper is
   exposed as `mambaAraModelPath()` so `tests/test_ara_controller.cpp` pins the behavior (deliberately without asserting
   absoluteness, since `dladdr` returns the module path exactly as the process was invoked).

**New test coverage:** `test_clap_plugin.cpp` had **no audio-path test at all**; it now creates → initializes → activates →
processes 512 frames through the real `models/model.bin`, checks finite/non-trivial output, bypass passthrough, and the
no-engine passthrough path, plus a webview-message test covering valid, out-of-range, malformed and non-finite values.

**Verification log (final revision):** configure + build exit 0 with zero `warning:`/`error:` lines; `ctest` **4/4 PASS**
(`test_mamba3` 2.7 s, `test_perf_budget` 4.4 s, `test_clap_plugin` 0.5 s, `test_ara_controller` 0.0 s); project-wide
clang-format diff loop clean; `ruff check --force-exclude .` clean; `black --check` clean (26 files);
`./build/tests/test_mamba3 /tmp/nope.bin` → SKIP message, exit 77.

---

## Findings Requiring Human Decision

Ordered by impact. Each is recorded in `docs/TODO.md` §P1.5 where it is actionable by the maintainers.

1. **CI is dead — needs a repository admin, not a patch (High).** Every run created since 2026-10-04 (three pushes, both
   `pull_request` runs, scheduled CodeQL) concluded `startup_failure` with 0 jobs and `path=BuildFailed`; the last run that
   executed was 2026-07-10 (`workflow_dispatch`). Still true at the time of writing: the pushes to `arena/01a105e2-clara` and the
   `pull_request` run opened by PR #2 (2026-10-04 18:43–18:45Z) also concluded `startup_failure` with 0 jobs, while all four
   workflows report `active`. `.github/workflows/ci.yml` parses as YAML (PyYAML), yamllint-clean except the
   document-start rule, has no tabs/dup keys/CRLF, is registered `active`, and is byte-identical to `main`. All workflows failing
   at once with zero jobs points to Actions spending limit/billing or Actions disabled for the repo. **Check** Settings →
   Billing → Actions and Settings → Actions → General. Until this is fixed, every CI claim in `CHANGELOG.md` (v0.1.1/v0.1.2:
   validator pin, model checks, X11-free build) is unverified.
2. ~~**VST3 cannot load the plugin as built (High).**~~ **RESOLVED 2026-10-04 after maintainer sign-off** — the CLAP output is
   renamed `MambaSuite.clap` and one cache variable (`CLAP_WRAPPER_OUTPUT_NAME`) now names both artefacts, so the stems cannot
   drift. Re-verified with the `dlopen` probe (`GetPluginFactory()` non-null from `~/.clap/MambaSuite.clap`). See
   *Post-review follow-up*. **The plugin file name is now a user-visible change: `MambaSuiteClap.clap` → `MambaSuite.clap`.**
3. ~~**`cmake --install` components are broken (Medium).**~~ **RESOLVED 2026-10-04 after maintainer sign-off** — all three
   components exit 0 and honour `--prefix`; on Linux the CLAP is a bare file (the bundle/bare-name collision was the cause of
   the exit 1), and `--component vst3` now installs the module itself to `lib/vst3/MambaSuite.vst3/Contents/<arch>/`. Guarded by
   the new `test_install_layout` CTest (findings 3.12, 5.9). A latent `COMPONENT` scoping bug in the ARA rule was found and fixed
   in the same pass.
4. **`-ffast-math` policy (Medium).** The flag is a deliberate performance choice, but it makes all `std::isfinite`/`std::isnan`
   folds unsafe. The code now uses `mambaIsFinite()`; the team should decide whether to keep `-ffast-math` everywhere or restrict
   it to the engine target (`-ffast-math` on a plugin that inspects host input for validity is a footgun).
5. **Repository layout / personal files (Low).** `SESSION_LOG.md` (104 KB), `omo-config.json` (unreferenced), and
   `.agent_context/nrinstruct.md` — which is *tracked* even though `.agent_context/` is in `.gitignore` — are stale personal or
   agent-tooling artifacts at the repository root. Same class: `archive/` still holds the ONNX era. Untracking/moving is a
   maintainer call.
6. **`.git` carries 574 MB of objects (Low–Medium).** 716 blobs, 263 over 1 MB (567.5 MB), of which 254 are
   `checkpoints_clara/checkpoint_*.pt` (≈2 MB each) and 12 are `.onnx` (largest 12 MB). Removing them means a history rewrite,
   which is explicitly out of scope for this review; note it for when the repo is next pruned or mirrored.

---

## Label Changes

`docs/TODO.md` is the project's own status surface and the task brief permits label corrections. Three labels were corrected:

| Item | Was | Now | Reason |
|---|---|---|---|
| `docs/TODO.md` §P1.5 (new section) | — | `[ ]` open items with explicit cause + who can fix | Records the CI outage, VST3 naming, install components, `-ffast-math`, stale root files. |
| `docs/BLUEPRINT_COMPARISON.md` | unlabelled (read as current findings) | `<!-- STATUS: historical -->` + banner | Sections 1–7 describe the pre-fix tree (LFS-pointer weights, failing tests, old 0.379 ms/sample engine); only the addendum is current. |
| `docs/CHANGELOG.md` | unlabelled (version numbers read as releases) | `<!-- STATUS: historical -->` + banner | Pre-0.1.0 internal numbering that overstated maturity; `/CHANGELOG.md` is the live changelog. |

Additionally, `README.md` gained a `<!-- STATUS: in progress -->` marker on the install section (rules verified broken, decision
pending) and a CI section explaining the unverified state; those are status labels, not code changes.

No other label was found wrong: the README status table's claims (`21 checks`, `1,250% of one core`, untrained fixture) were each
re-verified against the tree and kept.

No GitHub label needed correction: the repository has zero issues, and the only merged PR (#1) carries no labels at all, so
there was no mislabelling to fix. All four workflows (`CI`, `CodeQL`, `Dependency Graph`, `Security Risk Assessment`) are
registered `active` — which is exactly why their uniform `startup_failure` is an account-level condition rather than a
configuration one (see Finding 1). Confirmation at the time of writing: the pushes for this branch and the `pull_request`
event opened by PR #2 also produced `startup_failure` runs with 0 jobs.

---

## Detailed Findings by Dimension

### 1. Repository hygiene

**Good:** no secrets in tracked files or history; no build artifacts committed (`build/`, `*.o`, `*.so`, `*.clap`, `*.vst3` all
absent from `git ls-files`); 141 tracked files, ~15 MB working tree, dominated by the intentional 6.6 MB `model.bin`; 45-commit
history with meaningful messages; `models/*.bin` protected from line-ending conversion via `.gitattributes`.

| # | Finding | Severity | Status |
|---|---|---|---|
| 1.1 | `.agent_context/nrinstruct.md` is tracked although `.agent_context/` is ignored — 2026-06-04 session notes describing the pre-Mamba-3 architecture | Low | proposed (untrack) |
| 1.2 | `omo-config.json` is an unreferenced third-party agent-tool config at the root | Low | proposed (remove) |
| 1.3 | `SESSION_LOG.md` (104 KB) duplicates the changelog and TODO history at the root | Low | proposed |
| 1.4 | `.git` holds 574 MB of objects (254 `checkpoints_clara/checkpoint_*.pt` ≈ 2 MB each, 12 `.onnx`, largest 12 MB) — clone cost only, no working-tree impact | Low–Medium | research (history rewrite required) |
| 1.5 | `d-drive-wip` is an orphan branch (no common ancestor with `main`) — dead weight unless intentional | Low | research |
| 1.6 | `.gitignore` missed `build-*/` even though `CMakePresets.json` creates `build-macos*/`, `build-windows/`; and `venv_rocm/`, `get-pip.py` created by `scripts/setup_local.sh` | Low | **fixed** (`b7572f8`) |
| 1.7 | The branch `arena/01a1052d-clara` is already merged; leave for the owner to delete | Info | research |
| 1.8 | No topics, no milestones, zero issues, one merged PR — thin GitHub metadata | Low | proposed |

### 2. Documentation accuracy / staleness

**Good:** `CHANGELOG.md` is unusually honest about the untrained weights and the performance shortfall; `models/README.md`
documents the lost LFS blobs; `docs/TODO.md` carries measured numbers rather than estimates.

| # | Finding | Severity | Status |
|---|---|---|---|
| 2.1 | README claimed CI works; every run since 2026-10-04 is `startup_failure` with 0 jobs | High | **fixed** + escalated |
| 2.2 | README's "Running" section claimed the VST3 loads from the bundle; the wrapper cannot find `MambaSuiteClap.clap` | High | **fixed** (renamed in the follow-up — see below) |
| 2.3 | README claimed tests build with the plain build; they are opt-in (`-DBUILD_TESTING=ON`) | Medium | **fixed** |
| 2.4 | README's install instructions did not match reality (`--component clap` exits 1, `--component vst3` ignores `--prefix`) | Medium | **fixed** (documented) |
| 2.5 | README said "C++20 toolchain"; `CMAKE_CXX_STANDARD` is 17 | Low | **fixed** |
| 2.6 | README/CI note about `CMAKE_POLICY_VERSION_MINIMUM=3.5` is stale: a fresh configure with all four pinned dependencies succeeds without it (verified on CMake 4.4.3; the 3.4/3.4.3 `cmake_minimum_required` files in vst3sdk are never processed because clap-wrapper v0.9.1 globs SDK sources instead of `add_subdirectory`) | Low | **fixed** |
| 2.7 | `AGENTS.md` said CI builds "three platforms"; it builds two (no macOS job) | Medium | **fixed** |
| 2.8 | `AGENTS.md`/`MambaConfig.h` claimed `MAX_CHUNK_SIZE` is model semantics that "must match training"; nothing but a `static_assert` consumes it, and no training/export code reads it | Medium | **fixed** (marked unverified) |
| 2.9 | `docs/TRAINING_QUICKSTART.md`'s one-command setup ran `scripts/setup_cloud.sh`, which does not exist; the flag table listed wrong defaults (`--grad-accum` 4 vs actual 1, `--clip-seconds` 0.5 vs actual 4.0, a `--cache` flag that is not there) | Medium | **fixed** |
| 2.10 | `docs/BLUEPRINT_COMPARISON.md` read as current though it describes the pre-fix tree | Medium | **fixed** (historical banner) |
| 2.11 | `docs/CHANGELOG.md` 1.x numbering precedes the first release and overstated maturity | Low | **fixed** (historical banner) |
| 2.12 | `docs/adr/ADR-001-…md` had a broken relative link to `MAMBA3_MIGRATION_PLAN.md` | Low | **fixed** |
| 2.13 | `python/tools/validate_export.py` docstring said "Version 3"; `EXPECTED_VERSION = 2` | Low | **fixed** |
| 2.14 | `tests/test_mamba3.cpp` comment said the fixture agreement is `~1e-6`; the test prints `1.5e-3` | Low | **fixed** |
| 2.15 | Project URL is inconsistent across the tree (`scottmill306/CLARA` in `Platform.hpp:6` and the incubator checklist vs `SGM-Studios/clara` badge vs `sgm-studios/clara` GitHub homepage vs actual `sgm-audio/clara` origin). The badge, the incubator checklist and the `TRAINING.md` `curl \| bash` URL were corrected to `sgm-audio/clara` in the follow-up; `MAMBA_PROJECT_URL` in `src/core/Platform.hpp:8` still says `scottmill306/CLARA` and **is proposed, not changed** — it is emitted into the plugin descriptor (`mamba_plugin_info.url`), i.e. user-visible metadata, and the maintainer should pick the canonical URL before the repo goes public | Medium | in progress |
| 2.16 | Broken relative links: five in total — four in `archive/mamba1-onnx-era/README.md` (`MAMBA3_MIGRATION_PLAN.md`, `MI300X_TRAINING_PLAN.md`, `TRAINING_QUICKSTART.md`, a `train_clara_mamba3.py` that no longer exists) and one in ADR-001. All five now resolve. **Correction to an earlier intermediate count of nine:** the four `](x)` "links" in `docs/CLARA_OVERHAUL_PLAN.md` were a regex artifact — they are `self.conv_stem[0](x)` expressions inside a Python code block, not markdown. A code-block-aware scan of all 55 tracked markdown files finds **0 broken relative links** after these fixes (26 relative links checked; the rest are absolute/anchor URLs or code samples) | Low | **fixed** |
| 2.17 | README's "21 checks" figure was verified correct; no change needed | Info | fixed (verified) |

### 3. Project scaffolding

**Good:** CMake ≥ 3.14 with FetchContent pins for CLAP, ARA SDK, clap-wrapper and VST3 SDK; options for no-VST3 and no-native-GUI
builds; presets for Linux/macOS/Windows; tests opt-in with a `clara_test` aggregate target and a perf-budget regression guard;
pre-commit pins clang-format, black and ruff; `.clang-format` and `ruff.toml` present.

| # | Finding | Severity | Status |
|---|---|---|---|
| 3.1 | CI broken at the account level (see Finding 1) — the only verification gate the project has | High | escalated |
| 3.2 | **No `.editorconfig`**; editors could reflow C++/Python inconsistently | Low | **fixed** |
| 3.3 | No `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CODEOWNERS`, issue/PR templates, or `CITATION.cff` | Low | proposed |
| 3.4 | No `docs/README.md` index — 15 docs files with no entry point beyond README links | Low | proposed |
| 3.5 | `ruff.toml`'s `exclude = ["archive/"]` did not apply to pre-commit (explicit file paths): 20 findings in archived code would fail the hook. Fixed with `force-exclude = true` | Medium | **fixed** |
| 3.6 | `python/clara_kaggle.ipynb` is invisible to the pre-commit Python hooks (`identify` files it as `text/json/jupyter`, not `python`), so it drifted: 4 ruff findings. Now clean under explicit invocation | Low | **fixed** |
| 3.7 | `scripts/{bootstrap_local,launch_train}.sh` invoked `${repo}/train_clara.py`, which does not exist (it is `python/training/train_clara.py`) — the documented local training path could not work | Medium | **fixed** |
| 3.8 | `scripts/{run_test,run_train}.sh` told users to `pip install -r requirements.txt`; no `requirements.txt` exists (setup is `scripts/setup_local.sh`) | Low | **fixed** |
| 3.9 | `tests/CMakeLists.txt` had no `TIMEOUT` on `test_mamba3`; the sibling perf test did | Low | **fixed** |
| 3.10 | `clap-validator` is advisory in CI (step tolerates failure) and unavailable here; noted, not a defect | Info | research |
| 3.11 | `CMakePresets.json` `windows-msvc-release` sets `CMAKE_CXX_STANDARD` without `CMAKE_BUILD_TYPE`/multi-config handling — untestable in this environment | Low | research |
| 3.12 | No regression guard for the install layout or the CLAP/VST3 name pairing, which is why the breakage survived | Medium | **fixed** — new `test_install_layout` CTest (installs every component into a scratch prefix, checks exit codes, `--prefix` honouring, the name pairing and `models/` adjacency; negative-controlled against a scratch build configured with the old name) |
| 3.13 | A full `cmake --install` also drops the fetched CLAP SDK's headers and CMake package files under `include/clap/` and `lib/cmake/clap/` (the SDK's own install rules, not CLARA's) | Low | research (documented in README) |

### 4. Code quality

**Good:** the engine path allocates nothing on the audio thread (scratch allocated in `init()`, verified by reading every
allocation site); the SPSC queue is used correctly and has two concurrency tests; `-Wall -Wextra` clean after this review;
Python has no bare `except:`, no mutable default args, and no `TODO`/`FIXME` markers; error handling in the plugin uses `fprintf`
to stderr rather than exceptions, which is the right choice behind the CLAP C ABI.

| # | Finding | Severity | Status |
|---|---|---|---|
| 4.1 | Formatter drift in 3 C++ + 3 Python files (pinned versions) | Low | **fixed** |
| 4.2 | Two `-Wall -Wextra` warnings in tests (unused `N`, unused `sink`) | Low | **fixed** |
| 4.3 | `std::isfinite`/`std::isnan` used in code that is compiled with `-ffast-math` — silently folded away in Release | Medium | **fixed** |
| 4.4 | `deferredTaskQueue` is popped in `mamba_on_main_thread` but never pushed in production code; `presets`/`activePreset`, ARA `contentDirty_`/`getEngine()`, ARA `getAraEngine()` are written or defined but never read | Low | research (dead-code candidates; removal is unverifiable without more archaeology) |
| 4.5 | `test_numeric_sensitivity` is a *section* inside `test_mamba3`, not a binary — the README refers to it as if it tracked drift separately | Low | research |
| 4.6 | `torch.load(..., weights_only=False)` at 6 sites (`train_clara.py:944,1264`; `train_clara_kaggle.py:758,1008`; `train_local_radeon.py:709`; `inspect_ckpt.py:4`) — all load self-produced checkpoints; low risk, but `weights_only=True` where possible is cheap hardening | Low | research |
| 4.7 | `strncpy` used for CLAP port names without guaranteeing NUL (fixed-width `char[256]` arrays, source is a literal and `memset` precedes it — currently safe, fragile if names become dynamic) | Low | research |
| 4.8 | `python/` has no `requirements.txt`/`pyproject.toml`; dependencies are installed by a shell script (`setup_local.sh`) | Low | research |
| 4.9 | Deprecated `python/tools/dump_reference_output.py` shim retained; harmless, correctly documented | Info | — |

### 5. Bugs & correctness

**Good:** the Python↔C++ agreement test is genuinely meaningful (`max abs diff 1.55e-03`, `7.3e-04` relative — within float32
noise of a 2.14 peak); the perf-budget test measures against a same-machine calibration loop rather than wall-clock guesses; the
weight loader rejects size mismatches; the plugin's state format is versioned and its round-trip is tested.

| # | Finding | Severity | Status |
|---|---|---|---|
| 5.1 | Parameter values not clamped to the declared `[0, 1]` range; NaN passes through into every sample (dry signal is added back unconditionally) | High | **fixed** + tests |
| 5.2 | NaN/Inf guard defeated by `-ffast-math` (see 4.3) | Medium | **fixed** + tests |
| 5.3 | ARA playback renderer resolved weights against the host's working directory, so the engine almost never initialized | Medium | **fixed** + test |
| 5.4 | **No test covered the plugin's audio path** (`init` → `activate` → `process`) | Medium | **fixed** + test |
| 5.5 | `test_mamba3` failed hard when the model was absent, so a fresh environment without Python/torch reported a red failure instead of SKIP | Low | **fixed** |
| 5.6 | LayerNorm uses gamma=1/beta=0 with no weights loaded; the training script documents the same choice (`train_local_radeon.py:382,409`), so this is intentional, but worth a code comment near `WeightLoader` since the export carries no LN arrays | Low | research (documented) |
| 5.7 | Float32 drift (2.0 % of peak in 12 ms, 7.3 % in 93 ms vs a float64 oracle) and the ~12× real-time shortfall are known, measured, and tracked in `docs/TODO.md` P1 — not new findings | High | in progress (tracked upstream) |
| 5.8 | The repo has never been loaded in a DAW; verified here only with a minimal CLAP host harness (init OK, finite output) | Medium | research |
| 5.9 | `COMPONENT` in `install(TARGETS …)` binds to the artefact type it follows, so the ARA rule's `COMPONENT ara` applied to `LIBRARY` only — on Windows the `.dll` would have installed under the default component | Low | **fixed** (both TARGETS rules repeat `COMPONENT` per artefact type) |

---

## Post-review Follow-up (2026-10-04, maintainer-requested)

After the review was delivered, the maintainer asked for two changes to be applied rather than proposed — the VST3 name
reconciliation (Finding 2.2 / decision 2) and the install-component repair (decision 3) — and to prepare for the repository going
public. Work done, all on `arena/01a105e2-clara`:

**1. The CLAP output is renamed `MambaSuiteClap.clap` → `MambaSuite.clap`.**

One cache variable, `CLAP_WRAPPER_OUTPUT_NAME` (default `MambaSuite`), now sets `OUTPUT_NAME` on the CLAP target *and* names the
VST3, so the two stems cannot drift apart again — the invariant clap-wrapper depends on
(`wrapasvst3_entry.cpp`: `plugname = os::getBinaryName() + ".clap"`). The CMake *target* is still `MambaSuiteClap`; only the file
name changed. `CLAP_WRAPPER_OUTPUT_NAME` was previously declared next to the VST3 section and is now declared above the CLAP
target, where it takes effect.

Verified three ways:

| Probe | Old name only | New name |
|---|---|---|
| `dlopen` + `GetPluginFactory()` (VST3 module, `~/.clap` populated) | `nullptr` | factory created — `Mamba-3 SSM Processor`, vendor `Elite DSP`, id `com.mamba.audio.suite` |
| CLAP host harness → `plugin->init()` with `models/` beside the `.clap` | — | `true`; `false` without `models/` (control), so weight resolution is still load-bearing |
| `test_install_layout` CTest | — | PASS, and FAILS when pointed at a scratch build configured with the old name (negative control) |

This is a **user-visible change**: hosts that loaded `build/plugins/MambaSuiteClap.clap` by path must use
`build/plugins/MambaSuite.clap`. It is stated in `CHANGELOG.md` under *Unreleased* and in the README.

**2. The three `cmake --install` components are repaired.**

| Component | Before | After |
|---|---|---|
| `clap` | exit 1 — bare `.clap` file installed into the same path as the `.clap/` bundle directory; self-nested tree, models install then died with `Permission denied` | exit 0 → `lib/clap/MambaSuite.clap` + `lib/clap/models/` on Linux; bundle layout on macOS/Windows |
| `vst3` | exit 0 but wrote only weights to an absolute path in the build tree, ignoring `--prefix`; the module was never installed | exit 0 → `lib/vst3/MambaSuite.vst3/Contents/<arch>/` |
| `ara` | exit 0 | exit 0 (unchanged behaviour; `COMPONENT` scoping bug fixed, see 5.9) |

The Linux CLAP is now a bare DSO, not a bundle: `clap/include/clap/entry.h` says host search paths are scanned for "files
and/or bundles as appropriate in your OS", a bare file and a same-named bundle directory cannot coexist, and that collision was the
root cause of the exit 1. macOS/Windows keep the bundle layout, which is their convention.

**3. New regression guard: `test_install_layout`** (`tests/install_check.cmake`, registered in `tests/CMakeLists.txt`). It installs
every component into a scratch prefix, then asserts: all installs exit 0; the install honoured `--prefix`; exactly one `.clap`
installed; its name equals `CLAP_WRAPPER_OUTPUT_NAME`; `models/model.bin` sits beside it; and, when a VST3 was built, the module
installed and its stem matches. It SKIPs (exit 77) when the tree is not built, and it is multi-config aware.

**4. Repository-going-public prerequisites touched.** README CI badge and the stale repository URLs now point at
`sgm-audio/clara` (the badge previously pointed at `SGM-Studios/clara`, which 404s). `docs/TRAINING.md`'s `curl | bash` one-liner
pointed at a *different account* and a **nonexistent script**; it now uses the archived launcher that actually exists.
`docs/INCUBATOR_CHECKLIST.md` also claimed "three platforms" for CI (it is two — no macOS job). Remaining, and deliberately not
changed: `MAMBA_PROJECT_URL` in `src/core/Platform.hpp:8` (see Finding 2.15) — it is emitted into the plugin descriptor shown by
hosts, so the maintainer should decide the canonical URL. Note also that when the repository is made public, the anonymous CI
badge will start resolving only if the badge URL matches the final repository path.

## Recommended Next Steps

1. **Repo admin: fix CI.** Check Settings → Billing → Actions and Settings → Actions → General; one run of the existing workflow
   on `main` will re-validate everything else in this report. Nothing else in the project is verifiable until that happens.
2. ~~**Decide the VST3 naming.**~~ **Done** — `MambaSuite.clap` is now the built name and the pairing is guarded by
   `test_install_layout`. What remains is the one thing no local harness can prove: a load inside a real DAW (P2 in
   `docs/TODO.md`).
3. **Merge this branch's fixes** (`4d1f2a3` in particular) once CI can run: the clamping, `mambaIsFinite` and ARA path fixes are
   small, tested, and independent of any product decision.
4. ~~**Repair the install components.**~~ **Done** — see *Post-review Follow-up*; all three components install prefix-relative and
   the new `test_install_layout` CTest would catch a regression.
5. **Then resume the P1 work already in `docs/TODO.md`**: shrink the model (~12–24× needed), fix the float32 drift, train and
   ship a real model. The training scaffolding this review repaired (`launch_train.sh`, `bootstrap_local.sh`,
   `TRAINING_QUICKSTART.md`) is what makes that reachable.
6. **Housekeeping pass** (cheap, low risk): untrack `.agent_context/nrinstruct.md`, delete `omo-config.json`, move
   `SESSION_LOG.md` under `docs/`, add `CONTRIBUTING.md`/`SECURITY.md`/`docs/README.md`, and fix the remaining broken relative
   links.

---

*Every claim above was checked against `d6c64fe` + the ten commits on `arena/01a105e2-clara`. Claims that could not be verified
in this environment are explicitly marked "research". Reproductions used minimal local harnesses (a `dlopen` VST3 factory probe
and a CLAP host that creates/activates/processes a plugin); they are throwaway and deliberately not committed.*
