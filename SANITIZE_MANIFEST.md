# Sanitize manifest: clara (2026-10-31)

Branch `chore/sanitize`, cut from `main` @ `539be30`. 5 local commits, including this file. Nothing pushed.

## Removed
| Path | Files | Why |
|---|---|---|
| `archive/` | 44 | Old ONNX-era code and models, including `test_export.bin`. Superseded by the C++ engine. (From PR #3.) |
| `AGENTS.md`, `CLAUDE.md`, `.agent_context/`, `SESSION_LOG.md`, `REVIEW_REPORT.md`, `omo-config.json` | 6 | Agent notes and session logs. (From PR #3.) |
| `docs/` stale files | 9 | Superseded plans and reports. (From PR #3.) |
| `models/model.bin` | 1 | 6.6 MB of untrained, Xavier-random weights. CMake regenerates it with `export_mamba_weights.py --seed 42`. Two runs (torch venv, and system python3 with numpy) give the tracked sha256 `bc10f763…8639e`. Now in `.gitignore`. |

## Moved
- `docs/TRAINING.md` → `docs/training/TRAINING.md`
- `docs/TRAINING_QUICKSTART.md` → `docs/training/QUICKSTART.md`
- `docs/TRAIN_GUIDE.md` → `docs/training/PER_TRACK_PROCESSOR.md`
- `docs/MAMBA3_MIGRATION_PLAN.md`, `docs/PHASE2_ROADMAP.md` → `docs/design/`
- References fixed in the ADRs, `reference_forward.py`, `MambaConfig.h` and TRAINING.md. Link check: 0 broken.

## Added
`NOTICE` (VST3 SDK v3.8.0_build_66, MIT; clap-wrapper v0.14.0), and `.gitignore` entries for model.bin, `*.clap`, `*.vst3` and agent files. LICENSE is Apache-2.0 and unchanged. Scott's MIT rule is only for repos with no licence.

## Code fixes (tested)
- **Release flags:** `-ffast-math` → `-fno-math-errno -fno-trapping-math`. With clang 22, `-ffinite-math-only` folded `mambaIsFinite` and reassociated the bypass mix, and `test_clap_plugin` failed at :544 and :598.
  - Same box, alternating A/B, 3 runs each, 6 layers, 512-sample mono blocks: the engine went from 874–930 µs/sample to 195–211 µs/sample.
  - Why fast-math was 4.5× slower isn't known.
- **VST3 SDK and clap-wrapper:** VST3 SDK v3.7.6_build_18 (GPLv3/proprietary) → v3.8.0_build_66 (MIT). clap-wrapper v0.9.1 → v0.14.0.
  - The `MambaSuite_as_vst3` target is created explicitly, because clap-wrapper ≥ 0.10 only makes it when it's the top-level project.
  - Clang `-Wpragma-pack` in the SDK headers is demoted from `-Werror`.
  - VST3 probe: `ModuleEntry` is true and `GetPluginFactory` returns a factory.
- **Build wiring:**
  - X11 is linked through `X11::X11`.
  - model.bin generation moved to the top-level CMakeLists, so a non-test build also gets weights.
  - New `build_vst3_for_install` fixture, so `test_install_layout` can't run before the bundle exists.
- **`tests/clap_validator.cmake`:** `--invert-filter --test-filter` → `--exclude` (clap-validator 0.4.1).
- **ci.yml:**
  - numpy in the test jobs, no torch on Windows.
  - The validator step uses `validate`; the old `--no-gui` flag doesn't exist.
- **`process()` never read `in_events`,** so automation didn't reach the plugin while it was active (clap-validator param-set-events failed).
  - Events are now applied at block start.
  - New test `test_process_applies_param_events`: it fails with the fix reverted (checked) and passes with it.
- **Render extension:** hard real-time requirement is now false, and OFFLINE mode is accepted (new `test_render_modes`). The descriptor no longer says real time or ARA2.

## Doc claims changed (README fully rewritten)
- Removed the CI badge (there's no completed run; billing is blocked).
- "Mamba-3 MIMO" → Mamba-3-inspired.
- "Real time" → about 10× slower than real time: 195–211 µs/sample against the 20.8 µs budget at 48 kHz.
- ARA is described as not reachable from a host.
- Weights are described as generated and untrained.
- Added the clap-validator results, the before/after test table and a dependency table.
- CHANGELOG and docs/TODO.md updated. TODO gains:
  - the perf-budget re-baseline;
  - the validator failures;
  - "GUI exists, not host-tested".

## Tests, before and after (Shared 8-vCPU Intel Xeon VM (1 thread per core), Linux, zig clang 22.1.8, Release `-O3 -fno-math-errno -fno-trapping-math`)
| Check | Before (`main` @ 539be30) | After |
|---|---|---|
| ctest | 2/6 pass. Failed: test_perf_budget (ratio 3.35–3.47 > 3.0), test_clap_plugin (fast-math broke the finite check and the bypass mix), clap_validator (invalid flags), test_install_layout (no VST3 bundle built) | 7/7 pass (fresh build directory; the clap_validator ctest is quick mode and excludes `fuzz|process|param-set-wrong|state-reproducibility|state-buffered`; `-DCLAP_VALIDATOR_FULL=ON` runs everything) |
| clap-validator 0.4.1, full run | not runnable (bad flags) | 44 tests: 27 pass, 5 fail (state-reproducibility basic/binary/buffered; param-fuzz basic/bounds time out at 45 s), 1 warning, 11 skipped |
| `test_perf_budget`, engine µs/sample, 6 layers, d_model 128, mono, 512-sample blocks | 874–930 | 195–211 (ratio 0.37–0.40 against the 3.0 budget; the budget is now too loose and needs a re-baseline) |
| ruff, gitleaks dir, local-path grep | clean, clean, 1 hit | clean, clean, 0 |
Logs: `_logs/clara-before-ctest.log`, `clara-after-fresh.log`, `clara-after-ctest.log`.

## Needs Scott
- **Perf-budget re-baseline** (C15).
- **The model decision (C7):** a smaller model is required. See `/workspace/handoff/clara-runpod-training.md`.
- **The 5 clap-validator failures.**
- **ARA host reachability** (C11).
- **Descriptor name and vendor:** "Mamba-3 SSM Processor" / "Elite DSP" were left unchanged, and the homepage URL hasn't been checked.
- **CI billing.**
- **History rewrite approval** (below).

## History rewrite (prepared, not run on the real repo)

Nothing has been rewritten or pushed. The commands below were tried on a throwaway `--mirror` clone in `/tmp` on 2026-10-31. They exited 0, and afterwards `git log --all -p` and the commit messages had 0 matches for the local-path patterns. Pack size went from 508.33 MiB to 11.87 MiB. That scratch clone was deleted.

The secret scans (gitleaks `git` and `dir`, and trufflehog, over all history) found no secrets. So this rewrite is about local paths and large or unneeded files, not live keys. Scott says any exposed keys were rotated anyway.

`paths-clara.txt` (paths removed from every commit):
```text
checkpoints_clara/
archive/
models/model.bin
models/model_ref.bin
models/model_trained.bin
models/test_export.bin
models/mamba_analysis.onnx
models/mamba_analysis.onnx.data
models/mamba_rt.onnx
models/mamba_rt.onnx.data
models/mamba_rt_1x1.onnx
models/mamba_rt_1x1_complete.onnx
models/mamba_rt_1x1_int8.onnx
models/mamba_rt_9input_backup.onnx
models/mamba_rt_batched_npu.onnx
models/mamba_rt_batched_npu.onnx.data
models/mamba_rt_npu.onnx
AGENTS.md
CLAUDE.md
.agent_context/
SESSION_LOG.md
REVIEW_REPORT.md
omo-config.json
```

`replace.txt` is kept outside the repo, so the paths it removes aren't republished here. Use `/workspace/handoff/_history/replace-paths.txt`. It has 8 literal lines covering the Linux, Fedora Atomic (`/var/home`), macOS and Windows home-directory prefixes and the `x:` workspace URI. The `/var/home` line comes before the plain `/home` line.

Commands:
```bash
# 1. Merge chore/sanitize into the default branch first, so the rewrite covers the cleaned tree.
git clone --mirror https://github.com/sgm-audio/clara.git clara-rewrite.git
cd clara-rewrite.git
git filter-repo --invert-paths --paths-from-file ../paths-clara.txt --replace-text ../replace.txt
# 2. Check before pushing
git log --all -p | grep -cEf ../path-patterns.txt   # expect 0 (patterns file is next to replace.txt)
gitleaks git . --no-banner --redact                                                       # expect no leaks
git count-objects -vH
# 3. Push. Needs Scott's OK: this rewrites every commit hash.
git push --force --all origin && git push --force --tags origin
```

Caveats:
- GitHub keeps `refs/pull/*` for every PR, and a force-push can't touch them. The old objects stay reachable through those PR refs and cached commit URLs until GitHub Support runs a GC; ask them to purge after the push.
- Forks and existing clones keep the old history. Collaborators need to re-clone, and open PRs have to be recreated.
- `models/input_ref.bin`, `output_ref.bin` and `ref_meta.txt` are test fixtures and stay.
