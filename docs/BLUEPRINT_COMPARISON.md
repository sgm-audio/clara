# CLARA vs. the two blueprints — a verification-based comparison

**Repo:** `sgm-audio/clara` @ `7a175b8` (2026-07-09, `main`) · **Review date:** 2026-10-04
**Method:** not a doc read — I built the project, ran its tests, regenerated its model fixtures, benchmarked the inference core, and queried GitHub for the live CI/LFS state.

**Blueprint A** = the "cutting-edge blueprint" (Mamba-3 + LoRA + RunPod + JUCE + funnel + freelancing).
**Blueprint B** = the "reality check" (Mamba-3 as differentiator, ship audio software, evidence-first marketing, shrink until it runs).

---

## 1. TL;DR

1. **Both blueprints are directionally right about the architecture, and CLARA has already gone further than either assumes on the hard part.** The repo contains a hand-written C++ implementation of Mamba-3's three headline mechanisms — exponential/trapezoidal discretization, complex state via data-dependent RoPE, and MIMO rank-4 — with zero heap allocations on the audio thread (verified). Blueprint B's "verify that Mamba-3 is a real, reproducible paper before building a roadmap on it" test **passes**: it is an ICLR 2026 Oral (arXiv 2603.15569, Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu), and the repo mirrors its design.

2. **Neither blueprint describes CLARA's actual blocking problems, and both would be misled by its README.** Two measured facts dominate everything else:
   - **Nobody can get the model weights.** `models/*.bin` are Git LFS pointers; the LFS budget for the account is exhausted, so the LFS API returns `403 This repository exceeded its LFS budget`. A fresh clone contains 132-byte text files where the weights should be, and `mamba_init()` returns `false` on load failure → **the plugin will not load in a DAW at all**.
   - **The model is ~10–18× too heavy to run in real time on a CPU.** Measured: **0.38–0.42 ms per sample** with the repo's own Release flags (0.20–0.25 ms with `-march=native`). One mono 48 kHz stream needs **~1,800 % of one core** (≈ 955 % best case); stereo needs ~2× that. Real-time factor ≈ **0.05×**.

3. **Blueprint A's per-customer LoRA/JUCE/ONNX/RunPod specifics are either premature or already rejected for good reasons; Blueprint B's plan is the right one but underestimates the arithmetic.** B says "under 10 % of one core" — CLARA is at ~1,800 %. Closing that is a **model-size and execution-shape decision (~100× per-sample cost reduction)**, not a tuning exercise. The concrete levers are in §5.

---

## 2. What I actually verified

| Claim (README / AGENTS / SESSION_LOG) | Evidence | Verdict |
|---|---|---|
| "Builds on Linux: `cmake -B build && cmake --build build`" | Core (`libMamba3Core.a`) and ARA build. CLAP target fails on minimal Debian: `src/gui/MambaNativeGUI.hpp:30: fatal error: X11/Xlib.h: No such file`; CMake links `X11` (CMakeLists.txt:120) with no `find_package(X11)` | ⚠️ Needs undocumented X11 dev headers |
| "RT-safe (zero heap allocations on audio thread)" | LD_PRELOAD alloc counter: **0 allocations** across 64 `process_block()` calls (4,096 samples) | ✅ **True** |
| "Latency 0 samples — time-domain, no STFT" | `clap_latency_get()` returns 0; per-sample causal streaming in `SsmInferenceCore::process_block` | ✅ **True** |
| "Trained model — identity autoencoder; model.bin = briefly-trained" | `models/model.bin`, `model_trained.bin`, `*_ref.bin` are all **132-byte LFS pointers**; LFS API → `403 exceeded its LFS budget` | ❌ **Unshippable** |
| "4/4 CTest tests pass" | On the as-cloned tree, `test_mamba3` **aborts**: `terminate called after throwing an instance of 'std::bad_alloc'` (exit 134). `test_clap_plugin` can't build without X11. `clap_validator` is skipped (no binary; no cargo) | ❌ **Not reproducible** |
| `test_mamba3` = "Python↔C++ numerical equivalence, **bit-exact**" | The assertion is `max_diff < 0.05` — not bit-exact. It compares the **R=4 time-domain** engine against the **R=8 spectral (STFT, complex)** exporter; `dump_reference_output.py` says so itself: *"bit-exact Python↔C++ comparison is NOT possible. The C++ engine must first be updated to R=8."* It SKIPs when fixtures are absent and **crashes** when they are present as LFS pointers (reads "vers" as `uint32 N` ≈ 1.9e9 → 7.7 GB `std::vector`) | ❌ **Test cannot pass by construction** |
| Engine works on valid weights | Regenerated `export_mamba_weights.py --seed 42` (6,649,600 B, exactly the size the LFS pointer declares) → **7/7 engine tests pass**: init, silence, sine (stddev 0.85), conditioning changes output, determinism, state continuity, 1024-sample stability | ✅ Engine logic is sound |
| "GUI ❌ Deferred" (README) | `src/gui/MambaWebviewGUI.hpp` (262 lines) + `MambaNativeGUI.hpp` (491 lines, X11/Win32 blit) are wired into the plugin; INCUBATOR_CHECKLIST and SESSION_LOG 2026-07-07 both list GUI as done | ❌ **README is stale** |
| CI green (badge in README) | `gh run list`: last 3 CI runs = failure, cancelled, cancelled. The failing step is `wget .../clap-validator/releases/latest/download/clap-validator-linux-x86_64.tar.gz` — **no such asset exists** in 0.3.2 (`-ubuntu-18.04.tar.gz`) or 0.4.1 (`-ubuntu-22.04.zip`). The `build` job dies there, so `test` is skipped. `actions/checkout` is used **without `lfs: true`**, so CI never has real weights either | ❌ **Red on `main`** |
| Mamba-3 is a real, citable architecture | ICLR 2026 Oral; arXiv 2603.15569; trapezoidal + complex/RoPE + MIMO. Repo implements all three | ✅ **True** |
| LoRA adapters anywhere | `grep -ri lora` over the active tree: **0 hits** | ❌ Not implemented |
| "No ONNX Runtime / LibTorch / TensorRT" | Confirmed in CMakeLists and code; ONNX era archived in `archive/mamba1-onnx-era/` | ✅ True (deliberate) |

### Measured performance (this machine)

Intel Xeon @ 2.60 GHz, 2 vCPU container · model = regenerated seed-42 export (dimensions identical to the real one, so cost-relevant) · Release flags as in the repo (`-O3 -ffast-math`).

| Configuration | Per sample | 1 mono 48 kHz stream | Real-time factor |
|---|---|---|---|
| Repo Release flags, block 512 | **0.379 ms** | **1,820 %** of one core | 0.055× |
| Repo Release flags, stereo (2 engines) | 0.419 ms | 2,009 % | 0.050× |
| `-O3 -march=native -ffast-math`, mono | 0.199 ms | **955 %** | 0.105× |
| `-O3 -march=native -ffast-math`, stereo | 0.208 ms | 997 % | 0.100× |

Context: a naive 1612×128 matvec (the engine's dominant op, same loop style) reaches **11.3 GFLOP/s** here with generic flags and **35.5 GFLOP/s** with `-march=native`. So the implementation is already running at "naive matvec" efficiency — there is no cheap win left in the current structure.

Analytically, one sample costs ≈ **2.03 M MACs** (in_proj 206k + out_proj 33k + FiLM 33k + MIMO state/output 66k, × 6 layers). At 48 kHz that is ≈ **195 GFLOP/s for one mono stream**, ~390 GFLOP/s stereo. For a plugin that should live at 3–8 % of one core, the per-sample budget is ~15–35 GFLOP/s → **the model needs roughly a 100× per-sample cost reduction.**

---

## 3. Blueprint A, point by point

| Blueprint A | Repo reality | Who's right |
|---|---|---|
| Mamba-3 SSM is the right architecture for real-time audio; O(1) inference | Architecture choice: yes. But **O(1) per sample ≠ small per sample.** CLARA is exactly the proof: O(1) at 2 M MACs/sample is what breaks real-time | **A is right about the model, wrong to skip the cost.** This is the single most important correction to both blueprints |
| Train a base model on A100/H100 on RunPod | Repo trains locally (ROCm, 860M iGPU) and on Kaggle/MI300X; RunPod only survives in `archive/` | Neither matters — training works; GPU rental was never the blocker |
| Rank-4..16 LoRA per customer/preset, swap at runtime inside the plugin | **No LoRA code anywhere.** At 1.66 M params, runtime adapter swapping buys nothing over baking weights per preset (B's argument), and it cannot be shipped before the base model is shippable at all | **B** |
| Export via ONNX Runtime / LibTorch / TensorRT-LLM, inject in JUCE | The repo **tried ONNX and removed it** (`archive/mamba1-onnx-era`), and **deliberately rejected JUCE** (GPL vs proprietary ARA SDK; ~20 MB dependency; archived `ui-polish-design.md`). It ships CLAP + clap-wrapper VST3 + ARA with a hand-written engine and a software-rendered native GUI | **Repo's existing decisions beat A's suggestions** |
| "The moat is latency and sound quality vs Transformers/WaveNets" | True conceptually — but there is **no DAW demo, no audio sample, no measured CPU number** in the repo to make that claim with | **B** |
| Funnel: technical thread → OSS base plugin → newsletter → inbound DMs | The funnel is fine; it presupposes a stranger can run the plugin, which today they cannot (weights unavailable) | B's ordering (artifact first) |

**One-line verdict on A:** it is a plausible 2023-era playbook for *some* audio-AI business, but its technical specifics are ones this repo has already tested and outgrown, and its economics (LoRA per customer) ignore the real bottleneck (per-sample compute), not the training cost.

---

## 4. Blueprint B, point by point

| Blueprint B | Repo reality | Verdict |
|---|---|---|
| "Treat Mamba-3 as whatever paper and code you can actually cite and reproduce" | Paper is real (ICLR 2026 Oral); the repo reproduces the three mechanisms in C++ | ✅ B's test is passed here |
| "A custom state-space model with LoRA is a niche. It becomes income when a small causal model beats a conventional effect on a specific job" | No shipped effect yet; the roadmap's per-effect variants (Saturator/Compressor/Denoiser) are all "planned" | ✅ B describes the gap |
| "Shrink the model until a laptop can run it" | Measured 1,800 % of a core → needs ~100× reduction | ✅ **B's most important line, and it is currently unaddressed** |
| "Under 10 % of one core" target | Off by ~180× | B's *target* is right; B doesn't say how far away CLARA is |
| "Musicians buy knobs and presets, not rank values" | 3 params, no presets, no installers, no manual, no demo limitation (gaps G13, Phase 3) | ✅ B |
| "Post the sound, not the architecture" | No audio, no video, no demo link anywhere in the repo | ✅ B |
| Two separate codebases (training / plugin) | Actually true here: `python/training` vs `src/core`, joined only by a flat-binary weight format | ✅ Already followed |
| "Force it to be concrete: no host allocations in `processBlock`" | Verified: 0 allocations, deferred reset, SPSC queue, no locks | ✅ Already followed |
| 90-day sequence (conventional effect → tiny model → one adapter → demo) | CLARA is inverted: it did the hardest research step first and skipped the boring finished effect, the demo, and the price | ✅ B's sequence would have saved a lot of time |

**One-line verdict on B:** the strategy is correct for this repo, but it needs one addition — **the arithmetic.** Before "one flagship effect", CLARA must pass a per-sample compute budget, and the only way there is a smaller/blockier model. Everything else in B (evidence, presets, honest latency, price) is downstream and cheap.

---

## 5. What actually blocks CLARA, and in what order

### P0 — make the repo work for a stranger (days)

1. **Fix weight distribution.** Do not ship 6.6 MB of weights through LFS on a private repo with a blown budget. Either commit the `.bin` directly (fine at this size), or publish it as a Release asset plus `scripts/fetch_model.py`, and add a checksum. Then delete the LFS filter for these files.
2. **Fail loudly and safely on a bad model file.** `test_mamba3` aborts with `std::bad_alloc` on a non-model file: `WeightLoader` rejects it correctly, but `test_bit_exact()` then reads `N` from an untrusted header and allocates `N` floats. Validate `N * 4 + 4 == file_size` before allocating. Same for the plugin path.
3. **Make CI honest.** The clap-validator URL points to an asset that has never existed (`clap-validator-linux-x86_64.tar.gz` in a release that ships `clap-validator-<ver>-ubuntu-22.04.zip`). Pin the asset name, and use `actions/checkout` with `lfs: true` **or** fetch the model explicitly. A red badge in the README while the summary table says "✅ builds on Linux" is the cheapest credibility loss to fix.
4. **Fix or rename the "bit-exact" test.** `max_diff < 0.05` is a tolerance test, not bit-exact, and it compares an R=4 time-domain engine against an R=8 spectral reference — impossible by construction (the exporter says so in its own docstring). Either compare like-for-like against `python/reference_forward.py`, or move it behind H3 and rename it "cross-architecture sanity check".
5. **Sync the README with reality** (GUI done, ARA status, R=4 vs R=8 split-brain).

### P1 — pass a real-time budget (weeks)

Target: **≤ 5 % of one modern laptop core per stereo instance** (with headroom for 2× host oversampling and other plugins). At 48 kHz that is a per-sample budget of ~15–35 GFLOP/s; today's model needs ~390 GFLOP/s stereo. Levers, in order of payoff:

1. **Restructure per-sample matvec → block GEMM.** Today every sample re-reads a 6.65 MB weight file through naive row loops. Processing 64–512-sample tiles as GEMMs keeps weights in cache and raises ILP/vectorization: realistically **3–5×**. (`MAX_BLOCK_SIZE = 512` already exists.)
2. **Fix the RoPE trig.** `step_norm_rope()` computes `cos/sin` per **(head, rank, angle)** = 256 pairs per layer per sample — **3,072 transcendental calls per sample**, 4× redundant (the angle only depends on head and angle index), and none are cached. Hoist them out of the rank loop and precompute tables: likely **2–4×**, plus it removes a latency spike source.
3. **Cut the model's width — the actual decision.** `in_proj` alone (1612×128) is ~61 % of the MACs per sample. d_model 128 → 32–48, in_proj 1612 → ~350, MIMO rank 4 → 1 (or 2), 6 layers → 4: this is where the remaining **~10–30×** has to come from. Retrain and re-listen; if the effect can't survive that, it isn't a plugin candidate at this latency class.
4. **Add a CI compute-budget test.** A micro-benchmark that fails the build if per-sample cost regresses. This number — not the loss curve — decides whether the product exists.
5. Only after 1–4: quantization (int8/bf16 paths), and only if measurements still demand it.

### P2 — the artifact both blueprints ask for (after P1)

One flagship effect (denoiser / amp-cab match / dialogue cleanup are the realistic small ones), a 20–40 s same-loudness A/B in a real DAW, a CPU% overlay in the video, presets, installer, manual, demo limitation, honest latency and PDC. This is the "evidence" B is talking about, and it is currently absent.

### P3 — differentiators that are cheap only *after* the above

ARA conditioning, CLAP text/audio encoders, per-effect variants, and — only if a real user asks — adapter-style preset swapping. LoRA is a *result* of having a shippable base model, not a substitute for one.

---

## 6. Reproduction

```bash
# 1. Weights are LFS pointers; verify that first
head -c 60 models/model.bin            # "version https://git-lfs.github.com/spec/v1"
curl -s -X POST -H 'Accept: application/vnd.git-lfs+json' -H 'Content-Type: application/vnd.git-lfs+json' \
  -d '{"operation":"download","transfers":["basic"],"objects":[{"oid":"5bbde5682e2fb62bbeb6a57af0f2f2e9820c88497dfbca73ec7ed9a9b5dbb7f0","size":6649600}]}' \
  https://github.com/sgm-audio/clara.git/info/lfs/objects/batch   # -> 403 exceeded its LFS budget

# 2. Build (CMake 4.x needs the policy shim; sandbox workaround for CPM download, see below)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build --target Mamba3Core test_mamba3 --parallel

# 3. Test on the as-cloned weights -> abort, exit 134
stdbuf -oL ./build/tests/test_mamba3 models/model.bin

# 4. Regenerate valid fixtures elsewhere and re-test -> 7/7 engine tests pass
python3 python/tools/export_mamba_weights.py --output /tmp/models-regen/model.bin --seed 42
./build/tests/test_mamba3 /tmp/models-regen/model.bin    # ALL TESTS PASSED (bit_exact SKIPs)

# 5. Measure the engine (harness linked against build/libMamba3Core.a)
./bench /tmp/models-regen/model.bin     # 0.38 ms/sample mono @48k -> 1820% of one core
```

Notes on the sandbox: GitHub release-asset downloads are blocked here, so `clap-wrapper`'s CPM bootstrap needed a git-clone of `cpm-cmake/CPM.cmake@v0.38.6` placed at the expected path (and the guard patched in the fetched copy under `build/_deps/`). That is an environment artifact, not a repo defect. The `dump_reference_output.py` path additionally requires `torchaudio`, which is imported by `train_clara.py` but is not listed in the documented requirements.

---

## 7. Bottom line

- **Blueprint A** is a good story about a different company. Its technical specifics (ONNX/LibTorch/TensorRT, JUCE, runtime LoRA, rented A100s) are either already rejected here for defensible reasons or irrelevant to the thing that is actually broken.
- **Blueprint B** is the right plan for CLARA, and CLARA already satisfies its hardest engineering conditions (hand-written RT-safe SSM kernel, no heap on the audio thread, real Mamba-3 mechanisms, separate training/plugin codebases, honest latency semantics). But B's "shrink until a laptop can run it" is not a metaphor here: it is a **~100× per-sample compute reduction**, and it is the only path from "research artifact" to "plugin a stranger can load".
- **What neither blueprint has, and CLARA does:** a faithful, from-scratch C++ Mamba-3 with trapezoidal discretization, complex/RoPE state and MIMO rank-4, running allocation-free. That is worth protecting. Right now it is locked behind a Git LFS 403 and a 1,800 % CPU bill — both fixable, in that order.

---

# Addendum — 2026-10-04: what was implemented, and what the measurements changed

The review above proposed a fix order. This addendum records what was actually
done, with the measurements that confirm or correct the original estimates. The
proposals that were wrong are called out as such.

## P0 — reproducibility (done)

| Item | Change | Evidence |
|---|---|---|
| Weight distribution | Removed the Git LFS filter for `models/*.bin` and committed the model directly (6.6 MB). Documented the lost blobs and their oids in `models/README.md`. | `head -c 8 models/model.bin` no longer returns `version ` |
| Crash on non-model input | `test_mamba3` now validates a fixture's declared length against its actual size before allocating, and `WeightLoader` rejects files whose size does not exactly match the declared payload. | The as-cloned tree used to abort with `std::bad_alloc`; it now fails with a clear message |
| CI red on `main` | Pinned the clap-validator asset that exists (`0.4.1 … ubuntu-22.04.zip`, unzipped), added `libx11-dev`, added a model sanity step (size + LFS-pointer check) on Linux and Windows. | YAML validates; four jobs intact |
| "Bit-exact" test | Replaced with `test_python_reference`: fixtures come from `dump_engine_reference.py`, which runs the engine's *own* model through `python/reference_forward.py`. Documented the tolerance, not just the assertion. | 21 checks pass; agreement 1.0–1.8e-3 abs / 5–9e-4 rel |
| Docs vs reality | README, AGENTS, `models/README.md`, TODO, CHANGELOG, INCUBATOR_CHECKLIST updated. | — |

### Two findings that were not in the original review

1. **`python/reference_forward.py` could not be imported at all** — a multi-line
   generator expression inside an f-string is a `SyntaxError`. The file the C++
   engine was nominally "bit-exact against" had never run.
2. **The reference disagreed with the engine in five places**: RoPE pairing
   (half-split vs interleaved), missing `B_bias`/`C_bias`, missing Δ conditioning
   (skipped deliberately by an earlier author who was "isolating instability"),
   a broadcast bug in the D-skip, and state-normalization scope. All five were
   resolved in favour of the shipping engine, which is why the equivalence test
   is now meaningful rather than impossible.

## P1 — performance (partially done; the original estimate was too optimistic)

Implemented: batched FiLM/Δ conditioning per chunk, register-blocked GEMM kernels
for `in_proj`/`out_proj`, 4× fewer RoPE transcendental calls, and a compute-budget
test in CI. (The last item was proposed as part of P1 and is now `test_perf_budget`.)

| | Per sample (mono, 48 kHz) | % of one core | Stereo |
|---|---|---|---|
| Before | 0.379 ms | 1,820% | 0.419 ms → 2,009% |
| After | **0.262 ms** | **1,256%** | 0.259 ms → 1,242% |
| After, `-march=native` | 0.188 ms | 903% | 0.195 ms → 935% |

**The report's estimate was wrong.** It predicted "realistically 3–5x" from
restructuring the projections. The measured gain is **1.45x mono / 1.6x stereo**.
Why: per-component profiling shows `in_proj` (57% of cost) runs at ~13 GFLOP/s
before and ~17 GFLOP/s after, against a plain-loop matvec calibration of
11 GFLOP/s and a naive-FMA ceiling of ~35 GFLOP/s on this core. **The engine was
already near this CPU's floating-point issue limit** — it was not primarily
memory-bandwidth bound as the review assumed, so cache-friendly restructuring had
little room. Sweeping the GEMM tile shape from 2×16 to 8×16 moved the total by
under 5%, which is the same conclusion from another direction: block it, and you
are at the issue limit.

The recurrence is the one part with measured headroom left on the table: its inner
loop strides through `state[h][d][s]`, so it cannot vectorize. Reordering the state
to `[h][s][d]` would make the inner `d` loop contiguous (~34% of per-sample cost).
That is recorded in `docs/TODO.md` rather than done here, because the honest
conclusion is that **kernel tuning cannot close a 12x gap** — see below.

## The finding that changes the plan: numerical drift

Not in the original review at all, and now the most technically interesting item.
Comparing the float32 engine against a float64 oracle of the identical pipeline:

| Horizon | float32 vs float64 |
|---|---|
| 5.8 ms | 1.8e-3 (0.18% of peak) |
| 11.6 ms | 2.0e-2 (2.0%) |
| 23 ms | 3.6e-2 (3.6%) |
| 93 ms | 7.3e-2 (7.3%) |

Peak `|ssm_real|` sits at exactly 1.000 at the end of a 4096-sample run, i.e. the
periodic clamp is active — the state is saturated, and the clamp is behaving as a
band-aid over unstable dynamics (which is also what the "isolating instability"
comment in `reference_forward.py` was about). Output stays finite, so the old
long-run test passed regardless.

Consequences: different CPUs, compilers or FMA paths will produce measurably
different audio from the same input and weights; and any trained model inherited
this dynamic range. `test_numeric_sensitivity` now reports the drift on every CI
run instead of hiding it.

## What this means for the priority order

The review's P1 said "pass a real-time budget; the levers are kernel work first,
then model size". The measurements invert that: kernel work yielded 1.45x, and the
gap is ~12x for mono / ~24x for stereo, so **model size is not the last lever, it
is the only one**. Concretely, the remaining work is:

1. **Shrink the model** — `in_proj` (1612×128) is 61% of the MACs per sample;
   `d_model` 128 → 32–48, MIMO rank 4 → 1–2, 6 → 4 layers. Then retrain and listen.
2. **Fix the drift** — double accumulation in the recurrence, or fix the dynamics
   so the clamp never fires. Both are cheap; neither has been attempted.
3. **Then** revisit H3 (R=8 PerTrackProcessor): at 4.26M params it is *further*
   from real time than what ships today, so the target architecture and the
   real-time requirement are currently in conflict. That conflict should be
   resolved deliberately rather than discovered late.

Everything else in the original review (evidence, demos, pricing, presets) still
depends on the same two preconditions: an artifact that loads, and a cost that
fits a CPU budget.
