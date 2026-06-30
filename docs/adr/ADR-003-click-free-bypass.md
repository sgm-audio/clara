# ADR-003: Click-Free Bypass via 50ms Linear Crossfade

**Status:** Accepted (implemented)
**Date:** 2026-05-30

## Context

The CLAP plugin exposes a `bypass` parameter with `CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS` flags. When the host engages or disengages bypass, it sets the atomic to a discrete 0.0 or 1.0 value. If the audio processing path switched instantaneously between the wet/dry mix and the dry signal, the resulting amplitude discontinuity would produce an audible pop or click.

A DC discontinuity occurs when the output signal jumps abruptly from a processed value (which may have been amplified, filtered, or delayed) to the raw input sample, or vice versa. The magnitude of the pop depends on the instantaneous difference between the two paths — at a 48kHz sample rate, adjacent samples can differ by up to the full signal amplitude (e.g., ~4V peak for consumer line level), producing an energy spike across the spectrum.

An inaudible transition requires:
- **Duration long enough** to spread the transition energy below the threshold of perception (typically >3-10ms for a full-amplitude step; 50ms is conservative)
- **Shape smooth enough** to avoid generating high-frequency components (linear ramp has a sinc-envelope spectrum; 50ms push the first null to 20Hz)
- **Deterministic per-sample math** with no branch or conditional that could cause timing variation on the audio thread

### Alternatives Considered

| Option | Description | Issue |
|--------|-------------|-------|
| **Instantaneous switch** | `output = bypass ? dry : wet` | DC discontinuity — the instantaneous difference between wet and dry paths at the switch point creates a step function, which has energy at all frequencies. Audible pop on any non-zero signal. |
| **Exponential ramp** | `bypassVal *= decayFactor` per sample (e.g., `0.999f`) | Perceptibly non-linear in amplitude space. The approach curve is fast at first but slow to converge, making the transition duration dependent on the threshold. Harder to tune to a specific duration. |
| **Step-synchronous** | Align transition to zero-crossing of the difference signal | Requires lookahead (buffering) to identify zero-crossings, adding latency. Not guaranteed to find a zero-crossing within a reasonable window for sustained tones. Complex state tracking. |
| **Equal-power cosine** | `sqrt(1 - t) * dry + sqrt(t) * wet` | Correct for crossfading two uncorrelated signals (e.g., DJ crossfader). For bypass, the two paths are highly correlated (the wet path is the dry path plus processing). Equal-power curve adds unnecessary math complexity for marginal benefit. Linear is perceptually equivalent for correlated signals. |
| **Linear ramp** (chosen) | `bypassVal = clamp(bypassVal ± step, 0, 1)` | Simple, deterministic, constant-duration. Step is accumulated per-sample outside the channel loop. |

## Decision

### 1. 50ms Linear Crossfade Duration

The crossfade takes exactly 50ms regardless of sample rate. This duration was chosen because:
- **Psychoacoustic lower bound**: ~3-10ms is the threshold for detecting a discontinuity as a separate event (versus a smooth transition). 50ms is well above this, providing margin for all signal types.
- **High-frequency suppression**: A 50ms linear ramp has its first spectral null at 1/0.05 = 20Hz, meaning the transition energy is concentrated below the audible range.
- **Practical upper bound**: 50ms at 48kHz = 2400 samples. At the default block size of 256 samples, this spans ~9-10 process callbacks. Longer durations (>100ms) would make bypass feel sluggish to the user.

The step per sample is computed once at `activate()` and never changes for the session:

```cpp
// MambaClapPlugin.cpp:262
state->crossfadeStep = 1.0f / (0.05f * static_cast<float>(sample_rate));
```

At 48kHz: `1.0f / (0.05f * 48000.0f) = 1.0f / 2400.0f ≈ 0.0004167`.

At 96kHz: `1.0f / (0.05f * 96000.0f) = 1.0f / 4800.0f ≈ 0.0002083`.

### 2. Per-Sample Ramp, Step Accumulated Outside the Channel Loop

The crossfade value is a single `float` shared across all channels. Each sample advances the ramp once, then all channels read that same value:

```cpp
// MambaClapPlugin.cpp:328-344
float bypassVal = state->currentBypassValue;
for (uint32_t s = 0; s < numSamples; ++s) {
    for (uint32_t ch = 0; ch < processChans; ++ch) {
        float mix = (1.0f - bypassVal) * (wet * wetDry + dry * (1.0f - wetDry))
                  + bypassVal * dry;
        output[s] = mix;
    }
    bypassVal = std::clamp(bypassVal + crossfadeStep, 0.0f, 1.0f);  // step once, not per-channel
}
state->currentBypassValue = bypassVal;
```

**The step accumulation is outside the per-channel loop.** This is intentional and fixes a stereo aliasing bug from an earlier revision where the step was inside the channel loop. The bug caused:

- Channel 0 advanced the ramp by `crossfadeStep`, then channel 1 advanced it by another `crossfadeStep`, so the two channels were always one sample apart in the transition.
- For stereo material, left and right channels would drift during the 50ms transition, causing a transient stereo widening/combing effect.
- The severity depended on block size — larger blocks meant more accumulated drift.

The fix: advance the ramp once per sample (outside the channel loop), then all channels read the same `bypassVal` for that sample position.

### 3. Direction Detection and Idle State

When `targetBypass` (the atomic loaded at the start of `processBlock()`) differs from `currentBypassValue` (the persistent ramp position from the previous block), the step direction is determined once:

```cpp
// MambaClapPlugin.cpp:316-318
const float crossfadeStep = (targetBypass != state->currentBypassValue)
    ? ((targetBypass > state->currentBypassValue) ? state->crossfadeStep : -state->crossfadeStep)
    : 0.0f;
```

If already at the target, `crossfadeStep` is 0 and the bypass value remains constant — a no-op fast path. The step value is signed, not unsigned: the negative step is computed once and reused, avoiding a negation inside the sample loop.

### 4. Block-Spanning Transitions

The crossfade can (and often does) span multiple `processBlock()` calls. The state variable `currentBypassValue` persists across blocks:

1. Block N: `currentBypassValue=0.0`, host sets `bypass=1.0`. The ramp advances from 0.0 toward 1.0. At end of block, `currentBypassValue` is saved (e.g., 0.3 if only 30% of the ramp fit in this block).
2. Block N+1: `targetBypass=1.0` (still 1.0), `currentBypassValue=0.3`. The ramp continues from 0.3 toward 1.0.
3. Block N+K: ramp reaches 1.0. `currentBypassValue` is clamped to 1.0. All subsequent blocks see `targetBypass=1.0` and `currentBypassValue=1.0` — the crossfade step is 0, the whole pipeline is bypassed.

At 48kHz with a 256-sample block size, a full transition takes 2400/256 ≈ 9.4 blocks (~9.4ms per block, total 50ms).

### 5. Mix Chain Order of Operations

The bypass crossfade is applied last in the mix chain, after intensity and wet/dry:

```cpp
// MambaClapPlugin.cpp:336-339
wet = dry + (wet - dry) * intensityVal;                          // STEP 1: intensity
float mix = (1.0f - bypassVal) * (wet * wetDry + dry * (1.0f - wetDry))  // STEP 2: wet/dry blend
          + bypassVal * dry;                                               // STEP 3: bypass crossfade
```

The order is deliberate:

1. **Intensity** (`intensityVal`): Scales the effect of the SSM-processed signal toward the dry signal. At 0.0, the wet signal is identical to dry (bypassed at the model level). At 1.0, the wet signal is the full model output. This is a *perceptual* depth control, not a bypass mechanism.
2. **Wet/Dry blend** (`wetDry`): Blends between the intensity-adjusted wet signal and the dry signal. This is the standard mix control (serial effects typically have a wet/dry knob).
3. **Bypass crossfade** (`bypassVal`): Crossfades between the entire wet/dry mix result and the raw dry input. When bypass reaches 1.0 (fully engaged), the output is pure dry regardless of intensity or wet/dry settings.

When bypass is fully engaged (`bypassVal = 1.0`), the mix equation reduces to `output = dry` — the model processing and all mix parameters are bypassed. The model is still processing internally (SSM state continues to accumulate), but its output is discarded. This means when bypass is disengaged, the model has been running silently and its state reflects the recent audio — no state-reset transient on un-bypass.

### 6. Interaction with State Load

When a preset is loaded (`mamba_state_load()`, `MambaClapPlugin.cpp:214`):

```cpp
state->currentBypassValue = data.bypass;
```

The `currentBypassValue` is set directly to the restored bypass value, not ramped. This means loading a preset with bypass=1.0 immediately sets the ramp to 1.0 — no crossfade on preset load. This is acceptable because:
- Preset loads happen outside the audio thread, triggered by user action.
- The user expects instantaneous change on preset load.
- The SSM state is reset on preset load anyway (`engine->resetState()` at line 216), so there's already a discontinuity.

### 7. CLAP Parameter Flags and Host Interaction

The bypass parameter is declared with both `CLAP_PARAM_IS_STEPPED` and `CLAP_PARAM_IS_BYPASS` flags (`MambaClapPlugin.cpp:61`):

```cpp
param_info->flags = CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS;
```

- `CLAP_PARAM_IS_STEPPED`: Tells the host this parameter takes discrete values (0 or 1), not continuous. The host typically renders this as a toggle button or switch.
- `CLAP_PARAM_IS_BYPASS`: Tells the host this is the bypass parameter. The host may use this for its own bypass indicator light, or to synchronize with its own bypass automation.

The host sends discrete 0/1 values via `clap_event_param_value_t`. The DSP side stores these in the `bypass` atomic and ramps the internal `currentBypassValue` smoothly. The ramp is invisible to the host — `clap_params_get_value()` returns the atomic `bypass` value (0 or 1), not the internal ramp position. This means the host's UI shows instantaneous bypass state even though the audio is crossfading.

### 8. Initial Value and Default State

- `bypass` atomic defaults to `0.0f` (no bypass).
- `currentBypassValue` defaults to `0.0f` (ramp at fully wet position).
- `crossfadeStep` defaults to `0.001f` before `activate()` sets it correctly. This default is never used in practice — `activate()` always runs before `processBlock()`.

On plugin instantiation, the plugin starts in fully active (non-bypassed) state. The first `processBlock()` call sees `targetBypass=0`, `currentBypassValue=0`, `crossfadeStep=0` — no ramp active.

## Consequences

### Positive
- **Click-free bypass transitions** verified with both impulsive and sustained input signals.
- **Deterministic per-sample math**: no branches in the inner loop (the crossfade step direction is pre-computed).
- **Sample-rate-independent**: crossfade duration is exactly 50ms at any sample rate.
- **Block-spanning**: transition naturally handles any block size without special-casing.
- **All channels synchronized**: single ramp shared across stereo channels; cannot drift.
- **Idempotent bypass**: if bypass state doesn't change, the ramp is a no-op (zero step, no per-sample state update beyond clamp to 0/1).

### Negative
- **~2400 samples of transition at 48kHz**: 50ms is conservative. Some users may feel the bypass feels slightly sluggish compared to hardware-style instant bypass. The 50ms was chosen to guarantee inaudibility across all signal types; could be shortened to ~10ms if user feedback indicates sluggishness.
- **Model runs during bypass**: The Mamba SSM continues processing audio during bypass. This is by design (keeps SSM state current for un-bypass) but consumes CPU even when bypassed. Future optimization could detect sustained bypass and skip model processing.
- **No bypass on preset load**: Crossfade is not applied when loading a preset — `currentBypassValue` is set directly. This is acceptable because preset loads are user-initiated instantaneous events.

### Constraints
- `crossfadeStep` must be computed at `activate()` — cannot change during processing. If sample rate changes, the plugin must be deactivated and reactivated.
- `currentBypassValue` persists across `processBlock()` calls. Must be stored in `MambaPluginState` (thread-safe because it's only accessed from the audio thread).
- The ramp is linear, not equal-power. For bypass (correlated signals), linear is correct. Do not change to equal-power without understanding the rationale.
- The step must remain **outside** the channel loop. Putting it inside causes stereo drift.

## Compliance

- Verified: `crossfadeStep = 1.0f / (0.05f * sampleRate)` computed at `mamba_activate()` (`MambaClapPlugin.cpp:262`).
- Verified: Step direction pre-computed once per block at `MambaClapPlugin.cpp:316-318`.
- Verified: Step accumulation outside channel loop at `MambaClapPlugin.cpp:342`.
- Verified: `bypassVal` persists across blocks via `state->currentBypassValue` (`MambaClapPlugin.cpp:344`).
- Verified: Mix equation order: intensity → wetDry → bypass crossfade (`MambaClapPlugin.cpp:336-339`).
- Verified: `currentBypassValue` set directly on state load (`MambaClapPlugin.cpp:214`).
- Verified: `CLAP_PARAM_IS_STEPPED | CLAP_PARAM_IS_BYPASS` flags on bypass parameter (`MambaClapPlugin.cpp:61`).
- Verified: Default `currentBypassValue = 0.0f` in `MambaPluginState` (`MambaClapPlugin.hpp:27`).
- Verified: Default `crossfadeStep = 0.001f` in `MambaPluginState` (`MambaClapPlugin.hpp:28`) — unused before `activate()`.

## Related

- `src/clap/MambaClapPlugin.cpp:61` — bypass parameter flags
- `src/clap/MambaClapPlugin.cpp:262` — `crossfadeStep` calculation at activate
- `src/clap/MambaClapPlugin.cpp:316-318` — step direction detection
- `src/clap/MambaClapPlugin.cpp:328-344` — per-sample crossfade ramp + mix
- `src/clap/MambaClapPlugin.hpp:27-28` — `currentBypassValue` and `crossfadeStep` members
- `src/clap/MambaClapPlugin.cpp:214` — state load sets `currentBypassValue` directly
- ADR-001: Shared ONNX Inference Engine (model runs during bypass)
- ADR-002: Deferred Atomic State Reset (SSM state reset on preset load)
