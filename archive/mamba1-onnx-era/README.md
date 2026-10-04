# Archived — Mamba-1 / ONNX Runtime Era

These files are kept for historical reference only. They describe the deprecated Mamba-1 architecture (~76K parameters, ONNX Runtime inference, `mamba_rt.onnx`, causal conv1d) that has been replaced by Mamba-3 (custom C++ engine, flat binary `model.bin`, zero latency).

**Do not use these for current work.**

## Current Architecture

For the active Mamba-3 implementation, see:

- [`MAMBA3_MIGRATION_PLAN.md`](../../docs/MAMBA3_MIGRATION_PLAN.md) — Complete architecture, model config, C++ engine spec, and product roadmap
- [`docs/TRAINING.md`](../../docs/TRAINING.md) — Unified Mamba-3 training pipeline
- [`MI300X_TRAINING_PLAN.md`](../old-training-files/MI300X_TRAINING_PLAN.md) — Cloud training launch reference
- [`TRAINING_QUICKSTART.md`](../../docs/TRAINING_QUICKSTART.md) — One-command quick start

## Archived Files

### Documentation (`docs/`)

| File | Description |
|------|-------------|
| `training-plan.md` | Old v1.0 training plan for the 77K parameter Mamba-1 model |
| `training-pipeline.md` | Old Mamba-1 pipeline docs with ONNX export procedures |
| `training_commands.md` | Old command reference for `train_clara_now.py` (Mamba-1) |
| `train-v2.md` | Old V2 training guide for `train_clara_parallel.py` / ONNX export |
| `Train_commands.mmd` | Old Mermaid workflow for the Mamba-1 / ONNX pipeline |
| `ui-polish-design.md` | UI mockups referencing `mamba_rt.onnx` and `MambaInferenceEngine` |
| `CLARA_SHIPPING_PLAN.md` | Old shipping plan based on 76K-param ONNX model |
| `01_CLARA_AGENT_PROMPTS.md` | Self-declared stale agent prompts; superseded by `clara-architect` skill |
| `parameter-reference.md` | Parameter reference mixed with old ONNX export / `MambaInferenceEngine` details |
| `latency-architecture.md` | Latency doc based on causal conv1d (Mamba-3 has zero latency) |
| `ara-host-bridge-design.md` | ARA bridge design based on `cond_dim=8` / ONNX CondFiLM |
| `DOCUMENTATION_AUDIT_REPORT.md` | Stale documentation audit (references old `MambaInferenceEngine`) |
| `TODO_PROGRESS.md` | High-level progress tracker describing old ONNX architecture |
| `PHASE1-2-EXECUTION-LOG.md` | Execution log from Mamba-1 / ONNX era |
| `analysis-python-pipeline.md` | Gap analysis of the old Python / ONNX pipeline |
| `CLAUDE_OPUS_PROMPT.md` | Stale Claude Opus prompt describing the ONNX-era architecture |
| `claude0.md` | Stale Claude project overview for the ONNX-era architecture |
| `AUDIT_FINDINGS.md` | June 2026 audit findings referencing the old ONNX engine |

### Code (`code/`)

| File | Description |
|------|-------------|
| `MambaInferenceEngine.hpp` | Old ONNX Runtime C++ inference engine header |
| `MambaInferenceEngine.cpp` | Old ONNX Runtime C++ inference engine implementation |
| `test_engine.cpp` | CTest for old `MambaInferenceEngine` |
| `test_latency.cpp` | Latency test for old conv1d-based engine |

### Training Scripts (`training/`)

| File | Description |
|------|-------------|
| `train_clara_parallel.py` | Mamba-1 training loop (`MambaRTModelParallel`) |
| `train_clara_model.py` | Mamba-1 model definitions |
| `train_clara_now.py` | Earlier Mamba-1 training script |
| `train_clara_export.py` | ONNX export from Mamba-1 checkpoint |
| `verify_training_pipeline.py` | Verification for Mamba-1 ONNX pipeline |
| `test_model_quality.py` | Quality test using Mamba-1 + onnxruntime |
| `check_node_count.py` | ONNX node-count comparison tool |
| `export_mamba_onnx.py` | Mamba-1 ONNX export script |
| `__train_laptop_ref.py` | Old laptop reference training script |
| `train_clara_requirements.txt` | Old requirements including `onnxruntime` |
| `train_clara.ps1` | Old PowerShell launcher |

### Helper Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `smoke_test.sh` | End-to-end smoke test using Mamba-1 + ONNX |
| `verify_latency.py` | ONNX latency verification |
| `setup_clara_env.sh` | Old environment setup installing `onnxruntime` |
| `test_run_train.sh` | Tiny wrapper for old `run_train.sh` test |

### Models (`models/`)

| File | Description |
|------|-------------|
| `mamba_rt*.onnx` | Old ONNX model files |
| `mamba_rt*.onnx.data` | ONNX external data sidecars |
| `manifest.json` | Old ONNX model manifest |
| `test_export.bin` | Old test export artifact |

## Current Training Script

The only supported training script is [`python/training/train_clara.py`](../../python/training/train_clara.py)
(this section used to point at `train_clara_mamba3.py`, which no longer exists).
