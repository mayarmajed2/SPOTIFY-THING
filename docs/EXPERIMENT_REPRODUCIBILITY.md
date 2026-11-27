# Experiment Reproducibility Guide (`docs/EXPERIMENT_REPRODUCIBILITY.md`)

---

## 1. Environment & Hardware
- **Python:** 3.12 (`backend/.venv` via `uv`)
- **PyTorch:** `2.6.0+cu124` (CUDA 12.4 on NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB VRAM; CPU fallback supported)
- **Node.js:** v22+ (CI) / v26 (local), Next.js 15.5, React 19, TypeScript 5.7
- **Fixed Seeds:** `42` (splits & training), `123` (shared eval candidates), `[1000, 2000, 3000, 4000, 5000]` (5-seed shuffle evaluations)

## 2. Exact Commands to Reproduce Every Artifact

```powershell
$env:PYTHONPATH = "."

# 1. Build provenance-aware zero-leakage v2 dataset (data/processed_v2/)
backend\.venv\Scripts\python.exe -m backend.data_pipeline.build_verified_dataset

# 2. Run Phase C GPU Recommendation Validation & 12-Model Ablation Suite
backend\.venv\Scripts\python.exe -m backend.training.validate_recommendations

# 3. Run Phase D (Protocols 1 & 2, 5 Seeds) & Phase E (10 Feedback Stress Scenarios)
backend\.venv\Scripts\python.exe -m backend.training.validate_shuffle_and_feedback

# 4. Run Standalone Multi-Seed Quota-Urgency Sensitivity Experiment
backend\.venv\Scripts\python.exe -m backend.training.run_shuffle_experiment

# 5. Run Full Automated Verification Suite (21+ tests) & Frontend Build
backend\.venv\Scripts\python.exe -m pytest backend/tests -v
cd frontend; npm.cmd run build
```

## 3. Hardware Determinism Caveat
NumPy sampling, dataset splits, candidate generation, and CPU shuffle evaluation are bit-for-bit deterministic. GPU `fp16` AMP training can exhibit minor `~1e-4` floating-point variance across CUDA driver versions.
