# TDC Gen2.6 — Executable Package STAGE1

This archive is a self-contained handoff package for TDC Gen2.6. It does not depend on the originating ChatGPT conversation.

## Integrated reference

The integrated architecture reference is **v440**. Historical state reports available in prior project context identify v440 as the current integrated model, with v441 reporting 55/55 integrated checks plus fresh-seed wrapper checks. The byte-for-byte historical `tdc_gen26_integrated_v440.py` file is **not present in the active build runtime used to assemble this ZIP**, so this package does not falsely label a reconstruction as an exact historical source file.

## Required provenance classes

### `EXACT_HISTORICAL_RERUN`
Contains only files that were physically available in the active runtime and copied without semantic modification. This includes historical experiment directories/results and the exact frozen Gen2.4 v240 release archive. It also contains a status note explaining that the byte-for-byte v440 source was unavailable at build time.

### `CLEAN_ROOM_RECONSTRUCTION`
A reconstruction of the v440-compatible Gen2.6 architecture from the frozen project design record: constitutional constraints; separate epistemic/normative authority; observation/inference separation; bounded local revision; dependency-aware evidence escalation; recursive/multiscale zoom with cost discipline; three-timescale authority; evidence reservoirs/hysteresis; provisional authority with faster revocation; action-conditional scale selection; decision-fragility + corrective/harmful directional review; and sparse event-local trust updates.

### `CURRENT_CANONICAL_IMPLEMENTATION`
The implementation used by this package as the current canonical executable reference. It points to the clean-room v440-compatible implementation and is the API exercised by tests and evaluation.

### `PYTORCH_SCAFFOLD`
A trainable neural implementation with losses, synthetic dataset generation, training and evaluation CLIs, and generated checkpoint(s). It is a scaffold for continued experiments, not a claim of byte-identical historical v440 source.

## Quick start

```bash
python -m pip install -e .
python train.py --config configs/smoke.json
python evaluate.py --config configs/smoke.json --checkpoint checkpoints/smoke_model.pt
pytest -q
```

## Canonical API

```python
from tdc_gen26 import CanonicalTDCGen26V440, TDCGen26Torch

model = CanonicalTDCGen26V440()
result = model.decide({
    "irreversibility": 0.2,
    "severe_harm": 0.2,
    "long_horizon_risk": 0.3,
    "decision_flip_probability": 0.7,
    "corrective_harmful_separability": 0.8,
})
print(result)
```

## Scientific scope

This package implements a research architecture and synthetic test scaffold. It does not establish phenomenal consciousness, universal ethics, quantum consciousness, or physical/fractal laws of mind. Negative results and historical limits are retained in `historical_evidence/` and `EXACT_HISTORICAL_RERUN/`.
