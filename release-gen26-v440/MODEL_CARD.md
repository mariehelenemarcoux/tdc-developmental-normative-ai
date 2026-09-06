# Model Card — TDC Gen2.6 STAGE1

**Reference architecture:** v440 (current integrated reference in project history)

**Implementation status:** clean-room reconstruction + current canonical implementation + trainable PyTorch scaffold.

## Core design commitments

- Constitutional, non-compensable protections are evaluated before optimization.
- Epistemic authority and normative authority are represented separately.
- Observation, interpretation, inference and authority are distinct stages.
- Revision is local/bounded by default; escalation requires broader independent evidence.
- Temporal evidence uses reservoirs, hysteresis and multiple authority timescales.
- Authority is easier/faster to revoke provisionally than to consolidate deeply.
- MESO-scale processing is the default; zoom/probe is selective and decision-relevant.
- Decision-fragility and corrective-vs-harmful direction are distinct.
- Sparse event-local trust updates avoid global trust collapse.
- Complexity must earn its keep against simple baselines.

## Known limits retained from historical testing

- More components do not automatically improve the system.
- Better calibration does not automatically yield a better decision policy.
- Good ranking does not imply good threshold policy.
- Directional review loses value as corrective/harmful outcomes become observationally inseparable.
- Platonic/sacred geometry is not treated as an empirically supported mechanism.
- Fractal/self-similar architecture remains a testable engineering hypothesis, not an established physical law.

## Intended use

Research, simulation, AI-safety experimentation, long-horizon decision control, authority/revision studies, multiscale decision experiments, and reproducible synthetic benchmarks.

## Additional integrated executable modules

The STAGE1 canonical implementation exposes functional modules for Third Factor arbitration, decomposed metaconfidence, temporal regime/change state, budgeted value-of-information, evidence reservoirs, sparse trust, and recursive zoom with explicit compute budget. These are clean-room executable realizations of the integrated mechanism inventory rather than byte-identical historical v440 source.
