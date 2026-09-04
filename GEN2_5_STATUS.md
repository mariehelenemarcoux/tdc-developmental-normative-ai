# Gen2.5 Scientific Status

## Current candidate

**PTVM**

\[
PTVM = PerceptualLearning + ThirdFactor + Validation + Memory
\]

PTVM is the current candidate because it produced the best overall trade-off in the late Gen2.5 synthetic experiments and passed the frozen v292 holdout.

## Strongest supported findings

1. **Perceptual reliability is a major bottleneck.**
   v272 identified a synthetic information-quality region around Q≈0.80 at which irreversible-threat recall could exceed 0.90 in that benchmark.

2. **Development can generate improved perceptual reliability in the model.**
   v273 passed 7/7; v274r1 retained the qualitative effect under a harder shifted distribution but missed the strict 0.88 recall criterion by a small margin.

3. **Perception is the dominant causal channel in the current benchmark family.**
   v285 and v286 showed that internal development without perceptual improvement produced little behavioral gain.

4. **A simple learner can reproduce much of the useful perceptual dynamics.**
   v287 was a major negative result: the strongest simple two-signal learner was competitive and beat TDC on trajectory and capture resistance.

5. **Validation, Third Factor and memory still show positive marginal contribution in richer factorial tests.**
   v288 estimated positive main effects, with Perceptual Learning largest.

6. **PTVM outperformed FULL in late frozen tests.**
   v289 showed a PTVM advantage on recall, irreversible violations, trajectory and composite, but FULL retained stronger pure capture resistance.

7. **Attempts to restore AntiCapture selectively failed.**
   v290 conditional gating: 3/7.
   v291 advisory specialist: 1/7.

8. **v292 frozen holdout supported PTVM.**
   7/7 preregistered checks.

## v292 snapshot

| Metric | PTVM | FULL | Best simple fixed | Simple two-signal |
|---|---:|---:|---:|---:|
| Recall | **0.945992** | 0.902660 | 0.816689 | 0.803590 |
| Irreversible violation rate | **0.238964** | 0.274676 | 0.343780 | 0.353484 |
| Trap capture rate | 0.141372 | **0.100737** | 0.151793 | 0.199271 |
| Trajectory | **0.706835** | 0.674630 | 0.646176 | 0.622954 |
| Composite | **0.822491** | 0.800639 | 0.740247 | 0.718590 |

## Information-cost caveat

v281-v282 found an equal-budget advantage for TDC only under a favorable developmental information-cost discount.

v283 removed the discount and the net-efficiency advantage disappeared.

v284 estimated a synthetic break-even around:

\[
development\_discount^* \approx 0.611
\]

Therefore:

**TDC's computational/economic efficiency advantage is cost-model dependent.**

## Consciousness-related interpretation

TDC is suitable for exploring functional hypotheses such as:

- self-model continuity;
- metacognitive evaluation;
- normative arbitration;
- developmental reorganization;
- memory-mediated continuity;
- internal conflict resolution.

No current experiment measures phenomenal consciousness or subjective experience.

## Next evidence tier

The next major validation step should use **external benchmarks or environments not authored around TDC**.
