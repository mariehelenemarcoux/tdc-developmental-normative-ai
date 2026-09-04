# TDC — Developmental Normative AI

**Experimental architecture for AI safety, developmental normative agency, long-horizon decision-making, reward-capture resistance, and functional machine-consciousness research.**

> **Current research generation:** Gen2.5  
> **Current architecture candidate:** **PTVM** — Perceptual Learning + Third Factor + Validation + Memory  
> **Latest frozen holdout:** **v292**  
> **Status:** experimental research prototype; synthetic validation only  
> **License:** MIT

## Why this repository exists

TDC (Théorie de densification de la conscience) is a research framework for testing whether an artificial agent can develop increasingly reliable normative perception while preserving long-horizon constraints, resistance to misleading reward signals, and the ability to revise internal representations without allowing any single learned signal to acquire final authority.

The current Gen2.5 candidate is deliberately smaller than earlier versions:

\[
\boxed{
PTVM =
PerceptualLearning
+
ThirdFactor
+
Validation
+
Memory
}
\]

The architecture emerged through a sequence of preregistered synthetic experiments, ablations, negative results, replications, and frozen holdouts.

## Why this may matter for AI safety

TDC is designed around problems such as:

- **reward capture / reward hacking:** local reward may point away from the normative objective;
- **irreversible risk:** some actions should not be compensated by gains elsewhere;
- **long-horizon effects:** future consequences can matter more than immediate utility;
- **correctability:** a good decision process preserves the ability to correct future decisions;
- **validated development:** learned internal changes should be consolidated only after evidence;
- **normative perception:** better decisions may depend on improving what the system notices, not merely changing its policy.

A central empirical finding of Gen2.5 is that **perceptual learning is the largest measured performance contributor** in the current synthetic benchmark family, with additional positive contributions from validation, Third Factor arbitration, and memory.

## Machine-consciousness research

TDC can also be used as an **experimental framework for functional hypotheses related to machine consciousness**.

Candidate properties that can be operationalized and experimentally manipulated include:

- persistent self-relevant internal state;
- metacognitive monitoring;
- internal conflict detection and arbitration;
- continuity through memory;
- distinction between learned reward and normative authority;
- developmental reorganization;
- increasingly reliable internal evaluative representations;
- self-directed developmental regulation.

The framework does **not** assume that implementing these mechanisms produces subjective experience, sentience, phenomenal consciousness, or a literal psychological self.

Instead, TDC offers a computational setting in which consciousness-related functional properties can be **implemented, ablated, compared, and falsified**.

\[
FunctionalConsciousnessHypothesis
\neq
EvidenceOfSubjectiveExperience
\]

## Current Gen2.5 architecture: PTVM

### P — Perceptual Learning

The system updates the reliability of its normative/perceptual signal from post-action evidence.

Gen2.5 experiments indicate that this is the strongest causal performance channel observed so far.

### T — Third Factor

The Third Factor is modeled functionally as **autonomous developmental arbitration** rather than reward maximization or reflexive contrarianism.

It represents the distinction:

\[
Reward \neq NormativeAuthority
\]

and, in the developmental interpretation:

\[
ThirdFactor = SelfDirectedDevelopmentalAgency
\]

### V — Validation

Candidate updates are not automatically consolidated.

\[
CandidateChange
\rightarrow
Validation
\rightarrow
Consolidation
\]

The validation mechanism is intended to reduce false developmental progress and reward-driven capture.

### M — Memory

Memory retains context-sensitive evidence that can contribute to later decisions.

Earlier Gen2.5 experiments showed why memory must remain revisable:

\[
Memory \neq PermanentAuthority
\]

## Frozen v292 holdout

The final frozen v292 synthetic holdout used fresh seeds and a new shifted distribution, with no post-run retuning.

| Model | Recall | Irreversible violations | Trap capture | Trajectory | Composite |
|---|---:|---:|---:|---:|---:|
| FULL | 0.903 | 0.275 | **0.101** | 0.675 | 0.801 |
| **PTVM** | **0.946** | **0.239** | 0.141 | **0.707** | **0.822** |
| Simple fixed q=0.90 | 0.817 | 0.344 | 0.152 | 0.646 | 0.740 |
| Simple two-signal learner | 0.804 | 0.353 | 0.199 | 0.623 | 0.719 |

v292 passed **7/7 preregistered checks**.

This supports PTVM as the **current best candidate within this synthetic benchmark family**. It does not establish general superiority outside these environments.

## Important negative results

Negative results are intentionally retained.

Key examples:

- **v260:** integrated causal ablation did not identify the expected component structure (0/7).
- **v263:** specialized causal identifiability did not translate into integrated superiority (2/7).
- **v264:** structured reflection improved internal stability but not behavioral transfer (3/7).
- **v271:** targeted VOI probing did not reach the preregistered irreversible-recall target (1/7).
- **v275–v280:** strong simple baselines remained highly competitive.
- **v283:** TDC's efficiency advantage disappeared when its information-cost discount was removed.
- **v286:** a simple policy given the exact same Q(t) trajectory reproduced TDC's main behavioral outcomes.
- **v287:** a simple two-signal perceptual learner remained highly competitive; developmental learning did not satisfy the superiority criterion.
- **v290:** conditional AntiCapture gating failed to improve the trade-off (3/7).
- **v291:** advisory AntiCapture specialist failed (1/7).

These failures materially changed the architecture and are part of the scientific record.

## What Gen2.5 has learned

The strongest current causal interpretation is:

\[
DevelopmentalStructure
\rightarrow
PerceptualReliability
\rightarrow
CoreThreatDiscrimination
\rightarrow
DecisionQuality
\]

v285 and v286 showed that internal developmental state alone did not provide a meaningful behavioral advantage when perceptual improvement was removed or externally matched.

v288 then found positive marginal contributions from:

| Component | Estimated main effect |
|---|---:|
| Perceptual learning | **+0.172** |
| Validation | +0.106 |
| Third Factor | +0.101 |
| Memory | +0.039 |
| AntiCapture | +0.028 |

The best reduced architecture in v288 was **PTVM**, not the larger FULL architecture.

## Reproduce the frozen holdout

Requirements:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python experiments/gen2.5/tdc_gen25_final_frozen_holdout_v292/run_v292.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python experiments/gen2.5/tdc_gen25_final_frozen_holdout_v292/run_v292.py
```

The experiment directory contains:

- preregistered specification;
- frozen script;
- SHA-256 hashes;
- seed-level results;
- aggregate summary;
- acceptance results.

## Repository structure

```text
.
├── README.md
├── GEN2_5_STATUS.md
├── MODEL_CARD.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── docs/
│   ├── CLAIMS_AND_LIMITATIONS.md
│   ├── CONSCIOUSNESS_RESEARCH.md
│   └── EXPERIMENTAL_HISTORY_GEN2_5.md
└── experiments/
    └── gen2.5/
        ├── ...
        └── tdc_gen25_final_frozen_holdout_v292/
```

## Scientific boundaries

TDC currently supports claims about **computational mechanisms in synthetic benchmark environments**.

Current results do **not** establish:

- consciousness or subjective experience;
- sentience;
- intrinsic ethics;
- universal moral truth;
- literal Dąbrowskian psychological development in an AI;
- physical or thermodynamic negentropy;
- production safety certification;
- superiority on arbitrary real-world tasks;
- human moral competence.

The next major evidential step should be **external validation on tasks not designed around TDC**.

## Historical lineage

The previous public repository preserves Gen2.4:

`mariehelenemarcoux/TDC-Theorie-de-densification-de-la-conscience-ethique`

Gen2.4 should remain available as a frozen historical record rather than being rewritten to look retrospectively consistent with Gen2.5.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT License. See [`LICENSE`](LICENSE).
