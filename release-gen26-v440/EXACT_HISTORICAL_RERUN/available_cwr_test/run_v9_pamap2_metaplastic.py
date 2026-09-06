#!/usr/bin/env python3
"""
run_v9_pamap2_metaplastic.py

Semi-real PAMAP2 temporal evaluation for:
    CWR v0.1
vs
    CWR v0.1 + observable stability reconsolidation + metaplasticity

Requirements:
- Place this file next to run_v8a_pamap2.py
- Use the same prepared_pamap2/subject101.npz ... subject108.npz files
  produced by prepare_pamap2.py.
- No ground-truth "stable/drift" label is used by the adaptive algorithm.
  The stream generator knows the scenario only for evaluation.

Protocol:
- LOSO calibration exactly follows the original PAMAP2 benchmark.
- Test-subject windows are split into:
    stream pool (unlabeled online adaptation)
    held-out evaluation pool
- A Markov stream switches randomly between stable and injected local drift.
- Every block:
    1. run the frozen v0.1 residual detector;
    2. if suspect -> local repair, reset stability history;
    3. if no suspect -> accumulate residual stability evidence;
    4. after 2 no-suspect blocks with <=10% residual relative range,
       perform weak reconsolidation;
    5. alpha = clip(0.02 + 0.16*C + 0.10*Dhat, 0.02, 0.28)
       where C is temporal consistency and Dhat is recent detected-drift rate.

This is a semi-real test:
- real PAMAP2 sensor windows;
- experimentally injected sensor-frame perturbations from prepare_pamap2.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from run_v8a_pamap2 import (
    SUBJECTS, V, LAM, REL_POWER,
    load_subjects, stratified_sample,
    std_fit, std_apply,
    cwr_fit, cwr_adapt, gcca_apply,
    ridge, mse,
)

PI_STABLE = 0.85
P_SS = 0.90
STEPS = 120
BATCH = 32
EVAL_EVERY = 5
STABILITY_WINDOW = 2
RESIDUAL_REL_TOL = 0.10
VOLATILITY_WINDOW = 20

SCENARIOS = ("stable", "rotation", "axis_loss", "rotation_axis_loss")


def markov_states(n: int, rng: np.random.Generator):
    p_ds = PI_STABLE * (1 - P_SS) / (1 - PI_STABLE)
    stable = bool(rng.random() < PI_STABLE)
    out = []
    for _ in range(n):
        out.append("stable" if stable else "drift")
        if stable:
            stable = bool(rng.random() < P_SS)
        else:
            stable = bool(rng.random() < p_ds)
    return out


def residual_median(Xa, means, enc):
    P = np.stack([(Xa[i] - means[i]) @ enc[i] for i in range(V)])
    r = np.array([
        mse(
            P[i],
            np.mean(P[[j for j in range(V) if j != i]], axis=0),
        )
        for i in range(V)
    ])
    return float(np.median(r))


def reconsolidate(Xa, means, enc, rel, alpha):
    m2 = [z.copy() for z in means]
    e2 = [z.copy() for z in enc]

    P = np.stack([
        (Xa[i] - m2[i]) @ e2[i]
        for i in range(V)
    ])
    target = np.mean(P, axis=0)

    for i in range(V):
        mui = Xa[i].mean(0)
        B = ridge(Xa[i] - mui, target, LAM)
        m2[i] = (1 - alpha) * m2[i] + alpha * mui
        e2[i] = (1 - alpha) * e2[i] + alpha * B

    return m2, e2


def run(feature_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    S = load_subjects(feature_dir)

    missing = [s for s in SUBJECTS if s not in S]
    if missing:
        raise FileNotFoundError(
            f"Missing prepared subjects: {missing}. "
            "Run prepare_pamap2.py first."
        )

    rows = []

    for test_subject in SUBJECTS:
        rng_cal = np.random.default_rng(88000 + test_subject)
        train_subjects = [s for s in SUBJECTS if s != test_subject]

        ytrain = np.concatenate([S[s]["y"] for s in train_subjects])
        Xtrain = [
            np.concatenate([S[s][f"stable_v{v}"] for s in train_subjects], axis=0)
            for v in range(V)
        ]

        sel = stratified_sample(ytrain, 960, rng_cal)
        Xraw_cal = [X[sel] for X in Xtrain]
        ycal = ytrain[sel]

        med, mu, sd, Xcal = std_fit(Xraw_cal)
        cm, ce, Gcwr, rel = cwr_fit(Xcal)

        clf = LogisticRegression(
            max_iter=700, C=1.0, solver="lbfgs"
        ).fit(Gcwr, ycal)

        yt = S[test_subject]["y"]
        n = len(yt)

        for rep in range(3):
            rng = np.random.default_rng(
                9_900_000 + test_subject * 101 + rep
            )

            # Split test subject: online stream vs never-adapted evaluation set.
            perm = rng.permutation(n)
            cut = max(BATCH * 3, int(0.70 * n))
            stream_idx = perm[:cut]
            eval_idx = perm[cut:]

            if len(eval_idx) < 20:
                eval_idx = perm[-max(20, n // 4):]
                stream_idx = np.setdiff1d(
                    np.arange(n), eval_idx, assume_unique=False
                )

            # Standardize all prepared variants once with training calibration stats.
            variants = {}
            for scenario in SCENARIOS:
                if scenario == "stable":
                    raw = [
                        S[test_subject][f"stable_v{v}"]
                        for v in range(V)
                    ]
                else:
                    raw = [
                        S[test_subject][f"{scenario}_r{rep}_v{v}"]
                        for v in range(V)
                    ]
                variants[scenario] = std_apply(raw, med, mu, sd)

            states = markov_states(STEPS, rng)
            current_drift = "rotation"

            # Pre-generate stream scenario independently of model decisions.
            stream_scenarios = []
            previous = "stable"
            for state in states:
                if state == "stable":
                    scenario = "stable"
                else:
                    if previous == "stable":
                        current_drift = rng.choice(
                            ["rotation", "axis_loss", "rotation_axis_loss"]
                        )
                    scenario = current_drift
                stream_scenarios.append(scenario)
                previous = state

            for method in ("CWR_v01", "CWR_meta_observable"):
                means = [x.copy() for x in cm]
                enc = [x.copy() for x in ce]

                stable_hist = []
                detected_hist = []
                recon_count = 0

                for t, scenario in enumerate(stream_scenarios, start=1):
                    batch_idx = rng.choice(
                        stream_idx,
                        size=min(BATCH, len(stream_idx)),
                        replace=len(stream_idx) < BATCH,
                    )
                    Xa = [variants[scenario][v][batch_idx] for v in range(V)]

                    # Detector is always run; no hidden scenario label drives action.
                    ma, ea, suspects, residual = cwr_adapt(
                        Xa, means, enc
                    )
                    detected = int(len(suspects) > 0)
                    detected_hist.append(detected)
                    detected_hist = detected_hist[-VOLATILITY_WINDOW:]

                    if method == "CWR_v01":
                        means, enc = ma, ea

                    else:
                        if detected:
                            means, enc = ma, ea
                            stable_hist = []
                        else:
                            med_resid = float(np.median(residual))
                            stable_hist.append(med_resid)

                            if len(stable_hist) >= STABILITY_WINDOW:
                                recent = np.asarray(
                                    stable_hist[-STABILITY_WINDOW:],
                                    dtype=float,
                                )
                                relchg = (
                                    recent.max() - recent.min()
                                ) / (abs(recent.mean()) + 1e-12)

                                if relchg <= RESIDUAL_REL_TOL:
                                    C = float(
                                        np.exp(
                                            -relchg / RESIDUAL_REL_TOL
                                        )
                                    )
                                    Dhat = float(
                                        np.mean(detected_hist)
                                    )
                                    alpha = float(
                                        np.clip(
                                            0.02
                                            + 0.16 * C
                                            + 0.10 * Dhat,
                                            0.02,
                                            0.28,
                                        )
                                    )
                                    means, enc = reconsolidate(
                                        Xa, means, enc, rel, alpha
                                    )
                                    recon_count += 1

                    if t % EVAL_EVERY == 0 or t == STEPS:
                        # Evaluate on the current physical observation regime,
                        # but on held-out windows never used for adaptation.
                        Xev = [
                            variants[scenario][v][eval_idx]
                            for v in range(V)
                        ]
                        yev = yt[eval_idx]
                        G = gcca_apply(
                            Xev, means, enc, rel
                        )
                        pred = clf.predict(G)

                        rows.append({
                            "subject": test_subject,
                            "repeat": rep,
                            "step": t,
                            "stream_scenario": scenario,
                            "method": method,
                            "accuracy": accuracy_score(yev, pred),
                            "macro_f1": f1_score(
                                yev, pred,
                                average="macro",
                                zero_division=0,
                            ),
                            "detected_suspect": detected,
                            "n_suspects": len(suspects),
                            "recon_count": recon_count,
                        })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("method")[["accuracy", "macro_f1"]]
        .mean()
        .reset_index()
    )

    final = (
        df[df.step == STEPS]
        .groupby("method")[["accuracy", "macro_f1"]]
        .mean()
        .reset_index()
    )

    seedwise = (
        df.groupby(["subject", "repeat", "method"])
        .accuracy.mean()
        .unstack("method")
        .reset_index()
    )
    seedwise["meta_minus_v01"] = (
        seedwise["CWR_meta_observable"]
        - seedwise["CWR_v01"]
    )

    audit = pd.DataFrame([{
        "mean_gain": float(seedwise.meta_minus_v01.mean()),
        "median_gain": float(seedwise.meta_minus_v01.median()),
        "wins": int((seedwise.meta_minus_v01 > 0).sum()),
        "losses": int((seedwise.meta_minus_v01 < 0).sum()),
        "ties": int((seedwise.meta_minus_v01 == 0).sum()),
        "n_subject_repeat_pairs": len(seedwise),
    }])

    df.to_csv(output_dir / "cwr_v9_pamap2_metaplastic_trials.csv", index=False)
    summary.to_csv(output_dir / "cwr_v9_pamap2_metaplastic_summary.csv", index=False)
    final.to_csv(output_dir / "cwr_v9_pamap2_metaplastic_final.csv", index=False)
    seedwise.to_csv(output_dir / "cwr_v9_pamap2_metaplastic_paired.csv", index=False)
    audit.to_csv(output_dir / "cwr_v9_pamap2_metaplastic_audit.csv", index=False)

    print("\nMEAN PERFORMANCE")
    print(summary.round(4).to_string(index=False))
    print("\nFINAL PERFORMANCE")
    print(final.round(4).to_string(index=False))
    print("\nPAIRED AUDIT")
    print(audit.round(4).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("results_v9_meta"))
    args = p.parse_args()
    run(args.feature_dir, args.output_dir)


if __name__ == "__main__":
    main()
