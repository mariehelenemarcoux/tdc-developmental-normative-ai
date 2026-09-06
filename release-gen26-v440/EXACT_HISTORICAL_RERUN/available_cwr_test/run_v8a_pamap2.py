#!/usr/bin/env python3
"""
run_v8a_pamap2.py

Reproduce the frozen CWR v0.1 PAMAP2 replication benchmark.

Primary comparable cohort:
    subjects 101-108

Subject 109 is excluded from the primary multiclass inference because its
Protocol recording is not comparable for the selected endpoint.

Protocol:
- leave-one-subject-out
- three IMUs as three perspectives: wrist, chest, ankle
- 960 calibration windows sampled from the remaining subjects
- 120 unlabeled adaptation windows from the unseen subject
- CWR hyperparameters frozen from v0.1
- 3 perturbation repeats per unseen subject
- inferential unit = subject, not window

Conditions:
- stable
- wrist 3D frame rotation
- wrist axis loss
- wrist 3D frame rotation + axis loss

Baselines:
- frozen CWR
- plain GCCA
- low-rank ALS
- concatenated PCA
- supervised late fusion

Important limitation:
The PAMAP2 measurements are real. The frame-drift and axis-loss perturbations
are experimentally injected.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import binomtest, wilcoxon


V = 3
K = 6
LAM = 1.5
REL_POWER = 1.5

SUBJECTS = list(range(101, 109))
SCENARIOS = ("stable", "rotation", "axis_loss", "rotation_axis_loss")


def ridge(X: np.ndarray, Y: np.ndarray, lam: float = 0.8) -> np.ndarray:
    return np.linalg.solve(
        X.T @ X + lam * np.eye(X.shape[1]),
        X.T @ Y,
    )


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def invsqrt_psd(A: np.ndarray) -> np.ndarray:
    val, vec = np.linalg.eigh((A + A.T) / 2)
    val = np.maximum(val, 1e-10)
    return (vec * (1 / np.sqrt(val))) @ vec.T


def impute_fit(Xv):
    med = []
    out = []
    for X in Xv:
        m = np.nanmedian(X, axis=0)
        m = np.where(np.isfinite(m), m, 0.0)
        med.append(m)
        out.append(np.where(np.isfinite(X), X, m))
    return med, out


def impute_apply(Xv, med):
    return [
        np.where(np.isfinite(Xv[i]), Xv[i], med[i])
        for i in range(V)
    ]


def std_fit(Xv):
    med, Xi = impute_fit(Xv)
    mu = [x.mean(0) for x in Xi]
    sd = [x.std(0) + 1e-6 for x in Xi]
    Xs = [(Xi[i] - mu[i]) / sd[i] for i in range(V)]
    return med, mu, sd, Xs


def std_apply(Xv, med, mu, sd):
    Xi = impute_apply(Xv, med)
    return [(Xi[i] - mu[i]) / sd[i] for i in range(V)]


def fit_gcca(Xv, weights=None):
    """
    Linear MAXVAR-style GCCA computed in feature space.
    """
    n = len(Xv[0])
    if weights is None:
        weights = np.ones(V)

    means = []
    centered = []
    blocks = []

    for i, X in enumerate(Xv):
        m = X.mean(0)
        Xc = X - m
        means.append(m)
        centered.append(Xc)

        W = invsqrt_psd(
            Xc.T @ Xc + LAM * np.eye(X.shape[1])
        )
        blocks.append(np.sqrt(weights[i]) * Xc @ W)

    Ucat = np.concatenate(blocks, axis=1)
    u, s, vt = np.linalg.svd(Ucat, full_matrices=False)
    G = u[:, :K] * np.sqrt(n)

    enc = []
    res = []

    for Xc in centered:
        B = ridge(Xc, G, LAM)
        enc.append(B)
        res.append(mse(Xc @ B, G))

    return means, enc, G, np.asarray(res)


def gcca_apply(Xv, means, enc, weights):
    P = np.stack([
        (Xv[i] - means[i]) @ enc[i]
        for i in range(V)
    ])

    w = np.asarray(weights, float)
    w /= w.sum() + 1e-12

    return np.tensordot(w, P, axes=(0, 0))


def cwr_fit(Xcal):
    """
    Frozen CWR v0.1 calibration:
    - initial GCCA
    - reliability estimated from latent residual
    - reliability power = 1.5
    - weighted GCCA
    """
    m0, e0, G0, res = fit_gcca(Xcal, np.ones(V))

    rel = 1 / (res + 0.05)
    rel = (rel / rel.mean()) ** REL_POWER

    m, e, G, _ = fit_gcca(Xcal, rel)
    return m, e, G, rel


def cwr_adapt(Xa, means, enc):
    """
    Frozen local perspective realignment rule.
    """
    m = [z.copy() for z in means]
    e = [z.copy() for z in enc]

    P = np.stack([
        (Xa[i] - m[i]) @ e[i]
        for i in range(V)
    ])

    residual = np.array([
        mse(
            P[i],
            np.mean(
                P[[j for j in range(V) if j != i]],
                axis=0,
            ),
        )
        for i in range(V)
    ])

    med = np.median(residual) + 1e-12
    suspects = np.where(residual > 1.65 * med)[0]

    if len(suspects) > 1:
        suspects = np.array([int(np.argmax(residual))])

    for i in suspects:
        stable = [
            j for j in range(V)
            if j != i and j not in suspects
        ]
        target = np.mean(P[stable], axis=0)

        mui = Xa[i].mean(0)
        e[i] = ridge(Xa[i] - mui, target, LAM)
        m[i] = mui

    return m, e, suspects, residual


def fit_affine(X, Y, lam=1e-4):
    mx = X.mean(0)
    my = Y.mean(0)
    return mx, my, ridge(X - mx, Y - my, lam)


def apply_affine(X, transform):
    mx, my, B = transform
    return (X - mx) @ B + my


def pca_fit(Xv):
    C = np.concatenate(Xv, axis=1)
    mu = C.mean(0)
    Cc = C - mu

    u, s, vt = np.linalg.svd(
        Cc,
        full_matrices=False,
    )

    W = vt[:K].T
    return mu, W, Cc @ W


def pca_apply(Xv, mu, W):
    return (
        np.concatenate(Xv, axis=1) - mu
    ) @ W


def als_fit(Xv, iters=7):
    C = np.concatenate(Xv, axis=1)
    mu = C.mean(0)
    Cc = C - mu

    u, s, vt = np.linalg.svd(
        Cc,
        full_matrices=False,
    )

    G = u[:, :K] * s[:K]
    B = vt[:K].T

    for _ in range(iters):
        B = np.linalg.solve(
            G.T @ G + 0.5 * np.eye(K),
            G.T @ Cc,
        ).T

        G = (
            Cc @ B
            @ np.linalg.inv(
                B.T @ B + 0.5 * np.eye(K)
            )
        )

    return mu, B, G


def als_apply(Xv, mu, B):
    C = np.concatenate(Xv, axis=1)
    return (
        (C - mu)
        @ B
        @ np.linalg.inv(
            B.T @ B + 0.5 * np.eye(K)
        )
    )


def stratified_sample(y, n, rng):
    idx = np.arange(len(y))
    out = []

    classes = np.unique(y)
    base = n // len(classes)

    for c in classes:
        ic = idx[y == c]
        out.extend(
            rng.choice(
                ic,
                size=min(base, len(ic)),
                replace=False,
            )
        )

    remain = n - len(out)

    if remain > 0:
        pool = np.setdiff1d(
            idx,
            np.asarray(out, dtype=int),
            assume_unique=False,
        )
        out.extend(
            rng.choice(
                pool,
                size=min(remain, len(pool)),
                replace=False,
            )
        )

    return np.asarray(out, dtype=int)


def load_subjects(feature_dir: Path):
    S = {}

    for subject in range(101, 110):
        path = feature_dir / f"subject{subject}.npz"

        if not path.exists():
            continue

        z = np.load(path)
        S[subject] = {
            key: z[key]
            for key in z.files
        }

    return S


def run(feature_dir: Path, output_dir: Path):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    S = load_subjects(feature_dir)

    missing = [
        s for s in SUBJECTS
        if s not in S
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing prepared subjects: {missing}"
        )

    rows = []
    detector_rows = []

    for test_subject in SUBJECTS:
        rng_cal = np.random.default_rng(
            88000 + test_subject
        )

        train_subjects = [
            s for s in SUBJECTS
            if s != test_subject
        ]

        ytrain = np.concatenate([
            S[s]["y"]
            for s in train_subjects
        ])

        Xtrain = [
            np.concatenate([
                S[s][f"stable_v{v}"]
                for s in train_subjects
            ], axis=0)
            for v in range(V)
        ]

        sel = stratified_sample(
            ytrain,
            960,
            rng_cal,
        )

        Xraw_cal = [
            X[sel]
            for X in Xtrain
        ]
        ycal = ytrain[sel]

        med, mu, sd, Xcal = std_fit(
            Xraw_cal
        )

        cm, ce, Gcwr, rel = cwr_fit(
            Xcal
        )
        gm, ge, Ggcca, _ = fit_gcca(
            Xcal,
            np.ones(V),
        )
        pm, pW, Gpca = pca_fit(Xcal)
        am, aB, Gals = als_fit(Xcal)

        clf_cwr = LogisticRegression(
            max_iter=700,
            C=1.0,
            solver="lbfgs",
        ).fit(Gcwr, ycal)

        clf_gcca = LogisticRegression(
            max_iter=700,
            C=1.0,
            solver="lbfgs",
        ).fit(Ggcca, ycal)

        clf_pca = LogisticRegression(
            max_iter=700,
            C=1.0,
            solver="lbfgs",
        ).fit(Gpca, ycal)

        clf_als = LogisticRegression(
            max_iter=700,
            C=1.0,
            solver="lbfgs",
        ).fit(Gals, ycal)

        late = []

        for v in range(V):
            clf = LogisticRegression(
                max_iter=700,
                C=1.0,
                solver="lbfgs",
            )
            clf.fit(Xcal[v], ycal)
            late.append(clf)

        yt = S[test_subject]["y"]
        n = len(yt)

        for rep in range(3):
            rng = np.random.default_rng(
                990000
                + test_subject * 31
                + rep
            )

            nadapt = min(
                120,
                max(60, n // 5),
            )

            adapt_idx = rng.choice(
                np.arange(n),
                size=nadapt,
                replace=False,
            )

            test_idx = np.setdiff1d(
                np.arange(n),
                adapt_idx,
                assume_unique=False,
            )

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

                Xt = std_apply(
                    raw,
                    med,
                    mu,
                    sd,
                )

                Xa = [
                    x[adapt_idx]
                    for x in Xt
                ]

                Xte = [
                    x[test_idx]
                    for x in Xt
                ]

                yte = yt[test_idx]

                # Adaptive CWR
                ma, ea, suspects, residual = cwr_adapt(
                    Xa,
                    cm,
                    ce,
                )

                G_cwr = gcca_apply(
                    Xte,
                    ma,
                    ea,
                    rel,
                )

                # Frozen CWR
                G_frozen = gcca_apply(
                    Xte,
                    cm,
                    ce,
                    rel,
                )

                # Plain GCCA:
                # full adaptation on the same unlabeled block
                # followed by affine alignment back to the old gauge.
                gma, gea, Ga, _ = fit_gcca(
                    Xa,
                    np.ones(V),
                )

                G_old = gcca_apply(
                    Xa,
                    gm,
                    ge,
                    np.ones(V),
                )

                g_align = fit_affine(
                    Ga,
                    G_old,
                )

                G_gcca = apply_affine(
                    gcca_apply(
                        Xte,
                        gma,
                        gea,
                        np.ones(V),
                    ),
                    g_align,
                )

                # PCA adaptation + gauge alignment
                pma, pWa, Gpa = pca_fit(
                    Xa
                )

                p_align = fit_affine(
                    Gpa,
                    pca_apply(
                        Xa,
                        pm,
                        pW,
                    ),
                )

                G_pca = apply_affine(
                    pca_apply(
                        Xte,
                        pma,
                        pWa,
                    ),
                    p_align,
                )

                # ALS adaptation + gauge alignment
                ama, aBa, Gal = als_fit(
                    Xa
                )

                a_align = fit_affine(
                    Gal,
                    als_apply(
                        Xa,
                        am,
                        aB,
                    ),
                )

                G_als = apply_affine(
                    als_apply(
                        Xte,
                        ama,
                        aBa,
                    ),
                    a_align,
                )

                evaluated = (
                    ("CWR", G_cwr, clf_cwr),
                    ("CWR_frozen", G_frozen, clf_cwr),
                    ("plain_GCCA", G_gcca, clf_gcca),
                    ("ALS", G_als, clf_als),
                    ("concat_PCA", G_pca, clf_pca),
                )

                for method, G, clf in evaluated:
                    pred = clf.predict(G)

                    rows.append({
                        "subject": test_subject,
                        "repeat": rep,
                        "scenario": scenario,
                        "method": method,
                        "accuracy": accuracy_score(
                            yte,
                            pred,
                        ),
                        "macro_f1": f1_score(
                            yte,
                            pred,
                            average="macro",
                            zero_division=0,
                        ),
                    })

                # Supervised late fusion
                probs = np.mean([
                    late[v].predict_proba(
                        Xte[v]
                    )
                    for v in range(V)
                ], axis=0)

                pred = late[0].classes_[
                    np.argmax(
                        probs,
                        axis=1,
                    )
                ]

                rows.append({
                    "subject": test_subject,
                    "repeat": rep,
                    "scenario": scenario,
                    "method": "late_fusion",
                    "accuracy": accuracy_score(
                        yte,
                        pred,
                    ),
                    "macro_f1": f1_score(
                        yte,
                        pred,
                        average="macro",
                        zero_division=0,
                    ),
                })

                detector_rows.append({
                    "subject": test_subject,
                    "repeat": rep,
                    "scenario": scenario,
                    "wrist_realign": int(
                        0 in set(
                            suspects.tolist()
                        )
                    ),
                    "n_realign": len(suspects),
                    "suspects": ";".join(
                        map(
                            str,
                            suspects.tolist(),
                        )
                    ),
                    "resid_wrist": residual[0],
                    "resid_chest": residual[1],
                    "resid_ankle": residual[2],
                })

    df = pd.DataFrame(rows)
    detector = pd.DataFrame(
        detector_rows
    )

    summary = (
        df.groupby(
            ["scenario", "method"]
        )[["accuracy", "macro_f1"]]
        .mean()
        .reset_index()
    )

    stable = (
        df[df.scenario == "stable"]
        .set_index(
            ["subject", "repeat", "method"]
        )[["accuracy", "macro_f1"]]
    )

    stress = df[
        df.scenario != "stable"
    ].copy()

    for metric in (
        "accuracy",
        "macro_f1",
    ):
        stress[f"{metric}_drop"] = [
            float(
                stable.loc[
                    (
                        r.subject,
                        r.repeat,
                        r.method,
                    ),
                    metric,
                ]
                - getattr(r, metric)
            )
            for r in stress.itertuples()
        ]

    drop_summary = (
        stress.groupby(
            ["scenario", "method"]
        )[[
            "accuracy_drop",
            "macro_f1_drop",
        ]]
        .mean()
        .reset_index()
    )

    audit = []

    for scenario in SCENARIOS[1:]:
        W = (
            stress[
                stress.scenario == scenario
            ]
            .groupby(
                ["subject", "method"]
            )
            .accuracy_drop
            .mean()
            .unstack("method")
        )

        for baseline in (
            "CWR_frozen",
            "plain_GCCA",
            "ALS",
            "concat_PCA",
            "late_fusion",
        ):
            d = (
                W[baseline]
                - W["CWR"]
            )

            nz = d[
                np.abs(d) > 1e-12
            ]

            wins = int((nz > 0).sum())
            losses = int((nz < 0).sum())

            audit.append({
                "scenario": scenario,
                "baseline": baseline,
                "mean_drop_advantage_CWR": float(
                    d.mean()
                ),
                "wins_subjects": wins,
                "losses_subjects": losses,
                "sign_p_one_sided": (
                    binomtest(
                        wins,
                        wins + losses,
                        0.5,
                        alternative="greater",
                    ).pvalue
                    if wins + losses
                    else 1.0
                ),
                "wilcoxon_p_one_sided": (
                    wilcoxon(
                        d,
                        alternative="greater",
                    ).pvalue
                    if np.any(d != 0)
                    else 1.0
                ),
            })

    audit = pd.DataFrame(audit)

    aggregate = (
        stress.groupby(
            ["subject", "method"]
        )
        .accuracy_drop
        .mean()
        .unstack("method")
    )

    aggregate_audit = []

    for baseline in (
        "CWR_frozen",
        "plain_GCCA",
        "ALS",
        "concat_PCA",
        "late_fusion",
    ):
        d = (
            aggregate[baseline]
            - aggregate["CWR"]
        )

        nz = d[
            np.abs(d) > 1e-12
        ]

        wins = int((nz > 0).sum())
        losses = int((nz < 0).sum())

        aggregate_audit.append({
            "baseline": baseline,
            "mean_drop_advantage_CWR": float(
                d.mean()
            ),
            "wins_subjects": wins,
            "losses_subjects": losses,
            "sign_p_one_sided": (
                binomtest(
                    wins,
                    wins + losses,
                    0.5,
                    alternative="greater",
                ).pvalue
                if wins + losses
                else 1.0
            ),
            "wilcoxon_p_one_sided": (
                wilcoxon(
                    d,
                    alternative="greater",
                ).pvalue
                if np.any(d != 0)
                else 1.0
            ),
        })

    aggregate_audit = pd.DataFrame(
        aggregate_audit
    )

    df.to_csv(
        output_dir
        / "cwr_v8a_pamap2_results.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "cwr_v8a_pamap2_summary.csv",
        index=False,
    )

    stress.to_csv(
        output_dir
        / "cwr_v8a_pamap2_drops.csv",
        index=False,
    )

    drop_summary.to_csv(
        output_dir
        / "cwr_v8a_pamap2_drop_summary.csv",
        index=False,
    )

    audit.to_csv(
        output_dir
        / "cwr_v8a_pamap2_subject_audit.csv",
        index=False,
    )

    aggregate_audit.to_csv(
        output_dir
        / "cwr_v8a_pamap2_aggregate_audit.csv",
        index=False,
    )

    detector.to_csv(
        output_dir
        / "cwr_v8a_pamap2_detector.csv",
        index=False,
    )

    print(
        "\nMEAN ACCURACY\n"
    )

    print(
        summary.pivot(
            index="scenario",
            columns="method",
            values="accuracy",
        )
        .round(4)
        .to_string()
    )

    print(
        "\nMEAN ACCURACY DROP VS STABLE "
        "(lower is better)\n"
    )

    print(
        drop_summary.pivot(
            index="scenario",
            columns="method",
            values="accuracy_drop",
        )
        .round(4)
        .to_string()
    )

    print(
        "\nSUBJECT-LEVEL PAIRED AUDIT\n"
    )
    print(audit.to_string(index=False))

    print(
        "\nAGGREGATE AUDIT\n"
    )
    print(
        aggregate_audit.to_string(
            index=False
        )
    )

    print(
        "\nWRIST REALIGNMENT RATE\n"
    )
    print(
        detector.groupby(
            "scenario"
        )
        .wrist_realign
        .mean()
        .to_string()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help=(
            "Directory produced by "
            "prepare_pamap2.py"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
    )

    args = parser.parse_args()

    run(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
