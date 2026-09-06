#!/usr/bin/env python3
"""
prepare_pamap2.py

Prepare window-level multiview features for the CWR PAMAP2 benchmark.

Expected input:
    PAMAP2 Protocol files named subject101.dat ... subject109.dat

Each PAMAP2 row has 54 columns:
    0   timestamp
    1   activity_id
    2   heart_rate
    3:20   wrist IMU block
    20:37  chest IMU block
    37:54  ankle IMU block

For the benchmark, each IMU location is treated as one perspective:
    view 0 = wrist
    view 1 = chest
    view 2 = ankle

The script creates clean activity windows and stores four versions:
    stable
    rotation
    axis_loss
    rotation_axis_loss

Important:
- The underlying PAMAP2 recordings are real.
- The frame-rotation and axis-loss perturbations are experimentally injected.
- Labels are not used to construct the multiview representation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

WINDOW = 100
STRIDE = 50
PURITY = 0.95

SUBJECTS = list(range(101, 110))
SCENARIOS = ("stable", "rotation", "axis_loss", "rotation_axis_loss")

# PAMAP2 IMU blocks, excluding timestamp/activity/heart-rate.
# Wrist: cols 3:20
# Chest: cols 20:37
# Ankle: cols 37:54
VIEW_SLICES = (
    slice(3, 20),
    slice(20, 37),
    slice(37, 54),
)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a proper 3D rotation matrix."""
    A = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def feature_block(X: np.ndarray) -> np.ndarray:
    """
    Fixed, label-free window summary.
    NaNs are preserved here and imputed later using calibration statistics.
    """
    return np.concatenate([
        np.nanmean(X, axis=0),
        np.nanstd(X, axis=0),
        np.nanmin(X, axis=0),
        np.nanmax(X, axis=0),
        np.sqrt(np.nanmean(X * X, axis=0)),
    ])


def apply_rotation_and_axis_loss(
    imu: np.ndarray,
    rotation: np.ndarray | None,
    lost_axis: int | None,
) -> np.ndarray:
    """
    Apply one common 3D transform consistently to the tri-axial vector groups
    inside a PAMAP2 IMU block.

    PAMAP2 IMU block structure:
      0    temperature
      1:4  acceleration scale 16g
      4:7  acceleration scale 6g
      7:10 gyroscope
      10:13 magnetometer
      13:17 orientation quaternion

    The physical rotation is applied to the four tri-axial vector groups.
    Temperature and quaternion are left unchanged in this benchmark.
    """
    out = imu.copy()

    vector_groups = (
        slice(1, 4),
        slice(4, 7),
        slice(7, 10),
        slice(10, 13),
    )

    for sl in vector_groups:
        T = out[:, sl]
        if rotation is not None:
            T = T @ rotation.T
        if lost_axis is not None:
            T[:, lost_axis] = 0.0
        out[:, sl] = T

    return out


def prepare_subject(
    dat_path: Path,
    subject: int,
    out_dir: Path,
    repeats: int = 3,
) -> None:
    print(f"Reading {dat_path.name} ...")
    A = np.loadtxt(dat_path, dtype=np.float32)

    if A.ndim != 2 or A.shape[1] != 54:
        raise ValueError(
            f"{dat_path} has shape {A.shape}; expected 54 columns."
        )

    y_raw = A[:, 1].astype(np.int16)

    rng = np.random.default_rng(260000 + subject)
    rotations = [random_rotation(rng) for _ in range(repeats)]
    lost_axes = [int(rng.integers(0, 3)) for _ in range(repeats)]

    stable_views = [[], [], []]
    transformed = {
        scen: [
            [[], [], []] for _ in range(repeats)
        ]
        for scen in SCENARIOS
        if scen != "stable"
    }
    labels = []

    min_nonzero = int(PURITY * WINDOW)

    for start in range(0, len(A) - WINDOW + 1, STRIDE):
        yw = y_raw[start:start + WINDOW]
        nz = yw[yw > 0]

        if len(nz) < min_nonzero:
            continue

        vals, counts = np.unique(nz, return_counts=True)
        label = int(vals[np.argmax(counts)])

        if np.max(counts) < min_nonzero:
            continue

        W = A[start:start + WINDOW]

        # Stable views
        stable_blocks = []
        for v, sl in enumerate(VIEW_SLICES):
            imu = W[:, sl]
            f = feature_block(imu)
            stable_views[v].append(f)
            stable_blocks.append(imu)

        # Perturbed wrist view only; chest/ankle remain unchanged.
        wrist = stable_blocks[0]
        chest_f = feature_block(stable_blocks[1])
        ankle_f = feature_block(stable_blocks[2])

        for rep in range(repeats):
            R = rotations[rep]
            axis = lost_axes[rep]

            variants = {
                "rotation": apply_rotation_and_axis_loss(
                    wrist, rotation=R, lost_axis=None
                ),
                "axis_loss": apply_rotation_and_axis_loss(
                    wrist, rotation=None, lost_axis=axis
                ),
                "rotation_axis_loss": apply_rotation_and_axis_loss(
                    wrist, rotation=R, lost_axis=axis
                ),
            }

            for scen, wrist_variant in variants.items():
                transformed[scen][rep][0].append(feature_block(wrist_variant))
                transformed[scen][rep][1].append(chest_f)
                transformed[scen][rep][2].append(ankle_f)

        labels.append(label)

    payload = {
        "subject": np.array(subject, dtype=np.int16),
        "y": np.asarray(labels, dtype=np.int16),
    }

    for v in range(3):
        payload[f"stable_v{v}"] = np.asarray(
            stable_views[v], dtype=np.float64
        )

    for scen in ("rotation", "axis_loss", "rotation_axis_loss"):
        for rep in range(repeats):
            for v in range(3):
                payload[f"{scen}_r{rep}_v{v}"] = np.asarray(
                    transformed[scen][rep][v], dtype=np.float64
                )

    out_path = out_dir / f"subject{subject}.npz"
    np.savez_compressed(out_path, **payload)

    unique, counts = np.unique(payload["y"], return_counts=True)
    print(
        f"  saved {out_path.name}: {len(labels)} windows; "
        f"activities={dict(zip(unique.tolist(), counts.tolist()))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing subject101.dat ... subject109.dat",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("prepared_pamap2"),
        help="Directory for prepared .npz feature files",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of deterministic perturbation replicates",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for subject in SUBJECTS:
        dat_path = args.data_dir / f"subject{subject}.dat"
        if not dat_path.exists():
            print(f"WARNING: {dat_path.name} not found; skipping.")
            continue
        prepare_subject(
            dat_path=dat_path,
            subject=subject,
            out_dir=args.output_dir,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
