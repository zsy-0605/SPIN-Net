"""Training-only estimation and validation of fixed functional prototypes."""

from __future__ import annotations

import json
import fcntl
from pathlib import Path

import h5py
import numpy as np
import torch

from ..config import (
    CHUNK_SIZE,
    FUNCTIONAL_PROTOTYPES_PATH,
    FUNCTIONAL_PROTOTYPES_SIDECAR_PATH,
    N_CLASSES,
    N_ROIS,
    N_VERTICES,
    LOSS_EPSILON,
    PER_VERTEX_TOP_FRACTION,
    PROFILE_DIR,
    SESSION_IDS,
    STRICT_DATA_PROVENANCE,
)


def _profile_path(subject, session) -> Path:
    return Path(PROFILE_DIR) / f"sub{subject}" / f"sess{session}" / (
        f"sub{subject}_sess{session}_fs_LR_32k_roifs_LR_900.surf2surf_profile.mat"
    )


def _read_profile_chunk(dataset, start, end):
    if dataset.shape == (N_VERTICES, N_ROIS):
        return np.asarray(dataset[start:end, :], dtype=np.float64)
    if dataset.shape == (N_ROIS, N_VERTICES):
        return np.asarray(dataset[:, start:end], dtype=np.float64).T
    raise ValueError(
        f"Expected profile shape {(N_VERTICES, N_ROIS)} or {(N_ROIS, N_VERTICES)}; "
        f"got {dataset.shape}."
    )


def validate_prototype_provenance(sidecar_path, training_subjects, sessions=SESSION_IDS):
    path = Path(sidecar_path)
    if not path.exists():
        if STRICT_DATA_PROVENANCE:
            raise FileNotFoundError(f"Missing functional-prototype sidecar: {path}")
        return
    metadata = json.loads(path.read_text(encoding="utf-8"))
    declared_subjects = {str(value) for value in metadata.get("training_subjects", [])}
    expected_subjects = {str(value) for value in training_subjects}
    if declared_subjects != expected_subjects:
        raise ValueError("Functional prototypes were not estimated from exactly the training split.")
    if tuple(metadata.get("sessions", [])) != tuple(sessions):
        raise ValueError("Functional prototypes must use both configured sessions.")
    if metadata.get("group_weighting") != "probabilistic_theta":
        raise ValueError("Prototype sidecar must declare probabilistic_theta weighting.")
    if not np.isclose(
        float(metadata.get("per_vertex_top_fraction", -1.0)),
        PER_VERTEX_TOP_FRACTION,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Functional prototypes must be estimated from profiles with the "
            "configured retained-connection density."
        )


def load_functional_prototypes(
    training_subjects,
    path=FUNCTIONAL_PROTOTYPES_PATH,
    sidecar_path=FUNCTIONAL_PROTOTYPES_SIDECAR_PATH,
):
    validate_prototype_provenance(sidecar_path, training_subjects)
    prototypes = np.load(path)
    if prototypes.shape == (N_ROIS, N_CLASSES):
        prototypes = prototypes.T
    if prototypes.shape != (N_CLASSES, N_ROIS):
        raise ValueError(
            f"Functional prototypes must have shape {(N_CLASSES, N_ROIS)}; "
            f"got {prototypes.shape}."
        )
    if not np.isfinite(prototypes).all():
        raise ValueError("Functional prototypes contain non-finite values.")
    return torch.from_numpy(prototypes.astype(np.float32, copy=False))


def estimate_functional_prototypes(
    training_subjects,
    theta,
    output_path=FUNCTIONAL_PROTOTYPES_PATH,
    sidecar_path=FUNCTIONAL_PROTOTYPES_SIDECAR_PATH,
    sessions=SESSION_IDS,
):
    """Stream profiles and estimate one fixed 1,483-D prototype per network."""
    theta = torch.as_tensor(theta, dtype=torch.float64).cpu().numpy()
    if theta.shape != (N_VERTICES, N_CLASSES):
        raise ValueError(f"Theta must have shape {(N_VERTICES, N_CLASSES)}.")
    numerator = np.zeros((N_CLASSES, N_ROIS), dtype=np.float64)
    denominator = np.zeros(N_CLASSES, dtype=np.float64)
    for subject in training_subjects:
        for session in sessions:
            path = _profile_path(subject, session)
            with h5py.File(path, "r") as handle:
                if "profile_mat" not in handle:
                    raise KeyError(f"profile_mat is missing from {path}")
                dataset = handle["profile_mat"]
                for start in range(0, N_VERTICES, CHUNK_SIZE):
                    end = min(start + CHUNK_SIZE, N_VERTICES)
                    weights = theta[start:end]
                    profile = _read_profile_chunk(dataset, start, end)
                    if not np.isfinite(profile).all():
                        raise ValueError(f"Profile contains non-finite values: {path}")
                    if np.any(np.abs(profile) > 1.0 + 1e-6):
                        raise ValueError(
                            "Functional prototypes require sparse weighted "
                            f"Pearson-correlation profiles in [-1, 1]: {path}"
                        )
                    numerator += weights.T @ profile
                    denominator += weights.sum(axis=0)
    if np.any(denominator <= 0):
        raise ValueError("At least one network has no training-set prototype support.")
    prototypes = (
        numerator / (denominator[:, None] + LOSS_EPSILON)
    ).astype(np.float32)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, prototypes)
    sidecar = {
        "training_subjects": list(training_subjects),
        "sessions": list(sessions),
        "group_weighting": "probabilistic_theta",
        "per_vertex_top_fraction": PER_VERTEX_TOP_FRACTION,
        "roi_features": N_ROIS,
        "networks": N_CLASSES,
    }
    Path(sidecar_path).write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return torch.from_numpy(prototypes)


def load_or_estimate_functional_prototypes(training_subjects, theta):
    path = Path(FUNCTIONAL_PROTOTYPES_PATH)
    if path.exists():
        return load_functional_prototypes(training_subjects)
    # Slurm arrays may start several configurations simultaneously. A shared
    # advisory lock ensures the training-only prototype is estimated once.
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if path.exists():
            return load_functional_prototypes(training_subjects)
        return estimate_functional_prototypes(training_subjects, theta)
