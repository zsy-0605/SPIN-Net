"""Reproducible construction of manuscript inputs from run-level arrays."""

from __future__ import annotations

import numpy as np


def vertex_roi_pearson_profiles(vertex_time_series, roi_time_series, valid_mask):
    """Run-level Pearson profiles from aligned [time,vertex] and [time,ROI] data.

    ROI time series must use the same verified 1,483-seed order in every cohort.
    Surface resampling and seed selection precede this step.
    """
    vertices = np.asarray(vertex_time_series, dtype=np.float64)
    seeds = np.asarray(roi_time_series, dtype=np.float64)
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if vertices.ndim != 2 or seeds.ndim != 2 or vertices.shape[0] != seeds.shape[0] or vertices.shape[1] != mask.size or vertices.shape[0] < 2:
        raise ValueError("Time series must share time points and the fixed vertex mask.")
    if not np.isfinite(vertices[:,mask]).all() or not np.isfinite(seeds).all():
        raise ValueError("Valid time series must be finite.")
    x = vertices[:,mask] - vertices[:,mask].mean(axis=0)
    y = seeds - seeds.mean(axis=0)
    xnorm, ynorm = np.linalg.norm(x,axis=0), np.linalg.norm(y,axis=0)
    if np.any(xnorm==0) or np.any(ynorm==0):
        raise ValueError("Pearson correlation is undefined for constant time series.")
    result = np.zeros((mask.size,seeds.shape[1]),dtype=np.float32)
    result[mask] = np.clip((x/xnorm).T @ (y/ynorm),-1,1)
    return result


def build_session_weighted_profile(
    run_profiles: np.ndarray,
    valid_mask: np.ndarray,
    *,
    top_fraction: float = 0.10,
) -> np.ndarray:
    """Average two runs and retain each vertex's strongest correlations.

    Parameters
    ----------
    run_profiles:
        Array with shape ``[2, vertices, ROI seeds]`` containing run-level
        Pearson correlations for one imaging session.
    valid_mask:
        Boolean valid-cortex vector. Invalid rows are always zero and do not
        enter threshold estimation.
    top_fraction:
        Fraction of ROI connections retained independently for every valid
        vertex. Ties are resolved deterministically by ascending ROI index.
        Selected entries retain their averaged Pearson correlation values;
        all remaining entries are set to zero.
    """
    profiles = np.asarray(run_profiles)
    if profiles.ndim != 3 or profiles.shape[0] != 2:
        raise ValueError("run_profiles must have shape [2, vertices, ROI seeds].")
    if not np.issubdtype(profiles.dtype, np.number) or not np.all(
        np.isfinite(profiles)
    ):
        raise ValueError("run_profiles must contain finite numeric values.")
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if mask.size != profiles.shape[1]:
        raise ValueError("valid_mask must contain one entry per cortical vertex.")
    if not bool(mask.any()):
        raise ValueError("valid_mask contains no valid cortical vertices.")
    if not 0.0 < float(top_fraction) < 1.0:
        raise ValueError("top_fraction must lie strictly between 0 and 1.")

    session_mean = profiles.astype(np.float64, copy=False).mean(axis=0)
    valid_values = session_mean[mask]
    n_rois = session_mean.shape[1]
    n_keep = int(round(n_rois * float(top_fraction)))
    n_keep = min(max(n_keep, 1), n_rois)

    # Stable descending sort resolves equal correlations by ascending ROI
    # index and makes the retained set reproducible.
    selected = np.argsort(-valid_values, axis=1, kind="stable")[:, :n_keep]
    retained = np.zeros(valid_values.shape, dtype=np.float32)
    rows = np.arange(valid_values.shape[0])[:, None]
    retained[rows, selected] = valid_values[rows, selected]

    output = np.zeros(session_mean.shape, dtype=np.float32)
    output[mask] = retained
    return output


def construct_probabilistic_group_prior(
    training_labels: np.ndarray,
    valid_mask: np.ndarray,
    *,
    n_networks: int = 17,
) -> np.ndarray:
    """Construct ``Theta[v,k]`` from training-subject hard parcellations."""
    labels = np.asarray(training_labels)
    if labels.ndim != 2 or labels.shape[0] < 1:
        raise ValueError("training_labels must have shape [subjects, vertices].")
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if mask.size != labels.shape[1]:
        raise ValueError("valid_mask must contain one entry per cortical vertex.")
    valid_labels = labels[:, mask]
    if not np.issubdtype(valid_labels.dtype, np.number):
        raise TypeError("training_labels must be numeric.")
    if not np.all(np.isfinite(valid_labels)) or not np.all(
        valid_labels == np.floor(valid_labels)
    ):
        raise ValueError("Valid-cortex training labels must be finite integers.")
    if np.any((valid_labels < 1) | (valid_labels > n_networks)):
        raise ValueError(f"Valid-cortex labels must lie in 1..{n_networks}.")

    theta = np.zeros((labels.shape[1], n_networks), dtype=np.float32)
    for network_index in range(n_networks):
        theta[mask, network_index] = np.mean(
            valid_labels == network_index + 1, axis=0, dtype=np.float64
        )
    return theta
