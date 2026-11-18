import os

import h5py
import numpy as np

from ..config import (
    LH_GEODESIC_DISTANCE_PATH,
    N_VERTICES_PER_HEMI,
    RH_GEODESIC_DISTANCE_PATH,
)


def load_hemisphere_distance_matrices(
    lh_path=LH_GEODESIC_DISTANCE_PATH,
    rh_path=RH_GEODESIC_DISTANCE_PATH,
):
    """Load precomputed left/right hemisphere vertex distance matrices.

    The preferred storage format is ``.npy`` because it can be memory-mapped and
    chunk-sliced without loading the full 32k x 32k matrix into RAM.
    """
    if not lh_path or not rh_path:
        raise ValueError(
            "Both hemisphere-specific geodesic distance paths must be configured."
        )

    if not os.path.exists(lh_path) or not os.path.exists(rh_path):
        missing = [
            path for path in (lh_path, rh_path)
            if not os.path.exists(path)
        ]
        raise FileNotFoundError(
            "The geodesic compactness loss requires "
            "precomputed hemisphere distance matrices. Missing: "
            + ", ".join(missing)
        )

    lh_dist = _load_distance_matrix(lh_path)
    rh_dist = _load_distance_matrix(rh_path)
    _validate_distance_matrix(lh_dist, "left", lh_path)
    _validate_distance_matrix(rh_dist, "right", rh_path)
    print("Loaded precomputed hemisphere geodesic distance matrices")
    print(f"Left hemisphere distance path: {lh_path}")
    print(f"Right hemisphere distance path: {rh_path}")
    return {"lh": lh_dist, "rh": rh_dist}


def _load_distance_matrix(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path, mmap_mode="r")
    if ext == ".npz":
        data = np.load(path)
        for key in ("distance", "dist", "geodesic", "D", "arr_0"):
            if key in data:
                return data[key]
        raise KeyError(f"No distance matrix key found in {path}")
    if ext == ".mat":
        return _load_mat_distance(path)
    raise ValueError(f"Unsupported distance matrix format: {path}")


def _load_mat_distance(path):
    try:
        import scipy.io as sio

        mat = sio.loadmat(path)
        for key in ("distance", "dist", "geodesic", "D", "lh_dist", "rh_dist"):
            if key in mat:
                return mat[key]
        candidates = [
            value for key, value in mat.items()
            if not key.startswith("__") and getattr(value, "ndim", 0) == 2
        ]
        if candidates:
            return candidates[0]
    except (NotImplementedError, ValueError):
        if not h5py.is_hdf5(path):
            raise

    with h5py.File(path, "r") as f:
        for key in ("distance", "dist", "geodesic", "D", "lh_dist", "rh_dist"):
            if key in f:
                return HDF5DistanceMatrix(path, key, f[key].shape)
        candidates = [key for key in f.keys() if len(f[key].shape) == 2]
        if candidates:
            return HDF5DistanceMatrix(path, candidates[0], f[candidates[0]].shape)
    raise KeyError(f"No 2D distance matrix found in {path}")


class HDF5DistanceMatrix:
    """Slice a large v7.3 matrix without retaining a closed HDF5 dataset."""

    def __init__(self, path, dataset, shape):
        self.path, self.dataset, self.shape = str(path), dataset, tuple(shape)

    def __getitem__(self, key):
        with h5py.File(self.path, 'r') as handle:
            return np.asarray(handle[self.dataset][key])


def _validate_distance_matrix(distance_matrix, hemisphere, path):
    shape = distance_matrix.shape
    expected = (N_VERTICES_PER_HEMI, N_VERTICES_PER_HEMI)
    if shape != expected:
        raise ValueError(
            f"{hemisphere} hemisphere distance matrix should have shape "
            f"{expected}, got {shape} from {path}"
        )
