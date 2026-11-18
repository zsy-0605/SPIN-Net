# src/utils/graph_utils.py
import os
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import h5py
import torch


_MEDIAL_WALL_MASK_CACHE = {}


def normalize_adj(adj):
    """Symmetrically normalize the adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def load_neighbors(path, field):
    """Load adjacency data."""
    try:
        # Try loading with SciPy.
        mat_data = sio.loadmat(path)
        return mat_data[field]['vertexNbors'][0][0]
    except NotImplementedError:
        # Use h5py for MATLAB v7.3 files.
        with h5py.File(path, 'r') as f:
            return np.array(f[field]['vertexNbors']).T
    except Exception as e:
        print(f"Error loading file {path}: {str(e)}")
        raise

def load_medial_wall_mask(path):
    """Load the fs_LR_32k valid-cortex mask (True denotes a valid vertex)."""
    if path in _MEDIAL_WALL_MASK_CACHE:
        return _MEDIAL_WALL_MASK_CACHE[path].copy()

    if not os.path.exists(path):
        raise FileNotFoundError(f"Medial-wall mask not found: {path}")

    try:
        mat_data = sio.loadmat(path)
        if "medial_mask" not in mat_data:
            raise KeyError(f"'medial_mask' not found in {path}")
        mask = np.asarray(mat_data["medial_mask"]).reshape(-1)
    except NotImplementedError:
        with h5py.File(path, "r") as f:
            if "medial_mask" not in f:
                raise KeyError(f"'medial_mask' not found in {path}")
            mask = np.asarray(f["medial_mask"]).reshape(-1)

    mask = mask.astype(bool, copy=False)
    _MEDIAL_WALL_MASK_CACHE[path] = mask
    return mask.copy()


def load_adjacency_matrices(profile_dir, valid_mask=None):
    """Load the binary, undirected, medial-wall-masked one-ring surface graph."""
    try:
        # Load mesh files from the profile directory.
        lh_path = os.path.join(profile_dir, 'lh_avg_mesh.mat')
        rh_path = os.path.join(profile_dir, 'rh_avg_mesh.mat')
        
        print(f"\nLoading mesh files from root directory:")
        print(f"Left hemisphere path: {lh_path}")
        print(f"Right hemisphere path: {rh_path}")
        
        # Check that the files exist.
        if not os.path.exists(lh_path) or not os.path.exists(rh_path):
            raise FileNotFoundError(f"Mesh files not found in {profile_dir}")
            
        # Load adjacency data for both hemispheres.
        lh_nbors = load_neighbors(lh_path, 'lh_avg_mesh')
        rh_nbors = load_neighbors(rh_path, 'rh_avg_mesh')
        
        if valid_mask is None:
            from ..config import MEDIAL_WALL_MASK_PATH

            valid_mask = load_medial_wall_mask(MEDIAL_WALL_MASK_PATH)

        # Build the sparse adjacency matrix.
        adj_mat = build_sparse_adjacency_matrix(
            lh_nbors,
            rh_nbors,
            valid_mask=valid_mask
        )
        
        return adj_mat
        
    except Exception as e:
        print(f"\nError in load_adjacency_matrices:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise

def build_sparse_adjacency_matrix(lh_nbors, rh_nbors, valid_mask=None):
    """Build the fixed binary one-ring graph with separate hemispheres."""
    n_vertices = lh_nbors.shape[1] + rh_nbors.shape[1]
    rows = []
    cols = []
    
    # Process the left hemisphere.
    for i in range(lh_nbors.shape[1]):
        neighbors = lh_nbors[:, i]
        valid_neighbors = neighbors[neighbors > 0] - 1
        rows.extend([i] * len(valid_neighbors))
        cols.extend(valid_neighbors)
    
    # Process the right hemisphere.
    offset = lh_nbors.shape[1]
    for i in range(rh_nbors.shape[1]):
        neighbors = rh_nbors[:, i]
        valid_neighbors = neighbors[neighbors > 0] - 1
        rows.extend([i + offset] * len(valid_neighbors))
        cols.extend(valid_neighbors + offset)
    
    # Create the sparse matrix.
    adj = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_vertices, n_vertices),
        dtype=np.float32
    )

    # The cortical mesh is undirected even if a source neighbor table is not
    # perfectly symmetric.
    adj = adj.maximum(adj.transpose()).tocsr()

    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid_mask.size != n_vertices:
            raise ValueError(
                f"Expected {n_vertices} mask entries, got {valid_mask.size}"
            )
        valid_diag = sp.diags(valid_mask.astype(np.float32))
        adj = valid_diag @ adj @ valid_diag

    adj.data[:] = 1.0
    adj.eliminate_zeros()
    return adj.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a SciPy sparse matrix to a PyTorch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
