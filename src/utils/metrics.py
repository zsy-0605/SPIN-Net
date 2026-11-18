import torch
import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from ..config import *
from .graph_utils import load_medial_wall_mask
import torch.nn as nn
import traceback


def compute_connectivity_homogeneity(
    labels,
    connectivity_profiles,
    valid_mask=None,
    return_network_scores=False
):
    """Compute whole-cortex resting-state homogeneity.

    For each subject and network, this is the mean pairwise Pearson correlation
    between the connectivity profiles of all vertices assigned to that network.
    The subject-level score is the unweighted mean of all 17 network scores.
    """
    if isinstance(labels, dict):
        model_output = labels
        labels = model_output.get('labels')
        if valid_mask is None:
            valid_mask = model_output.get('valid_mask')
    if labels is None or connectivity_profiles is None:
        raise ValueError(
            "Whole-cortex labels and matching vertex connectivity profiles are required."
        )

    labels = torch.as_tensor(labels)
    connectivity_profiles = torch.as_tensor(connectivity_profiles)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if connectivity_profiles.dim() == 2:
        connectivity_profiles = connectivity_profiles.unsqueeze(0)
    if labels.dim() != 2 or connectivity_profiles.dim() != 3:
        raise ValueError(
            "Expected labels [B,V] and connectivity profiles [B,V,R]."
        )
    if labels.shape[:2] != connectivity_profiles.shape[:2]:
        raise ValueError(
            "Labels and connectivity profiles must share batch and vertex "
            f"dimensions; got {tuple(labels.shape)} and "
            f"{tuple(connectivity_profiles.shape)}."
        )

    batch_size, n_vertices = labels.shape
    device = connectivity_profiles.device
    profiles = connectivity_profiles.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.long)

    if valid_mask is None:
        if n_vertices == N_VERTICES:
            valid_mask = torch.from_numpy(
                load_medial_wall_mask(MEDIAL_WALL_MASK_PATH)
            )
        else:
            valid_mask = labels != 0
    valid_mask = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)
    if valid_mask.dim() == 1:
        valid_mask = valid_mask.unsqueeze(0).expand(batch_size, -1)
    if valid_mask.shape != labels.shape:
        raise ValueError(
            f"Expected valid mask shape {tuple(labels.shape)}, "
            f"got {tuple(valid_mask.shape)}."
        )

    subject_scores = []
    all_network_scores = []
    for subject_index in range(batch_size):
        network_scores = []
        for network_id in range(1, N_CLASSES + 1):
            network_mask = (
                valid_mask[subject_index]
                & (labels[subject_index] == network_id)
            )
            n_network_vertices = int(network_mask.sum().item())
            if n_network_vertices < 2:
                raise ValueError(
                    f"Subject {subject_index}, network {network_id} has "
                    f"{n_network_vertices} valid vertices; pairwise Pearson "
                    "homogeneity requires at least two."
                )

            network_profiles = profiles[subject_index, network_mask]
            network_score = _mean_pairwise_pearson(
                network_profiles,
                context=f"subject {subject_index}, network {network_id}"
            )
            network_scores.append(network_score)

        network_scores = torch.stack(network_scores)
        all_network_scores.append(network_scores)
        subject_scores.append(network_scores.mean())

    overall_score = torch.stack(subject_scores).mean()
    if return_network_scores:
        return {
            'overall': overall_score,
            'subject_scores': torch.stack(subject_scores),
            'network_scores': torch.stack(all_network_scores)
        }
    return overall_score


def compute_connectivity_homogeneity_from_file(
    labels,
    profile_path,
    valid_mask=None,
    dataset_name='profile_mat'
):
    """Memory-efficient whole-cortex homogeneity from one HDF5 profile file."""
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1).cpu()
    if labels.numel() != N_VERTICES:
        raise ValueError(
            f"Expected {N_VERTICES} whole-cortex labels, got {labels.numel()}."
        )
    if valid_mask is None:
        valid_mask = torch.from_numpy(
            load_medial_wall_mask(MEDIAL_WALL_MASK_PATH)
        )
    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool).reshape(-1).cpu()
    if valid_mask.numel() != N_VERTICES:
        raise ValueError(
            f"Expected {N_VERTICES} valid-mask entries, got {valid_mask.numel()}."
        )

    network_scores = []
    with h5py.File(profile_path, 'r') as profile_file:
        if dataset_name not in profile_file:
            raise KeyError(f"'{dataset_name}' not found in {profile_path}")
        profile_dataset = profile_file[dataset_name]
        vertex_major = profile_dataset.shape == (N_VERTICES, N_ROIS)
        roi_major = profile_dataset.shape == (N_ROIS, N_VERTICES)
        if not vertex_major and not roi_major:
            raise ValueError(
                f"Unexpected profile shape {profile_dataset.shape} in {profile_path}."
            )

        for network_id in range(1, N_CLASSES + 1):
            network_indices = torch.nonzero(
                valid_mask & (labels == network_id),
                as_tuple=False
            ).squeeze(1).cpu().numpy()
            if network_indices.size < 2:
                raise ValueError(
                    f"Network {network_id} has {network_indices.size} valid vertices."
                )
            if vertex_major:
                network_profiles = np.asarray(
                    profile_dataset[network_indices, :]
                )
            else:
                network_profiles = np.asarray(
                    profile_dataset[:, network_indices]
                ).T
            network_scores.append(
                _mean_pairwise_pearson(
                    torch.from_numpy(network_profiles).to(torch.float32),
                    context=f"network {network_id} in {profile_path}"
                )
            )

    return torch.stack(network_scores).mean()


def _mean_pairwise_pearson(network_profiles, context):
    """Exact mean off-diagonal Pearson correlation without an n-by-n matrix."""
    centered = (
        network_profiles
        - network_profiles.mean(dim=1, keepdim=True)
    )
    norms = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
    if torch.any(norms <= torch.finfo(centered.dtype).eps):
        raise ValueError(
            f"{context} contains a constant connectivity profile with "
            "undefined Pearson correlation."
        )
    normalized = centered / norms

    summed_profile = normalized.sum(dim=0)
    off_diagonal_sum = (
        summed_profile.square().sum()
        - normalized.square().sum()
    )
    n_vertices = network_profiles.size(0)
    denominator = n_vertices * (n_vertices - 1)
    return (off_diagonal_sum / denominator).clamp(-1.0, 1.0)


def compute_vertexwise_connectivity_homogeneity_from_file(
    labels,
    profile_path,
    valid_mask=None,
    dataset_name="profile_mat",
):
    """Return each vertex's mean correlation to its assigned network.

    This is the vertex-resolved counterpart of whole-network homogeneity and
    supports the chunk-boundary analysis. Invalid vertices are returned as
    ``nan``. The mean over vertices of a network equals that network's mean
    off-diagonal pairwise Pearson correlation.
    """
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1).cpu()
    if labels.numel() != N_VERTICES:
        raise ValueError(f"Expected {N_VERTICES} labels, got {labels.numel()}.")
    if valid_mask is None:
        valid_mask = torch.from_numpy(
            load_medial_wall_mask(MEDIAL_WALL_MASK_PATH)
        )
    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool).reshape(-1).cpu()
    if valid_mask.numel() != N_VERTICES:
        raise ValueError(f"Expected {N_VERTICES} valid-mask entries.")

    output = torch.full((N_VERTICES,), float("nan"), dtype=torch.float32)
    with h5py.File(profile_path, "r") as profile_file:
        if dataset_name not in profile_file:
            raise KeyError(f"{dataset_name!r} not found in {profile_path}")
        dataset = profile_file[dataset_name]
        vertex_major = dataset.shape == (N_VERTICES, N_ROIS)
        roi_major = dataset.shape == (N_ROIS, N_VERTICES)
        if not vertex_major and not roi_major:
            raise ValueError(
                f"Unexpected profile shape {dataset.shape} in {profile_path}."
            )
        for network_id in range(1, N_CLASSES + 1):
            indices = torch.nonzero(
                valid_mask & (labels == network_id), as_tuple=False
            ).squeeze(1)
            if indices.numel() < 2:
                raise ValueError(
                    f"Network {network_id} has {indices.numel()} valid vertices."
                )
            numpy_indices = indices.numpy()
            if vertex_major:
                values = np.asarray(dataset[numpy_indices, :])
            else:
                values = np.asarray(dataset[:, numpy_indices]).T
            profiles = torch.from_numpy(values).to(torch.float32)
            centered = profiles - profiles.mean(dim=1, keepdim=True)
            norms = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
            if torch.any(norms <= torch.finfo(centered.dtype).eps):
                raise ValueError(
                    f"Network {network_id} in {profile_path} contains a "
                    "constant connectivity profile."
                )
            normalized = centered / norms
            network_sum = normalized.sum(dim=0, keepdim=True)
            per_vertex = (
                normalized * (network_sum - normalized)
            ).sum(dim=1) / (indices.numel() - 1)
            output[indices] = per_vertex
    return output



def compute_spatial_smoothness(features, adj_matrix, chunk_info=None):
    """Evaluate the spatial smoothness of features."""
    try:
        # Ensure features are a 3D tensor.
        features = features.to(dtype=torch.float32)
        if len(features.shape) == 2:
            features = features.unsqueeze(0)
            
        batch_size, chunk_size, hidden_dim = features.shape
        
        # Check chunk_info.
        if chunk_info is None:
            chunk_info = {
                'start': 0,
                'end': chunk_size,
                'original_size': chunk_size
            }
            
        start_idx = chunk_info['start']
        original_size = chunk_info['original_size']
        
        # Use only valid data.
        features = features[:, :original_size]

        # Extract the adjacency matrix for this chunk.
        if adj_matrix.is_sparse:
            indices = adj_matrix._indices()
            values = adj_matrix._values()
            
            mask = ((indices[0] >= start_idx) & 
                   (indices[0] < start_idx + original_size) &
                   (indices[1] >= start_idx) & 
                   (indices[1] < start_idx + original_size))
                   
            chunk_indices = indices[:, mask].clone()
            chunk_indices = chunk_indices - start_idx
            chunk_values = values[mask].to(dtype=torch.float32)
            
            adj_chunk = torch.sparse_coo_tensor(
                chunk_indices,
                chunk_values,
                size=(original_size, original_size),
                device=features.device,
                dtype=torch.float32
            ).coalesce()
        else:
            adj_chunk = adj_matrix[
                start_idx:start_idx + original_size, 
                start_idx:start_idx + original_size
            ].to(dtype=torch.float32)

        # Compute the degree matrix.
        if adj_chunk.is_sparse:
            D = torch.sparse.sum(adj_chunk, dim=1).to_dense()
        else:
            D = adj_chunk.sum(dim=1)
        D = torch.diag(D)
        
        # Construct the graph Laplacian.
        L = D - adj_chunk if not adj_chunk.is_sparse else None
        
        total_smoothness = 0.0
        
        for b in range(batch_size):
            # Compute smoothness for each sample in the batch.
            batch_features = features[b]  # [original_size, hidden_dim]
            
            if adj_chunk.is_sparse:
                Lx = torch.sparse.mm(adj_chunk, batch_features)
                Lx = D.mm(batch_features) - Lx
            else:
                Lx = L.mm(batch_features)
                
            smoothness = (batch_features * Lx).sum() / (original_size * original_size)
            total_smoothness += smoothness.item()

        avg_smoothness = total_smoothness / batch_size
        return torch.tensor(avg_smoothness, device=features.device)
        
    except Exception as e:
        print(f"\nError in compute_spatial_smoothness:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        return torch.tensor(0.0, device=features.device)

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, features, adj, chunk_info=None):
        """Compute the smoothness loss.
        Args:
            features: [batch_size, chunk_size, dim] or [chunk_size, dim]
            adj: Adjacency matrix
            chunk_info: Chunk metadata
        """
        try:
            # Ensure features are a 3D tensor.
            if len(features.shape) == 2:
                features = features.unsqueeze(0)
                
            # Ensure chunk_info is available.
            if chunk_info is None:
                chunk_info = {
                    'start': 0,
                    'original_size': features.size(1)
                }
                
            return compute_spatial_smoothness(features, adj, chunk_info)
            
        except Exception as e:
            print(f"\nError in smoothness_loss:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Features shape: {features.shape}")
            print(f"Chunk info: {chunk_info}")
            print("Full traceback:")
            traceback.print_exc()
            return torch.tensor(0.0, device=features.device)
        

def compute_laplacian_matrix(adj_matrix, start_idx, size, device):
    """Compute the graph Laplacian."""
    if adj_matrix.is_sparse:
        indices = adj_matrix._indices()
        values = adj_matrix._values()
        
        # Select valid edges within the current chunk.
        mask = ((indices[0] >= start_idx) & 
               (indices[0] < start_idx + size) &
               (indices[1] >= start_idx) & 
               (indices[1] < start_idx + size))
               
        chunk_indices = indices[:, mask].clone()
        chunk_indices = chunk_indices - start_idx
        chunk_values = values[mask].to(dtype=torch.float32)
        
        # Construct the adjacency matrix for this chunk.
        adj_chunk = torch.sparse_coo_tensor(
            chunk_indices,
            chunk_values,
            size=(size, size),
            device=device,
            dtype=torch.float32
        ).coalesce()
        
        # Compute the degree matrix.
        D = torch.sparse.sum(adj_chunk, dim=1).to_dense()
        D = torch.diag(D)
        
        # Construct the graph Laplacian L = D - A.
        L = torch.sparse_coo_tensor(
            adj_chunk._indices(),
            -adj_chunk._values(),
            size=adj_chunk.size(),
            device=device,
            dtype=torch.float32
        )
        L = L + torch.sparse_coo_tensor(
            torch.arange(size, device=device).repeat(2, 1),
            D.diagonal(),
            size=L.size(),
            device=device,
            dtype=torch.float32
        ).coalesce()
        
    else:
        adj_chunk = adj_matrix[
            start_idx:start_idx + size, 
            start_idx:start_idx + size
        ].to(dtype=torch.float32)
        D = torch.diag(adj_chunk.sum(dim=1))
        L = D - adj_chunk
        
    return L


def evaluate_network_structure(features, n_clusters=N_CLASSES):
    """Evaluate network structure.
    
    Args:
        features: [batch_size, n_vertices, n_features] Feature tensor
        n_clusters: Number of clusters
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        # Prepare the data.
        features_np = features.detach().cpu().numpy()
        features_np = features_np.reshape(-1, features_np.shape[-1])
        
        # Perform clustering.
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(features_np)
        
        # Compute evaluation metrics.
        silhouette = metrics.silhouette_score(
            features_np, 
            labels,
            sample_size=10000  # Use subsampling to reduce computation.
        )
        
        calinski = metrics.calinski_harabasz_score(
            features_np,
            labels
        )
        
        davies = metrics.davies_bouldin_score(
            features_np,
            labels
        )
        
        return {
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies
        }
        
    except Exception as e:
        print(f"Error in evaluate_network_structure: {str(e)}")
        return {
            'silhouette': 0.0,
            'calinski_harabasz': 0.0,
            'davies_bouldin': float('inf')
        }
