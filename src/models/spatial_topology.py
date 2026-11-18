"""Cortical graph propagation and spatial probability regularizers."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base_modules import MemoryEfficientGraphConvolution
from ..config import (
    DISTANCE_THRESHOLD,
    GRAPH_HOPS,
    HIDDEN_DIM,
    LAMBDA_D,
    N_CLASSES,
    N_GCN_LAYERS,
    N_VERTICES_PER_HEMI,
    LOSS_EPSILON,
)


def _dense_binary_chunk(adj, start: int, size: int, device) -> torch.Tensor:
    """Extract the within-chunk portion of a binary one-ring graph."""
    if adj.is_sparse:
        coalesced = adj.coalesce()
        idx, values = coalesced.indices(), coalesced.values()
        keep = (
            (idx[0] >= start) & (idx[0] < start + size)
            & (idx[1] >= start) & (idx[1] < start + size)
        )
        local = idx[:, keep] - start
        dense = torch.zeros(size, size, device=device, dtype=torch.float32)
        if local.numel():
            dense[local[0], local[1]] = (values[keep] != 0).float().to(device)
    else:
        dense = (adj[start:start + size, start:start + size] != 0).to(
            device=device, dtype=torch.float32
        )
    dense.fill_diagonal_(0.0)
    return torch.maximum(dense, dense.transpose(0, 1))


def k_hop_reachability(one_ring: torch.Tensor, hops: int = GRAPH_HOPS) -> torch.Tensor:
    """Return ``I[sum_{r=0}^hops B^r > 0]``."""
    n = one_ring.shape[0]
    identity = torch.eye(n, device=one_ring.device, dtype=torch.float32)
    binary = (one_ring != 0).float()
    reach = identity.clone()
    frontier = identity
    for _ in range(hops):
        frontier = (frontier @ binary > 0).float()
        reach = torch.maximum(reach, frontier)
    return reach


def symmetric_normalize(adjacency: torch.Tensor) -> torch.Tensor:
    degree = adjacency.sum(dim=1)
    inv_sqrt = torch.where(degree > 0, degree.rsqrt(), torch.zeros_like(degree))
    return inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]


class SpatialTopologyModule(nn.Module):
    """Two residual LayerNorm GCN layers over normalized four-hop reachability."""

    def __init__(
        self,
        dim: int = HIDDEN_DIM,
        num_gcn_layers: int = N_GCN_LAYERS,
        distance_threshold: float = DISTANCE_THRESHOLD,
        lambda_d: float = LAMBDA_D,
        distance_matrices=None,
    ) -> None:
        super().__init__()
        self.gcn_layers = nn.ModuleList(
            MemoryEfficientGraphConvolution(dim, dim) for _ in range(num_gcn_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(num_gcn_layers))
        self.smoothness_loss = SmoothnessLoss()
        self.geodesic_constraint = GraphGeodesicCompactnessLoss(
            distance_threshold, lambda_d, distance_matrices
        )
        self._adjacency_chunk_cache: dict[tuple, torch.Tensor] = {}

    def forward(self, x, adj, chunk_info=None, valid_mask=None, return_loss=False):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if chunk_info is None:
            raise ValueError("chunk_info is required for graph propagation.")
        batch, padded_size, dim = x.shape
        start = int(chunk_info["start"])
        size = int(chunk_info.get("original_size", chunk_info["end"] - start))
        mask = valid_mask[:, :size].bool() if valid_mask is not None else torch.ones(
            batch, size, device=x.device, dtype=torch.bool
        )
        # The split mask is fixed, so all paired samples must agree.
        if not torch.equal(mask, mask[0:1].expand_as(mask)):
            raise ValueError("All samples must share the fixed cortical valid mask.")
        graph, valid_indices = self._extract_adjacency_chunk(
            adj, start, size, mask[0], x.device
        )
        valid_x = x[:, :size, :][:, valid_indices, :]
        for layer, norm in zip(self.gcn_layers, self.norms):
            valid_x = norm(layer(valid_x, graph, chunk_info) + valid_x)
        output = torch.zeros(batch, padded_size, dim, device=x.device, dtype=x.dtype)
        output[:, valid_indices, :] = valid_x
        return (output, {}) if return_loss else output

    def _extract_adjacency_chunk(self, adj, start, size, valid_mask, device):
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        key = (int(start), int(size), tuple(valid_indices.cpu().tolist()), str(device))
        if key not in self._adjacency_chunk_cache:
            one_ring = _dense_binary_chunk(adj, start, size, device)
            one_ring = one_ring[valid_indices][:, valid_indices]
            reach = k_hop_reachability(one_ring, GRAPH_HOPS)
            self._adjacency_chunk_cache[key] = symmetric_normalize(reach).to_sparse().coalesce()
        return self._adjacency_chunk_cache[key], valid_indices


class SmoothnessLoss(nn.Module):
    """One-ring graph-Laplacian loss on fused probabilities ``P``."""

    def forward(self, probabilities, adj, chunk_info, valid_mask=None):
        if probabilities.ndim == 2:
            probabilities = probabilities.unsqueeze(0)
        batch, _, classes = probabilities.shape
        start = int(chunk_info["start"])
        size = int(chunk_info.get("original_size", chunk_info["end"] - start))
        one_ring = _dense_binary_chunk(adj, start, size, probabilities.device)
        total = probabilities.sum() * 0.0
        for b in range(batch):
            mask = (
                valid_mask[b, :size].bool() if valid_mask is not None
                else torch.ones(size, device=probabilities.device, dtype=torch.bool)
            )
            graph = one_ring * mask[:, None] * mask[None, :]
            degree = graph.sum(dim=1)
            p = probabilities[b, :size] * mask[:, None]
            laplacian_p = degree[:, None] * p - graph @ p
            # This equals sum_(directed edges) ||P_i-P_j||^2 / 2.
            numerator = (p * laplacian_p).sum()
            denominator = classes * degree.sum().clamp_min(1.0)
            total = total + numerator / denominator
        return total / batch


class GraphGeodesicCompactnessLoss(nn.Module):
    """Penalize same-network probability assigned to distant within-hemi pairs."""

    def __init__(self, distance_threshold=DISTANCE_THRESHOLD, lambda_d=LAMBDA_D,
                 distance_matrices=None):
        super().__init__()
        if not isinstance(distance_matrices, dict) or not {"lh", "rh"} <= set(distance_matrices):
            raise ValueError("Precomputed lh/rh geodesic distance matrices are required.")
        self.distance_threshold = float(distance_threshold)
        self.lambda_d = float(lambda_d)
        self.distance_matrices = distance_matrices

    def forward(self, probabilities, adj, chunk_info, valid_mask=None):
        del adj
        if probabilities.ndim == 2:
            probabilities = probabilities.unsqueeze(0)
        if probabilities.shape[-1] != N_CLASSES:
            raise ValueError(f"Expected {N_CLASSES} network probabilities.")
        batch = probabilities.shape[0]
        start = int(chunk_info["start"])
        size = int(chunk_info.get("original_size", chunk_info["end"] - start))
        p = probabilities[:, :size]
        far = self._extract_distance_far_mask(start, size, p.device)
        total = p.sum() * 0.0
        for b in range(batch):
            sample_far = far
            if valid_mask is not None:
                mask = valid_mask[b, :size].bool()
                sample_far = far * (mask[:, None] & mask[None, :])
            denominator = sample_far.sum() + LOSS_EPSILON
            total = total + ((p[b] @ p[b].T) * sample_far).sum() / denominator
        return self.lambda_d * total / batch

    def _extract_distance_far_mask(self, start, size, device):
        end = start + size
        far = torch.zeros(size, size, device=device, dtype=torch.float32)
        lh_end = min(end, N_VERTICES_PER_HEMI)
        if start < lh_end:
            local_end = lh_end - start
            distance = self._slice(self.distance_matrices["lh"], start, lh_end, device)
            far[:local_end, :local_end] = (distance > self.distance_threshold).float()
        rh_start = max(start, N_VERTICES_PER_HEMI)
        if rh_start < end:
            local_start = rh_start - start
            hs, he = rh_start - N_VERTICES_PER_HEMI, end - N_VERTICES_PER_HEMI
            distance = self._slice(self.distance_matrices["rh"], hs, he, device)
            far[local_start:, local_start:] = (distance > self.distance_threshold).float()
        return far

    @staticmethod
    def _slice(matrix, start, end, device):
        value = matrix[start:end, start:end]
        return value.to(device=device, dtype=torch.float32) if torch.is_tensor(value) else torch.as_tensor(
            value, device=device, dtype=torch.float32
        )
