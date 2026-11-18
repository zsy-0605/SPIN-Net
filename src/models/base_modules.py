"""Reusable neural-network operators."""

from __future__ import annotations

import torch
import torch.nn as nn


class MemoryEfficientGraphConvolution(nn.Module):
    """Compute ``A_hat (H W + b)`` for a batched vertex representation."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, chunk_info=None):
        del chunk_info
        if x.ndim != 3:
            raise ValueError(f"Expected graph features [B,V,D], got {tuple(x.shape)}")
        transformed = self.linear(x.float())
        if adjacency.shape != (x.shape[1], x.shape[1]):
            raise ValueError(
                f"Graph shape {tuple(adjacency.shape)} does not match {x.shape[1]} vertices."
            )
        if adjacency.is_sparse:
            graph = adjacency.coalesce().to(device=x.device, dtype=torch.float32)
            return torch.stack([
                torch.sparse.mm(graph, sample) for sample in transformed
            ])
        graph = adjacency.to(device=x.device, dtype=torch.float32)
        return torch.matmul(graph.unsqueeze(0), transformed)
