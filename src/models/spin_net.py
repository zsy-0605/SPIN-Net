"""Manuscript-aligned SPIN-Net model and loss construction."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from .adaptive_prior_classification import (
    AdaptivePriorClassificationModule,
    SPINNetLoss,
)
from .individualized_representation import IndividualizedRepresentationModule
from .spatial_topology import SpatialTopologyModule
from ..config import (
    CHUNK_SIZE,
    COMPUTE_DEVICE,
    HIDDEN_DIM,
    LOSS_EPSILON,
    N_CLASSES,
    N_ROIS,
    N_VALID_VERTICES,
    N_VERTICES,
    PROFILE_DIR,
)
from ..utils.distance_utils import load_hemisphere_distance_matrices
from ..utils.graph_utils import load_adjacency_matrices


SPINNET_VARIANTS = {"full", "without_in", "without_s", "without_p", "fixed_alpha"}


class SPINNet(nn.Module):
    """Individualization, spatial propagation, and adaptive prior fusion."""

    def __init__(
        self,
        input_dim: int = N_ROIS,
        hidden_dim: int = HIDDEN_DIM,
        num_classes: int = N_CLASSES,
        variant: str = "full",
        fixed_alpha: float | None = None,
        adjacency=None,
        distance_matrices=None,
        functional_prototypes: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if variant not in SPINNET_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; expected {sorted(SPINNET_VARIANTS)}")
        if variant == "fixed_alpha" and fixed_alpha is None:
            raise ValueError("fixed_alpha variant requires a fusion weight.")
        if variant != "fixed_alpha" and fixed_alpha is not None:
            raise ValueError("fixed_alpha is only valid for the fixed_alpha variant.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.variant = variant
        self.fixed_alpha = None if fixed_alpha is None else float(fixed_alpha)
        self.architecture_version = 3

        if variant == "without_in":
            self.individualized_representation = None
            self.input_adapter = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            )
        else:
            self.individualized_representation = IndividualizedRepresentationModule(
                input_dim=input_dim, hidden_dim=hidden_dim
            )
            self.input_adapter = None

        if variant == "without_s":
            self.spatial_topology = None
        else:
            if distance_matrices is None:
                distance_matrices = load_hemisphere_distance_matrices()
            self.spatial_topology = SpatialTopologyModule(
                dim=hidden_dim, distance_matrices=distance_matrices
            )

        if variant == "without_p":
            self.adaptive_prior_classification = None
            self.individualized_head = nn.Linear(hidden_dim, num_classes)
        else:
            self.adaptive_prior_classification = AdaptivePriorClassificationModule(
                dim=hidden_dim,
                num_classes=num_classes,
                fixed_alpha=self.fixed_alpha,
            )
            self.individualized_head = None

        if adjacency is None:
            adjacency = load_adjacency_matrices(PROFILE_DIR)
        self.register_buffer("adj_matrix", self._to_sparse_tensor(adjacency).coalesce())
        if functional_prototypes is None:
            prototype_tensor = torch.empty(0, input_dim, dtype=torch.float32)
        else:
            prototype_tensor = torch.as_tensor(functional_prototypes, dtype=torch.float32)
            self._validate_prototypes(prototype_tensor)
        self.register_buffer("functional_prototypes", prototype_tensor)
        self.apply(self._init_weights)

    @staticmethod
    def _to_sparse_tensor(matrix):
        if torch.is_tensor(matrix):
            return matrix.to_sparse() if not matrix.is_sparse else matrix
        coo = sp.coo_matrix(matrix)
        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
        values = torch.from_numpy(coo.data.astype(np.float32, copy=False))
        return torch.sparse_coo_tensor(indices, values, coo.shape)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_prototypes(self, prototypes):
        if tuple(prototypes.shape) != (self.num_classes, self.input_dim):
            raise ValueError(
                f"Functional prototypes must have shape {(self.num_classes, self.input_dim)}; "
                f"got {tuple(prototypes.shape)}"
            )
        if not torch.isfinite(prototypes).all():
            raise ValueError("Functional prototypes contain non-finite values.")

    def forward(self, data, *, compute_losses: bool | None = None):
        """Run SPIN-Net on one cortical chunk.

        Training/validation batches contain paired sessions and set
        ``compute_losses=True``.  At inference, ``compute_losses=False``
        permits a single session to be processed independently, as described
        in the manuscript.  The default follows ``module.training``.
        """
        if compute_losses is None:
            compute_losses = self.training
        device = next(self.parameters()).device
        profiles = data["profile"].to(device=device, dtype=torch.float32)
        valid_mask = data.get("valid_mask")
        if not torch.is_tensor(valid_mask):
            raise ValueError("valid_mask is required to exclude invalid cortex.")
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)
        metadata = data.get("chunk_info")
        batch, chunk_size, rois = profiles.shape
        if rois != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} ROI features, got {rois}.")
        common_chunk = self._validate_chunk_batch(
            metadata, batch, require_cross_session_pairs=compute_losses
        )

        if self.individualized_representation is None:
            h_ind = self.input_adapter(profiles)
            h_ind = h_ind * valid_mask.unsqueeze(-1)
            losses = {}
        else:
            representation = self.individualized_representation(
                profiles, metadata, valid_mask, return_loss=compute_losses
            )
            if compute_losses:
                h_ind, losses = representation
            else:
                h_ind, losses = representation, {}

        if self.spatial_topology is None:
            h = h_ind
        else:
            h = self.spatial_topology(
                h_ind, self.adj_matrix, common_chunk, valid_mask,
                return_loss=False
            )

        if self.adaptive_prior_classification is None:
            individualized_logits = self.individualized_head(h)
            q = torch.softmax(individualized_logits, dim=-1) * valid_mask.unsqueeze(-1)
            p = q
            gate = None
        else:
            theta = data.get("theta")
            if torch.is_tensor(theta):
                theta = theta.to(device=device, dtype=torch.float32)
            fusion_result = self.adaptive_prior_classification(
                h, theta, common_chunk, valid_mask,
                return_loss=compute_losses
            )
            if compute_losses:
                fusion, prior_losses = fusion_result
            else:
                fusion, prior_losses = fusion_result, {}
            individualized_logits = fusion["individualized_logits"]
            q = fusion["individualized_probabilities"]
            p = fusion["probabilities"]
            gate = fusion["gate"]
            losses.update(prior_losses)

        if compute_losses:
            prototypes = data.get("functional_prototypes")
            if prototypes is None:
                prototypes = self.functional_prototypes
            prototypes = torch.as_tensor(
                prototypes, device=device, dtype=torch.float32
            )
            self._validate_prototypes(prototypes)
            losses["functional_assignment_loss"] = self._functional_assignment_loss(
                profiles, prototypes, q, p, valid_mask
            )
            losses["temporal_consistency_loss"] = self._temporal_consistency_loss(
                q, gate, metadata, valid_mask
            )
            if self.spatial_topology is not None:
                smooth = self.spatial_topology.smoothness_loss(
                    p, self.adj_matrix, common_chunk, valid_mask
                )
                geodesic = self.spatial_topology.geodesic_constraint(
                    p, self.adj_matrix, common_chunk, valid_mask
                )
                losses.update(
                    smoothness_loss=smooth,
                    geodesic_loss=geodesic,
                    total_spatial_loss=smooth + geodesic,
                )

        labels = (p.argmax(dim=-1) + 1).masked_fill(~valid_mask, 0)
        output_gate = (
            gate.squeeze(-1) if gate is not None
            else torch.zeros_like(valid_mask, dtype=p.dtype)
        )
        # Mean valid count over all 33 chunks is exact for a complete epoch.
        mean_valid = N_VALID_VERTICES / math.ceil(N_VERTICES / CHUNK_SIZE)
        chunk_weight = valid_mask.sum(dim=1).float().mean() / mean_valid
        return {
            "features": h,
            "individualized_logits": individualized_logits,
            "individualized_probabilities": q,
            "probabilities": p,
            # Kept for consumers that expect logits; log(P) preserves argmax.
            "logits": p.clamp_min(LOSS_EPSILON).log(),
            "labels": labels,
            "gate": output_gate,
            "valid_mask": valid_mask,
            "chunk_info": common_chunk,
            "batch_chunk_info": metadata,
            "variant": self.variant,
            "fixed_alpha": self.fixed_alpha,
            "chunk_weight": chunk_weight,
            "losses": losses,
        }

    @staticmethod
    def _functional_assignment_loss(profiles, prototypes, q, p, valid_mask):
        x = torch.nn.functional.normalize(profiles, dim=-1, eps=LOSS_EPSILON)
        mu = torch.nn.functional.normalize(prototypes, dim=-1, eps=LOSS_EPSILON)
        distance = 1.0 - torch.einsum("bvr,kr->bvk", x, mu)
        expected_q = (q * distance).sum(dim=-1)
        expected_p = (p * distance).sum(dim=-1)
        per_vertex = 0.5 * (expected_q + expected_p)
        return per_vertex[valid_mask].mean() if valid_mask.any() else per_vertex.sum() * 0.0

    @staticmethod
    def _paired_indices(metadata):
        for i, left in enumerate(metadata):
            if left.get("session") != 1:
                continue
            matches = [
                j for j, right in enumerate(metadata)
                if right.get("subject") == left.get("subject")
                and right.get("session") == 2
                and right.get("start") == left.get("start")
                and right.get("end") == left.get("end")
            ]
            if len(matches) != 1:
                raise ValueError("Each session-1 sample must have exactly one session-2 match.")
            yield i, matches[0]

    @classmethod
    def _temporal_consistency_loss(cls, q, gate, metadata, valid_mask):
        terms = []
        for i, j in cls._paired_indices(metadata):
            mask = valid_mask[i] & valid_mask[j]
            if not mask.any():
                continue
            term = (q[i, mask] - q[j, mask]).pow(2).mean()
            if gate is not None:
                term = term + (gate[i, mask] - gate[j, mask]).pow(2).mean()
            terms.append(term)
        # Normalize by 2 N_b n_valid for N_b matched pairs.
        return 0.5 * torch.stack(terms).mean() if terms else q.sum() * 0.0

    @staticmethod
    def _validate_chunk_batch(
        metadata, batch_size, *, require_cross_session_pairs: bool
    ):
        if not isinstance(metadata, list) or len(metadata) != batch_size:
            raise ValueError("chunk_info must contain one metadata record per sample.")
        first = metadata[0]
        interval = (first.get("start"), first.get("end"), first.get("original_size"))
        if any(
            (item.get("start"), item.get("end"), item.get("original_size")) != interval
            for item in metadata
        ):
            raise ValueError("All samples in a batch must use the same cortical chunk.")
        if require_cross_session_pairs:
            for item in metadata:
                if not any(
                    other is not item
                    and other.get("subject") == item.get("subject")
                    and other.get("session") != item.get("session")
                    for other in metadata
                ):
                    raise ValueError(
                        "Every training/validation sample requires a "
                        "cross-session partner."
                    )
        return first


def create_spin_net_model(
    device=None,
    variant="full",
    fixed_alpha=None,
    functional_prototypes=None,
    adjacency=None,
    distance_matrices=None,
):
    device = device or COMPUTE_DEVICE
    model = SPINNet(
        variant=variant,
        fixed_alpha=fixed_alpha,
        functional_prototypes=functional_prototypes,
        adjacency=adjacency,
        distance_matrices=distance_matrices,
    ).to(device)
    return model, SPINNetLoss(device=device, variant=variant)


def count_parameters(model):
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    }


def print_model_summary(model):
    counts = count_parameters(model)
    print(
        f"SPIN-Net variant={model.variant}, parameters={counts['total']:,}, "
        f"trainable={counts['trainable']:,}"
    )
