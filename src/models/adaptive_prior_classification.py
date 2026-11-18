"""Adaptive population-prior fusion and training objective."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import (
    COMPUTE_DEVICE,
    HIDDEN_DIM,
    LAMBDA_CON,
    LAMBDA_FUNC,
    LAMBDA_PRIOR,
    LAMBDA_SPA,
    LAMBDA_TEMP,
    LOSS_EPSILON,
    N_CLASSES,
)


def _expand_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(reference.shape[:2], dtype=torch.bool, device=reference.device)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(reference.shape[0], -1)
    if tuple(mask.shape) != tuple(reference.shape[:2]):
        raise ValueError(
            f"valid_mask must have shape {tuple(reference.shape[:2])}; "
            f"got {tuple(mask.shape)}"
        )
    return mask.to(device=reference.device, dtype=torch.bool)


def validate_probability_prior(
    theta: torch.Tensor,
    reference: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast and validate the fixed probabilistic group prior ``Theta``."""
    if theta is None:
        raise ValueError("A probabilistic group prior Theta is required.")
    if theta.is_sparse:
        theta = theta.to_dense()
    theta = theta.to(device=reference.device, dtype=reference.dtype)
    if theta.ndim == 2:
        theta = theta.unsqueeze(0).expand(reference.shape[0], -1, -1)
    elif theta.ndim == 3 and theta.shape[0] == 1:
        theta = theta.expand(reference.shape[0], -1, -1)
    if tuple(theta.shape) != (*reference.shape[:2], reference.shape[-1]):
        raise ValueError(
            "Theta and individualized probabilities must have identical shapes; "
            f"got {tuple(theta.shape)} and {tuple(reference.shape)}."
        )
    if not torch.isfinite(theta).all() or torch.any(theta < 0):
        raise ValueError("Theta must contain finite non-negative probabilities.")
    sums = theta.sum(dim=-1)
    if sums[valid_mask].numel() and not torch.allclose(
        sums[valid_mask], torch.ones_like(sums[valid_mask]), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Theta must sum to one at every valid cortical vertex.")
    if torch.any(theta[~valid_mask] != 0):
        raise ValueError("Theta must be zero at invalid/medial-wall vertices.")
    return theta


class PriorConsistencyLoss(nn.Module):
    """``KL(Theta || Q)`` averaged over valid vertices.

    The term is independent of the adaptive gate. Applying it to the
    individualized distribution ``Q`` prevents the trivial ``alpha=0``
    solution.
    """

    def forward(
        self,
        individualized_probabilities: torch.Tensor,
        prior_probabilities: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = _expand_mask(valid_mask, individualized_probabilities)
        theta = validate_probability_prior(
            prior_probabilities, individualized_probabilities, mask
        )
        q = individualized_probabilities.clamp_min(LOSS_EPSILON)
        theta_safe = theta.clamp_min(LOSS_EPSILON)
        per_vertex = (theta * (theta_safe.log() - q.log())).sum(dim=-1)
        return per_vertex[mask].mean() if mask.any() else per_vertex.sum() * 0.0


class AdaptivePriorClassificationModule(nn.Module):
    """Classify individual features and fuse ``Q`` with ``Theta`` directly.

    ``Q = softmax(W_ind H + b)``
    ``alpha = sigmoid(w_g^T H + b_g)``
    ``P = alpha Q + (1-alpha) Theta``
    """

    def __init__(
        self,
        dim: int = HIDDEN_DIM,
        num_classes: int = N_CLASSES,
        fixed_alpha: float | None = None,
    ) -> None:
        super().__init__()
        if fixed_alpha is not None and not 0.0 <= float(fixed_alpha) <= 1.0:
            raise ValueError("fixed_alpha must lie in [0, 1].")
        self.num_classes = int(num_classes)
        self.fixed_alpha = None if fixed_alpha is None else float(fixed_alpha)
        self.individualized_head = nn.Linear(dim, num_classes)
        self.gate_head = nn.Linear(dim, 1) if fixed_alpha is None else None
        self.prior_consistency_loss = PriorConsistencyLoss()

    def forward(
        self,
        x: torch.Tensor,
        theta: torch.Tensor,
        chunk_info=None,
        valid_mask: torch.Tensor | None = None,
        return_loss: bool = False,
        return_gate: bool = False,
    ):
        del chunk_info
        individualized_logits = self.individualized_head(x)
        q = torch.softmax(individualized_logits, dim=-1)
        mask = _expand_mask(valid_mask, q)
        theta = validate_probability_prior(theta, q, mask)
        if self.gate_head is None:
            alpha = torch.full(
                (*x.shape[:2], 1), self.fixed_alpha, device=x.device, dtype=x.dtype
            )
        else:
            alpha = torch.sigmoid(self.gate_head(x))
        p = alpha * q + (1.0 - alpha) * theta

        mask_float = mask.unsqueeze(-1).to(x.dtype)
        q = q * mask_float
        p = p * mask_float
        alpha = alpha * mask_float
        outputs = {
            "individualized_logits": individualized_logits,
            "individualized_probabilities": q,
            "probabilities": p,
            "gate": alpha,
        }
        losses = {
            "prior_consistency_loss": self.prior_consistency_loss(q, theta, mask)
        }
        if return_loss and return_gate:
            return outputs, losses, alpha
        if return_loss:
            return outputs, losses
        if return_gate:
            return outputs, alpha
        return outputs


class SPINNetLoss(nn.Module):
    """Weighted SPIN-Net objective."""

    def __init__(self, device=None, variant: str = "full") -> None:
        super().__init__()
        if variant not in {"full", "fixed_alpha", "without_in", "without_s", "without_p"}:
            raise ValueError(f"Unknown SPIN-Net variant: {variant}")
        self.device = device or COMPUTE_DEVICE
        self.variant = variant

    def forward(self, model_output, chunk_info=None, chunk_weight=None):
        del chunk_info
        losses = model_output["losses"]
        required = {"functional_assignment_loss", "temporal_consistency_loss"}
        if self.variant != "without_in":
            required.add("inter_subject_loss")
        if self.variant != "without_s":
            required.add("total_spatial_loss")
        if self.variant != "without_p":
            required.add("prior_consistency_loss")
        missing = required.difference(losses)
        if missing:
            raise KeyError(f"Missing objective terms for {self.variant}: {sorted(missing)}")

        def scalar(name):
            value = losses[name]
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            return value.mean() if value.ndim else value

        if chunk_weight is None:
            chunk_weight = model_output.get("chunk_weight", 1.0)
        chunk_weight = torch.as_tensor(
            chunk_weight, device=self.device, dtype=torch.float32
        )
        vertex_objective = LAMBDA_FUNC * scalar("functional_assignment_loss")
        vertex_objective = vertex_objective + LAMBDA_TEMP * scalar(
            "temporal_consistency_loss"
        )
        if self.variant != "without_p":
            vertex_objective = vertex_objective + LAMBDA_PRIOR * scalar(
                "prior_consistency_loss"
            )
        if self.variant != "without_s":
            vertex_objective = vertex_objective + LAMBDA_SPA * scalar(
                "total_spatial_loss"
            )
        total = chunk_weight * vertex_objective
        if self.variant != "without_in":
            total = total + LAMBDA_CON * scalar("inter_subject_loss")
        return total


def clip_gradients(model, max_norm: float = 1.0):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
