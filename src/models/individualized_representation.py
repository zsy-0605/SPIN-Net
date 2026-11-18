# src/models/individualized_representation.py
# Module 1: Individualized Functional Representation Module.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ..config import *


class IndividualizedRepresentationModule(nn.Module):
    """Individualized functional representation module.

    Learn a stable and discriminative functional representation for each subject.
    Combine feature embedding with cross-session stability and between-subject contrastive learning.
    """
    
    def __init__(self, input_dim=N_ROIS, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        
        # Encode each vertex connectivity profile as a sequence of ROI tokens.
        self.feature_embedding = ROITokenLearnableQueryAttentionEncoder(
            n_rois=input_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            token_dim=ROI_TOKEN_DIM,
            dropout=dropout
        )
        
        # Contrastive learning across subjects.
        self.inter_subject_contrast = InterSubjectContrastModule(
            dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout
        )
        
    def forward(self, x, chunk_info=None, valid_mask=None, return_loss=False):
        """
        Forward pass.
        Args:
            x: Input connectivity profiles [batch_size, chunk_size, 1483]
            chunk_info: Chunk metadata for contrastive learning
            valid_mask: Valid-cortex vertex mask [batch_size, chunk_size]
            return_loss: Whether to return loss terms
        Returns:
            features: Individualized functional representations [batch_size, chunk_size, hidden_dim]
            losses: Loss dictionary when return_loss=True
        """
        # ROI-token learnable-query attention pooling: connectivity profiles -> 256-dimensional vertex embeddings.
        embedded_features = self.feature_embedding(x)  # [B, T, 256]
        
        # Encourage discriminative representations across subjects.
        if return_loss:
            contrastive_features, inter_loss = self.inter_subject_contrast(
                embedded_features,
                chunk_info=chunk_info,
                valid_mask=valid_mask,
                return_loss=True
            )
            return contrastive_features, {
                'inter_subject_loss': inter_loss
            }

        return self.inter_subject_contrast(
            embedded_features,
            chunk_info=chunk_info,
            valid_mask=valid_mask,
            return_loss=False
        )


class ROITokenLearnableQueryAttentionEncoder(nn.Module):
    """Learnable-query attention encoder over ROI tokens.

    A dense 1483x1483 attention matrix per vertex is prohibitively expensive for
    chunk-wise training. This module uses learnable-query attention pooling over ROI
    tokens while keeping memory linear in the number of ROIs.
    """

    def __init__(self, n_rois=N_ROIS, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS,
                 token_dim=ROI_TOKEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.n_rois = n_rois
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.token_dim = token_dim
        self.inner_dim = n_heads * token_dim
        self.activation_checkpointing = bool(CONFIG["model"].get("attention_activation_checkpointing", True))

        self.roi_embedding = nn.Parameter(torch.empty(n_rois, self.inner_dim))
        self.summary_query = nn.Parameter(torch.empty(n_heads, token_dim))
        self.value_projection = nn.Linear(1, self.inner_dim)
        self.k_proj = nn.Linear(self.inner_dim, self.inner_dim)
        self.v_proj = nn.Linear(self.inner_dim, self.inner_dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.inner_dim),
            nn.Linear(self.inner_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.roi_embedding, std=0.02)
        nn.init.normal_(self.summary_query, std=0.02)

    def forward(self, x):
        batch_size, chunk_size, n_rois = x.shape
        if n_rois != self.n_rois:
            raise ValueError(f"Expected {self.n_rois} ROI tokens, got {n_rois}")

        flat = x.reshape(-1, n_rois)
        outputs = []
        for start in range(0, flat.size(0), ROI_ATTENTION_VERTEX_CHUNK):
            stop = min(start + ROI_ATTENTION_VERTEX_CHUNK, flat.size(0))
            values = flat[start:stop]
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self._encode_vertices, values,
                    use_reentrant=False, preserve_rng_state=True))
            else:
                outputs.append(self._encode_vertices(values))

        return torch.cat(outputs, dim=0).view(batch_size, chunk_size, self.hidden_dim)

    def _encode_vertices(self, values):
        tokens = self.value_projection(values.unsqueeze(-1)) + self.roi_embedding.unsqueeze(0)
        n_vertices = values.shape[0]
        k = self.k_proj(tokens).view(n_vertices, self.n_rois, self.n_heads, self.token_dim).transpose(1,2)
        v = self.v_proj(tokens).view(n_vertices, self.n_rois, self.n_heads, self.token_dim).transpose(1,2)
        scores = torch.einsum('hd,bhrd->bhr',self.summary_query,k) / self.token_dim**0.5
        weights = self.dropout(torch.softmax(scores,dim=-1))
        pooled = torch.einsum('bhr,bhrd->bhd',weights,v).reshape(n_vertices,self.inner_dim)
        return self.out_proj(pooled)


class InterSubjectContrastModule(nn.Module):
    """Contrastive learning module across subjects.

    Compute InfoNCE directly on attention-pooling representations.

    There is no additional projection or subject encoder; the attention-pooling
    output has 256 dimensions and is passed directly to the Spatial module.
    """
    
    def __init__(self, dim, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(temperature=CONTRASTIVE_TEMPERATURE)
        
    def forward(self, x, chunk_info=None, valid_mask=None, return_loss=False):
        """
        Forward pass.
        Args:
            x: Subject features [batch_size, chunk_size, dim]
            chunk_info: Chunk metadata for positive and negative pairs
            valid_mask: Valid-cortex vertex mask [batch_size, chunk_size]
            return_loss: Whether to return loss terms
        """
        if return_loss:
            loss = self.contrastive_loss(
                x,
                chunk_info,
                valid_mask=valid_mask
            )
            return x, loss
            
        return x


class ContrastiveLoss(nn.Module):
    """Session-1 anchored InfoNCE with session-2 candidates.

    The positive is the same subject and same chunk in session 2.  Session-2
    samples from the other subjects in the mini-batch are the negatives.
    """
    
    def __init__(self, temperature=CONTRASTIVE_TEMPERATURE, resolution=None,
                 n_groups=None, sampled_vertices=None):
        super().__init__()
        self.temperature = temperature
        self.resolution = resolution or CONFIG["loss"].get("contrastive_resolution", "chunk")
        self.n_groups = int(n_groups or CONFIG["loss"].get("contrastive_groups", 4))
        self.sampled_vertices = int(sampled_vertices or CONFIG["loss"].get("contrastive_sampled_vertices", 64))
        if self.resolution not in {"none", "chunk", "four_groups", "sampled_vertices"}:
            raise ValueError("Unknown contrastive resolution.")
        if self.temperature <= 0 or self.n_groups < 1 or self.sampled_vertices < 1:
            raise ValueError("Temperature and contrastive set sizes must be positive.")
        
    def forward(self, features, chunk_info=None, valid_mask=None):
        """
        Compute the InfoNCE contrastive loss.
        Args:
            features: [batch_size, chunk_size, dim] Encoded subject features
            chunk_info: Chunk metadata containing subject and session identifiers
            valid_mask: Valid-cortex vertex mask [batch_size, chunk_size]
        """
        if features.dim() != 3:
            raise ValueError("ContrastiveLoss: features期望为 [B, N, D]")

        batch_size, n_vertices, dim = features.shape
        
        # Average over valid vertices to form a subject-specific chunk representation.
        if valid_mask is None:
            raise ValueError(
                "ContrastiveLoss requires a valid-cortex mask."
            )
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0).expand(batch_size, -1)
        valid_mask = valid_mask[:, :n_vertices].to(
            device=features.device,
            dtype=features.dtype
        )
        if not torch.equal(valid_mask.bool(), valid_mask[0:1].bool().expand_as(valid_mask)):
            raise ValueError("Contrastive samples must share the fixed valid-cortex mask.")
        if self.resolution == "none":
            return features.sum() * 0.0
        indices = torch.nonzero(valid_mask[0].bool(), as_tuple=False).flatten()
        if indices.numel() == 0:
            raise ValueError("InfoNCE requires valid vertices.")
        if self.resolution == "chunk":
            pooled = features[:, indices].mean(dim=1, keepdim=True)
        elif self.resolution == "four_groups":
            if indices.numel() < self.n_groups:
                raise ValueError("Fewer valid vertices than contrastive groups.")
            groups = torch.tensor_split(indices, self.n_groups)
            pooled = torch.stack([features[:, group].mean(dim=1) for group in groups], dim=1)
        else:
            if indices.numel() < self.sampled_vertices:
                raise ValueError("Fewer valid vertices than the requested sample.")
            selected = indices[torch.randperm(indices.numel(), device=features.device)[:self.sampled_vertices]]
            # A single draw is shared by all subjects and both sessions.
            pooled = features[:, selected]
        pooled = F.normalize(pooled, dim=-1)

        # Extract subject and session identifiers from chunk_info.
        subjects = []
        sessions = []
        chunk_starts = []
        if isinstance(chunk_info, list):
            for ci in chunk_info:
                subjects.append(ci.get('subject', -1))
                sessions.append(ci.get('session', -1))
                chunk_starts.append(ci.get('start', -1))
        elif isinstance(chunk_info, dict):
            for _ in range(batch_size):
                subjects.append(chunk_info.get('subject', -1))
                sessions.append(chunk_info.get('session', -1))
                chunk_starts.append(chunk_info.get('start', -1))
        else:
            raise ValueError(
                "ContrastiveLoss requires subject, session, and chunk metadata."
            )

        # HCP subject IDs may be serialized as integers or numeric strings.
        subject_keys = [str(subject) for subject in subjects]
        subject_codes = {key:index for index,key in enumerate(dict.fromkeys(subject_keys))}
        subjects = torch.tensor([subject_codes[key] for key in subject_keys], device=pooled.device)
        sessions = torch.tensor(sessions, device=pooled.device)
        chunk_starts = torch.tensor(chunk_starts, device=pooled.device)

        anchors = torch.nonzero(sessions == 1, as_tuple=False).flatten()
        candidates = torch.nonzero(sessions == 2, as_tuple=False).flatten()
        if anchors.numel() == 0 or candidates.numel() == 0:
            raise ValueError("InfoNCE requires both session 1 and session 2 samples.")
        if not torch.all(chunk_starts == chunk_starts[0]):
            raise ValueError("InfoNCE candidates must use the same cortical chunk.")
        if len(set(subjects[candidates].tolist())) != candidates.numel():
            raise ValueError("Session-2 candidates must contain each subject once.")
        logits = torch.einsum('agd,cgd->gac', pooled[anchors], pooled[candidates]) / self.temperature
        target_rows = []
        for anchor in anchors.tolist():
            matches = torch.nonzero(
                (subjects[candidates] == subjects[anchor])
                & (chunk_starts[candidates] == chunk_starts[anchor]),
                as_tuple=False,
            ).flatten()
            if matches.numel() != 1:
                raise ValueError(
                    "Each session-1 InfoNCE anchor requires exactly one "
                    "same-subject, same-chunk session-2 positive."
                )
            target_rows.append(matches.item())
        targets = torch.tensor(target_rows, device=pooled.device, dtype=torch.long)
        return F.cross_entropy(logits.reshape(-1, candidates.numel()),
                               targets.repeat(pooled.shape[1]))
