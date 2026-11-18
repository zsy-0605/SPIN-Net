"""Generate a whole-cortex parcellation from one prepared session or ADNI scan."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from src.config import CHUNK_SIZE, COMPUTE_DEVICE, CONFIG
from src.models.spin_net import SPINNet


def parcellate_single_session(model, profile, theta, valid_mask, *, chunk_size=CHUNK_SIZE):
    """Read a single [V,R] (or [R,V]) profile chunk by chunk, without pairing.

    Profiles must already use the training ROI order, vertex mask and sparse
    correlation representation. No model parameters or training references
    are estimated from this input. HDF5 datasets can be streamed directly.
    """
    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    n_vertices = mask.size
    prior = np.asarray(theta)
    if prior.shape != (n_vertices, model.num_classes):
        raise ValueError("Theta and mask must share the full-cortex vertex indexing.")
    if profile.shape == (n_vertices, model.input_dim):
        transposed = False
    elif profile.shape == (model.input_dim, n_vertices):
        transposed = True
    else:
        raise ValueError("Unexpected vertex-to-ROI profile shape.")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive.")
    labels = np.zeros(n_vertices, dtype=np.int64)
    probabilities = np.zeros((n_vertices, model.num_classes), dtype=np.float32)
    gate = np.zeros(n_vertices, dtype=np.float32)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0, n_vertices, int(chunk_size)):
            end = min(start + int(chunk_size), n_vertices)
            values = np.asarray(profile[:, start:end]).T if transposed else np.asarray(profile[start:end, :])
            if not np.isfinite(values).all() or np.any(np.abs(values) > 1.0 + 1e-6):
                raise ValueError("Profiles must contain finite sparse weighted correlations in [-1,1].")
            if not mask[start:end].any():
                continue
            data = {
                'profile': torch.as_tensor(values, dtype=torch.float32, device=device)[None],
                'theta': torch.as_tensor(prior[start:end], dtype=torch.float32, device=device),
                'valid_mask': torch.as_tensor(mask[start:end], device=device)[None],
                'chunk_info': [{'subject': 0, 'session': 1, 'start': start, 'end': end, 'original_size': end-start}],
            }
            result = model(data, compute_losses=False)
            labels[start:end] = result['labels'][0].cpu().numpy()
            probabilities[start:end] = result['probabilities'][0].cpu().numpy()
            gate[start:end] = result['gate'][0].cpu().numpy()
    return {'labels': labels, 'probabilities': probabilities, 'gate': gate, 'valid_mask': mask}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--profile', required=True, help='HDF5 .mat file containing profile_mat')
    parser.add_argument('--theta', required=True, help='Training-derived [64984,17] .npy prior')
    parser.add_argument('--valid-mask', required=True, help='Fixed training mask as .npy')
    parser.add_argument('--output', required=True, help='Output .npz')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg, state = checkpoint['config'], checkpoint['model_state']
    if cfg.get('architecture_version') != 3:
        raise ValueError("Checkpoint architecture must match the current SPIN-Net model.")
    resolved = cfg.get('resolved_experiment', {})
    architecture_keys = ('hidden_dim','attention_heads','roi_token_dim','graph_layers',
                         'graph_hops','residual_connections','layer_normalization')
    if any(resolved.get('model', {}).get(key) != CONFIG['model'].get(key) for key in architecture_keys):
        raise ValueError("Use the checkpoint's model configuration via SPIN_CONFIG.")
    if resolved.get('data', {}).get('chunk_size') != CHUNK_SIZE:
        raise ValueError("Inference chunk size must match the trained model.")
    # Geodesic matrices are used only in the training loss. No distance data
    # are read or sliced during this inference-only forward pass.
    model = SPINNet(input_dim=cfg['input_dim'], hidden_dim=cfg['hidden_dim'],
        num_classes=cfg['num_classes'], variant=cfg['variant'], fixed_alpha=cfg['fixed_alpha'],
        adjacency=state['adj_matrix'], distance_matrices={'lh': None, 'rh': None},
        functional_prototypes=state['functional_prototypes']).to(COMPUTE_DEVICE)
    model.load_state_dict(state)
    with h5py.File(args.profile, 'r') as handle:
        result = parcellate_single_session(model, handle['profile_mat'],
            np.load(args.theta), np.load(args.valid_mask))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)


if __name__ == '__main__':
    main()
