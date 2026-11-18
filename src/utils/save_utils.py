import scipy.io as sio
import os
import torch
from ..config import N_VERTICES_PER_HEMI

def save_subject_parcellation(subject_id, predictions, save_dir):
    """Save the parcellation for one subject."""
    filename = f'Ind_parcellation_SPINNet_sub{subject_id}.mat'
    filepath = os.path.join(save_dir, filename)
    
    # Extract left- and right-hemisphere labels from the predictions.
    lh_predictions = predictions[:N_VERTICES_PER_HEMI]
    rh_predictions = predictions[N_VERTICES_PER_HEMI:]
    
    # Convert to MATLAB format.
    mat_dict = {
        'lh_labels': lh_predictions.cpu().numpy(),
        'rh_labels': rh_predictions.cpu().numpy()
    }
    
    sio.savemat(filepath, mat_dict)

def save_all_parcellations(predictions_dict, save_dir):
    """Save parcellations for all subjects.
    
    Args:
        predictions_dict: {subject_id: (lh_labels, rh_labels)} 
        save_dir: Output directory.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for subject_id, (lh_labels, rh_labels) in predictions_dict.items():
        full_predictions = torch.cat([lh_labels.cpu(), rh_labels.cpu()])
        save_subject_parcellation(subject_id, full_predictions, save_dir)
