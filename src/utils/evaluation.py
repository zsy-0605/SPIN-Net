import torch
import numpy as np
import os
import scipy.io as sio
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from .metrics import (
    compute_connectivity_homogeneity,
    compute_spatial_smoothness,
    evaluate_network_structure
)
from ..config import *

class NetworkEvaluator():
    """Network evaluator."""
    
    def __init__(self):
        self.metrics_history = {
            'homogeneity': [],
            'smoothness': [],
            'network_structure': [],
            'test_metrics': {}  # Store test-set metrics.
        }
        
    def evaluate(
        self,
        features,
        adj_matrix,
        chunk_info=None,
        labels=None,
        connectivity_profiles=None,
        valid_mask=None,
        include_structure=False,
    ):
        """Evaluate available metrics without substituting a chunk proxy for homogeneity."""
        try:
            # Compute smoothness.
            smoothness = compute_spatial_smoothness(features, adj_matrix, chunk_info)
            
            # Optional network-structure diagnostic.
            structure_metrics = (
                evaluate_network_structure(features) if include_structure else {}
            )
            
            results = {
                'smoothness': float(smoothness),
                **{k: float(v) for k, v in structure_metrics.items()}
            }
            if labels is not None and connectivity_profiles is not None:
                homogeneity = compute_connectivity_homogeneity(
                    labels,
                    connectivity_profiles,
                    valid_mask=valid_mask
                )
                results['homogeneity'] = float(homogeneity)
            return results
                    
        except Exception as e:
            print(f"\nError in NetworkEvaluator.evaluate:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            raise
    

    def evaluate_test_subject(self, subject_id, predictions, connectivity_data):
        """Evaluate one test subject.
        
        Args:
            subject_id: Subject identifier
            predictions: (lh_labels, rh_labels) predicted labels
            connectivity_data: Connectivity data for this subject
            
        Returns:
            dict: Evaluation metrics for this subject
        """
        try:
            # Concatenate labels from both hemispheres.
            lh_labels, rh_labels = predictions
            full_labels = torch.cat([lh_labels, rh_labels])
            
            # Homogeneity is evaluated from full-cortex labels and connectivity
            # profiles; it is not approximated from chunk-level latent features.
            metrics = {
                'homogeneity': compute_connectivity_homogeneity(
                    full_labels,
                    connectivity_data
                )
            }
            
            # Record metrics for this subject.
            self.metrics_history['test_metrics'][subject_id] = metrics
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating test subject {subject_id}: {str(e)}")
            raise

    def evaluate_all_test_results(self, predictions_dict, connectivity_data):
        """Evaluate all test-set results.
        
        Args:
            predictions_dict: {subject_id: (lh_labels, rh_labels)}
            connectivity_data: {subject_id: connectivity_matrix}
            
        Returns:
            dict: Mean evaluation metrics
        """
        try:
            all_metrics = []
            for subject_id, predictions in predictions_dict.items():
                subject_metrics = self.evaluate_test_subject(
                    subject_id,
                    predictions,
                    connectivity_data[subject_id]
                )
                all_metrics.append(subject_metrics)
            
            # Compute mean metrics.
            avg_metrics = {}
            for metric in all_metrics[0].keys():
                values = [m[metric] for m in all_metrics]
                avg_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
                
            # Add the overall evaluation score.
            avg_metrics['overall'] = self._compute_overall_score(avg_metrics)
            
            return avg_metrics
            
        except Exception as e:
            print(f"Error in evaluate_all_test_results: {str(e)}")
            raise
            
    def _compute_overall_score(self, metrics):
        """Compute the overall evaluation score."""
        try:
            # Set metric weights.
            weights = {
                'homogeneity': 0.4,
                'smoothness': 0.3,
                'silhouette': 0.3
            }
            
            # Compute the weighted score.
            score = sum(
                metrics[k]['mean'] * weights[k]
                for k in weights.keys()
            )
            
            return {
                'score': score,
                'weights': weights
            }
            
        except Exception as e:
            print(f"Error computing overall score: {str(e)}")
            raise
            
    def get_trend(self, metric_name, window_size=5):
        """Compute the trend of a metric.
        
        Args:
            metric_name: Metric name
            window_size: Window size
            
        Returns:
            float: Trend score; positive values indicate an increasing trend
        """
        if len(self.metrics_history[metric_name]) < window_size:
            return 0.0
            
        recent_values = self.metrics_history[metric_name][-window_size:]
        trend = sum(y - x for x, y in zip(recent_values[:-1], recent_values[1:]))
        return trend / (window_size - 1)
        
    def should_stop_early(self, patience=5, min_improvement=1e-4):
        """Check whether training should stop early.
        
        Args:
            patience: Number of epochs without sufficient improvement
            min_improvement: Minimum improvement threshold
            
        Returns:
            bool: Whether to stop training
        """
        if len(self.metrics_history['homogeneity']) < patience:
            return False
            
        recent_homogeneity = self.metrics_history['homogeneity'][-patience:]
        best_so_far = max(self.metrics_history['homogeneity'][:-patience])
        
        return all(h - best_so_far < min_improvement for h in recent_homogeneity)

    def save_evaluation_results(self, save_dir):
        """Save evaluation results.
        
        Args:
            save_dir: Output path.
        """
        try:
            results = {
                'metrics_history': self.metrics_history,
                'test_metrics': self.metrics_history['test_metrics']
            }
            
            save_path = os.path.join(save_dir, 'evaluation_results.mat')
            sio.savemat(save_path, results)
            print(f"Evaluation results saved to {save_path}")
            
        except Exception as e:
            print(f"Error saving evaluation results: {str(e)}")

    def evaluate_reproducibility(
        self,
        subject_predictions1,
        subject_predictions2,
        return_details=False
    ):
        """Evaluate matched and direct test--retest reproducibility."""
        try:
            matched_scores = []
            direct_scores = []
            nmi_scores = []
            per_subject = {}
            
            for subject_id in subject_predictions1:
                if subject_id not in subject_predictions2:
                    continue
                    
                lh1, rh1 = subject_predictions1[subject_id]
                lh2, rh2 = subject_predictions2[subject_id]
                
                # Move data to the correct device.
                lh1, rh1 = lh1.to(COMPUTE_DEVICE), rh1.to(COMPUTE_DEVICE)
                lh2, rh2 = lh2.to(COMPUTE_DEVICE), rh2.to(COMPUTE_DEVICE)

                dice_matrix = self.compute_network_dice_matrix(lh1, rh1, lh2, rh2)
                direct_score = torch.diag(dice_matrix).mean()
                matched_network, permutation = self.hungarian_network_dice(
                    dice_matrix
                )
                matched_score = matched_network.mean()
                matched_scores.append(matched_score)
                direct_scores.append(direct_score)
                if return_details:
                    nmi_score = self.compute_nmi(lh1, rh1, lh2, rh2)
                    nmi_scores.append(nmi_score)

                if return_details:
                    per_subject[subject_id] = {
                        'matched': matched_score,
                        'direct': direct_score,
                        'nmi': nmi_score,
                        'network_dice_matched': matched_network,
                        'network_dice_direct': torch.diag(dice_matrix),
                        'hungarian_permutation': permutation,
                        'identity_correspondence': bool(
                            torch.equal(
                                permutation,
                                torch.arange(N_CLASSES, device=permutation.device)
                            )
                        ),
                    }

            if not direct_scores:
                empty = torch.tensor(0.0, device=COMPUTE_DEVICE)
                if return_details:
                    return {
                        'matched': empty,
                        'direct': empty,
                        'nmi': empty,
                        'identity_count': 0,
                        'identity_fraction': 0.0,
                        'per_subject': per_subject
                    }
                return empty

            matched_mean = torch.stack(matched_scores).mean()
            direct_mean = torch.stack(direct_scores).mean()
            if return_details:
                nmi_mean = torch.stack(nmi_scores).mean()
                identity_count = sum(
                    int(item['identity_correspondence'])
                    for item in per_subject.values()
                )
                return {
                    'matched': matched_mean,
                    'direct': direct_mean,
                    'nmi': nmi_mean,
                    'identity_count': identity_count,
                    'identity_fraction': identity_count / len(per_subject),
                    'per_subject': per_subject
                }
            return matched_mean
            
        except Exception as e:
            print(f"Error in evaluate_reproducibility: {str(e)}")
            raise

    @staticmethod
    def hungarian_network_dice(dice_matrix):
        """Return optimal one-to-one Dice values and the session-2 permutation."""
        matrix = dice_matrix.detach().cpu().numpy()
        row_indices, column_indices = linear_sum_assignment(-matrix)
        order = np.argsort(row_indices)
        column_indices = column_indices[order]
        permutation = torch.as_tensor(
            column_indices, device=dice_matrix.device, dtype=torch.long
        )
        rows = torch.arange(N_CLASSES, device=dice_matrix.device)
        return dice_matrix[rows, permutation], permutation

    @staticmethod
    def compute_network_dice_matrix(lh1, rh1, lh2, rh2):
        """Compute the 17-by-17 network Dice overlap matrix between two sessions."""
        labels1 = torch.cat([lh1.reshape(-1), rh1.reshape(-1)]).to(
            COMPUTE_DEVICE, dtype=torch.long
        )
        labels2 = torch.cat([lh2.reshape(-1), rh2.reshape(-1)]).to(
            COMPUTE_DEVICE, dtype=torch.long
        )
        if labels1.numel() != labels2.numel():
            raise ValueError(
                "Parcellations must contain the same number of cortical vertices "
                f"(got {labels1.numel()} and {labels2.numel()})."
            )

        valid1 = (labels1 >= 1) & (labels1 <= N_CLASSES)
        valid2 = (labels2 >= 1) & (labels2 <= N_CLASSES)
        sizes1 = torch.bincount(
            labels1[valid1] - 1, minlength=N_CLASSES
        ).float()
        sizes2 = torch.bincount(
            labels2[valid2] - 1, minlength=N_CLASSES
        ).float()

        paired = valid1 & valid2
        pair_ids = (
            (labels1[paired] - 1) * N_CLASSES + (labels2[paired] - 1)
        )
        intersections = torch.bincount(
            pair_ids, minlength=N_CLASSES * N_CLASSES
        ).reshape(N_CLASSES, N_CLASSES).float()

        denominators = sizes1[:, None] + sizes2[None, :]
        return torch.where(
            denominators > 0,
            2.0 * intersections / denominators.clamp_min(1.0),
            torch.zeros_like(denominators)
        )

    @staticmethod
    def compute_nmi(lh1, rh1, lh2, rh2):
        """Compute normalized mutual information after concatenating both hemispheres.

        Exclude vertices outside the shared ``1..N_CLASSES`` label space in either parcellation
        so that shared medial-wall zero labels do not inflate NMI.
        """
        labels1 = torch.cat([lh1.reshape(-1), rh1.reshape(-1)]).detach().cpu()
        labels2 = torch.cat([lh2.reshape(-1), rh2.reshape(-1)]).detach().cpu()

        if labels1.numel() != labels2.numel():
            raise ValueError(
                "Parcellations must contain the same number of cortical vertices "
                f"(got {labels1.numel()} and {labels2.numel()})."
            )

        valid = (
            (labels1 >= 1) & (labels1 <= N_CLASSES)
            & (labels2 >= 1) & (labels2 <= N_CLASSES)
        )
        if not bool(valid.any()):
            return torch.tensor(0.0, device=COMPUTE_DEVICE)

        score = normalized_mutual_info_score(
            labels1[valid].numpy(),
            labels2[valid].numpy(),
            average_method='arithmetic'
        )
        return torch.tensor(
            float(score), device=COMPUTE_DEVICE, dtype=torch.float32
        )

    def evaluate_similarity(self, subject_predictions, return_details=False):
        """Return a descriptive all-pairs, same-session similarity summary.

        This general diagnostic is not the manuscript's inferential
        between-subject reference. The reported analysis uses
        ``restricted_derangement_reproducibility`` to enforce one-to-one,
        non-self, cross-family Session-1/Session-2 matching.
        """
        try:
            similarity_scores = []
            nmi_scores = []
            per_pair = []
            subjects = list(subject_predictions.keys())
            
            if len(subjects) < 2:
                empty = torch.tensor(0.0, device=COMPUTE_DEVICE)
                if return_details:
                    return {'dice': empty, 'nmi': empty, 'per_pair': per_pair}
                return empty
                
            for i in range(len(subjects)):
                for j in range(i+1, len(subjects)):
                    lh1, rh1 = subject_predictions[subjects[i]]
                    lh2, rh2 = subject_predictions[subjects[j]]
                    
                    # Move data to the correct device.
                    lh1, rh1 = lh1.to(COMPUTE_DEVICE), rh1.to(COMPUTE_DEVICE)
                    lh2, rh2 = lh2.to(COMPUTE_DEVICE), rh2.to(COMPUTE_DEVICE)

                    # Label semantics are shared between subjects, so
                    # corresponding networks are compared directly without
                    # post-hoc relabeling.
                    dice_matrix = self.compute_network_dice_matrix(
                        lh1, rh1, lh2, rh2
                    )
                    network_scores = torch.diag(dice_matrix)
                    pair_dice = network_scores.mean()
                    similarity_scores.append(pair_dice)
                    if return_details:
                        pair_nmi = self.compute_nmi(lh1, rh1, lh2, rh2)
                        nmi_scores.append(pair_nmi)
                        per_pair.append({
                            'subject1': subjects[i],
                            'subject2': subjects[j],
                            'dice': pair_dice,
                            'nmi': pair_nmi,
                            'network_dice': network_scores
                        })
            
            if not similarity_scores:
                dice_mean = torch.tensor(0.0, device=COMPUTE_DEVICE)
                nmi_mean = torch.tensor(0.0, device=COMPUTE_DEVICE)
            else:
                dice_mean = torch.stack(similarity_scores).mean()

            if return_details:
                nmi_mean = (
                    torch.stack(nmi_scores).mean()
                    if nmi_scores
                    else torch.tensor(0.0, device=COMPUTE_DEVICE)
                )
                return {
                    'dice': dice_mean,
                    'nmi': nmi_mean,
                    'per_pair': per_pair
                }
            return dice_mean
            
        except Exception as e:
            print(f"Error in evaluate_similarity: {str(e)}")
            raise
