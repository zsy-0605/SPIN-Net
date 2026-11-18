import os
import torch
import numpy as np
import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.models.spin_net import create_spin_net_model
from src.data.data_loader import (
    get_dataloader,
    load_priors,
    load_and_validate_split_manifest,
    validate_profile_inventory,
)
from src.analysis.reproducibility import restricted_derangement_reproducibility
from src.data.prototypes import load_or_estimate_functional_prototypes
from src.utils.logger import setup_logger
from src.utils.metrics import compute_connectivity_homogeneity_from_file
from src.utils.evaluation import NetworkEvaluator
from src.utils.graph_utils import load_medial_wall_mask
from src.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    COMPUTE_DEVICE,
    GRADIENT_CLIP,
    LEARNING_RATE,
    MEDIAL_WALL_MASK_PATH,
    MIN_LEARNING_RATE,
    NUM_EPOCHS,
    N_CLASSES,
    N_VERTICES,
    N_VERTICES_PER_HEMI,
    PROFILE_DIR,
    RANDOM_SEED,
    REPRODUCIBILITY_DERANGEMENTS,
    RESULTS_DIR,
    SAVE_SESSION_PROBABILITIES,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VAL_SUBJECTS,
    WEIGHT_DECAY,
    config_snapshot,
    ensure_output_dirs,
    validate_input_paths,
)
import warnings
import scipy.io as sio
import random
import json
import time

warnings.filterwarnings("ignore", message="pixdim[1,2,3] should be non-zero; setting 0 dims to 1")
warnings.filterwarnings('ignore', message='.*pixdim.*')
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')
warnings.filterwarnings('ignore', message='.*torch.amp.autocast.*')
warnings.filterwarnings('ignore', module='nibabel')

class Trainer:
    def __init__(self, args):
        self.args = args
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)
        ensure_output_dirs()
        validate_input_paths(require_geodesics=args.variant != "without_s")
        self.logger = setup_logger()
        self.device = COMPUTE_DEVICE
        self.experiment_name = self._experiment_name(
            args.variant,
            args.fixed_alpha
        )
        self.contrastive_resolution = config_snapshot()["loss"].get("contrastive_resolution", "chunk")
        if self.contrastive_resolution != "chunk":
            self.experiment_name += "_contrastive_" + self.contrastive_resolution
        self.checkpoint_dir = os.path.join(
            CHECKPOINT_DIR,
            self.experiment_name
        )
        self.results_dir = os.path.join(
            RESULTS_DIR,
            self.experiment_name
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.logger.info(f'Using device: {self.device}')
        self.logger.info(f'Experiment configuration: {self.experiment_name}')
        
        # Initialize the evaluator.
        self.evaluator = NetworkEvaluator()
        
        # Initialize training state.
        self.current_epoch = 0 
        self.best_metric = float('inf')
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        
        # Initialize the data and model.
        self.setup_data()
        self.setup_model()

    @staticmethod
    def _experiment_name(variant, fixed_alpha):
        if variant == "fixed_alpha":
            if fixed_alpha is None:
                raise ValueError(
                    "--variant fixed_alpha requires --fixed_alpha."
                )
            if not 0.0 <= float(fixed_alpha) <= 1.0:
                raise ValueError("--fixed_alpha must lie in [0, 1].")
            alpha_token = f"{float(fixed_alpha):.3f}".rstrip("0").rstrip(".")
            return f"fixed_alpha_{alpha_token.replace('.', 'p')}"
        if fixed_alpha is not None:
            raise ValueError(
                "--fixed_alpha is only valid with --variant fixed_alpha."
            )
        return variant
    def setup_data(self):
        """Initialize the datasets."""
        self.train_subjects = TRAIN_SUBJECTS
        self.val_subjects = VAL_SUBJECTS
        self.test_subjects = TEST_SUBJECTS
        split_manifest = load_and_validate_split_manifest()
        if split_manifest["family_ids"] is None:
            raise ValueError(
                "The manuscript analysis requires family IDs for the restricted "
                "between-subject reference."
            )
        self.test_family_ids = dict(zip(
            split_manifest["test"],
            split_manifest["family_ids"]["test"],
        ))
        self.session_ids = list(range(1, 3))
        n_profiles = validate_profile_inventory(
            self.train_subjects + self.val_subjects + self.test_subjects,
            self.session_ids,
        )
        self.logger.info("Validated %d subject/session profile files.", n_profiles)
        prior_bundle = load_priors()
        self.functional_prototypes = load_or_estimate_functional_prototypes(
            self.train_subjects, prior_bundle["theta"]
        )

        # Create the data loaders.
        include_prior = self.args.variant != "without_p"
        self.train_loader = get_dataloader(
            self.train_subjects, 
            self.session_ids,
            batch_size=self.args.batch_size,
            shuffle=True,
            include_prior=include_prior
        )
        
        self.val_loader = get_dataloader(
            self.val_subjects, 
            self.session_ids,
            batch_size=self.args.batch_size,
            shuffle=False,
            include_prior=include_prior
        )
            
        self.test_loader = get_dataloader(
            self.test_subjects,
            self.session_ids,
            batch_size=self.args.batch_size,
            shuffle=False,
            include_prior=include_prior,
            paired_sessions=False,
        )


    def setup_model(self):
        """Initialize the SPIN-Net model."""
        # Create the SPIN-Net model and loss function.
        self.model, self.criterion = create_spin_net_model(
            device=self.device,
            variant=self.args.variant,
            fixed_alpha=self.args.fixed_alpha,
            functional_prototypes=self.functional_prototypes,
        )
        
        # Print the model summary.
        from src.models.spin_net import print_model_summary
        print_model_summary(self.model)

        # Optimizer.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, len(self.train_loader) * self.args.epochs),
            eta_min=MIN_LEARNING_RATE
        )

        # Clip gradients.
        self.max_grad_norm = GRADIENT_CLIP

    def train_step(self, data):
        try:
            # Move tensors to the target device with the required dtypes.
            data = {
                k: (
                    v.to(self.device, dtype=torch.bool)
                    if k == 'valid_mask' and torch.is_tensor(v)
                    else v.to(self.device, dtype=torch.float32)
                    if torch.is_tensor(v)
                    else v
                )
                for k, v in data.items()
            }

            self.optimizer.zero_grad()
            
            with torch.amp.autocast(device_type='cuda', enabled=False):
                # Forward pass.
                output = self.model(data, compute_losses=True)
                
                # Compute the combined SPIN-Net objective.
                total_loss = self.criterion(output, data.get('chunk_info'))
                
                # Collect module losses for monitoring.
                losses = output['losses']
                regular_loss = sum(losses.values()) if losses else torch.tensor(0.0)

            # Backpropagation.
            total_loss.backward()
            
            # Clip gradients.
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
            
            self.optimizer.step()
            self.scheduler.step()

            return {
                'objective_loss': total_loss.item(),
                'regular_loss': regular_loss.item(),
                'model_output': output
            }

        except Exception as e:
            print(f"\nError in train_step:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            print(f"Data shapes:")
            for k, v in data.items():
                if torch.is_tensor(v):
                    print(f"{k}: {v.shape}")
            raise

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_regular_loss = 0
        epoch_metrics = {
            'smoothness': []
        }
        num_batches = len(self.train_loader)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        epoch_started = time.perf_counter()
        
        try:
            for batch_idx, data in enumerate(self.train_loader):
                # Perform one training step.
                step_results = self.train_step(data)
                
                # Accumulate losses.
                total_regular_loss += step_results['regular_loss']
                total_loss += step_results['objective_loss']

                # Evaluate network quality.
                with torch.no_grad():
                    smoothness = step_results['model_output']['losses'].get(
                        'smoothness_loss', torch.tensor(0.0)
                    )
                    metrics = {'smoothness': float(smoothness.detach().cpu())}
                    for k, v in metrics.items():
                        if k in epoch_metrics:
                            epoch_metrics[k].append(v)

                # Log progress every 10 batches.
                if batch_idx % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.info(
                        f'Epoch {epoch}, Batch {batch_idx}/{num_batches}, LR: {current_lr:.6f}\n'
                        f'Objective Loss: {step_results["objective_loss"]:.4f}, '
                        f'Regular Monitor: {step_results["regular_loss"]:.4f}, '
                        f'Metrics: Smoothness={metrics["smoothness"]:.4f}'
                    )

            # Compute epoch-level metrics.
            avg_loss = total_loss / num_batches
            avg_regular_loss = total_regular_loss / num_batches
            avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

            # Log the epoch summary.
            self.logger.info(
                f"\n=== Epoch {epoch} Summary ===\n"
                f"Average Loss: {avg_loss:.4f}\n"
                f"Average Regular Loss: {avg_regular_loss:.4f}\n"
                f"Average Metrics:\n"
                f"  Smoothness: {avg_metrics['smoothness']:.4f}\n"
                f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            performance = {
                'epoch': int(epoch),
                'training_seconds': time.perf_counter() - epoch_started,
                'peak_allocated_bytes': int(torch.cuda.max_memory_allocated(self.device)) if self.device.type == 'cuda' else None,
                'device': str(self.device),
                'gpu_name': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
                'contrastive_resolution': self.contrastive_resolution,
            }
            with open(os.path.join(self.results_dir, 'training_performance.jsonl'), 'a') as stream:
                stream.write(json.dumps(performance) + '\n')
            return float(avg_loss), avg_metrics

        except Exception as e:
            self.logger.error(f"Error in epoch {epoch}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        total_regular_loss = 0
        val_metrics = {
            'smoothness': []
        }
        num_batches = len(self.val_loader)

        try:
            with torch.no_grad():
                for data in self.val_loader:
                    # Prepare the input data.
                    data = {
                        k: (
                            v.to(self.device, dtype=torch.bool)
                            if k == 'valid_mask' and torch.is_tensor(v)
                            else v.to(self.device, dtype=torch.float32)
                            if torch.is_tensor(v)
                            else v
                        )
                        for k, v in data.items()
                    }
                    
                    # Forward pass.
                    model_output = self.model(data, compute_losses=True)
                    
                    # Compute the loss.
                    regular_loss = self.criterion(model_output)
                    
                    # Accumulate losses.
                    total_regular_loss += regular_loss.item()
                    total_loss += regular_loss.item()
                    
                    smoothness = model_output['losses'].get(
                        'smoothness_loss', torch.tensor(0.0)
                    )
                    metrics = {'smoothness': float(smoothness.detach().cpu())}
                    
                    for k, v in metrics.items():
                        if k in val_metrics:
                            val_metrics[k].append(v)

                # Compute mean values.
                avg_loss = total_loss / num_batches
                avg_regular_loss = total_regular_loss / num_batches
                avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}

                # Log validation results.
                self.logger.info(
                    f"\n=== Validation Results ===\n"
                    f"Average Loss: {avg_loss:.4f}\n"
                    f"Average Regular Loss: {avg_regular_loss:.4f}\n"
                    f"Average Metrics:\n"
                    f"  Smoothness: {avg_metrics['smoothness']:.4f}"
                )

                return float(avg_loss), avg_metrics

        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}")
            raise

    def test(self):
        """Evaluate and save parcellations, including reproducibility and similarity."""
        self.model.eval()
        
        # Combined parcellations use argmax after averaging the two session probability maps for each subject.
        # Session-specific parcellations are saved separately for reproducibility analysis.
        subject_predictions = {}
        session_predictions = {}
        valid_cortex_mask = torch.from_numpy(
            load_medial_wall_mask(MEDIAL_WALL_MASK_PATH)
        ).bool()
        
        try:
            with torch.no_grad():
                for data in self.test_loader:
                    # Prepare the input data.
                    data = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in data.items()}

                    # Forward pass.
                    # Each session is inferred independently; no paired-session
                    # information or training loss is consumed here.
                    model_output = self.model(data, compute_losses=False)
                    predictions = model_output['labels']
                    probabilities = model_output['probabilities']
                    gate_values = model_output['gate']
                    valid_masks = model_output['valid_mask']

                    chunk_infos = data['chunk_info'] if isinstance(data['chunk_info'], list) else [data['chunk_info']]
                    for batch_idx, chunk_info in enumerate(chunk_infos):
                        subject_id = chunk_info['subject']
                        session_id = chunk_info['session']

                        session_key = (subject_id, session_id)
                        if session_key not in session_predictions:
                            session_predictions[session_key] = {
                                'labels': torch.zeros(
                                    N_VERTICES,
                                    dtype=torch.long
                                ),
                                'gate': torch.zeros(
                                    N_VERTICES,
                                    dtype=torch.float32
                                ),
                                'probabilities': torch.zeros(
                                    N_VERTICES,
                                    N_CLASSES,
                                    dtype=torch.float32
                                ) if SAVE_SESSION_PROBABILITIES else None
                            }
                        if subject_id not in subject_predictions:
                            subject_predictions[subject_id] = {
                                'probability_sum': torch.zeros(
                                    N_VERTICES,
                                    N_CLASSES,
                                    dtype=torch.float32
                                ),
                                'labels': torch.zeros(
                                    N_VERTICES,
                                    dtype=torch.long
                                ),
                                'adaptive_gate': torch.zeros(
                                    N_VERTICES,
                                    dtype=torch.float32
                                )
                            }
                        
                        start_idx = chunk_info['start']
                        original_size = chunk_info.get(
                            'original_size',
                            chunk_info['end'] - start_idx
                        )
                        chunk_valid = valid_masks[batch_idx, :original_size]
                        global_slice = slice(
                            start_idx,
                            start_idx + original_size
                        )
                        session_predictions[session_key]['labels'][global_slice] = (
                            predictions[batch_idx, :original_size].detach().cpu()
                        )
                        session_predictions[session_key]['gate'][global_slice] = (
                            gate_values[batch_idx, :original_size]
                            * chunk_valid.to(gate_values.dtype)
                        ).detach().cpu()
                        if SAVE_SESSION_PROBABILITIES:
                            session_predictions[session_key]['probabilities'][
                                global_slice
                            ] = (
                                probabilities[batch_idx, :original_size]
                                * chunk_valid.unsqueeze(-1).to(probabilities.dtype)
                            ).detach().cpu()
                        # The optional two-session aggregate is formed only
                        # after independent forward passes.  It is not used for
                        # test-retest Dice or cross-session homogeneity.
                        subject_predictions[subject_id]['probability_sum'][
                            global_slice
                        ] += (
                            probabilities[batch_idx, :original_size]
                            * chunk_valid.unsqueeze(-1).to(probabilities.dtype)
                            / len(self.session_ids)
                        ).detach().cpu()
                        subject_predictions[subject_id]['adaptive_gate'][
                            global_slice
                        ] += (
                            gate_values[batch_idx, :original_size]
                            * chunk_valid.to(gate_values.dtype)
                            / len(self.session_ids)
                        ).detach().cpu()

                for subject_id, aggregate in subject_predictions.items():
                    probability_sum = aggregate.pop('probability_sum')
                    labels = probability_sum.argmax(dim=-1) + 1
                    labels[~valid_cortex_mask] = 0
                    aggregate['labels'] = labels
                
                # Compute reproducibility and similarity metrics.
                reproducibility_dict = {}
                for subject_id in self.test_subjects:
                    session1_key = (subject_id, 1)
                    session2_key = (subject_id, 2)
                    if session1_key in session_predictions and session2_key in session_predictions:
                        labels1 = session_predictions[session1_key]['labels']
                        labels2 = session_predictions[session2_key]['labels']
                        reproducibility_dict[subject_id] = (
                            labels1[:N_VERTICES_PER_HEMI],
                            labels1[N_VERTICES_PER_HEMI:],
                            labels2[:N_VERTICES_PER_HEMI],
                            labels2[N_VERTICES_PER_HEMI:]
                        )
                
                # Hungarian matching is retained as the manuscript safeguard;
                # corresponding-label Dice is saved as a sensitivity result.
                reproducibility_details = self.evaluator.evaluate_reproducibility(
                    {k: (v[0], v[1]) for k, v in reproducibility_dict.items()},
                    {k: (v[2], v[3]) for k, v in reproducibility_dict.items()},
                    return_details=True
                )
                reproducibility = reproducibility_details['matched']

                # For each subject and network, compute the mean pairwise Pearson correlation
                # of whole-cortex connectivity profiles, then average equally across the 17 networks.
                homogeneity_subject_scores = []
                homogeneity_by_subject = {}
                for subject_id, session_labels in reproducibility_dict.items():
                    session_scores = []
                    label_pairs = (
                        (session_labels[0], session_labels[1]),
                        (session_labels[2], session_labels[3])
                    )
                    # Cross-session validation: labels inferred from one
                    # session are scored using the other session's profiles.
                    for label_session_index, (lh_labels, rh_labels) in enumerate(
                        label_pairs
                    ):
                        profile_session_id = self.session_ids[1 - label_session_index]
                        full_labels = torch.cat([lh_labels, rh_labels])
                        profile_path = os.path.join(
                            PROFILE_DIR,
                            f'sub{subject_id}',
                            f'sess{profile_session_id}',
                            f'sub{subject_id}_sess{profile_session_id}_fs_LR_32k_roifs_LR_900.surf2surf_profile.mat'
                        )
                        session_scores.append(
                            compute_connectivity_homogeneity_from_file(
                                full_labels,
                                profile_path,
                                valid_mask=valid_cortex_mask
                            )
                        )
                    subject_homogeneity = torch.stack(session_scores).mean()
                    homogeneity_subject_scores.append(subject_homogeneity)
                    homogeneity_by_subject[subject_id] = subject_homogeneity
                homogeneity = torch.stack(homogeneity_subject_scores).mean()

                # Manuscript reference: Session-2 identities are subjected to
                # 10,000 restricted one-to-one derangements. Self-pairings and
                # within-family pairings are prohibited.
                session1_maps = {
                    subject_id: session_predictions[(subject_id, 1)]['labels']
                    for subject_id in self.test_subjects
                }
                session2_maps = {
                    subject_id: session_predictions[(subject_id, 2)]['labels']
                    for subject_id in self.test_subjects
                }
                reference = restricted_derangement_reproducibility(
                    session1_maps,
                    session2_maps,
                    self.test_family_ids,
                    n_permutations=REPRODUCIBILITY_DERANGEMENTS,
                    random_state=RANDOM_SEED,
                )
                similarity = float(np.mean(reference['between_dice_matched']))
                between_subject_nmi = float(np.mean(reference['between_nmi']))
                
                # Save evaluation results.
                evaluation_results = {
                    'reproducibility': reproducibility.item(),
                    'reproducibility_matched': reproducibility_details['matched'].item(),
                    'reproducibility_direct': reproducibility_details['direct'].item(),
                    'reproducibility_nmi': reproducibility_details['nmi'].item(),
                    'homogeneity': homogeneity.item(),
                    'between_subject_dice': similarity,
                    'between_subject_nmi': between_subject_nmi,
                    'dice_p_adjusted': reference['p_adjusted_dice'],
                    'nmi_p_adjusted': reference['p_adjusted_nmi'],
                }
                
                # Save parcellations and evaluation results.
                for subject_id in subject_predictions:
                    session1 = session_predictions[(subject_id, 1)]
                    session2 = session_predictions[(subject_id, 2)]
                    subject_predictions[subject_id].update({
                        'session1_labels': session1['labels'],
                        'session2_labels': session2['labels'],
                        'session1_gate': session1['gate'],
                        'session2_gate': session2['gate'],
                        'session1_probabilities': session1['probabilities'],
                        'session2_probabilities': session2['probabilities'],
                    })
                    self.save_subject_results(
                        subject_id,
                        subject_predictions[subject_id]
                    )
                
                # Save evaluation results.
                self.save_test_evaluation(evaluation_results)
                self.save_subject_level_metrics(
                    reproducibility_details,
                    homogeneity_by_subject
                )
                self.save_reproducibility_reference(reference)
                
                self.logger.info(f"\nTest Results:")
                self.logger.info(f"Reproducibility (direct): {reproducibility_details['direct'].item():.4f}")
                self.logger.info(f"Reproducibility (NMI): {reproducibility_details['nmi'].item():.4f}")
                self.logger.info(f"Resting-state homogeneity: {homogeneity.item():.4f}")
                self.logger.info(f"Between-subject Dice: {similarity:.4f}")
                self.logger.info(f"Between-subject NMI: {between_subject_nmi:.4f}")
                self.logger.info(
                    "Restricted-reference Holm-adjusted P: Dice=%.6g, NMI=%.6g",
                    reference['p_adjusted_dice'],
                    reference['p_adjusted_nmi'],
                )
                
        except Exception as e:
            self.logger.error(f"Error in test phase: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    
    def save_subject_level_metrics(
        self,
        reproducibility_details,
        homogeneity_by_subject
    ):
        """Save observed per-subject values required for paired statistics."""
        try:
            per_subject = reproducibility_details['per_subject']
            subject_ids = [
                subject_id for subject_id in self.test_subjects
                if subject_id in per_subject
                and subject_id in homogeneity_by_subject
            ]
            if not subject_ids:
                raise ValueError("No complete subject-level metrics were produced.")

            save_dict = {
                'subject_ids': np.asarray(subject_ids).reshape(-1, 1),
                'dice_matched': np.asarray([
                    per_subject[s]['matched'].item() for s in subject_ids
                ]).reshape(-1, 1),
                'dice_direct': np.asarray([
                    per_subject[s]['direct'].item() for s in subject_ids
                ]).reshape(-1, 1),
                'nmi': np.asarray([
                    per_subject[s]['nmi'].item() for s in subject_ids
                ]).reshape(-1, 1),
                'homogeneity': np.asarray([
                    homogeneity_by_subject[s].item() for s in subject_ids
                ]).reshape(-1, 1),
                'network_dice_direct': np.stack([
                    per_subject[s]['network_dice_direct'].detach().cpu().numpy()
                    for s in subject_ids
                ]),
                'network_dice_matched': np.stack([
                    per_subject[s]['network_dice_matched'].detach().cpu().numpy()
                    for s in subject_ids
                ]),
                'hungarian_permutation': np.stack([
                    per_subject[s]['hungarian_permutation'].detach().cpu().numpy()
                    for s in subject_ids
                ]),
                'identity_correspondence': np.asarray([
                    per_subject[s]['identity_correspondence'] for s in subject_ids
                ], dtype=np.uint8).reshape(-1, 1),
            }
            save_path = os.path.join(
                self.results_dir,
                'subject_level_metrics.mat'
            )
            sio.savemat(save_path, save_dict)
            self.logger.info(
                f"Subject-level metrics saved to {save_path}"
            )
        except Exception as e:
            self.logger.error(
                f"Error saving subject-level metrics: {str(e)}"
            )
            raise

    def save_reproducibility_reference(self, reference):
        """Save the observed and restricted-derangement distributions."""
        save_path = os.path.join(
            self.results_dir, 'restricted_reproducibility_reference.mat'
        )
        sio.savemat(save_path, {
            key: value for key, value in reference.items()
            if key != 'subjects'
        } | {
            'subject_ids': np.asarray(reference['subjects']).reshape(-1, 1)
        })
        self.logger.info(
            "Restricted reproducibility reference saved to %s", save_path
        )

    def save_checkpoint(self, path):
        """Save a SPIN-Net checkpoint."""
        try:
            checkpoint = {
                'epoch': self.current_epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'best_metric': self.best_metric,
                'model_type': 'SPIN-Net',
                'config': {
                    'input_dim': self.model.input_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'num_classes': self.model.num_classes,
                    'architecture_version': self.model.architecture_version,
                    'variant': self.model.variant,
                    'fixed_alpha': self.model.fixed_alpha,
                    'contrastive_resolution': self.contrastive_resolution,
                    'resolved_experiment': config_snapshot(),
                }
            }

            # Ensure the directory exists.
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save the checkpoint.
            torch.save(checkpoint, path)
            self.logger.info(f"Checkpoint saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {str(e)}")
            raise
            
    def load_checkpoint(self, path):
        """Load a checkpoint."""
        try:
            if path is None:
                self.logger.warning("No checkpoint path provided, using initial model state")
                return
                
            if not os.path.exists(path):
                raise FileNotFoundError(f"Checkpoint file not found: {path}")
                    
            self.logger.info(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path)
            checkpoint_config = checkpoint.get('config', {})
            checkpoint_architecture = checkpoint_config.get(
                'architecture_version'
            )
            if checkpoint_architecture != self.model.architecture_version:
                raise ValueError(
                    "Checkpoint architecture is incompatible with the current model. "
                    "Retrain this configuration "
                    f"with architecture_version={self.model.architecture_version}."
                )
            checkpoint_variant = checkpoint_config.get('variant', 'full')
            if checkpoint_config.get('contrastive_resolution', 'chunk') != self.contrastive_resolution:
                raise ValueError("Checkpoint contrastive resolution differs from the requested experiment.")
            checkpoint_alpha = checkpoint_config.get('fixed_alpha')
            if checkpoint_variant != self.model.variant:
                raise ValueError(
                    f"Checkpoint variant '{checkpoint_variant}' does not match "
                    f"requested variant '{self.model.variant}'."
                )
            if checkpoint_alpha != self.model.fixed_alpha:
                raise ValueError(
                    f"Checkpoint fixed_alpha={checkpoint_alpha} does not match "
                    f"requested fixed_alpha={self.model.fixed_alpha}."
                )
            
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.current_epoch = checkpoint['epoch']
            self.best_metric = checkpoint['best_metric']
            
            self.logger.info("Checkpoint loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {str(e)}")
            if path is not None:  # Raise only when a checkpoint path was supplied but loading failed.
                raise

    def run(self):
        """Run the complete training, validation, and evaluation pipeline."""
        try:
            best_checkpoint_path = None

            # Training phase.
            if not self.args.test_only:
                self.logger.info("Starting training...")
                best_val_loss = float('inf')
                
                for epoch in range(self.args.epochs):
                    self.current_epoch = epoch
                    train_loss, train_metrics = self.train_epoch(epoch)
                    val_loss, val_metrics = self.validate()
                    
                    # The learning-rate scheduler already advances after each optimizer step.
                    
                    # Save the best model.
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint_path = os.path.join(
                            self.checkpoint_dir,
                            f'model_epoch_{epoch}.pt'
                        )
                        self.save_checkpoint(best_checkpoint_path)
                        
                # Save the final model.
                final_checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    'model_final.pt'
                )
                self.save_checkpoint(final_checkpoint_path)
                
                # Evaluate using the best checkpoint.
                if best_checkpoint_path is not None:
                    self.load_checkpoint(best_checkpoint_path)
                
            elif self.args.checkpoint:
                # Evaluation-only mode requires a checkpoint.
                self.load_checkpoint(self.args.checkpoint)
                
            # Run evaluation.
            self.test()
                
        except Exception as e:
            self.logger.error(f"Error in training/testing: {str(e)}")
            raise

def parse_args():
    parser = argparse.ArgumentParser(description='SPIN-Net训练脚本')
    parser.add_argument(
        '--batch_size', type=int, default=BATCH_SIZE,
        help='每个mini-batch中的被试对数（每对含两个session）'
    )
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='训练轮数')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--test_only', action='store_true', help='仅测试模式')
    parser.add_argument(
        '--variant',
        choices=['full', 'without_in', 'without_s', 'without_p', 'fixed_alpha'],
        default='full',
        help='论文主模型或消融配置；每个配置必须独立训练'
    )
    parser.add_argument(
        '--fixed_alpha',
        type=float,
        default=None,
        help='固定门控权重，仅与 --variant fixed_alpha 一起使用'
    )
    return parser.parse_args()

if __name__ == '__main__':
    warnings.filterwarnings("ignore", message="pixdim[1,2,3] should be non-zero; setting 0 dims to 1")
    args = parse_args()
    if args.test_only and not args.checkpoint:
        raise ValueError("--test_only requires a trained --checkpoint.")
    trainer = Trainer(args)
    trainer.run()
