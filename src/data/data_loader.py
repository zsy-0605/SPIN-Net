import os
import json
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Sampler
from ..config import *
from ..utils.graph_utils import load_medial_wall_mask

_PRIORS_CACHE = None


def _read_json_sidecar(path, description, strict=STRICT_DATA_PROVENANCE):
    """Read provenance JSON without ever inventing missing metadata."""
    if not os.path.exists(path):
        if strict:
            raise FileNotFoundError(
                f"Missing {description}: {path}. This metadata is required to "
                "verify data preparation. Provide the JSON sidecar, or set "
                "SPIN_STRICT_DATA_PROVENANCE=0 for legacy/synthetic tests."
            )
        return None
    try:
        with open(path, "r", encoding="utf-8") as sidecar_file:
            metadata = json.load(sidecar_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {description}: {path}") from exc
    except OSError as exc:
        raise OSError(f"Unable to read {description}: {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return metadata


def _id_key(value, field_name):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{field_name} values must be non-empty strings or integers.")
    key = str(value).strip()
    if not key:
        raise ValueError(f"{field_name} values must not be empty.")
    return key


def _manifest_split_entry(entry, split_name):
    """Return (subject_ids, family_ids-or-None) from one manifest split."""
    families = None
    if isinstance(entry, list):
        if entry and all(isinstance(item, dict) for item in entry):
            if not all("subject_id" in item for item in entry):
                raise ValueError(
                    f"Every record in split '{split_name}' must contain subject_id."
                )
            subject_ids = [item["subject_id"] for item in entry]
            has_family = ["family_id" in item for item in entry]
            if any(has_family) and not all(has_family):
                raise ValueError(
                    f"Split '{split_name}' has incomplete family_id metadata."
                )
            if all(has_family):
                families = [item["family_id"] for item in entry]
        elif any(isinstance(item, dict) for item in entry):
            raise ValueError(
                f"Split '{split_name}' cannot mix subject records and scalar IDs."
            )
        else:
            subject_ids = entry
    elif isinstance(entry, dict):
        subject_ids = entry.get("subject_ids")
        families = entry.get("family_ids")
        if not isinstance(subject_ids, list):
            raise ValueError(
                f"Split '{split_name}' must contain a subject_ids list."
            )
        if families is not None and not isinstance(families, list):
            raise ValueError(
                f"family_ids for split '{split_name}' must be a list."
            )
    else:
        raise ValueError(
            f"Split '{split_name}' must be a list or an object with subject_ids."
        )

    if families is not None and len(families) != len(subject_ids):
        raise ValueError(
            f"Split '{split_name}' has {len(subject_ids)} subject IDs but "
            f"{len(families)} family IDs."
        )
    return list(subject_ids), None if families is None else list(families)


def load_and_validate_split_manifest(
    manifest_path=None,
    strict=None,
):
    """Validate the 400/100/200 subject- and family-level split."""
    manifest_path = SPLIT_MANIFEST_PATH if manifest_path is None else manifest_path
    strict = STRICT_DATA_PROVENANCE if strict is None else strict
    manifest = _read_json_sidecar(
        manifest_path, "subject split manifest", strict=strict
    )
    if manifest is None:
        return {
            "train": [],
            "validation": [],
            "test": [],
            "family_ids": None,
        }

    split_root = manifest.get("splits", manifest)
    if not isinstance(split_root, dict):
        raise ValueError("Split manifest 'splits' field must be a JSON object.")

    source_names = {
        "train": "train",
        "validation": "validation" if "validation" in split_root else "val",
        "test": "test",
    }
    missing = [
        source for source in source_names.values()
        if source not in split_root
    ]
    if missing:
        raise ValueError(
            "Split manifest must define train, validation (or val), and test; "
            f"missing: {', '.join(missing)}."
        )

    splits = {}
    family_lists = {}
    for split_name, source_name in source_names.items():
        subjects, families = _manifest_split_entry(
            split_root[source_name], split_name
        )
        expected_size = EXPECTED_SPLIT_SIZES[split_name]
        if len(subjects) != expected_size:
            raise ValueError(
                f"Split '{split_name}' must contain {expected_size} subjects; "
                f"found {len(subjects)}."
            )
        keys = [_id_key(value, f"{split_name}.subject_id") for value in subjects]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Split '{split_name}' contains duplicate subject IDs.")
        splits[split_name] = subjects
        family_lists[split_name] = families

    subject_sets = {
        name: {_id_key(value, f"{name}.subject_id") for value in values}
        for name, values in splits.items()
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = subject_sets[left] & subject_sets[right]
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise ValueError(
                f"Subject leakage between '{left}' and '{right}' splits: {examples}."
            )

    # A compact parallel family_ids object at the manifest root is also accepted.
    root_families = manifest.get("family_ids")
    if root_families is not None:
        if not isinstance(root_families, dict):
            raise ValueError("Top-level family_ids must be an object keyed by split.")
        for split_name, source_name in source_names.items():
            if family_lists[split_name] is not None:
                continue
            family_values = root_families.get(
                source_name, root_families.get(split_name)
            )
            if family_values is not None:
                if not isinstance(family_values, list):
                    raise ValueError(
                        f"family_ids for split '{split_name}' must be a list."
                    )
                if len(family_values) != len(splits[split_name]):
                    raise ValueError(
                        f"family_ids for split '{split_name}' must align one-to-one "
                        "with subject_ids."
                    )
                family_lists[split_name] = family_values

    has_family_metadata = [
        family_lists[name] is not None for name in ("train", "validation", "test")
    ]
    if any(has_family_metadata) and not all(has_family_metadata):
        raise ValueError(
            "Family metadata is incomplete: provide family IDs for all three "
            "splits or omit family IDs entirely."
        )
    if strict and not all(has_family_metadata):
        raise ValueError(
            "Strict provenance validation requires family IDs for train, "
            "validation, and test so family-wise split leakage can be checked."
        )
    if all(has_family_metadata):
        family_sets = {}
        for split_name, families in family_lists.items():
            family_sets[split_name] = {
                _id_key(value, f"{split_name}.family_id") for value in families
            }
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = family_sets[left] & family_sets[right]
            if overlap:
                examples = ", ".join(sorted(overlap)[:5])
                raise ValueError(
                    f"Family leakage between '{left}' and '{right}' splits: "
                    f"{examples}."
                )

    return {
        **splits,
        "family_ids": family_lists if all(has_family_metadata) else None,
    }


def validate_group_atlas_sidecar(
    training_subjects,
    sidecar_path=None,
    strict=None,
):
    """Verify that the group atlas was made from exactly the training split."""
    sidecar_path = (
        GROUP_ATLAS_SIDECAR_PATH if sidecar_path is None else sidecar_path
    )
    strict = STRICT_DATA_PROVENANCE if strict is None else strict
    metadata = _read_json_sidecar(
        sidecar_path, "group-atlas provenance sidecar", strict=strict
    )
    if metadata is None:
        return None
    atlas_subjects = metadata.get("training_subjects")
    if not isinstance(atlas_subjects, list):
        raise ValueError(
            "Group-atlas sidecar must contain a training_subjects list."
        )
    atlas_keys = [_id_key(value, "training_subjects") for value in atlas_subjects]
    split_keys = [_id_key(value, "train.subject_id") for value in training_subjects]
    if len(set(atlas_keys)) != len(atlas_keys):
        raise ValueError("Group-atlas sidecar contains duplicate training subjects.")
    if set(atlas_keys) != set(split_keys):
        missing = sorted(set(split_keys) - set(atlas_keys))
        unexpected = sorted(set(atlas_keys) - set(split_keys))
        raise ValueError(
            "Group atlas training_subjects do not match the training split "
            f"(missing={missing[:5]}, unexpected={unexpected[:5]})."
        )
    prior_type = str(metadata.get("prior_type", "")).strip().lower()
    if strict and prior_type not in {"probabilistic", "probability", "soft"}:
        raise ValueError(
            "Group-atlas sidecar must declare prior_type='probabilistic'."
        )
    source_method = str(
        metadata.get("source_method", metadata.get("construction_method", ""))
    ).strip().lower().replace("_", "-")
    if strict and source_method not in {"ms-hbm", "mshbm"}:
        raise ValueError(
            "Group-atlas sidecar must declare source_method='MS-HBM' to match "
            "the manuscript prior construction."
        )
    if "networks" in metadata and int(metadata["networks"]) != N_CLASSES:
        raise ValueError(f"Group prior must contain {N_CLASSES} networks.")
    return metadata


def _session_run_specifications(metadata):
    sessions = metadata.get("sessions")
    if isinstance(sessions, dict):
        session_specs = list(sessions.values())
    elif isinstance(sessions, list):
        session_specs = sessions
    else:
        raise ValueError(
            "Profile sidecar must contain two session specifications in 'sessions'."
        )
    if len(session_specs) != 2:
        raise ValueError(
            f"Profile sidecar must describe exactly two sessions; found "
            f"{len(session_specs)}."
        )
    return session_specs


def validate_profile_sidecar(
    sidecar_path=None,
    strict=None,
):
    """Verify session averaging and per-vertex top-10% weighted profiles."""
    sidecar_path = PROFILE_SIDECAR_PATH if sidecar_path is None else sidecar_path
    strict = STRICT_DATA_PROVENANCE if strict is None else strict
    metadata = _read_json_sidecar(
        sidecar_path, "connectivity-profile provenance sidecar", strict=strict
    )
    if metadata is None:
        return None

    default_aggregation = metadata.get(
        "session_aggregation", metadata.get("run_aggregation")
    )
    for index, session_spec in enumerate(
        _session_run_specifications(metadata), start=1
    ):
        if isinstance(session_spec, list):
            runs = session_spec
            aggregation = default_aggregation
        elif isinstance(session_spec, dict):
            runs = session_spec.get("runs")
            aggregation = session_spec.get(
                "aggregation",
                session_spec.get("run_aggregation", default_aggregation),
            )
        else:
            raise ValueError(
                f"Session {index} specification must be a run list or JSON object."
            )
        if not isinstance(runs, list) or len(runs) != 2:
            raise ValueError(
                f"Session {index} must declare exactly two input runs."
            )
        if not isinstance(aggregation, str) or aggregation.strip().lower() not in {
            "mean", "average", "averaged"
        }:
            raise ValueError(
                f"Session {index} must declare mean aggregation across its two runs."
            )

    top_fraction = metadata.get("per_vertex_top_fraction")
    if (
        isinstance(top_fraction, bool)
        or not isinstance(top_fraction, (int, float))
        or not np.isclose(
            float(top_fraction),
            float(CONFIG["data"]["per_vertex_top_fraction"]),
            rtol=0.0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Profile sidecar per_vertex_top_fraction must equal the configured "
            f"value {CONFIG['data']['per_vertex_top_fraction']}."
        )
    if metadata.get("profile_value_mode") != "sparse_weighted":
        raise ValueError(
            "Profile sidecar profile_value_mode must be sparse_weighted."
        )
    return metadata


def validate_data_provenance(subject_ids=None, require_group_atlas=False):
    """Run all data-provenance checks."""
    splits = load_and_validate_split_manifest()
    validate_profile_sidecar()
    if subject_ids is not None and os.path.exists(SPLIT_MANIFEST_PATH):
        declared = {
            _id_key(value, "manifest.subject_id")
            for split_name in ("train", "validation", "test")
            for value in splits[split_name]
        }
        unknown = {
            _id_key(value, "requested.subject_id") for value in subject_ids
        } - declared
        if unknown:
            raise ValueError(
                "Requested subjects are absent from the split manifest: "
                + ", ".join(sorted(unknown)[:5])
            )
    if require_group_atlas:
        validate_group_atlas_sidecar(splits["train"])
    return splits


def validate_profile_inventory(subject_ids, session_ids=SESSION_IDS):
    """Require one prepared weighted profile for every declared subject/session."""
    session_ids = tuple(session_ids)
    missing = []
    for subject in subject_ids:
        for session in session_ids:
            path = os.path.join(
                PROFILE_DIR,
                f"sub{subject}",
                f"sess{session}",
                f"sub{subject}_sess{session}_fs_LR_32k_roifs_LR_900.surf2surf_profile.mat",
            )
            if not os.path.isfile(path):
                missing.append(path)
    if missing:
        preview = "\n".join(missing[:10])
        suffix = f"\n... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} declared subject/session profile files:\n"
            f"{preview}{suffix}"
        )
    return len(subject_ids) * len(session_ids)

class MemoryEfficientDataset(Dataset):
   def __init__(
       self,
       profile_dir,
       subject_ids,
       session_ids,
       include_prior=True
   ):
       self.profile_dir = profile_dir
       self.subject_ids = list(subject_ids)
       self.session_ids = list(session_ids)
       self.include_prior = bool(include_prior)
       validate_data_provenance(subject_ids=self.subject_ids)
       self.valid_mask = torch.from_numpy(
           load_medial_wall_mask(MEDIAL_WALL_MASK_PATH)
       ).bool()
       if self.valid_mask.numel() != N_VERTICES:
           raise ValueError(f"Expected {N_VERTICES} mask rows.")
       if int(self.valid_mask.sum()) != N_VALID_VERTICES:
           raise ValueError(
               f"Expected {N_VALID_VERTICES} valid cortical vertices, "
               f"found {int(self.valid_mask.sum())}."
           )
       self.chunk_size = CHUNK_SIZE
       self.total_chunks = (N_VERTICES + CHUNK_SIZE - 1) // CHUNK_SIZE

   def __len__(self):
       return len(self.subject_ids) * len(self.session_ids) * self.total_chunks
   
   def load_profile_chunk(self, sub_id, sess_id, start_idx, end_idx):
        profile_path = os.path.join(
            self.profile_dir,
            f'sub{sub_id}',
            f'sess{sess_id}',
            f'sub{sub_id}_sess{sess_id}_fs_LR_32k_roifs_LR_900.surf2surf_profile.mat'
        )
        
        try:
            if not os.path.exists(profile_path):
                raise FileNotFoundError(f"Profile file not found: {profile_path}")
                
            with h5py.File(profile_path, 'r') as f:
                profile_ds = f['profile_mat']
                
                if profile_ds.shape == (N_VERTICES, N_ROIS):
                    chunk_data = profile_ds[start_idx:end_idx, :]
                elif profile_ds.shape == (N_ROIS, N_VERTICES):
                    chunk_data = profile_ds[:, start_idx:end_idx].T
                else:
                    raise ValueError(
                        f"Incorrect profile matrix shape: expected {(N_VERTICES, N_ROIS)}, "
                        f"or {(N_ROIS, N_VERTICES)}, got {profile_ds.shape}"
                    )

                # Convert to a tensor.
                data = torch.from_numpy(chunk_data).float()
                if not torch.isfinite(data).all():
                    raise ValueError(f"Profile contains non-finite values: {profile_path}")
                if torch.any(torch.abs(data) > 1.0 + 1e-6):
                    raise ValueError(
                        "Profile input must contain sparse weighted Pearson "
                        "correlations in [-1, 1]: "
                        f"{profile_path}"
                    )
                
                # Preserve retained correlation weights without row normalization.
                return data
                    
        except Exception as e:
            print(f"\nError loading profile data:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

   def __getitem__(self, idx):
        try:
            # Compute sample indices.
            chunk_idx = idx % self.total_chunks
            temp_idx = idx // self.total_chunks
            subject_idx = temp_idx // len(self.session_ids)
            session_idx = temp_idx % len(self.session_ids)
            
            # Get subject and session identifiers.
            sub_id = self.subject_ids[subject_idx]
            sess_id = self.session_ids[session_idx]
            
            # Compute the chunk interval.
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, N_VERTICES)
            chunk_size = end_idx - start_idx
            
            # Load data without adding extra dimensions.
            profile_data = self.load_profile_chunk(
                sub_id, sess_id, start_idx, end_idx
            )  # [chunk_size, N_ROIS]
            
            valid_mask_chunk = self.valid_mask[start_idx:end_idx]
            if self.include_prior:
                priors = load_priors()
                theta_chunk = priors['theta'][start_idx:end_idx]
            else:
                # Without P, do not load the group atlas; use a placeholder to preserve the collate interface.
                # The model does not use this tensor.
                theta_chunk = torch.zeros(
                    end_idx - start_idx,
                    N_CLASSES,
                    dtype=PROFILE_DTYPE
                )
            
            return {
                'profile': profile_data,  # [chunk_size, N_ROIS]
                'theta': theta_chunk,     # [chunk_size, N_CLASSES]
                'valid_mask': valid_mask_chunk,
                'chunk_info': {
                    'start': start_idx,
                    'end': end_idx,
                    'subject': sub_id,
                    'session': sess_id,
                    'original_size': end_idx - start_idx
                }
            }
            
        except Exception as e:
            print(f"\nError in __getitem__:")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


class CrossSessionChunkBatchSampler(Sampler):
   """Batch ``batch_size`` subject pairs from the same cortical chunk."""

   def __init__(self, dataset, batch_size=BATCH_SIZE, shuffle=True):
       self.dataset = dataset
       if len(dataset.session_ids) != 2:
           raise ValueError("Cross-session sampling requires exactly two sessions.")
       self.subject_pairs_per_batch = int(batch_size)
       if self.subject_pairs_per_batch < 1:
           raise ValueError("batch_size denotes subject pairs and must be positive.")
       self.shuffle = shuffle



class IndependentChunkBatchSampler(Sampler):
   """Batch independent subject/session samples from one cortical chunk.

   Unlike :class:`CrossSessionChunkBatchSampler`, this sampler does not place
   paired sessions in the same batch.  It is intended for validation-free
   inference, where a parcellation must be obtainable from one session alone.
   """

   def __init__(self, dataset, batch_size=1, shuffle=False):
       self.dataset = dataset
       self.batch_size = int(batch_size)
       if self.batch_size < 1:
           raise ValueError("batch_size must be positive.")
       self.shuffle = bool(shuffle)

   def __iter__(self):
       import random

       sample_indices = [
           (subject_idx, session_idx)
           for subject_idx in range(len(self.dataset.subject_ids))
           for session_idx in range(len(self.dataset.session_ids))
       ]
       chunk_indices = list(range(self.dataset.total_chunks))
       if self.shuffle:
           random.shuffle(chunk_indices)

       for chunk_idx in chunk_indices:
           current = sample_indices[:]
           if self.shuffle:
               random.shuffle(current)
           for start in range(0, len(current), self.batch_size):
               batch = []
               for subject_idx, session_idx in current[start:start + self.batch_size]:
                   temporary = (
                       subject_idx * len(self.dataset.session_ids) + session_idx
                   )
                   batch.append(
                       temporary * self.dataset.total_chunks + chunk_idx
                   )
               if batch:
                   yield batch

   def __len__(self):
       import math

       samples = len(self.dataset.subject_ids) * len(self.dataset.session_ids)
       return self.dataset.total_chunks * math.ceil(samples / self.batch_size)

def sparse_batch_collate(batch):
    """Collate samples into a padded batch."""
    try:
        # Find the largest sample size in the batch.
        max_size = max([sample['profile'].size(0) for sample in batch])
        
        batch_dict = {
            'profile': [],
            'theta': [],
            'valid_mask': [],
            'chunk_info': []
        }
        
        # Process each sample in the batch.
        for sample in batch:
            curr_size = sample['profile'].size(0)
            
            # Process connectivity profiles.
            if curr_size < max_size:
                pad_size = max_size - curr_size
                profile_pad = torch.zeros(
                    pad_size, 
                    N_ROIS,  # Use the configured dimension.
                    dtype=sample['profile'].dtype,
                    device=sample['profile'].device
                )
                padded_profile = torch.cat([sample['profile'], profile_pad], dim=0)
            else:
                padded_profile = sample['profile']
            batch_dict['profile'].append(padded_profile)
            
            # Process Theta.    
            if curr_size < max_size:
                theta_pad = torch.zeros(
                    pad_size, 
                    N_CLASSES,  # Use the configured dimension.
                    dtype=sample['theta'].dtype,
                    device=sample['theta'].device
                )
                padded_theta = torch.cat([sample['theta'], theta_pad], dim=0)
            else:
                padded_theta = sample['theta']
            batch_dict['theta'].append(padded_theta)

            # Medial-wall mask processing
            if curr_size < max_size:
                mask_pad = torch.zeros(
                    pad_size,
                    dtype=torch.bool,
                    device=sample['valid_mask'].device
                )
                padded_mask = torch.cat([sample['valid_mask'], mask_pad], dim=0)
            else:
                padded_mask = sample['valid_mask']
            batch_dict['valid_mask'].append(padded_mask)
                
            # Update chunk metadata.
            chunk_info = sample['chunk_info'].copy()
            chunk_info.update({
                'padded_size': max_size,
                'original_size': curr_size
            })
            batch_dict['chunk_info'].append(chunk_info)
            
        # Build the output dictionary.
        output = {
            'profile': torch.stack(batch_dict['profile']),  # [batch_size, max_size, N_ROIS]
            'theta': torch.stack(batch_dict['theta']),      # [batch_size, max_size, N_CLASSES]
            'valid_mask': torch.stack(batch_dict['valid_mask']),
            'chunk_info': batch_dict['chunk_info']
        }
            
        return output
        
    except Exception as e:
        print(f"\nError in sparse_batch_collate:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print(f"Batch shapes:")
        for k, v in batch[0].items():
            if torch.is_tensor(v):
                print(f"{k}: {v.shape}")
        raise

def get_dataloader(
    subject_ids,
    session_ids,
    batch_size=BATCH_SIZE,
    shuffle=True,
    include_prior=True,
    paired_sessions=True,
):
    dataset = MemoryEfficientDataset(
        PROFILE_DIR,
        subject_ids,
        session_ids,
        include_prior=include_prior
    )
    if paired_sessions:
        batch_sampler = CrossSessionChunkBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    else:
        batch_sampler = IndependentChunkBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        collate_fn=sparse_batch_collate
    )

def _mat_mapping(path):
    """Read a MATLAB mapping while preserving array orientation."""
    try:
        import scipy.io as sio
        return {key: value for key, value in sio.loadmat(path).items() if not key.startswith("__")}
    except NotImplementedError:
        with h5py.File(path, "r") as handle:
            return {key: np.asarray(value) for key, value in handle.items()}


def _as_vertex_probability(array, expected_vertices):
    value = np.asarray(array, dtype=np.float32)
    value = np.squeeze(value)
    if value.shape == (N_CLASSES, expected_vertices):
        value = value.T
    if value.shape != (expected_vertices, N_CLASSES):
        raise ValueError(
            f"Expected probability array {(expected_vertices, N_CLASSES)}, got {value.shape}."
        )
    return value


def load_priors():
    """Load the fixed training-set probabilistic spatial prior ``Theta``."""
    global _PRIORS_CACHE
    if _PRIORS_CACHE is not None:
        return _PRIORS_CACHE

    try:
        validate_data_provenance(require_group_atlas=True)
        print(f"\nLoading group atlas from {GROUP_ATLAS_PATH}")

        atlas_data = _mat_mapping(GROUP_ATLAS_PATH)
        full_names = ("theta", "Theta", "probabilities", "group_probabilities")
        hemi_name_pairs = (
            ("lh_theta", "rh_theta"),
            ("lh_probabilities", "rh_probabilities"),
            ("lh_prob", "rh_prob"),
        )
        theta_np = None
        for name in full_names:
            if name in atlas_data:
                theta_np = _as_vertex_probability(atlas_data[name], N_VERTICES)
                break
        if theta_np is None:
            for left_name, right_name in hemi_name_pairs:
                if left_name in atlas_data and right_name in atlas_data:
                    theta_np = np.concatenate([
                        _as_vertex_probability(atlas_data[left_name], N_VERTICES_PER_HEMI),
                        _as_vertex_probability(atlas_data[right_name], N_VERTICES_PER_HEMI),
                    ])
                    break
        if theta_np is None:
            raise KeyError(
                "The group-prior file must contain a probabilistic Theta array "
                "(V x 17), not only hard atlas labels. Accepted fields include "
                "theta or paired lh_theta/rh_theta."
            )

        valid_mask = torch.from_numpy(
            load_medial_wall_mask(MEDIAL_WALL_MASK_PATH)
        ).bool()
        if valid_mask.numel() != N_VERTICES:
            raise ValueError(
                f"Expected {N_VERTICES} medial-wall mask entries, "
                f"got {valid_mask.numel()}."
            )
        theta = torch.from_numpy(theta_np).to(PROFILE_DTYPE)
        if not torch.isfinite(theta).all() or torch.any(theta < 0):
            raise ValueError("The probabilistic group prior must be finite and non-negative.")
        sums = theta.sum(dim=-1)
        if not torch.allclose(sums[valid_mask], torch.ones_like(sums[valid_mask]), atol=1e-5):
            raise ValueError("Theta must sum to one at each valid cortical vertex.")
        theta[~valid_mask] = 0.0
        atlas_labels = (theta.argmax(dim=-1) + 1).masked_fill(~valid_mask, 0)

        print("Loaded training-set group atlas:")
        print(f"labels shape: {atlas_labels.shape}")
        print(f"probabilistic prior shape: {theta.shape}")

        _PRIORS_CACHE = {
            'labels': atlas_labels,
            'theta': theta,
            'valid_mask': valid_mask
        }
        return _PRIORS_CACHE
    except Exception as e:
        print(f"\nError in load_priors:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print(f"Group atlas exists: {os.path.exists(GROUP_ATLAS_PATH)}")
        print(f"Group atlas path: {GROUP_ATLAS_PATH}")
        raise
