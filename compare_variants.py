#!/usr/bin/env python3
"""Compute paired statistics from saved subject-level results."""

import argparse
import json
import math

import numpy as np
import scipy.io as sio

from src.utils.statistics import compute_paired_statistics
from src.analysis.behavior_prediction import holm_adjust


METRICS = {
    "homogeneity": "homogeneity",
    "dice": "dice_matched",
    "nmi": "nmi"
}


def _subject_key(value):
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Invalid subject identifier shape: {array.shape}")
    scalar = array[0]
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return str(scalar).strip()


def load_subject_metrics(path):
    data = sio.loadmat(path, squeeze_me=True)
    if "subject_ids" not in data:
        raise KeyError(f"'subject_ids' is missing from {path}")
    subject_values = np.atleast_1d(data["subject_ids"])
    subjects = [_subject_key(value) for value in subject_values]
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Duplicate subject IDs in {path}")

    metrics = {}
    for output_name, mat_name in METRICS.items():
        if mat_name not in data:
            raise KeyError(f"'{mat_name}' is missing from {path}")
        values = np.asarray(data[mat_name], dtype=float).reshape(-1)
        if values.size != len(subjects):
            raise ValueError(
                f"{mat_name} has {values.size} values for "
                f"{len(subjects)} subjects in {path}"
            )
        metrics[output_name] = dict(zip(subjects, values))
    return subjects, metrics


def compare(full_path, variant_paths):
    full_subjects, full_metrics = load_subject_metrics(full_path)
    results = {}
    for variant_name, variant_path in variant_paths.items():
        variant_subjects, variant_metrics = load_subject_metrics(variant_path)
        if set(variant_subjects) != set(full_subjects):
            missing = sorted(set(full_subjects) - set(variant_subjects))
            extra = sorted(set(variant_subjects) - set(full_subjects))
            raise ValueError(
                f"Subject mismatch for {variant_name}; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

        results[variant_name] = {}
        for metric_name in METRICS:
            full_values = [
                full_metrics[metric_name][subject]
                for subject in full_subjects
            ]
            variant_values = [
                variant_metrics[metric_name][subject]
                for subject in full_subjects
            ]
            results[variant_name][metric_name] = (
                compute_paired_statistics(
                    full_values,
                    variant_values
                ).to_dict()
            )
    # The manuscript treats module-removal and fixed-gating comparisons as
    # separate multiple-comparison families for each outcome.
    families = {}
    for name in variant_paths:
        normalized = name.lower().replace("-", "_")
        if normalized.startswith("fixed_alpha"):
            family = "fixed_gating"
        elif normalized in {"without_in", "without_s", "without_p"}:
            family = "module_removal"
        else:
            family = f"standalone:{name}"
        families.setdefault(family, []).append(name)
    for metric_name in METRICS:
        for family, names in families.items():
            raw = np.asarray([
                results[name][metric_name]["p_value_two_sided"] for name in names
            ])
            adjusted = holm_adjust(raw)
            for index, name in enumerate(names):
                results[name][metric_name]["p_adjusted_holm"] = float(adjusted[index])
                results[name][metric_name]["holm_family"] = family
    return results


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate paired r, Cohen's dz, t, exact P, and CI directly "
            "from matched subject-level SPIN-Net results."
        )
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Variant name and subject_level_metrics.mat; may be repeated"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path; otherwise print to stdout"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    variants = {}
    for specification in args.variant:
        if "=" not in specification:
            raise ValueError(
                "--variant must use NAME=PATH syntax."
            )
        name, path = specification.split("=", 1)
        name = name.strip()
        if not name or name in variants:
            raise ValueError(
                f"Variant name must be non-empty and unique: {name!r}"
            )
        variants[name] = path

    payload = _json_safe(compare(args.full, variants))
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered)
            output_file.write("\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
