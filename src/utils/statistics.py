"""Paired, subject-level statistics.

The functions in this module operate on the observed score from each subject.
In particular, Cohen's paired-sample effect size is computed from the observed
within-subject differences; no correlation is assumed or reconstructed from
group-level summary statistics.

No SciPy dependency is required, so the reporting calculation can also run in
lightweight environments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PairedStatistics:
    """Complete summary of a two-sided paired-sample comparison.

    ``mean_difference`` is defined as ``full - variant``. Standard deviations
    use Bessel's correction (sample SD, denominator ``n - 1``).
    """

    n: int
    full_mean: float
    full_sd: float
    variant_mean: float
    variant_sd: float
    paired_r: float
    mean_difference: float
    difference_sd: float
    confidence_level: float
    ci_low: float
    ci_high: float
    cohen_dz: float
    t_statistic: float
    degrees_freedom: int
    p_value_two_sided: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a serialization-friendly representation."""

        return asdict(self)


def family_block_sign_flip_pvalue(
    differences,
    family_groups,
    *,
    n_permutations=10_000,
    random_state=0,
):
    """Two-sided paired randomization test with HCP families as blocks.

    All observations belonging to the same family receive the same random
    sign. The statistic is the mean paired difference, matching the
    family-aware inference described in the revised manuscript.
    """
    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    groups = np.asarray(family_groups).reshape(-1)
    if values.size < 2 or groups.size != values.size:
        raise ValueError(
            "differences and family_groups must contain the same two or more entries."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("differences must be finite.")
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be positive.")
    _, inverse = np.unique(groups.astype(str), return_inverse=True)
    n_blocks = int(inverse.max()) + 1
    rng = np.random.default_rng(random_state)
    observed = float(values.mean())
    exceedances = 0
    for _ in range(int(n_permutations)):
        block_signs = rng.choice((-1.0, 1.0), size=n_blocks)
        permuted = float(np.mean(values * block_signs[inverse]))
        exceedances += int(abs(permuted) >= abs(observed))
    return (1.0 + exceedances) / (int(n_permutations) + 1.0)


def compute_paired_statistics(
    full_scores: Iterable[float],
    variant_scores: Iterable[float],
    confidence_level: float = 0.95,
) -> PairedStatistics:
    """Compute statistics from paired subject-level scores.

    Parameters
    ----------
    full_scores:
        One score per subject from the full model.
    variant_scores:
        Scores from the same subjects, in the same order, for the comparison
        model.
    confidence_level:
        Confidence level for the two-sided Student-t interval of the mean
        difference.

    Returns
    -------
    PairedStatistics
        Means and sample SDs, observed Pearson paired correlation, mean
        difference and its confidence interval, paired Cohen's ``d_z``,
        paired ``t`` statistic with ``n - 1`` degrees of freedom, and the
        unrounded two-sided Student-t p-value.

    Notes
    -----
    A positive difference, ``d_z``, or ``t`` means that the full model scored
    higher. If either input is constant, its Pearson correlation is undefined
    and ``paired_r`` is ``nan``. For constant non-zero differences, the
    limiting values ``|d_z| = |t| = inf`` and ``p = 0`` are returned. If all
    differences are zero, the effect and test statistics are undefined
    (``nan``) and the p-value is reported as 1.
    """

    full = _as_finite_tuple(full_scores, "full_scores")
    variant = _as_finite_tuple(variant_scores, "variant_scores")
    if len(full) != len(variant):
        raise ValueError(
            "Paired inputs must have the same length; "
            f"got {len(full)} and {len(variant)}."
        )
    if len(full) < 2:
        raise ValueError("At least two paired subjects are required.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")

    n = len(full)
    degrees_freedom = n - 1
    full_mean = math.fsum(full) / n
    variant_mean = math.fsum(variant) / n
    full_deviations = tuple(value - full_mean for value in full)
    variant_deviations = tuple(value - variant_mean for value in variant)
    full_sum_squares = math.fsum(value * value for value in full_deviations)
    variant_sum_squares = math.fsum(
        value * value for value in variant_deviations
    )
    full_sd = math.sqrt(full_sum_squares / degrees_freedom)
    variant_sd = math.sqrt(variant_sum_squares / degrees_freedom)

    correlation_denominator = math.sqrt(
        full_sum_squares * variant_sum_squares
    )
    if correlation_denominator == 0.0:
        paired_r = math.nan
    else:
        covariance_sum = math.fsum(
            full_deviation * variant_deviation
            for full_deviation, variant_deviation in zip(
                full_deviations, variant_deviations
            )
        )
        paired_r = max(
            -1.0, min(1.0, covariance_sum / correlation_denominator)
        )

    differences = tuple(
        full_value - variant_value
        for full_value, variant_value in zip(full, variant)
    )
    mean_difference = math.fsum(differences) / n
    difference_sum_squares = math.fsum(
        (difference - mean_difference) ** 2 for difference in differences
    )
    difference_sd = math.sqrt(difference_sum_squares / degrees_freedom)

    if difference_sd == 0.0:
        ci_low = mean_difference
        ci_high = mean_difference
        if mean_difference == 0.0:
            cohen_dz = math.nan
            t_statistic = math.nan
            p_value = 1.0
        else:
            sign = math.copysign(1.0, mean_difference)
            cohen_dz = sign * math.inf
            t_statistic = sign * math.inf
            p_value = 0.0
    else:
        standard_error = difference_sd / math.sqrt(n)
        cohen_dz = mean_difference / difference_sd
        t_statistic = mean_difference / standard_error
        p_value = _student_t_two_sided_p(t_statistic, degrees_freedom)
        alpha = 1.0 - confidence_level
        critical_t = _student_t_two_sided_critical_value(
            alpha, degrees_freedom
        )
        margin = critical_t * standard_error
        ci_low = mean_difference - margin
        ci_high = mean_difference + margin

    return PairedStatistics(
        n=n,
        full_mean=full_mean,
        full_sd=full_sd,
        variant_mean=variant_mean,
        variant_sd=variant_sd,
        paired_r=paired_r,
        mean_difference=mean_difference,
        difference_sd=difference_sd,
        confidence_level=confidence_level,
        ci_low=ci_low,
        ci_high=ci_high,
        cohen_dz=cohen_dz,
        t_statistic=t_statistic,
        degrees_freedom=degrees_freedom,
        p_value_two_sided=p_value,
    )


def _as_finite_tuple(values: Iterable[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of numeric values.") from error
    except ValueError as error:
        raise ValueError(f"{name} contains a non-numeric value.") from error
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _student_t_two_sided_p(t_statistic: float, degrees_freedom: int) -> float:
    """Two-sided survival probability of Student's t distribution."""

    if degrees_freedom < 1:
        raise ValueError("degrees_freedom must be at least 1.")
    if math.isnan(t_statistic):
        return math.nan
    if math.isinf(t_statistic):
        return 0.0
    t_squared = t_statistic * t_statistic
    x = degrees_freedom / (degrees_freedom + t_squared)
    return _regularized_incomplete_beta(
        x, degrees_freedom / 2.0, 0.5
    )


def _student_t_two_sided_critical_value(
    alpha: float, degrees_freedom: int
) -> float:
    """Return positive ``t`` satisfying ``P(|T| >= t) = alpha``."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")
    lower = 0.0
    upper = 1.0
    while _student_t_two_sided_p(upper, degrees_freedom) > alpha:
        upper *= 2.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _student_t_two_sided_p(midpoint, degrees_freedom) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta ``I_x(a, b)`` in double precision."""

    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie between 0 and 1.")
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive.")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    log_beta_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    beta_term = math.exp(log_beta_term)
    if x < (a + 1.0) / (a + b + 2.0):
        value = beta_term * _beta_continued_fraction(a, b, x) / a
    else:
        value = (
            1.0
            - beta_term
            * _beta_continued_fraction(b, a, 1.0 - x)
            / b
        )
    return max(0.0, min(1.0, value))


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Lentz evaluation used by the incomplete-beta calculation."""

    max_iterations = 200
    tolerance = 3.0e-14
    tiny = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d

    for iteration in range(1, max_iterations + 1):
        twice_iteration = 2 * iteration
        coefficient = (
            iteration
            * (b - iteration)
            * x
            / ((qam + twice_iteration) * (a + twice_iteration))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        coefficient = (
            -(a + iteration)
            * (qab + iteration)
            * x
            / ((a + twice_iteration) * (qap + twice_iteration))
        )
        d = 1.0 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= tolerance:
            return result

    raise ArithmeticError("Incomplete-beta continued fraction did not converge.")
