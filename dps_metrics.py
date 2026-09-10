#!/usr/bin/env python
"""Quantitative comparison of recovered EoR delay power spectra between runs.

Written to answer a referee-style question on the combined-systematics section
of the paper: the text says the DPS recovery "suffers" relative to Case III
without saying by how much, or whether the loss is one of *accuracy* (the
posterior mean sits further from the truth) or of *precision* (the credible
intervals are wider).  Those are different claims and this module measures
them separately.

For each run it reduces the DPS chain to three numbers over the delay bins
that survive the foreground cut:

* **Precision** -- the width of the credible interval drawn as the error bar in
  Figure 7, and, between two runs, the median of the per-bin ratio of those
  widths.  This is the ``X`` in "error bars widen by a median factor of X".
* **Accuracy** -- the normalised deviation

  .. math::

      z(\\tau) = \\frac{\\mu(\\tau) - P_{\\mathrm{true}}(\\tau)}{\\sigma(\\tau)}

  reported as the median :math:`|z|` and the number of bins beyond
  :math:`2\\sigma` and :math:`3\\sigma`.  These are the ``Y`` and ``Z``.
* **Fractional deviation** -- median :math:`|\\mu - P_{\\mathrm{true}}| /
  P_{\\mathrm{true}}`, the unnormalised version of the same thing, comparable
  to the "fractional error" the notebooks already print.

The distinction the numbers are there to settle: if median :math:`|z|` is
roughly equal between the two runs while the widths grow, the combined case is
no more *biased*, only less *constraining* -- a mild degradation.  If median
:math:`|z|` grows too, the wider intervals are not absorbing the larger
residuals and the degradation is a real one.  `verdict()` applies that rule.

Conventions are taken from the notebooks so the numbers match the published
figures:

* the posterior point estimate is ``np.average(ps_sample, weights=ln_post)``,
  the posterior-weighted mean of the Figure 7 cell (see ``estimator`` for the
  caveat);
* :math:`\\sigma` is ``np.std(ps_sample, axis=0)``, as in the lower panel of
  ``errors_components.pdf``;
* the credible interval is the central ``conf_interval`` percent, 95 by
  default;
* the foreground-dominated bins dropped are indices 27-33, the ``rm =
  np.arange(27, 34)`` of that same cell, which is :math:`|\\tau| \\lesssim
  350`~ns at ``Nfreqs = 60``.

Note the sign of ``z``: it is *recovered minus true*, matching
``plot_delta_bsys.py``, and so is the negative of the ``(True - mu)/sigma``
panel of ``errors_components.pdf``.  Only :math:`|z|` is reported in the
summaries, so the choice affects the per-bin arrays alone.

Use from a notebook::

    import dps_metrics as dm

    comp = dm.compare_runs(
        reference=dm.load_run(result_dir_250k + 'low_dl_fr_20', niter=250000),
        target=dm.load_run(result_dir_sim + 'caseiv', niter=250000),
        reference_label='Case III',
        target_label='Combined',
    )
    print(dm.summary_text(comp))
    print(dm.paper_sentence(comp))

Or from the command line, on the machine holding the run outputs::

    conda run -n py10 python dps_metrics.py \\
        --reference 'Case III=/nvme2/scratch/sohini/hydra-pspec-systematic/paper_plots/250k_run/low_dl_fr_20' \\
        --target 'Combined=/nvme2/scratch/sohini/hydra-pspec-systematic/paper_plots/sim_data/caseiv' \\
        --niter 250000

Run it with no arguments for a smoke test on synthetic chains::

    conda run -n py10 python dps_metrics.py --selftest
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────

# Foreground-dominated delay bins dropped before any statistic is formed.
# `rm = np.arange(27, 34)` in the `errors_components.pdf` cell of
# paper_plots_c_v2.ipynb; |tau| <~ 350 ns for Nfreqs = 60.
FG_EXCLUDE_DEFAULT: Tuple[int, int] = (27, 34)

# Central credible interval drawn as the error bar in Figure 7, per cent.
CONF_INTERVAL_DEFAULT: float = 95.0

# Burn-in as a percentage of the chain, as in the notebook Configuration cell.
NBURN_PC_DEFAULT: float = 10.0

# Data dimensions of the runs.
NTIMES_DEFAULT: int = 80
NFREQS_DEFAULT: int = 60

# |z| beyond which a delay bin is counted as discrepant.
SIGMA_THRESHOLDS: Tuple[float, ...] = (2.0, 3.0)

# Below this fractional change in median |z|, the accuracy is called unchanged
# and the degradation is reported as one of precision alone.
ACCURACY_TOLERANCE: float = 0.25

# Below this fractional change in the credible-interval width, the precision is
# called unchanged too, and nothing has measurably degraded.
WIDTH_TOLERANCE: float = 0.10

# File names written by a hydra-pspec run.
CHAIN_FILE = 'dps-eor.npy'
LNPOST_FILE = 'ln-post.npy'
EOR_TRUE_FILE = 'eor_true.npy'


# ── Records ────────────────────────────────────────────────────────────────

@dataclass
class RunData:
    """The three arrays a DPS comparison needs from one Gibbs run.

    Attributes
    ----------
    label : str
        Name of the run, used in the summary table.
    ps_chain : numpy.ndarray
        DPS samples, shape ``(nsamples, ndelays)``, burn-in already removed.
    ps_true : numpy.ndarray
        True EoR delay power spectrum of the run, shape ``(ndelays,)``.
    ln_post : numpy.ndarray, optional
        Log-posterior of each retained sample, shape ``(nsamples,)``.  Used
        only by the ``'weighted'`` estimator.
    """

    label: str
    ps_chain: np.ndarray
    ps_true: np.ndarray
    ln_post: Optional[np.ndarray] = None


@dataclass
class DPSPosterior:
    """Per-bin posterior summary of one run, over the kept delay bins.

    Attributes
    ----------
    label : str
        Name of the run.
    mean : numpy.ndarray
        Posterior point estimate of ``P(tau)``.
    std : numpy.ndarray
        Posterior standard deviation.
    lower, upper : numpy.ndarray
        Bounds of the central credible interval.
    width : numpy.ndarray
        ``upper - lower``; the length of the Figure 7 error bar.
    conf_interval : float
        Credible interval used, per cent.
    estimator : str
        Point estimator used for `mean`.
    """

    label: str
    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    conf_interval: float
    estimator: str


@dataclass
class CaseMetrics:
    """Scalar accuracy and precision summary of one run.

    Attributes
    ----------
    label : str
        Name of the run.
    n_bins : int
        Number of delay bins the statistics were formed over.
    median_width : float
        Median credible-interval width.
    median_std : float
        Median posterior standard deviation.
    median_abs_z, mean_abs_z : float
        Median and mean of ``|z|``.
    n_beyond : dict
        Number of bins with ``|z|`` above each threshold in
        `SIGMA_THRESHOLDS`, keyed by the threshold as a string.
    reduced_chi2 : float
        ``mean(z**2)`` over the kept bins.
    median_frac_dev : float
        Median of ``|mean - true| / true``.
    """

    label: str
    n_bins: int
    median_width: float
    median_std: float
    median_abs_z: float
    mean_abs_z: float
    n_beyond: Dict[str, int]
    reduced_chi2: float
    median_frac_dev: float


@dataclass
class Comparison:
    """Reference-versus-target comparison and the verdict it supports.

    Attributes
    ----------
    reference, target : CaseMetrics
        Summaries of the two runs.
    width_ratio_median : float
        Median over bins of ``width_target / width_reference`` -- the ``X`` of
        the paper sentence.
    width_ratio_range : tuple of float
        16th and 84th percentiles of the same per-bin ratio.
    std_ratio_median : float
        The same ratio formed from the posterior standard deviations.
    abs_z_ratio : float
        ``median_abs_z`` of the target over that of the reference.
    verdict : str
        ``'mildly degraded'`` or ``'significantly degraded'``; see `verdict`.
    n_bins : int
        Number of delay bins compared.
    """

    reference: CaseMetrics
    target: CaseMetrics
    width_ratio_median: float
    width_ratio_range: Tuple[float, float]
    std_ratio_median: float
    abs_z_ratio: float
    verdict: str
    n_bins: int


# ── Delay bins ─────────────────────────────────────────────────────────────

def delay_power_spectrum(s: np.ndarray) -> np.ndarray:
    """Delay power spectrum of a visibility array.

    Identical to the ``calc_ps`` helper of the paper notebooks, reproduced
    here so the module can be run on ``.npy`` files without the notebook.

    Parameters
    ----------
    s : numpy.ndarray
        Visibilities, shape ``(nobs, nfreqs)``.

    Returns
    -------
    numpy.ndarray
        Real delay power spectrum, shape ``(nfreqs,)``.
    """
    axes = (1,)
    sk = np.fft.ifftshift(s, axes=axes)
    sk = np.fft.fftn(sk, axes=axes)
    sk = np.fft.fftshift(sk, axes=axes)
    _, nfreqs = sk.shape
    return np.mean(sk * sk.conj(), axis=0).real / nfreqs


def foreground_mask(
    ndelays: int,
    exclude: Tuple[int, int] = FG_EXCLUDE_DEFAULT,
) -> np.ndarray:
    """Boolean mask dropping the foreground-dominated delay bins.

    Parameters
    ----------
    ndelays : int
        Number of delay bins.
    exclude : tuple of int, optional
        Half-open index range to drop, ``(start, stop)``.  Defaults to the
        notebook's ``np.arange(27, 34)``.

    Returns
    -------
    numpy.ndarray
        Boolean array of length `ndelays`, ``True`` where a bin is kept.
    """
    start, stop = exclude
    if not 0 <= start <= stop <= ndelays:
        raise ValueError(
            f'exclude={exclude} is not a valid range for {ndelays} bins'
        )
    keep = np.ones(ndelays, dtype=bool)
    keep[start:stop] = False
    if not keep.any():
        raise ValueError('exclude removes every delay bin')
    return keep


def mask_from_delays(delays: np.ndarray, tau_min: float) -> np.ndarray:
    """Boolean mask keeping the bins with ``|tau| > tau_min``.

    An alternative to `foreground_mask` for when the cut is more naturally
    stated in nanoseconds than in bin indices.

    Parameters
    ----------
    delays : numpy.ndarray
        Delay of each bin, in ns.  An astropy ``Quantity`` should be passed as
        ``delays.value``.
    tau_min : float
        Cut, in ns.

    Returns
    -------
    numpy.ndarray
        Boolean array of the same length as `delays`.
    """
    keep = np.abs(np.asarray(delays, dtype=float)) > tau_min
    if not keep.any():
        raise ValueError(f'tau_min={tau_min} removes every delay bin')
    return keep


# ── Posterior summary ──────────────────────────────────────────────────────

def point_estimate(
    ps_chain: np.ndarray,
    ln_post: Optional[np.ndarray] = None,
    estimator: str = 'weighted',
) -> np.ndarray:
    """Posterior point estimate of the DPS, per delay bin.

    Parameters
    ----------
    ps_chain : numpy.ndarray
        DPS samples, shape ``(nsamples, ndelays)``.
    ln_post : numpy.ndarray, optional
        Log-posterior of each sample.  Required by ``'weighted'``.
    estimator : {'weighted', 'mean', 'median'}, optional
        ``'weighted'`` reproduces the Figure 7 cell exactly:
        ``np.average(ps_chain, weights=ln_post)``.  Note that this weights by
        the *log* posterior, so the weights are negative wherever the log
        posterior is, and the result is then not a posterior mean and need not
        even lie within the range of the samples; a warning is issued when
        that happens.  ``'mean'`` and ``'median'`` are the unweighted
        alternatives, and ``'mean'`` is what the ``sigma`` of
        ``errors_components.pdf`` is formed about.

    Returns
    -------
    numpy.ndarray
        Point estimate, shape ``(ndelays,)``.
    """
    if estimator == 'mean':
        return ps_chain.mean(axis=0)
    if estimator == 'median':
        return np.median(ps_chain, axis=0)
    if estimator != 'weighted':
        raise ValueError(
            f"estimator must be 'weighted', 'mean' or 'median', got {estimator!r}"
        )
    if ln_post is None:
        raise ValueError("estimator='weighted' needs ln_post")
    if ln_post.shape[0] != ps_chain.shape[0]:
        raise ValueError(
            f'ln_post has {ln_post.shape[0]} samples, chain has '
            f'{ps_chain.shape[0]}'
        )
    if np.any(ln_post < 0):
        warnings.warn(
            'ln_post contains negative values, so the weighted mean is a '
            'weighting by the log posterior rather than by the posterior. '
            "This reproduces the published figure; pass estimator='mean' for "
            'the plain posterior mean.',
            RuntimeWarning,
            stacklevel=2,
        )
    return np.average(ps_chain, weights=ln_post, axis=0)


def posterior_summary(
    run: RunData,
    mask: Optional[np.ndarray] = None,
    conf_interval: float = CONF_INTERVAL_DEFAULT,
    estimator: str = 'weighted',
) -> DPSPosterior:
    """Reduce a DPS chain to its per-bin posterior summary.

    Parameters
    ----------
    run : RunData
        The run to summarise.
    mask : numpy.ndarray, optional
        Boolean mask over delay bins.  Defaults to `foreground_mask`.
    conf_interval : float, optional
        Central credible interval, per cent.
    estimator : str, optional
        Passed to `point_estimate`.

    Returns
    -------
    DPSPosterior
        Summary over the kept bins.
    """
    if run.ps_chain.ndim != 2:
        raise ValueError(
            f'ps_chain must be 2-D (nsamples, ndelays), got shape '
            f'{run.ps_chain.shape}'
        )
    ndelays = run.ps_chain.shape[1]
    if run.ps_true.shape[0] != ndelays:
        raise ValueError(
            f'ps_true has {run.ps_true.shape[0]} bins, chain has {ndelays}'
        )
    if not 0 < conf_interval < 100:
        raise ValueError(f'conf_interval must be in (0, 100), got {conf_interval}')

    keep = foreground_mask(ndelays) if mask is None else np.asarray(mask, dtype=bool)
    if keep.shape[0] != ndelays:
        raise ValueError(
            f'mask has {keep.shape[0]} bins, chain has {ndelays}'
        )

    mean = point_estimate(run.ps_chain, run.ln_post, estimator)[keep]
    std = run.ps_chain.std(axis=0)[keep]

    percentile = conf_interval / 2 + 50
    upper = np.percentile(run.ps_chain, percentile, axis=0)[keep]
    lower = np.percentile(run.ps_chain, 100 - percentile, axis=0)[keep]

    return DPSPosterior(
        label=run.label,
        mean=mean,
        std=std,
        lower=lower,
        upper=upper,
        width=upper - lower,
        conf_interval=conf_interval,
        estimator=estimator,
    )


def normalised_deviation(
    post: DPSPosterior,
    ps_true: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-bin ``z = (mean - true) / std``.

    Parameters
    ----------
    post : DPSPosterior
        Posterior summary, already restricted to the kept bins.
    ps_true : numpy.ndarray
        True DPS over *all* bins, or over the kept bins if `mask` is ``None``
        and the lengths already agree.
    mask : numpy.ndarray, optional
        The mask `post` was formed with, used to restrict `ps_true`.

    Returns
    -------
    numpy.ndarray
        ``z`` over the kept bins.  Sign convention is recovered minus true,
        the negative of the ``errors_components.pdf`` panel.
    """
    true = np.asarray(ps_true, dtype=float)
    if mask is not None:
        true = true[np.asarray(mask, dtype=bool)]
    if true.shape != post.mean.shape:
        raise ValueError(
            f'ps_true has {true.shape[0]} kept bins, posterior has '
            f'{post.mean.shape[0]}'
        )
    if np.any(post.std <= 0):
        raise ValueError('posterior standard deviation is zero in some bins')
    return (post.mean - true) / post.std


# ── Metrics ────────────────────────────────────────────────────────────────

def case_metrics(
    run: RunData,
    mask: Optional[np.ndarray] = None,
    conf_interval: float = CONF_INTERVAL_DEFAULT,
    estimator: str = 'weighted',
) -> Tuple[CaseMetrics, DPSPosterior, np.ndarray]:
    """Accuracy and precision summary of one run.

    Parameters
    ----------
    run : RunData
        The run to summarise.
    mask : numpy.ndarray, optional
        Boolean mask over delay bins.  Defaults to `foreground_mask`.
    conf_interval : float, optional
        Central credible interval, per cent.
    estimator : str, optional
        Passed to `point_estimate`.

    Returns
    -------
    CaseMetrics
        The scalar summary.
    DPSPosterior
        The per-bin posterior it was formed from.
    numpy.ndarray
        The per-bin ``z``.
    """
    ndelays = run.ps_chain.shape[1]
    keep = foreground_mask(ndelays) if mask is None else np.asarray(mask, dtype=bool)

    post = posterior_summary(run, keep, conf_interval, estimator)
    z = normalised_deviation(post, run.ps_true, keep)

    true_kept = np.asarray(run.ps_true, dtype=float)[keep]
    if np.any(true_kept == 0):
        raise ValueError('true DPS is zero in some kept bins')
    frac_dev = np.abs(post.mean - true_kept) / np.abs(true_kept)

    metrics = CaseMetrics(
        label=run.label,
        n_bins=int(keep.sum()),
        median_width=float(np.median(post.width)),
        median_std=float(np.median(post.std)),
        median_abs_z=float(np.median(np.abs(z))),
        mean_abs_z=float(np.mean(np.abs(z))),
        n_beyond={
            f'{t:g}': int(np.sum(np.abs(z) > t)) for t in SIGMA_THRESHOLDS
        },
        reduced_chi2=float(np.mean(z ** 2)),
        median_frac_dev=float(np.median(frac_dev)),
    )
    return metrics, post, z


def verdict(
    reference: CaseMetrics,
    target: CaseMetrics,
    tolerance: float = ACCURACY_TOLERANCE,
    width_ratio: Optional[float] = None,
    width_tolerance: float = WIDTH_TOLERANCE,
) -> str:
    """Classify the degradation, if there is one.

    The rule the paper sentence turns on.  Widened credible intervals alone are
    a loss of precision, not of accuracy; only if the normalised deviation
    grows as well -- meaning the wider intervals are *not* absorbing the larger
    residuals -- is the degradation called significant.  And if neither moves,
    nothing has degraded, which is a result in its own right and must not be
    reported as a mild degradation.

    Parameters
    ----------
    reference, target : CaseMetrics
        Summaries of the two runs.
    tolerance : float, optional
        Fractional growth in median ``|z|`` below which accuracy counts as
        unchanged.
    width_ratio : float, optional
        Median per-bin ratio of the credible-interval widths.  Defaults to the
        ratio of the two median widths, which is close but not identical.
    width_tolerance : float, optional
        Fractional growth in that ratio below which precision counts as
        unchanged.

    Returns
    -------
    str
        ``'not measurably degraded'``, ``'mildly degraded'`` or
        ``'significantly degraded'``.
    """
    if reference.median_abs_z <= 0:
        raise ValueError('reference median |z| is zero; cannot form a ratio')
    if target.median_abs_z / reference.median_abs_z - 1.0 > tolerance:
        return 'significantly degraded'

    if width_ratio is None:
        if reference.median_width <= 0:
            raise ValueError(
                'reference credible-interval width is zero; cannot form a ratio'
            )
        width_ratio = target.median_width / reference.median_width
    if width_ratio - 1.0 > width_tolerance:
        return 'mildly degraded'
    return 'not measurably degraded'


def compare_runs(
    reference: RunData,
    target: RunData,
    mask: Optional[np.ndarray] = None,
    conf_interval: float = CONF_INTERVAL_DEFAULT,
    estimator: str = 'weighted',
    tolerance: float = ACCURACY_TOLERANCE,
) -> Comparison:
    """Compare a target run against a reference run.

    Parameters
    ----------
    reference : RunData
        The control run -- Case III, for the paper's combined-case section.
    target : RunData
        The run under test -- the combined case.
    mask : numpy.ndarray, optional
        Boolean mask over delay bins.  Defaults to `foreground_mask`.
    conf_interval : float, optional
        Central credible interval, per cent.
    estimator : str, optional
        Passed to `point_estimate`.
    tolerance : float, optional
        Passed to `verdict`.

    Returns
    -------
    Comparison
        The two summaries, the width ratios between them, and the verdict.
    """
    if reference.ps_chain.shape[1] != target.ps_chain.shape[1]:
        raise ValueError(
            f'runs have different numbers of delay bins: '
            f'{reference.ps_chain.shape[1]} and {target.ps_chain.shape[1]}'
        )

    ref_metrics, ref_post, _ = case_metrics(reference, mask, conf_interval, estimator)
    tgt_metrics, tgt_post, _ = case_metrics(target, mask, conf_interval, estimator)

    if np.any(ref_post.width <= 0):
        raise ValueError('reference credible-interval width is zero in some bins')
    width_ratio = tgt_post.width / ref_post.width
    std_ratio = tgt_post.std / ref_post.std

    return Comparison(
        reference=ref_metrics,
        target=tgt_metrics,
        width_ratio_median=float(np.median(width_ratio)),
        width_ratio_range=(
            float(np.percentile(width_ratio, 16)),
            float(np.percentile(width_ratio, 84)),
        ),
        std_ratio_median=float(np.median(std_ratio)),
        abs_z_ratio=float(tgt_metrics.median_abs_z / ref_metrics.median_abs_z),
        verdict=verdict(
            ref_metrics, tgt_metrics, tolerance,
            width_ratio=float(np.median(width_ratio)),
        ),
        n_bins=ref_metrics.n_bins,
    )


# ── Reporting ──────────────────────────────────────────────────────────────

def summary_text(comp: Comparison) -> str:
    """Render a comparison as a plain-text table.

    Parameters
    ----------
    comp : Comparison
        The comparison to render.

    Returns
    -------
    str
        Multi-line table, one column per run plus the target-to-reference
        ratio.
    """
    ref, tgt = comp.reference, comp.target
    rows = [
        ('delay bins compared', f'{ref.n_bins:d}', f'{tgt.n_bins:d}', ''),
        ('median CI width', f'{ref.median_width:.4g}',
         f'{tgt.median_width:.4g}', f'{comp.width_ratio_median:.2f}x'),
        ('median posterior sigma', f'{ref.median_std:.4g}',
         f'{tgt.median_std:.4g}', f'{comp.std_ratio_median:.2f}x'),
        ('median |z|', f'{ref.median_abs_z:.3f}', f'{tgt.median_abs_z:.3f}',
         f'{comp.abs_z_ratio:.2f}x'),
        ('mean |z|', f'{ref.mean_abs_z:.3f}', f'{tgt.mean_abs_z:.3f}', ''),
        ('reduced chi^2', f'{ref.reduced_chi2:.3f}',
         f'{tgt.reduced_chi2:.3f}', ''),
        ('median frac. deviation', f'{ref.median_frac_dev:.3f}',
         f'{tgt.median_frac_dev:.3f}', ''),
    ]
    for t in SIGMA_THRESHOLDS:
        key = f'{t:g}'
        rows.append((
            f'bins with |z| > {key}',
            f'{ref.n_beyond[key]:d}/{ref.n_bins:d}',
            f'{tgt.n_beyond[key]:d}/{tgt.n_bins:d}',
            '',
        ))

    headers = ('metric', ref.label, tgt.label, 'ratio')
    widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows))
        for i in range(4)
    ]
    line = '  '.join('-' * w for w in widths)

    out = ['  '.join(h.ljust(widths[i]) for i, h in enumerate(headers)), line]
    out += ['  '.join(r[i].ljust(widths[i]) for i in range(4)) for r in rows]
    out += [line, f'verdict: {comp.verdict}']
    return '\n'.join(out)


def paper_sentence(comp: Comparison) -> str:
    """The replacement sentence for the paper, with the numbers filled in.

    Substitutes the measured values into the wording that replaces "the DPS
    recovery suffers" in the combined-systematics section.

    Parameters
    ----------
    comp : Comparison
        The comparison to describe.

    Returns
    -------
    str
        A LaTeX sentence, ready to paste.
    """
    ref, tgt = comp.reference, comp.target
    lo, hi = comp.width_ratio_range

    bins = (
        f'{comp.n_bins} delay bins outside the foreground-dominated region'
    )
    accuracy = (
        f'the median deviation of the recovered DPS from the true DPS, '
        f'normalised by the posterior standard deviation, '
        f'{"changes only from" if comp.verdict == "not measurably degraded" else "changes from"} '
        f'${ref.median_abs_z:.2f}\\sigma$ to ${tgt.median_abs_z:.2f}\\sigma$'
    )
    discrepant = (
        f'The number of bins discrepant at more than $2\\sigma$ is '
        f'{ref.n_beyond["2"]} of {ref.n_bins} for the reference case and '
        f'{tgt.n_beyond["2"]} of {tgt.n_bins} for the combined case.'
    )

    if comp.verdict == 'not measurably degraded':
        return (
            f'The recovered EoR DPS for the combined case is essentially '
            f'indistinguishable from that of the control case. Across the '
            f'{bins}, the median width of the credible interval differs by '
            f'{abs(comp.width_ratio_median - 1.0) * 100:.0f}~per cent between '
            f'the two, and {accuracy}. {discrepant} Injecting systematics at '
            f'multiple Fourier mode pairs rather than one therefore neither '
            f'widens the posterior on the EoR DPS nor moves it further from '
            f'the truth, at the level these tests can resolve.'
        )

    if comp.verdict == 'mildly degraded':
        tail = (
            'The loss is therefore predominantly one of precision rather than '
            'accuracy --- the combined case constrains the EoR power spectrum '
            'less tightly, but is not substantially more biased.'
        )
    else:
        tail = (
            'The enlarged credible intervals therefore do not fully absorb the '
            'larger residuals: the combined case is both less constraining and '
            'measurably more biased than any individual case.'
        )
    return (
        f'Despite these improvements in sampling efficiency, the DPS recovery '
        f'is {comp.verdict}: the posterior credible intervals widen by a '
        f'median factor of {comp.width_ratio_median:.2f} '
        f'(16th--84th percentile {lo:.2f}--{hi:.2f}) across the {bins}, while '
        f'{accuracy}. {discrepant} {tail}'
    )


def to_dict(comp: Comparison) -> Dict[str, object]:
    """Comparison as a plain dictionary, for JSON output.

    Parameters
    ----------
    comp : Comparison
        The comparison to convert.

    Returns
    -------
    dict
        Nested dictionary of the comparison's fields.
    """
    return asdict(comp)


# ── Loading ────────────────────────────────────────────────────────────────

def load_run(
    run_dir: str,
    label: Optional[str] = None,
    niter: Optional[int] = None,
    nburn_pc: float = NBURN_PC_DEFAULT,
    ntimes: int = NTIMES_DEFAULT,
    nfreqs: int = NFREQS_DEFAULT,
) -> RunData:
    """Load one run's DPS chain, log-posterior and true EoR spectrum.

    Reads ``dps-eor.npy``, ``ln-post.npy`` and ``eor_true.npy`` from `run_dir`,
    trims to `niter` samples, discards `nburn_pc` per cent as burn-in, and
    forms the true DPS with `delay_power_spectrum` -- the same steps the
    Figure 7 cell takes.

    Parameters
    ----------
    run_dir : str
        Directory holding the run outputs.
    label : str, optional
        Name for the run.  Defaults to the directory's basename.
    niter : int, optional
        Number of samples to use.  Defaults to the whole chain.
    nburn_pc : float, optional
        Burn-in as a percentage of `niter`.
    ntimes, nfreqs : int, optional
        Dimensions the true EoR array is trimmed to.

    Returns
    -------
    RunData
        The loaded run, burn-in removed.
    """
    path = Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f'no such run directory: {path}')
    for name in (CHAIN_FILE, LNPOST_FILE, EOR_TRUE_FILE):
        if not (path / name).is_file():
            raise FileNotFoundError(f'{path} has no {name}')

    ps_chain = np.load(path / CHAIN_FILE)
    ln_post = np.load(path / LNPOST_FILE)
    eor_true = np.load(path / EOR_TRUE_FILE)[:ntimes, :nfreqs]

    n_avail = ps_chain.shape[0]
    n_use = n_avail if niter is None else int(niter)
    if n_use > n_avail:
        raise ValueError(
            f'{path.name}: asked for {n_use} samples, only {n_avail} on disk'
        )
    if not 0 <= nburn_pc < 100:
        raise ValueError(f'nburn_pc must be in [0, 100), got {nburn_pc}')

    nburn = int(n_use * nburn_pc / 100)
    return RunData(
        label=label or path.name,
        ps_chain=ps_chain[nburn:n_use],
        ps_true=delay_power_spectrum(eor_true),
        ln_post=ln_post[nburn:n_use],
    )


# ── Smoke test ─────────────────────────────────────────────────────────────

def make_demo_run(
    label: str,
    ndelays: int = NFREQS_DEFAULT,
    nsamples: int = 4000,
    width_scale: float = 1.0,
    bias_sigma: float = 0.5,
    seed: int = 0,
) -> RunData:
    """Synthetic run for tests and the smoke test.

    Builds a lognormal DPS chain about a smooth true spectrum, with the
    posterior width and the offset of the mean from the truth both under the
    caller's control, so that a comparison of two demo runs has a known
    answer.

    Parameters
    ----------
    label : str
        Name for the run.
    ndelays : int, optional
        Number of delay bins.
    nsamples : int, optional
        Number of chain samples.
    width_scale : float, optional
        Multiplies the posterior scatter; ``2.0`` gives roughly twice the
        credible-interval width of ``1.0``.
    bias_sigma : float, optional
        Offset of the posterior mean from the truth, in units of the
        posterior standard deviation.
    seed : int, optional
        Seed of the random generator.

    Returns
    -------
    RunData
        The synthetic run.
    """
    rng = np.random.default_rng(seed)
    tau = np.linspace(-1.0, 1.0, ndelays)
    ps_true = 1e-2 * np.exp(-(tau ** 2) / 0.5) + 1e-4

    sigma_rel = 0.15 * width_scale
    offset = bias_sigma * sigma_rel
    draws = rng.normal(
        loc=np.log(ps_true) + offset,
        scale=sigma_rel,
        size=(nsamples, ndelays),
    )
    ps_chain = np.exp(draws)
    ln_post = -rng.chisquare(df=10, size=nsamples)

    return RunData(
        label=label,
        ps_chain=ps_chain,
        ps_true=ps_true,
        ln_post=ln_post,
    )


def selftest(estimator: str = 'mean') -> Comparison:
    """Run the metrics on synthetic chains with a known answer.

    The target is built with twice the posterior width of the reference and
    the same offset *in units of sigma*, so a correct implementation reports a
    width ratio near 2 and an unchanged median ``|z|`` -- the "mildly
    degraded" branch.

    Parameters
    ----------
    estimator : str, optional
        Passed to `point_estimate`.

    Returns
    -------
    Comparison
        The comparison of the two synthetic runs.
    """
    reference = make_demo_run('Reference', width_scale=1.0, bias_sigma=0.5, seed=1)
    target = make_demo_run('Target', width_scale=2.0, bias_sigma=0.5, seed=2)
    return compare_runs(reference, target, estimator=estimator)


# ── Command line ───────────────────────────────────────────────────────────

def _parse_run_spec(spec: str) -> Tuple[Optional[str], str]:
    """Split a ``LABEL=PATH`` command-line argument.

    Parameters
    ----------
    spec : str
        Either ``PATH`` or ``LABEL=PATH``.

    Returns
    -------
    tuple
        ``(label, path)``, with `label` ``None`` if none was given.
    """
    if '=' in spec:
        label, path = spec.split('=', 1)
        return label.strip(), path.strip()
    return None, spec.strip()


def _parse_exclude(spec: str) -> Tuple[int, int]:
    """Parse a ``START:STOP`` delay-bin exclusion range.

    Parameters
    ----------
    spec : str
        Half-open range, e.g. ``'27:34'``.

    Returns
    -------
    tuple of int
        ``(start, stop)``.
    """
    parts = spec.split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f'--exclude wants START:STOP, got {spec!r}'
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'--exclude wants integers, got {spec!r}'
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser of the module.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    p = argparse.ArgumentParser(
        description=(
            'Accuracy and precision of a recovered EoR delay power spectrum, '
            'relative to a reference run.'
        ),
    )
    p.add_argument(
        '--reference', metavar='[LABEL=]DIR',
        help='control run, e.g. the Case III output directory',
    )
    p.add_argument(
        '--target', metavar='[LABEL=]DIR',
        help='run under test, e.g. the combined-case output directory',
    )
    p.add_argument(
        '--niter', type=int, default=None,
        help='samples to use from each chain (default: the whole chain)',
    )
    p.add_argument(
        '--burn-pc', type=float, default=NBURN_PC_DEFAULT,
        help=f'burn-in as a percentage of --niter (default: {NBURN_PC_DEFAULT:g})',
    )
    p.add_argument(
        '--conf', type=float, default=CONF_INTERVAL_DEFAULT,
        help=f'credible interval, per cent (default: {CONF_INTERVAL_DEFAULT:g})',
    )
    p.add_argument(
        '--exclude', type=_parse_exclude, default=FG_EXCLUDE_DEFAULT,
        metavar='START:STOP',
        help=(
            'foreground-dominated delay bins to drop, half-open '
            f'(default: {FG_EXCLUDE_DEFAULT[0]}:{FG_EXCLUDE_DEFAULT[1]})'
        ),
    )
    p.add_argument(
        '--estimator', choices=('weighted', 'mean', 'median'), default='weighted',
        help=(
            "posterior point estimate; 'weighted' reproduces the published "
            "figure (default: weighted)"
        ),
    )
    p.add_argument(
        '--ntimes', type=int, default=NTIMES_DEFAULT,
        help=f'LST samples (default: {NTIMES_DEFAULT})',
    )
    p.add_argument(
        '--nfreqs', type=int, default=NFREQS_DEFAULT,
        help=f'frequency channels (default: {NFREQS_DEFAULT})',
    )
    p.add_argument(
        '--json', metavar='PATH', default=None,
        help='also write the metrics to this file as JSON',
    )
    p.add_argument(
        '--selftest', action='store_true',
        help='run on synthetic chains instead of run outputs',
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    args = build_parser().parse_args(argv)

    if args.selftest or not (args.reference and args.target):
        if not args.selftest:
            print('No --reference/--target given; running the self-test.\n')
        comp = selftest()
        print('Synthetic chains: target has 2x the posterior width of the '
              'reference,\nand the same offset in units of sigma.\n')
    else:
        ref_label, ref_dir = _parse_run_spec(args.reference)
        tgt_label, tgt_dir = _parse_run_spec(args.target)
        reference = load_run(
            ref_dir, ref_label, args.niter, args.burn_pc, args.ntimes, args.nfreqs,
        )
        target = load_run(
            tgt_dir, tgt_label, args.niter, args.burn_pc, args.ntimes, args.nfreqs,
        )
        mask = foreground_mask(reference.ps_chain.shape[1], args.exclude)
        comp = compare_runs(
            reference, target, mask, args.conf, args.estimator,
        )

    print(summary_text(comp))
    print()
    print('Paper sentence:')
    print()
    print(paper_sentence(comp))

    if args.json:
        Path(args.json).write_text(json.dumps(to_dict(comp), indent=2))
        print(f'\nMetrics written to {args.json}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
