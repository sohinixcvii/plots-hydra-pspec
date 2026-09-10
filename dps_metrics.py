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
from typing import Dict, List, Optional, Sequence, Tuple

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

# Foreground modes used from fgmodes.npy, as in the notebook Configuration.
NFGMODES_DEFAULT: int = 10

# Every `stride`-th sample is used when averaging the sky over a chain.  The
# posterior predictive mean converges long before the chain is exhausted, and
# gcr-eor.npy runs to tens of GB at full length.
SKY_STRIDE_DEFAULT: int = 50

# File names written by a hydra-pspec run.
CHAIN_FILE = 'dps-eor.npy'
LNPOST_FILE = 'ln-post.npy'
EOR_TRUE_FILE = 'eor_true.npy'
BSYS_FILE = 'b-sys.npy'
EOR_GCR_FILE = 'gcr-eor.npy'
FG_AMPS_FILE = 'fg-amps.npy'
FG_MODES_FILE = 'fgmodes.npy'
FG_TRUE_FILE = 'fg_true.npy'


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

def _render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    footer: Optional[str] = None,
) -> str:
    """Render rows of pre-formatted strings as a fixed-width table.

    Columns are sized to their widest cell, rules are drawn above and below
    the body, and every cell is left-aligned, which keeps labels and numbers
    scannable side by side in a terminal.

    Parameters
    ----------
    headers : sequence of str
        Column headings; sets the column count.
    rows : sequence of sequence of str
        Body cells, already formatted.  Every row must match `headers` in
        length.
    footer : str, optional
        A line placed below the closing rule, e.g. a verdict.

    Returns
    -------
    str
        The rendered table, without a trailing newline.
    """
    ncols = len(headers)
    for r in rows:
        if len(r) != ncols:
            raise ValueError(
                f'row has {len(r)} cells, headers have {ncols}: {list(r)}'
            )

    widths = [
        max(len(str(headers[i])), max((len(str(r[i])) for r in rows), default=0))
        for i in range(ncols)
    ]
    rule = '  '.join('-' * w for w in widths)

    out = ['  '.join(str(headers[i]).ljust(widths[i]) for i in range(ncols)),
           rule]
    out += ['  '.join(str(r[i]).ljust(widths[i]) for i in range(ncols))
            for r in rows]
    out.append(rule)
    if footer:
        out.append(footer)
    return '\n'.join(out)


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
    return _render_table(headers, rows, footer=f'verdict: {comp.verdict}')


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


# ── b_sys posterior spread ─────────────────────────────────────────────────

@dataclass
class BsysSpread:
    """Marginal posterior spread of one complex systematics amplitude.

    Attributes
    ----------
    label : str
        Parameter label, e.g. ``'b_sys,1'``.
    std_real, std_imag : float
        Posterior standard deviation of the real and imaginary parts.
    std_total : float
        ``sqrt(std_real**2 + std_imag**2)``, the spread of the complex
        amplitude as a whole.
    """

    label: str
    std_real: float
    std_imag: float
    std_total: float


def load_bsys(
    run_dir: str,
    niter: Optional[int] = None,
    nburn_pc: float = NBURN_PC_DEFAULT,
) -> np.ndarray:
    """Load a run's systematics-amplitude chain, burn-in removed.

    Parameters
    ----------
    run_dir : str
        Directory holding the run outputs.
    niter : int, optional
        Number of samples to use.  Defaults to the whole chain.
    nburn_pc : float, optional
        Burn-in as a percentage of `niter`.

    Returns
    -------
    numpy.ndarray
        Complex chain of shape ``(nsamples, nparams)``.
    """
    path = Path(run_dir)
    if not (path / BSYS_FILE).is_file():
        raise FileNotFoundError(f'{path} has no {BSYS_FILE}')

    chain = np.load(path / BSYS_FILE)
    n_avail = chain.shape[0]
    n_use = n_avail if niter is None else int(niter)
    if n_use > n_avail:
        raise ValueError(
            f'{path.name}: asked for {n_use} samples, only {n_avail} on disk'
        )
    if not 0 <= nburn_pc < 100:
        raise ValueError(f'nburn_pc must be in [0, 100), got {nburn_pc}')
    return chain[int(n_use * nburn_pc / 100):n_use]


def bsys_spread(
    chain: np.ndarray,
    labels: Optional[Sequence[str]] = None,
) -> List[BsysSpread]:
    """Marginal posterior spread of every parameter in a b_sys chain.

    Parameters
    ----------
    chain : numpy.ndarray
        Complex chain of shape ``(nsamples, nparams)``.
    labels : sequence of str, optional
        Parameter labels.  Default ``'b_sys,1'`` upward.

    Returns
    -------
    list of BsysSpread
        One record per parameter, in chain order.
    """
    if chain.ndim != 2:
        raise ValueError(
            f'chain must be 2-D (nsamples, nparams), got shape {chain.shape}'
        )
    nparams = chain.shape[1]
    names = list(labels) if labels is not None else [
        f'b_sys,{i + 1}' for i in range(nparams)
    ]
    if len(names) != nparams:
        raise ValueError(f'{len(names)} labels for {nparams} parameters')

    out = []
    for i, name in enumerate(names):
        sr = float(np.std(chain[:, i].real))
        si = float(np.std(chain[:, i].imag))
        out.append(BsysSpread(name, sr, si, float(np.hypot(sr, si))))
    return out


def compare_bsys_spread(
    reference: np.ndarray,
    target: np.ndarray,
    reference_label: str = 'reference',
    target_label: str = 'target',
    indices: Optional[Sequence[Tuple[int, int]]] = None,
) -> str:
    """Table of posterior spread in two runs, and the ratio between them.

    The check behind the combined-case correlation-time argument.  That
    argument holds that a parameter speeds up because it acquires a degenerate
    partner *inside* the systematics block, which adds posterior variance that
    is refreshed exactly at every iteration and so dilutes the slow
    foreground-degenerate component.  The prediction is that the marginal
    posterior width of the affected parameters is **larger** in the combined
    run than in isolation.  A ratio at or below one refutes it.

    Parameters
    ----------
    reference, target : numpy.ndarray
        Complex b_sys chains, burn-in already removed.
    reference_label, target_label : str, optional
        Column headings.
    indices : sequence of (int, int), optional
        Parameter pairs to compare, as ``(reference index, target index)``.
        Defaults to matching by position over the parameters the two runs
        share, which is the right mapping when the combined run lists the
        individual case's modes first.

    Returns
    -------
    str
        A plain-text table.
    """
    if reference.ndim != 2 or target.ndim != 2:
        raise ValueError('both chains must be 2-D (nsamples, nparams)')

    if indices is None:
        n = min(reference.shape[1], target.shape[1])
        if reference.shape[1] != target.shape[1]:
            warnings.warn(
                f'matching {n} parameters by position, but the runs hold '
                f'{reference.shape[1]} and {target.shape[1]}. That is only '
                'the right mapping if the target lists the reference\'s modes '
                'first -- for the combined run, b_sys,1-4 are the Case I '
                'modes, 5-8 Case II and 9-12 Case III. Pass indices (or --map) '
                'to say so explicitly.',
                RuntimeWarning,
                stacklevel=2,
            )
        indices = [(i, i) for i in range(n)]
    for a, b in indices:
        if not 0 <= a < reference.shape[1]:
            raise ValueError(f'reference has no parameter {a}')
        if not 0 <= b < target.shape[1]:
            raise ValueError(f'target has no parameter {b}')

    ref_all = bsys_spread(reference)
    tgt_all = bsys_spread(target)

    rows = []
    for a, b in indices:
        r, t = ref_all[a], tgt_all[b]
        ratio = t.std_total / r.std_total if r.std_total > 0 else float('nan')
        verdict_ = ('wider' if ratio > 1.05 else
                    'narrower' if ratio < 0.95 else 'unchanged')
        rows.append((
            f'{r.label} -> {t.label}',
            f'{r.std_total:.4g}',
            f'{t.std_total:.4g}',
            f'{ratio:.2f}x',
            verdict_,
        ))

    headers = ('parameter', reference_label, target_label, 'ratio', '')
    return _render_table(headers, rows)


def bsys_correlation(
    chain: np.ndarray,
    i: int,
    j: int,
) -> Dict[str, float]:
    """Pearson correlation between two systematics amplitudes.

    The second half of the combined-case check: the partner that is supposed
    to absorb the variance should be strongly correlated with the parameter
    that speeds up.

    Parameters
    ----------
    chain : numpy.ndarray
        Complex b_sys chain of shape ``(nsamples, nparams)``.
    i, j : int
        Zero-based parameter indices.

    Returns
    -------
    dict
        ``'real'`` and ``'imag'`` correlation coefficients, and ``'max_abs'``,
        the larger of the two in absolute value.
    """
    for k in (i, j):
        if not 0 <= k < chain.shape[1]:
            raise ValueError(f'chain has no parameter {k}')
    if i == j:
        raise ValueError('i and j must name different parameters')

    out = {}
    for part in ('real', 'imag'):
        a = getattr(chain[:, i], part)
        b = getattr(chain[:, j], part)
        if a.std() == 0 or b.std() == 0:
            out[part] = float('nan')
        else:
            out[part] = float(np.corrcoef(a, b)[0, 1])
    out['max_abs'] = float(max(abs(out['real']), abs(out['imag'])))
    return out


# ── Sky residuals ──────────────────────────────────────────────────────────

@dataclass
class SkyResidual:
    """RMS of the sky residual for one run.

    Attributes
    ----------
    label : str
        Name of the run.
    rms_residual : float
        RMS of ``sky_true - mean(sky samples)`` over all times and channels.
    rms_sky : float
        RMS of the true sky, for scale.
    rms_eor : float
        RMS of the true EoR component alone.
    fractional : float
        ``rms_residual / rms_sky``.
    relative_to_eor : float
        ``rms_residual / rms_eor`` -- the residual measured against the signal
        the analysis is actually after.
    nsamples : int
        Chain samples averaged over.
    """

    label: str
    rms_residual: float
    rms_sky: float
    rms_eor: float
    fractional: float
    relative_to_eor: float
    nsamples: int


def sky_residual_rms(
    run_dir: str,
    label: Optional[str] = None,
    niter: Optional[int] = None,
    nburn_pc: float = NBURN_PC_DEFAULT,
    stride: int = SKY_STRIDE_DEFAULT,
    ntimes: int = NTIMES_DEFAULT,
    nfreqs: int = NFREQS_DEFAULT,
    nfgmodes: int = NFGMODES_DEFAULT,
) -> SkyResidual:
    """RMS residual between the true sky and its posterior predictive mean.

    The sky of each Gibbs sample is ``eor_gcr[i] + (fgmodes @ fg_amps[i].T).T``,
    the sum the notebook forms for Figures 4 and 9.  The chain is memory-mapped
    and averaged in place, so nothing of order ``(niter, ntimes, nfreqs)`` is
    ever allocated.

    Parameters
    ----------
    run_dir : str
        Directory holding the run outputs.
    label : str, optional
        Name for the run.  Defaults to the directory's basename.
    niter : int, optional
        Number of samples to draw from.  Defaults to the whole chain.
    nburn_pc : float, optional
        Burn-in as a percentage of `niter`.
    stride : int, optional
        Use every `stride`-th sample after burn-in.
    ntimes, nfreqs : int, optional
        Dimensions the true arrays are trimmed to.
    nfgmodes : int, optional
        Foreground modes used from ``fgmodes.npy``.

    Returns
    -------
    SkyResidual
        The RMS residual and the scales it should be read against.
    """
    path = Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f'no such run directory: {path}')
    for name in (EOR_GCR_FILE, FG_AMPS_FILE, FG_MODES_FILE,
                 EOR_TRUE_FILE, FG_TRUE_FILE):
        if not (path / name).is_file():
            raise FileNotFoundError(f'{path} has no {name}')
    if stride < 1:
        raise ValueError(f'stride must be >= 1, got {stride}')
    if not 0 <= nburn_pc < 100:
        raise ValueError(f'nburn_pc must be in [0, 100), got {nburn_pc}')

    eor_gcr = np.load(path / EOR_GCR_FILE, mmap_mode='r')
    fg_amps = np.load(path / FG_AMPS_FILE, mmap_mode='r')
    fgmodes = np.load(path / FG_MODES_FILE)[:, :nfgmodes]
    eor_true = np.load(path / EOR_TRUE_FILE)[:ntimes, :nfreqs]
    fg_true = np.load(path / FG_TRUE_FILE)[:ntimes, :nfreqs]

    n_avail = eor_gcr.shape[0]
    n_use = n_avail if niter is None else int(niter)
    if n_use > n_avail:
        raise ValueError(
            f'{path.name}: asked for {n_use} samples, only {n_avail} on disk'
        )
    nburn = int(n_use * nburn_pc / 100)

    total = np.zeros((ntimes, nfreqs), dtype=complex)
    count = 0
    for i in range(nburn, n_use, stride):
        fg_vis = (fgmodes @ fg_amps[i][:, :nfgmodes].T).T
        total += eor_gcr[i][:ntimes, :nfreqs] + fg_vis[:ntimes, :nfreqs]
        count += 1
    if count == 0:
        raise ValueError('burn-in and stride leave no samples to average')

    sky_true = eor_true + fg_true
    residual = sky_true - total / count

    rms = lambda x: float(np.sqrt(np.mean(np.abs(x) ** 2)))
    rms_residual, rms_sky, rms_eor = rms(residual), rms(sky_true), rms(eor_true)

    return SkyResidual(
        label=label or path.name,
        rms_residual=rms_residual,
        rms_sky=rms_sky,
        rms_eor=rms_eor,
        fractional=rms_residual / rms_sky if rms_sky > 0 else float('nan'),
        relative_to_eor=rms_residual / rms_eor if rms_eor > 0 else float('nan'),
        nsamples=count,
    )


def sky_residual_table(residuals: Sequence[SkyResidual]) -> str:
    """Render sky-residual records as a plain-text table.

    Parameters
    ----------
    residuals : sequence of SkyResidual
        One record per run, in the order they should appear.

    Returns
    -------
    str
        A plain-text table, one row per run.
    """
    if not residuals:
        return 'no runs'
    headers = ('case', 'RMS residual', 'RMS sky', 'RMS EoR',
               'resid/sky', 'resid/EoR', 'samples')
    rows = [
        (r.label, f'{r.rms_residual:.4g}', f'{r.rms_sky:.4g}',
         f'{r.rms_eor:.4g}', f'{r.fractional:.3g}',
         f'{r.relative_to_eor:.3g}', f'{r.nsamples:d}')
        for r in residuals
    ]
    return _render_table(headers, rows)


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
        '--task', choices=('dps', 'bsys', 'sky'), default='dps',
        help=(
            "which comparison to run: 'dps' (default) the delay power "
            "spectrum of --target against --reference; 'bsys' the posterior "
            "spread of the systematics amplitudes in the same two runs, with "
            "the partner correlation from --pair; 'sky' the RMS sky residual "
            'of every run given by --runs'
        ),
    )
    p.add_argument(
        '--runs', metavar='[LABEL=]DIR', nargs='+', default=None,
        help="run directories for --task sky, in the order to table them",
    )
    p.add_argument(
        '--map', metavar='I:J[,I:J...]', dest='index_map', default=None,
        help=(
            'for --task bsys, which reference parameter matches which target '
            'parameter, 1-based, e.g. 1:9,2:10,3:11,4:12 to compare an '
            'individual Case III run against the combined run. Without it the '
            'parameters are matched by position, which is right only for '
            'Case I'
        ),
    )
    p.add_argument(
        '--pair', metavar='I,J', default=None,
        help=(
            'for --task bsys, the 1-based parameter pair whose correlation '
            'is reported in the target run, e.g. 1,9'
        ),
    )
    p.add_argument(
        '--stride', type=int, default=SKY_STRIDE_DEFAULT,
        help=(
            'for --task sky, use every STRIDE-th sample when averaging '
            f'(default: {SKY_STRIDE_DEFAULT})'
        ),
    )
    p.add_argument(
        '--nfgmodes', type=int, default=NFGMODES_DEFAULT,
        help=f'foreground modes used (default: {NFGMODES_DEFAULT})',
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


def _parse_pair(spec: str) -> Tuple[int, int]:
    """Parse a 1-based ``I,J`` parameter pair into 0-based indices.

    Parameters
    ----------
    spec : str
        Pair as given on the command line, e.g. ``'1,9'``.

    Returns
    -------
    tuple of int
        Zero-based ``(i, j)``.
    """
    parts = spec.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f'--pair wants I,J, got {spec!r}')
    try:
        i, j = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'--pair wants integers, got {spec!r}'
        ) from exc
    if i < 1 or j < 1:
        raise argparse.ArgumentTypeError('--pair indices are 1-based')
    return i - 1, j - 1


def _parse_index_map(spec: str) -> List[Tuple[int, int]]:
    """Parse a 1-based ``I:J,I:J`` parameter mapping into 0-based pairs.

    Parameters
    ----------
    spec : str
        Mapping as given on the command line, e.g. ``'1:9,2:10'``.

    Returns
    -------
    list of (int, int)
        Zero-based ``(reference index, target index)`` pairs.
    """
    pairs = []
    for chunk in spec.split(','):
        parts = chunk.split(':')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f'--map wants I:J pairs, got {chunk!r}'
            )
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f'--map wants integers, got {chunk!r}'
            ) from exc
        if a < 1 or b < 1:
            raise argparse.ArgumentTypeError('--map indices are 1-based')
        pairs.append((a - 1, b - 1))
    if not pairs:
        raise argparse.ArgumentTypeError('--map is empty')
    return pairs


def _run_bsys(args: argparse.Namespace) -> int:
    """Handle ``--task bsys``: posterior spread, and the partner correlation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.

    Returns
    -------
    int
        Process exit status.
    """
    if not (args.reference and args.target):
        print('--task bsys needs --reference and --target')
        return 2

    ref_label, ref_dir = _parse_run_spec(args.reference)
    tgt_label, tgt_dir = _parse_run_spec(args.target)
    reference = load_bsys(ref_dir, args.niter, args.burn_pc)
    target = load_bsys(tgt_dir, args.niter, args.burn_pc)

    indices = _parse_index_map(args.index_map) if args.index_map else None

    print('Marginal posterior spread of the systematics amplitudes')
    print('(the combined-case argument predicts a ratio above one)\n')
    print(compare_bsys_spread(
        reference, target,
        ref_label or Path(ref_dir).name,
        tgt_label or Path(tgt_dir).name,
        indices=indices,
    ))

    if args.pair:
        i, j = _parse_pair(args.pair)
        r = bsys_correlation(target, i, j)
        print(
            f'\nPartner correlation in {tgt_label or Path(tgt_dir).name}, '
            f'b_sys,{i + 1} vs b_sys,{j + 1}:'
        )
        print(f'  real {r["real"]:+.3f}   imag {r["imag"]:+.3f}   '
              f'max |r| {r["max_abs"]:.3f}')
    return 0


def _run_sky(args: argparse.Namespace) -> int:
    """Handle ``--task sky``: RMS sky residual for every run given.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line.

    Returns
    -------
    int
        Process exit status.
    """
    if not args.runs:
        print('--task sky needs --runs')
        return 2

    residuals = []
    for spec in args.runs:
        label, run_dir = _parse_run_spec(spec)
        residuals.append(sky_residual_rms(
            run_dir, label, args.niter, args.burn_pc, args.stride,
            args.ntimes, args.nfreqs, args.nfgmodes,
        ))

    print('RMS residual between the true sky and its posterior predictive mean\n')
    print(sky_residual_table(residuals))

    if args.json:
        Path(args.json).write_text(
            json.dumps([asdict(r) for r in residuals], indent=2)
        )
        print(f'\nMetrics written to {args.json}')
    return 0


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

    if args.task == 'sky':
        return _run_sky(args)
    if args.task == 'bsys':
        return _run_bsys(args)

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
