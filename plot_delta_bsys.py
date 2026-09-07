#!/usr/bin/env python
"""Interval plot of the systematic-amplitude residuals ``delta b_sys``.

A replacement for the corner plot of the systematic amplitudes
(``paper_plots_c_v2_single_case.ipynb``, Figure 8) that shows the quantity the
corner plot is really about -- how far the posterior of each amplitude sits
from its true value -- instead of the amplitudes themselves.

Every parameter gets one row.  The row is centred on

.. math::

    \\Delta b_{\\mathrm{sys},i} = b_{\\mathrm{sys},i} - b^{\\mathrm{true}}_{\\mathrm{sys},i}

so the truth is the vertical line at zero running down the middle of the
figure, and the axis is symmetric about it.  On the row are

* graded credible-interval bars -- thick for :math:`1\\sigma`, thinner for
  :math:`2\\sigma` and :math:`3\\sigma`, from the same quantiles the corner plot
  used (0.15865/0.84135, 0.02275/0.97725, 0.00135/0.99865);
* the posterior median, with the ``median +hi -lo`` deviations of the corner
  plot's diagonal titles;
* the posterior mean with a :math:`\\pm\\sigma` error bar, drawn just below the
  interval bar so that the standard deviation is visible as a length and not
  only as a number.

The numbers behind all of that are printed alongside each row, so the figure
carries the same summary statistics as the corner plot's titles (median, the
plus/minus deviations, mean and sigma) without the ndim x ndim panel grid.
This is deliberately *not* a violin plot: no kernel-density estimate is drawn,
only the exact sample quantiles.

The module holds no data of its own.  The notebook passes it the real chain::

    import plot_delta_bsys as pdb

    fig, stats = pdb.plot_delta_bsys(
        np.abs(b_sys_gcr[:Niter]),
        np.abs(sys_amps_true),
        nsigma=3,
    )

Run the module directly for a smoke test on a synthetic chain::

    conda run -n py10 python plot_delta_bsys.py
    conda run -n py10 python plot_delta_bsys.py --save /tmp/delta_bsys.pdf
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ── Configuration ──────────────────────────────────────────────────────────

# Notebook palette (`colors` in the plot-specifications cell).
PAPER_COLORS: List[str] = ['#1d3557', '#ca6702', '#81babc', '#e63946', '#ffc8dd']

# Line widths of the graded interval bars, thickest (1 sigma) first.
BAR_WIDTHS: Tuple[float, ...] = (11.0, 5.5, 2.2)

# Vertical offset, in row units, of the mean +/- sigma error bar.
MEAN_OFFSET: float = 0.22

# Fraction of the half-range left as padding at each end of the x axis.
XPAD: float = 0.12


# ── Statistics ─────────────────────────────────────────────────────────────

@dataclass
class DeltaSummary:
    """Posterior summary of one systematic amplitude residual.

    Attributes
    ----------
    label : str
        Parameter label, used on the y axis of the figure.
    truth : float
        True value the residual is measured from.
    mean : float
        Posterior mean of ``delta b_sys``.
    median : float
        Posterior median of ``delta b_sys``.
    std : float
        Posterior standard deviation (sigma).
    lower : list of float
        Lower credible bounds of ``delta b_sys``, 1 sigma first.
    upper : list of float
        Upper credible bounds of ``delta b_sys``, 1 sigma first.
    nsamples : int
        Number of samples the summary was computed from.
    """

    label: str
    truth: float
    mean: float
    median: float
    std: float
    lower: List[float]
    upper: List[float]
    nsamples: int

    @property
    def minus(self) -> List[float]:
        """Distances from the median down to each lower bound (positive)."""
        return [self.median - lo for lo in self.lower]

    @property
    def plus(self) -> List[float]:
        """Distances from the median up to each upper bound (positive)."""
        return [hi - self.median for hi in self.upper]

    @property
    def pull(self) -> float:
        """Mean residual in units of sigma, ``mean / std`` (0 if sigma is 0)."""
        return self.mean / self.std if self.std > 0. else 0.


def sigma_quantiles(nsigma: int) -> List[Tuple[float, float]]:
    """Two-sided Gaussian quantile pairs for 1 .. ``nsigma`` sigma.

    Parameters
    ----------
    nsigma : int
        Highest sigma level, 1, 2 or 3, as in the notebook's ``corner_plot``.

    Returns
    -------
    list of tuple of float
        ``(lower, upper)`` quantiles, 1 sigma first, e.g. ``(0.15865,
        0.84135)`` for one sigma.

    Raises
    ------
    ValueError
        If ``nsigma`` is not 1, 2 or 3.
    """
    if nsigma not in (1, 2, 3):
        raise ValueError('nsigma must be 1, 2, or 3.')

    pairs = []
    for k in range(1, nsigma + 1):
        upper = 0.5 * (1. + math.erf(k / math.sqrt(2.)))
        pairs.append((1. - upper, upper))
    return pairs


def delta_samples(
    samples: np.ndarray,
    truths: Sequence[float],
    burn: int = 0,
    thin: int = 1,
) -> np.ndarray:
    """Residual chain ``samples - truths``, after burn-in and thinning.

    Parameters
    ----------
    samples : numpy.ndarray
        Real samples of shape ``(nsamples, ndim)``.  Complex chains must be
        reduced first (``np.abs``, ``np.real``, ...), exactly as they are for
        the corner plot.
    truths : sequence of float
        The ``ndim`` true values, reduced the same way as ``samples``.
    burn : int, optional
        Number of leading samples to discard.  Default 0, matching the corner
        plot.
    thin : int, optional
        Keep every ``thin``-th sample.  Default 1.

    Returns
    -------
    numpy.ndarray
        Residuals of shape ``(nkept, ndim)``.

    Raises
    ------
    ValueError
        If ``samples`` is not 2D, if ``truths`` does not match its second
        axis, if ``thin`` is not positive, or if burn-in leaves no samples.
    TypeError
        If ``samples`` or ``truths`` are complex.
    """
    S = np.asarray(samples)
    t = np.asarray(truths)

    if S.ndim != 2:
        raise ValueError(f'samples must be 2D (nsamples, ndim), got {S.shape}')
    if np.iscomplexobj(S) or np.iscomplexobj(t):
        raise TypeError('samples and truths must be real; reduce the complex '
                        'chain first, e.g. np.abs(b_sys_gcr).')
    if t.shape != (S.shape[1],):
        raise ValueError(f'truths has shape {t.shape}, expected {(S.shape[1],)}')
    if thin < 1:
        raise ValueError(f'thin must be >= 1, got {thin}')
    if burn < 0:
        raise ValueError(f'burn must be >= 0, got {burn}')

    kept = S[burn::thin]
    if kept.shape[0] == 0:
        raise ValueError(f'burn={burn}, thin={thin} leave no samples of '
                         f'{S.shape[0]}')

    return kept - t[None, :]


def summarise_delta(
    deltas: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    truths: Optional[Sequence[float]] = None,
    nsigma: int = 3,
) -> List[DeltaSummary]:
    """Summarise a residual chain parameter by parameter.

    Parameters
    ----------
    deltas : numpy.ndarray
        Residuals of shape ``(nsamples, ndim)``, from `delta_samples`.
    labels : sequence of str, optional
        Parameter labels.  Default ``r'$b_{sys,i}$'`` for ``i = 1 .. ndim``.
    truths : sequence of float, optional
        True values, carried through into the summaries.  Default zeros.
    nsigma : int, optional
        Highest sigma level summarised.  Default 3, as in the corner plot.

    Returns
    -------
    list of DeltaSummary
        One summary per parameter, in column order.

    Raises
    ------
    ValueError
        If ``deltas`` is not 2D or ``labels`` / ``truths`` have the wrong
        length.
    """
    D = np.asarray(deltas)
    if D.ndim != 2:
        raise ValueError(f'deltas must be 2D (nsamples, ndim), got {D.shape}')

    ndim = D.shape[1]
    names = list(labels) if labels is not None else default_labels(ndim)
    if len(names) != ndim:
        raise ValueError(f'got {len(names)} labels for {ndim} parameters')

    true_vals = np.zeros(ndim) if truths is None else np.asarray(truths, float)
    if true_vals.shape != (ndim,):
        raise ValueError(f'got {true_vals.shape} truths for {ndim} parameters')

    quantiles = sigma_quantiles(nsigma)
    flat = [q for pair in quantiles for q in pair]
    bounds = np.percentile(D, [100. * q for q in flat], axis=0)

    summaries = []
    for i in range(ndim):
        summaries.append(DeltaSummary(
            label=names[i],
            truth=float(true_vals[i]),
            mean=float(D[:, i].mean()),
            median=float(np.median(D[:, i])),
            std=float(D[:, i].std()),
            lower=[float(bounds[2 * k, i]) for k in range(len(quantiles))],
            upper=[float(bounds[2 * k + 1, i]) for k in range(len(quantiles))],
            nsamples=int(D.shape[0]),
        ))
    return summaries


def default_labels(ndim: int) -> List[str]:
    """``[r'$b_{sys,1}$', ...]`` for ``ndim`` parameters.

    Parameters
    ----------
    ndim : int
        Number of systematic amplitudes.

    Returns
    -------
    list of str
        Math-mode labels, numbered from 1.
    """
    return [rf'$b_{{sys,{i}}}$' for i in range(1, ndim + 1)]


def common_exponent(value: float, threshold: int = 3) -> int:
    """Power of ten worth factoring out of a group of numbers.

    Parameters
    ----------
    value : float
        Largest magnitude of the group that will be written together.
    threshold : int, optional
        Smallest ``|exponent|`` worth factoring out.  Default 3, so numbers
        between 0.001 and 1000 keep their plain decimal form.

    Returns
    -------
    int
        Exponent to factor out; 0 means "write the numbers as they are".
    """
    if not np.isfinite(value) or value == 0.:
        return 0

    exponent = int(math.floor(math.log10(abs(value))))
    return exponent if abs(exponent) >= threshold else 0


def decimals_for(value: float, sig: int = 3) -> int:
    """Decimal places giving ``value`` ``sig`` significant digits.

    Parameters
    ----------
    value : float
        Reference value, usually the smallest quantity on the line.
    sig : int, optional
        Significant digits wanted.  Default 3.

    Returns
    -------
    int
        Number of decimal places, never negative.
    """
    if not np.isfinite(value) or value == 0.:
        return sig - 1

    return max(0, sig - 1 - int(math.floor(math.log10(abs(value)))))


def _fixed_decimals(reference: float, centre: float, sig: int) -> int:
    """Decimals set by ``reference``, widened if ``centre`` would round to 0.

    Parameters
    ----------
    reference : float
        Quantity that sets the precision (a deviation or sigma).
    centre : float
        Quantity that must not be printed as a bare zero (a median or mean).
    sig : int
        Significant digits wanted for ``reference``.

    Returns
    -------
    int
        Decimal places to use for every number on the line.
    """
    decimals = decimals_for(reference, sig)

    if centre != 0. and round(centre, decimals) == 0.:
        # Give the centre one significant digit rather than print '0.000'.
        decimals = min(decimals_for(centre, 1), decimals + 3)

    return decimals


def _times_power(exponent: int) -> str:
    """``\\times 10^{e}`` for a non-zero exponent, empty string otherwise.

    Parameters
    ----------
    exponent : int
        Factored-out power of ten.

    Returns
    -------
    str
        Math-mode fragment, without surrounding ``$``.
    """
    return '' if exponent == 0 else rf'\times 10^{{{exponent}}}'


def format_triplet(
    centre: float,
    plus: float,
    minus: float,
    sig: int = 3,
    exponent: Optional[int] = None,
) -> str:
    """``centre +plus -minus`` as math-text, with a shared power of ten.

    Parameters
    ----------
    centre : float
        Central value (the median).
    plus, minus : float
        Positive distances up to the upper and down to the lower bound.
    sig : int, optional
        Significant digits of the deviations.  Default 3.
    exponent : int, optional
        Power of ten to factor out.  Default: chosen from the deviations by
        `common_exponent`.  Pass one to keep several rows consistent.

    Returns
    -------
    str
        Math-mode body, e.g. ``0.0000^{+0.0124}_{-0.0121}``, or
        ``\\left(...\\right)\\times 10^{-6}`` when everything is tiny.

    Notes
    -----
    All three numbers share one power of ten so that a mantissa's own
    exponent never collides with the ``+``/``-`` superscripts, which
    matplotlib's math-text renders as a second superscript.
    """
    reference = max(abs(plus), abs(minus)) or abs(centre) or 1.
    if exponent is None:
        exponent = common_exponent(reference)
    scale = 10. ** exponent

    decimals = _fixed_decimals(reference / scale, centre / scale, sig)
    body = (f'{centre / scale:.{decimals}f}'
            f'^{{+{plus / scale:.{decimals}f}}}'
            f'_{{-{minus / scale:.{decimals}f}}}')

    if exponent == 0:
        return body
    return rf'\left({body}\right){_times_power(exponent)}'


def format_mean_sigma(
    mean: float,
    std: float,
    sig: int = 3,
    exponent: Optional[int] = None,
) -> str:
    """``\\mu = ..., \\sigma = ...`` as math-text, with a shared power of ten.

    Parameters
    ----------
    mean : float
        Posterior mean of the residual.
    std : float
        Posterior standard deviation.
    sig : int, optional
        Significant digits of sigma.  Default 3.
    exponent : int, optional
        Power of ten to factor out.  Default: chosen from sigma by
        `common_exponent`.

    Returns
    -------
    str
        Math-mode body, without surrounding ``$``.
    """
    reference = abs(std) or abs(mean) or 1.
    if exponent is None:
        exponent = common_exponent(reference)
    scale = 10. ** exponent

    decimals = _fixed_decimals(reference / scale, mean / scale, sig)
    power = _times_power(exponent)

    return (rf'\mu = {mean / scale:.{decimals}f}{power}, \ '
            rf'\sigma = {std / scale:.{decimals}f}{power}')


def shared_exponent(
    summaries: Sequence[DeltaSummary],
    nsigma: Optional[int] = None,
    max_spread: int = 2,
) -> Optional[int]:
    """Power of ten that suits every row, or None if the rows are too unalike.

    Parameters
    ----------
    summaries : sequence of DeltaSummary
        Parameter summaries, one per row of the figure.
    nsigma : int, optional
        Sigma level whose deviations set the magnitudes.  Default: the highest
        level held by the summaries.
    max_spread : int, optional
        Largest range of magnitudes, in decades, still worth writing on one
        common scale.  Default 2.

    Returns
    -------
    int or None
        Exponent to factor out of every row (0 meaning "plain decimals"), or
        None when the rows span more than ``max_spread`` decades and are
        better formatted one by one.
    """
    exponents = []
    for summary in summaries:
        k = (len(summary.lower) if nsigma is None else nsigma) - 1
        reference = (max(abs(summary.plus[k]), abs(summary.minus[k]))
                     or abs(summary.median) or abs(summary.std))
        if reference > 0. and np.isfinite(reference):
            exponents.append(int(math.floor(math.log10(reference))))

    if not exponents:
        return 0
    if max(exponents) - min(exponents) > max_spread:
        return None

    return common_exponent(10. ** max(exponents))


def format_summary(
    summary: DeltaSummary,
    sig: int = 3,
    nsigma: Optional[int] = None,
    exponent: Optional[int] = None,
) -> str:
    """Two-line annotation carrying the corner plot's diagonal-title numbers.

    Parameters
    ----------
    summary : DeltaSummary
        Summary of one parameter.
    sig : int, optional
        Significant digits for every number.  Default 3, matching the corner
        plot's ``title_fmt='.3g'``.
    nsigma : int, optional
        Sigma level whose deviations are quoted.  Default: the highest level
        held by ``summary``.
    exponent : int, optional
        Power of ten factored out of both lines.  Default: chosen per line.
        `shared_exponent` supplies one that suits every row of a figure.

    Returns
    -------
    str
        ``median +hi -lo`` on the first line; mean, sigma and the mean in
        units of sigma on the second, as a matplotlib math-text string.

    Raises
    ------
    ValueError
        If ``nsigma`` exceeds the levels held by ``summary``.
    """
    level = len(summary.lower) if nsigma is None else nsigma
    if not 1 <= level <= len(summary.lower):
        raise ValueError(f'summary holds {len(summary.lower)} sigma levels, '
                         f'asked for {level}')

    k = level - 1
    median = format_triplet(summary.median, summary.plus[k], summary.minus[k],
                            sig=sig, exponent=exponent)
    mean_sigma = format_mean_sigma(summary.mean, summary.std, sig=sig,
                                   exponent=exponent)

    return (rf'$\Delta = {median}$   (${level}\sigma$)'
            '\n'
            rf'${mean_sigma}$, '
            rf'$\mu/\sigma = {summary.pull:.2f}$')


# ── Plot ───────────────────────────────────────────────────────────────────

def _scale_factors(
    summaries: Sequence[DeltaSummary],
    units: str,
) -> np.ndarray:
    """Per-parameter divisor turning residuals into the requested units.

    Parameters
    ----------
    summaries : sequence of DeltaSummary
        Parameter summaries.
    units : {'absolute', 'sigma'}
        ``'absolute'`` leaves the residuals alone; ``'sigma'`` divides each by
        its own posterior standard deviation.

    Returns
    -------
    numpy.ndarray
        One positive divisor per parameter.

    Raises
    ------
    ValueError
        If ``units`` is not one of the two accepted values.
    """
    if units == 'absolute':
        return np.ones(len(summaries))
    if units == 'sigma':
        return np.array([s.std if s.std > 0. else 1. for s in summaries])
    raise ValueError(f"units must be 'absolute' or 'sigma', got {units!r}")


def _draw_row(
    ax: plt.Axes,
    y: float,
    summary: DeltaSummary,
    scale: float,
    colors: Sequence[str],
    bar_widths: Sequence[float],
    marker_size: float,
) -> None:
    """Draw the interval bars and markers of one parameter onto ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    y : float
        Row centre in data coordinates.
    summary : DeltaSummary
        Summary of the parameter drawn on this row.
    scale : float
        Divisor applied to every abscissa (see `_scale_factors`).
    colors : sequence of str
        Palette; ``colors[0]`` draws the intervals, ``colors[1]`` the mean.
    bar_widths : sequence of float
        Line widths of the interval bars, thickest (1 sigma) first.
    marker_size : float
        Size of the median and mean markers.
    """
    nlevels = len(summary.lower)

    # Widest interval first, so the thicker inner bars are drawn on top.
    for k in reversed(range(nlevels)):
        width = bar_widths[min(k, len(bar_widths) - 1)]
        ax.plot(
            [summary.lower[k] / scale, summary.upper[k] / scale],
            [y, y],
            color=colors[0],
            linewidth=width,
            alpha=1.0 - 0.15 * k,
            solid_capstyle='butt',
            zorder=3 + k,
        )

    # Mean +/- sigma, offset below the interval bar so the two do not overlap.
    ax.errorbar(
        summary.mean / scale,
        y + MEAN_OFFSET,
        xerr=summary.std / scale,
        fmt='D',
        markersize=marker_size * 0.55,
        color=colors[1],
        ecolor=colors[1],
        elinewidth=2.0,
        capsize=6,
        capthick=2.0,
        zorder=7,
    )

    # Median on top of the interval bars.
    ax.plot(
        summary.median / scale,
        y,
        marker='o',
        markersize=marker_size,
        markerfacecolor='white',
        markeredgecolor=colors[0],
        markeredgewidth=2.0,
        linestyle='none',
        zorder=8,
    )


def _axis_limit(
    summaries: Sequence[DeltaSummary],
    scales: np.ndarray,
    pad: float = XPAD,
) -> float:
    """Half-width of a symmetric x axis holding every drawn element.

    Parameters
    ----------
    summaries : sequence of DeltaSummary
        Parameter summaries.
    scales : numpy.ndarray
        Per-parameter divisors from `_scale_factors`.
    pad : float, optional
        Extra fraction of the half-range added at both ends.  Default `XPAD`.

    Returns
    -------
    float
        Positive half-width; the axis then runs from ``-limit`` to ``+limit``,
        centred on zero.
    """
    reach = [0.]
    for s, scale in zip(summaries, scales):
        reach.extend(abs(v) / scale for v in s.lower)
        reach.extend(abs(v) / scale for v in s.upper)
        reach.append((abs(s.mean) + s.std) / scale)

    limit = max(reach)
    if limit <= 0.:
        return 1.
    return limit * (1. + pad)


def plot_delta_bsys(
    samples: np.ndarray,
    truths: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    burn: int = 0,
    thin: int = 1,
    nsigma: int = 3,
    units: str = 'absolute',
    annotate: bool = True,
    annotation_sig: int = 3,
    colors: Optional[Sequence[str]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    row_height: float = 1.15,
    label_fontsize: float = 30.,
    tick_fontsize: float = 24.,
    annotation_fontsize: float = 20.,
    legend_fontsize: float = 22.,
    marker_size: float = 13.,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    legend_loc: str = 'outside',
    legend_ncol: int = 3,
) -> Tuple[plt.Figure, List[DeltaSummary]]:
    """Draw the ``delta b_sys`` interval plot.

    Parameters
    ----------
    samples : numpy.ndarray
        Real chain of shape ``(nsamples, ndim)``, e.g. ``np.abs(b_sys_gcr)``.
    truths : sequence of float
        True amplitudes, reduced the same way as ``samples``.
    labels : sequence of str, optional
        Parameter labels.  Default ``r'$b_{sys,i}$'``.
    burn : int, optional
        Samples discarded from the head of the chain.  Default 0.
    thin : int, optional
        Keep every ``thin``-th sample.  Default 1.
    nsigma : int, optional
        Highest credible interval drawn and quoted.  Default 3.
    units : {'absolute', 'sigma'}, optional
        Draw the residuals in their own units, or divided by each parameter's
        sigma.  Default ``'absolute'``.
    annotate : bool, optional
        Print the summary numbers beside each row.  Default True.
    annotation_sig : int, optional
        Significant digits in the annotations.  Default 3, as in the corner
        plot's titles.
    colors : sequence of str, optional
        Palette; ``colors[0]`` intervals, ``colors[1]`` mean, ``colors[3]``
        zero line.  Default `PAPER_COLORS`.
    fig, ax : matplotlib objects, optional
        Draw into an existing figure/axes instead of making one.
    figsize : tuple of float, optional
        Figure size in inches.  Default scales with the number of parameters.
    row_height : float, optional
        Vertical spacing between parameter rows, in data units.  Default 1.15.
    label_fontsize, tick_fontsize, annotation_fontsize, legend_fontsize : float, optional
        Font sizes of the axis labels, ticks, row annotations and legend.
    marker_size : float, optional
        Size of the median marker.  Default 13.
    title : str, optional
        Figure title.  Default: none.
    xlabel : str, optional
        Override the x-axis label.
    legend_loc : str, optional
        ``'outside'`` (default) puts the key above the rows, where it cannot
        cover one; anything else is passed to ``ax.legend`` as a location.
    legend_ncol : int, optional
        Number of legend columns.  Default 3.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure drawn into.
    summaries : list of DeltaSummary
        The numbers behind it, in parameter order.

    Notes
    -----
    The zero line is the truth, so it is placed at the centre of a symmetric
    x axis.  Nothing here estimates a density: the bars are sample quantiles.
    """
    deltas = delta_samples(samples, truths, burn=burn, thin=thin)
    summaries = summarise_delta(deltas, labels=labels, truths=truths,
                                nsigma=nsigma)
    ndim = len(summaries)

    palette = list(colors) if colors is not None else PAPER_COLORS
    if len(palette) < 4:
        palette = list(palette) + PAPER_COLORS[len(palette):]

    scales = _scale_factors(summaries, units)

    own_figure = ax is None
    if ax is None:
        if figsize is None:
            width = 20. if annotate else 14.
            figsize = (width, max(5.0, 1.5 * ndim + 3.0))
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    # Rows top-to-bottom in parameter order.
    ys = row_height * np.arange(ndim)

    # The truth, down the middle of the figure.
    ax.axvline(0., color=palette[3], linestyle='--', linewidth=2.5, zorder=2)

    for y, summary, scale in zip(ys, summaries, scales):
        _draw_row(ax, y, summary, scale, palette, BAR_WIDTHS, marker_size)

    limit = _axis_limit(summaries, scales)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(ys[-1] + row_height * 0.75, ys[0] - row_height * 0.75)

    ax.set_yticks(ys)
    ax.set_yticklabels([s.label for s in summaries], fontsize=label_fontsize)
    ax.tick_params(axis='both', direction='out', length=6, width=1.5,
                   labelsize=tick_fontsize)

    if xlabel is None:
        xlabel = (r'$\Delta b_{\mathrm{sys}} = b_{\mathrm{sys}} - '
                  r'b^{\mathrm{true}}_{\mathrm{sys}}$')
        if units == 'sigma':
            xlabel += r' $[\sigma]$'
        else:
            xlabel += ' (arbitrary units)'
    ax.set_xlabel(xlabel, fontsize=label_fontsize)

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    if annotate:
        # One power of ten for the whole column, where the rows allow it.
        exponent = shared_exponent(summaries, nsigma=nsigma)
        for y, summary in zip(ys, summaries):
            ax.annotate(
                format_summary(summary, sig=annotation_sig, nsigma=nsigma,
                               exponent=exponent),
                xy=(1.02, y),
                xycoords=('axes fraction', 'data'),
                va='center',
                ha='left',
                fontsize=annotation_fontsize,
                annotation_clip=False,
            )

    handles = [
        Line2D([0], [0], color=palette[3], ls='--', lw=2.5, label='Truth'),
        Line2D([0], [0], color=palette[0], lw=BAR_WIDTHS[0],
               label=r'$1\sigma$ interval'),
    ]
    for k in range(1, min(nsigma, len(BAR_WIDTHS))):
        handles.append(Line2D([0], [0], color=palette[0],
                              lw=BAR_WIDTHS[k], alpha=1.0 - 0.15 * k,
                              label=rf'${k + 1}\sigma$ interval'))
    handles += [
        Line2D([0], [0], color=palette[0], marker='o', ls='none',
               markersize=marker_size, markerfacecolor='white',
               markeredgewidth=2.0, label='Median'),
        Line2D([0], [0], color=palette[1], marker='D', ls='none',
               markersize=marker_size * 0.55, label=r'Mean $\pm\ \sigma$'),
    ]
    if legend_loc == 'outside':
        # Above the rows, so no row can be covered by the key.
        ax.legend(handles=handles, loc='lower center',
                  bbox_to_anchor=(0.5, 1.01), frameon=False,
                  fontsize=legend_fontsize, ncol=legend_ncol,
                  columnspacing=1.6, handlelength=2.4, borderaxespad=0.)
    else:
        ax.legend(handles=handles, loc=legend_loc, frameon=False,
                  fontsize=legend_fontsize, ncol=legend_ncol)

    if title:
        if legend_loc == 'outside':
            fig.suptitle(title, fontsize=label_fontsize)
        else:
            ax.set_title(title, fontsize=label_fontsize)

    if own_figure:
        # Skipped for a caller-supplied axes: its geometry is the caller's.
        fig.tight_layout()

    return fig, summaries


def summary_text(summaries: Sequence[DeltaSummary], fmt: str = '.3g') -> str:
    """Plain-text table of the summaries, for printing next to the figure.

    Parameters
    ----------
    summaries : sequence of DeltaSummary
        Parameter summaries.
    fmt : str, optional
        Format spec for every number.  Default ``'.3g'``.

    Returns
    -------
    str
        One header line and one line per parameter: truth, mean, median,
        sigma, the highest-sigma deviations and the pull.
    """
    nsig = len(summaries[0].lower) if summaries else 0
    head = (f'{"param":>12} {"truth":>12} {"mean":>12} {"median":>12} '
            f'{"sigma":>12} {f"+{nsig}sig":>12} {f"-{nsig}sig":>12} '
            f'{"mu/sig":>8}')
    lines = [head, '-' * len(head)]

    for i, s in enumerate(summaries, start=1):
        lines.append(
            f'{f"b_sys,{i}":>12} {s.truth:>12{fmt}} {s.mean:>12{fmt}} '
            f'{s.median:>12{fmt}} {s.std:>12{fmt}} '
            f'{s.plus[-1]:>12{fmt}} {s.minus[-1]:>12{fmt}} {s.pull:>8.2f}'
        )
    return '\n'.join(lines)


# ── Demo / smoke test ──────────────────────────────────────────────────────

def make_demo_chain(
    truths: Sequence[float] = (12., 20.4),
    nsamples: int = 5000,
    sigmas: Optional[Sequence[float]] = None,
    seed: int = 0,
) -> np.ndarray:
    """Synthetic chain around ``truths``, for tests and the module's demo.

    Parameters
    ----------
    truths : sequence of float, optional
        True amplitudes the chain scatters around.
    nsamples : int, optional
        Chain length.  Default 5000.
    sigmas : sequence of float, optional
        Per-parameter scatter.  Default: 0.1 per cent of each truth.
    seed : int, optional
        Seed of the random generator.  Default 0.

    Returns
    -------
    numpy.ndarray
        Real samples of shape ``(nsamples, ndim)``.
    """
    t = np.asarray(truths, float)
    s = (1e-3 * np.abs(t) if sigmas is None else np.asarray(sigmas, float))
    rng = np.random.default_rng(seed)
    return t[None, :] + s[None, :] * rng.standard_normal((nsamples, t.size))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the demo command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse.  Default: ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: ``save``, ``nsigma``, ``units``, ``ndim``.
    """
    parser = argparse.ArgumentParser(
        description='Smoke test of the delta b_sys interval plot on a '
                    'synthetic chain.')
    parser.add_argument('--save', default=None,
                        help='write the demo figure to this path')
    parser.add_argument('--nsigma', type=int, default=3, choices=(1, 2, 3),
                        help='highest credible interval drawn (default 3)')
    parser.add_argument('--units', default='absolute',
                        choices=('absolute', 'sigma'),
                        help="x-axis units (default 'absolute')")
    parser.add_argument('--ndim', type=int, default=4,
                        help='number of synthetic amplitudes (default 4)')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Draw the demo figure and print its summary table.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments.  Default: ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status, 0 on success.
    """
    args = parse_args(argv)

    mpl.use('Agg')
    truths = np.linspace(4., 20., args.ndim)
    chain = make_demo_chain(truths, nsamples=20000)

    fig, summaries = plot_delta_bsys(
        chain, truths, nsigma=args.nsigma, units=args.units,
        title='Demo: synthetic chain',
    )
    print(summary_text(summaries))

    if args.save:
        fig.savefig(args.save, bbox_inches='tight', dpi=150)
        print(f'\nwrote {args.save}')
    else:
        print('\n(no --save given; figure built but not written)')

    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
