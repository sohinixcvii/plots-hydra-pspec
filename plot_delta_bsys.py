#!/usr/bin/env python
"""Interval plot of the systematic-amplitude residuals ``delta b_sys``.

A replacement for the corner plot of the systematic amplitudes
(``paper_plots_c_v2_single_case.ipynb``, Figure 8) that shows the quantity the
corner plot is really about -- how far the posterior of each amplitude sits
from its true value -- instead of the amplitudes themselves.

Every parameter gets one row, and the real and imaginary parts of that
amplitude are drawn as two point sets on it, offset a little above and below
the row centre.  Each is centred on

.. math::

    \\Delta b_{\\mathrm{sys},i} = b_{\\mathrm{sys},i} - b^{\\mathrm{true}}_{\\mathrm{sys},i}

so the truth is the vertical line at zero running down the middle of the
figure, and the axis is symmetric about it.  On each point set are

* graded credible-interval bars -- thick for :math:`1\\sigma`, thinner for
  :math:`2\\sigma` and :math:`3\\sigma`, from the same quantiles the corner plot
  used (0.15865/0.84135, 0.02275/0.97725, 0.00135/0.99865);
* the posterior median, as an open circle;
* the posterior mean with a :math:`\\pm\\sigma` error bar, drawn just below the
  interval bar so that the standard deviation is visible as a length.

The figure carries no text of its own beyond the axis labels and the key.  The
numbers -- median, the plus/minus deviations, mean, sigma -- come back with the
figure as `DeltaSummary` records and print as a table with `summary_text`;
``annotate=True`` puts them beside the rows instead.

This is deliberately *not* a violin plot: no kernel-density estimate is drawn,
only the exact sample quantiles.

The module holds no data of its own.  The notebook passes it the chain::

    import plot_delta_bsys as pdb

    fig, stats = pdb.plot_delta_bsys(
        b_sys_gcr[:Niter],          # complex -> real and imaginary parts
        sys_amps_true,
        nsigma=3,
    )
    print(pdb.summary_text(stats))

A real chain (``np.abs(b_sys_gcr)``, the corner plot's own reduction) draws a
single point set per row instead; ``components`` chooses explicitly.

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

# Neutral colour of the shape entries in the key (bar widths, markers): they
# describe every component rather than one of them.
NEUTRAL_COLOR: str = '#555555'

# Vertical offset of the mean +/- sigma error bar below its interval bar, as a
# fraction of the space one point set owns.  Only used by
# ``statistic='both'``; the single-statistic styles centre on the row.
MEAN_OFFSET: float = 0.42

# The statistics a point set can be drawn as.  'intervals' is the graded
# credible intervals with the posterior median; 'mean' is the posterior mean
# with a +/- sigma error bar; 'both' draws them one above the other, which is
# the pair a reader is unlikely to need at once.
STATISTICS: Tuple[str, ...] = ('intervals', 'mean', 'both')

# How the credible intervals are named in the key.  'collapsed' gives them one
# entry naming every level; 'graded' gives one entry per level, three swatches
# that differ only in line width.
INTERVAL_KEYS: Tuple[str, ...] = ('collapsed', 'graded')

# Vertical separation of the components of one parameter, as a fraction of the
# row height.
COMPONENT_GAP: float = 0.42

# Fraction of the half-range left as padding at each end of the x axis.
XPAD: float = 0.12

# The reductions a chain can be drawn through.
COMPONENT_FUNCS = {
    'real': np.real,
    'imag': np.imag,
    'abs': np.abs,
    'value': lambda x: x,      # a chain that is already real
}

# How each reduction is named in the key.
COMPONENT_LABELS = {
    'real': r'$\mathrm{Re}$',
    'imag': r'$\mathrm{Im}$',
    'abs': r'$|\cdot|$',
    'value': '',
}


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
    component : str
        Which reduction of the chain this is -- ``'real'``, ``'imag'``,
        ``'abs'``, or ``'value'`` for a chain that was already real.
    """

    label: str
    truth: float
    mean: float
    median: float
    std: float
    lower: List[float]
    upper: List[float]
    nsamples: int
    component: str = 'value'

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


def component_chains(
    samples: np.ndarray,
    truths: Sequence[complex],
    components: Optional[Sequence[str]] = None,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Split a chain into the real chains that will be drawn.

    Parameters
    ----------
    samples : numpy.ndarray
        Chain of shape ``(nsamples, ndim)``, complex or real.
    truths : sequence of complex
        The ``ndim`` true amplitudes, in the same form as ``samples``.
    components : sequence of str, optional
        Reductions to draw, from `COMPONENT_FUNCS`.  Default:
        ``('real', 'imag')`` for a complex chain, ``('value',)`` for a real
        one.

    Returns
    -------
    list of tuple
        ``(component, samples, truths)``, one entry per component, with real
        arrays.

    Raises
    ------
    ValueError
        If ``samples`` is not 2D, ``truths`` does not match its second axis,
        or a component name is not one of `COMPONENT_FUNCS`.

    Notes
    -----
    The reduction is applied to the samples and to the truths separately, and
    the residual is taken afterwards -- the same thing for the linear parts,
    and for ``'abs'`` it is what the corner plot does.
    """
    S = np.asarray(samples)
    t = np.asarray(truths)

    if S.ndim != 2:
        raise ValueError(f'samples must be 2D (nsamples, ndim), got {S.shape}')
    if t.shape != (S.shape[1],):
        raise ValueError(f'truths has shape {t.shape}, expected {(S.shape[1],)}')

    if components is None:
        components = ('real', 'imag') if np.iscomplexobj(S) else ('value',)
    components = list(components)

    if not components:
        raise ValueError('components is empty')
    unknown = [c for c in components if c not in COMPONENT_FUNCS]
    if unknown:
        raise ValueError(f'unknown components {unknown}; choose from '
                         f'{sorted(COMPONENT_FUNCS)}')

    chains = []
    for name in components:
        reduce = COMPONENT_FUNCS[name]
        chains.append((name,
                       np.asarray(reduce(S), dtype=float),
                       np.asarray(reduce(t), dtype=float)))
    return chains


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
    component: str = 'value',
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
    component : str, optional
        Reduction these residuals came through, recorded on each summary.
        Default ``'value'``.

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
            component=component,
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
    color: str,
    bar_widths: Sequence[float],
    marker_size: float,
    mean_offset: float,
    statistic: str = 'intervals',
) -> None:
    """Draw one point set onto ``ax``, as the statistic asked for.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw into.
    y : float
        Centre of this point set, in data coordinates.
    summary : DeltaSummary
        Summary of the parameter component drawn here.
    scale : float
        Divisor applied to every abscissa (see `_scale_factors`).
    color : str
        Colour of the bars and both markers; one colour per component.
    bar_widths : sequence of float
        Line widths of the interval bars, thickest (1 sigma) first.
    marker_size : float
        Size of the median marker.
    mean_offset : float
        Vertical offset of the mean error bar below the interval bar.  Applied
        only when both statistics are drawn.
    statistic : {'intervals', 'mean', 'both'}, optional
        Which statistic this point set shows.  Default ``'intervals'``.
    """
    if statistic not in STATISTICS:
        raise ValueError(
            f'statistic must be one of {STATISTICS}, got {statistic!r}'
        )

    draw_intervals = statistic in ('intervals', 'both')
    draw_mean = statistic in ('mean', 'both')
    # With one statistic on the row it sits on the row centre; with both, the
    # mean drops below so the two do not overlap.
    mean_y = y + mean_offset if statistic == 'both' else y

    if draw_mean:
        ax.errorbar(
            summary.mean / scale,
            mean_y,
            xerr=summary.std / scale,
            fmt='D',
            markersize=marker_size * (0.55 if statistic == 'both' else 0.8),
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=5,
            capthick=2.0,
            zorder=7,
        )

    if not draw_intervals:
        return

    nlevels = len(summary.lower)

    # Widest interval first, so the thicker inner bars are drawn on top.
    for k in reversed(range(nlevels)):
        width = bar_widths[min(k, len(bar_widths) - 1)]
        ax.plot(
            [summary.lower[k] / scale, summary.upper[k] / scale],
            [y, y],
            color=color,
            linewidth=width,
            alpha=1.0 - 0.15 * k,
            solid_capstyle='butt',
            zorder=3 + k,
        )

    # Median on top of the interval bars.
    ax.plot(
        summary.median / scale,
        y,
        marker='o',
        markersize=marker_size,
        markerfacecolor='white',
        markeredgecolor=color,
        markeredgewidth=2.0,
        linestyle='none',
        zorder=8,
    )


def _axis_limit(
    summaries: Sequence[DeltaSummary],
    scales: np.ndarray,
    pad: float = XPAD,
    statistic: str = 'intervals',
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
    statistic : {'intervals', 'mean', 'both'}, optional
        Which statistic is drawn; only what is on the figure sets the limit,
        so dropping one can tighten the axis.  Default ``'intervals'``.

    Returns
    -------
    float
        Positive half-width; the axis then runs from ``-limit`` to ``+limit``,
        centred on zero.
    """
    if statistic not in STATISTICS:
        raise ValueError(
            f'statistic must be one of {STATISTICS}, got {statistic!r}'
        )
    reach = [0.]
    for s, scale in zip(summaries, scales):
        if statistic in ('intervals', 'both'):
            reach.extend(abs(v) / scale for v in s.lower)
            reach.extend(abs(v) / scale for v in s.upper)
        if statistic in ('mean', 'both'):
            reach.append((abs(s.mean) + s.std) / scale)

    limit = max(reach)
    if limit <= 0.:
        return 1.
    return limit * (1. + pad)


def _legend_handles(
    component_names: Sequence[str],
    component_colors: Sequence[str],
    nsigma: int,
    zero_color: str,
    marker_size: float,
    statistic: str = 'intervals',
    interval_key: str = 'collapsed',
) -> List[Line2D]:
    """Key entries: the truth line, one per component, then the shapes.

    Parameters
    ----------
    component_names : sequence of str
        Components drawn, from `COMPONENT_FUNCS`.
    component_colors : sequence of str
        Their colours, in the same order.
    nsigma : int
        Highest credible interval drawn.
    zero_color : str
        Colour of the line through zero.
    marker_size : float
        Size of the median marker.
    statistic : {'intervals', 'mean', 'both'}, optional
        Which statistic is drawn.  The key describes only what is on the
        figure.  Default ``'intervals'``.
    interval_key : {'collapsed', 'graded'}, optional
        ``'collapsed'``, the default, gives the credible intervals a single
        entry naming every level; ``'graded'`` gives one entry per level.
        Collapsed suits a figure whose caption already maps thick, medium and
        thin to the levels, and spends one slot instead of `nsigma`.  Ignored
        when only one level is drawn.

    Returns
    -------
    list of matplotlib.lines.Line2D
        Proxy artists for ``ax.legend``.

    Notes
    -----
    The component entries carry the colours; the interval widths and the two
    markers describe every component alike, so they are drawn in
    `NEUTRAL_COLOR` whenever more than one component is on the figure.
    """
    handles = [Line2D([0], [0], color=zero_color, ls='--', lw=2.5,
                      label='Truth')]

    named = [n for n in component_names if COMPONENT_LABELS.get(n, n)]
    if len(named) > 1:
        for name, color in zip(component_names, component_colors):
            handles.append(Line2D([0], [0], color=color, lw=BAR_WIDTHS[0],
                                  label=COMPONENT_LABELS.get(name, name)))
        shape_color = NEUTRAL_COLOR
    else:
        shape_color = component_colors[0]

    if statistic not in STATISTICS:
        raise ValueError(
            f'statistic must be one of {STATISTICS}, got {statistic!r}'
        )

    if interval_key not in INTERVAL_KEYS:
        raise ValueError(
            f'interval_key must be one of {INTERVAL_KEYS}, got {interval_key!r}'
        )

    if statistic in ('intervals', 'both'):
        nlevels = min(nsigma, len(BAR_WIDTHS))
        if interval_key == 'collapsed' and nlevels > 1:
            levels = '/'.join(rf'${k + 1}\sigma$' for k in range(nlevels))
            # Two lines: one long entry would set the width of its whole
            # legend column and strand the entries beside it.
            handles.append(Line2D([0], [0], color=shape_color,
                                  lw=BAR_WIDTHS[0],
                                  label=f'Interval\n({levels})'))
        else:
            for k in range(nlevels):
                handles.append(Line2D([0], [0], color=shape_color,
                                      lw=BAR_WIDTHS[k],
                                      alpha=1.0 - 0.15 * k,
                                      label=rf'${k + 1}\sigma$ interval'))
        handles.append(
            Line2D([0], [0], color=shape_color, marker='o', ls='none',
                   markersize=marker_size, markerfacecolor='white',
                   markeredgewidth=2.0, label='Median')
        )

    if statistic in ('mean', 'both'):
        handles.append(
            Line2D([0], [0], color=shape_color, marker='D', ls='none',
                   markersize=marker_size * (0.55 if statistic == 'both'
                                             else 0.8),
                   label=r'Mean $\pm\ \sigma$')
        )
    return handles


def component_offsets(ncomponents: int, row_height: float) -> np.ndarray:
    """Vertical offsets of a parameter's components about its row centre.

    Parameters
    ----------
    ncomponents : int
        Number of components drawn per parameter.
    row_height : float
        Spacing between parameter rows, in data units.

    Returns
    -------
    numpy.ndarray
        One offset per component, symmetric about zero; a single zero when
        there is only one component.
    """
    if ncomponents < 2:
        return np.zeros(max(ncomponents, 1))

    span = COMPONENT_GAP * row_height
    return span * (np.arange(ncomponents) - 0.5 * (ncomponents - 1))


def plot_delta_bsys(
    samples: np.ndarray,
    truths: Sequence[complex],
    labels: Optional[Sequence[str]] = None,
    components: Optional[Sequence[str]] = None,
    burn: int = 0,
    thin: int = 1,
    nsigma: int = 3,
    statistic: str = 'intervals',
    interval_key: str = 'collapsed',
    units: str = 'absolute',
    annotate: bool = False,
    annotation_sig: int = 3,
    colors: Optional[Sequence[str]] = None,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    row_height: float = 1.15,
    label_fontsize: float = 50.,
    tick_fontsize: float = 48.,
    annotation_fontsize: float = 40.,
    legend_fontsize: float = 44.,
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
        Chain of shape ``(nsamples, ndim)``.  A complex chain is drawn as its
        real and imaginary parts, two point sets on each parameter's row; a
        real chain (``np.abs(b_sys_gcr)``, as the corner plot uses) gives one.
    truths : sequence of complex
        True amplitudes, in the same form as ``samples``.
    labels : sequence of str, optional
        Parameter labels.  Default ``r'$b_{sys,i}$'``.
    components : sequence of str, optional
        Reductions to draw, from `COMPONENT_FUNCS` -- ``'real'``, ``'imag'``,
        ``'abs'``.  Default: ``('real', 'imag')`` for a complex chain.
    burn : int, optional
        Samples discarded from the head of the chain.  Default 0.
    thin : int, optional
        Keep every ``thin``-th sample.  Default 1.
    nsigma : int, optional
        Highest credible interval drawn.  Default 3.
    statistic : {'intervals', 'mean', 'both'}, optional
        What each point set shows.  ``'intervals'``, the default, draws the
        graded credible intervals with the posterior median on top;
        ``'mean'`` draws the posterior mean with a ``+/- sigma`` error bar;
        ``'both'`` draws the two one above the other.  The credible intervals
        are what supports reading a bias off the figure at a stated
        confidence level, so they are the default; the mean and its error bar
        say much the same thing again and are rarely worth the second marker.
    interval_key : {'collapsed', 'graded'}, optional
        Whether the credible intervals get one key entry naming every level
        (default) or one entry each.  The caption of the paper figure already
        maps thick, medium and thin to the levels, so one entry suffices there.
    units : {'absolute', 'sigma'}, optional
        Draw the residuals in their own units, or divided by each point set's
        sigma.  Default ``'absolute'``.
    annotate : bool, optional
        Print the summary numbers beside each row.  Default False: the figure
        carries no text, and the numbers come back with it instead.
    annotation_sig : int, optional
        Significant digits in the annotations.  Default 3, as in the corner
        plot's titles.
    colors : sequence of str, optional
        Palette; ``colors[0]``, ``colors[1]``, ... colour the components in
        order and ``colors[3]`` the zero line.  Default `PAPER_COLORS`.
    fig, ax : matplotlib objects, optional
        Draw into an existing figure/axes instead of making one.
    figsize : tuple of float, optional
        Figure size in inches.  Default scales with the number of parameters
        and components.
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
        The numbers behind it, parameter by parameter and, within a
        parameter, component by component.

    Notes
    -----
    The zero line is the truth, so it is placed at the centre of a symmetric
    x axis.  Nothing here estimates a density: the bars are sample quantiles.
    """
    chains = component_chains(samples, truths, components)
    names = [name for name, _, _ in chains]

    per_component = [
        summarise_delta(
            delta_samples(part, part_truths, burn=burn, thin=thin),
            labels=labels, truths=part_truths, nsigma=nsigma, component=name,
        )
        for name, part, part_truths in chains
    ]
    ndim = len(per_component[0])

    # Parameter-major, so the returned list reads down the figure.
    summaries = [per_component[c][i]
                 for i in range(ndim) for c in range(len(chains))]
    scales = _scale_factors(summaries, units)
    scale_of = {id(s): sc for s, sc in zip(summaries, scales)}

    palette = list(colors) if colors is not None else PAPER_COLORS
    if len(palette) < 4:
        palette = list(palette) + PAPER_COLORS[len(palette):]
    component_colors = [palette[c % len(palette)] for c in range(len(chains))]

    own_figure = ax is None
    if ax is None:
        if figsize is None:
            width = 20. if annotate else 14.
            height = max(5.0, (1.1 + 0.5 * len(chains)) * ndim + 3.0)
            figsize = (width, height)
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    # Rows top-to-bottom in parameter order; components offset within a row.
    ys = row_height * np.arange(ndim)
    offsets = component_offsets(len(chains), row_height)
    span = (abs(offsets[1] - offsets[0]) if len(chains) > 1
            else 0.5 * row_height)
    mean_offset = MEAN_OFFSET * span

    # The truth, down the middle of the figure.
    ax.axvline(0., color=palette[3], linestyle='--', linewidth=2.5, zorder=2)

    for c, summaries_c in enumerate(per_component):
        for y, summary in zip(ys + offsets[c], summaries_c):
            _draw_row(ax, y, summary, scale_of[id(summary)],
                      component_colors[c], BAR_WIDTHS, marker_size,
                      mean_offset, statistic)

    limit = _axis_limit(summaries, scales, statistic=statistic)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(ys[-1] + row_height * 0.75, ys[0] - row_height * 0.75)

    ax.set_yticks(ys)
    ax.set_yticklabels([s.label for s in per_component[0]],
                       fontsize=label_fontsize)
    ax.tick_params(axis='both', direction='out', length=6, width=1.5,
                   labelsize=tick_fontsize)

    if xlabel is None:
        xlabel = (r'$\Delta b_{\mathrm{sys}} = b_{\mathrm{sys}} - '
                  r'b_{\mathrm{sys}, \mathrm{true}}$')
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
        for c, summaries_c in enumerate(per_component):
            for y, summary in zip(ys + offsets[c], summaries_c):
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

    handles = _legend_handles(names, component_colors, nsigma, palette[3],
                              marker_size, statistic, interval_key)
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
        Parameter summaries, as returned by `plot_delta_bsys`.
    fmt : str, optional
        Format spec for every number.  Default ``'.3g'``.

    Returns
    -------
    str
        One header line and one line per point set: parameter, component,
        truth, mean, median, sigma, the highest-sigma deviations and the mean
        in units of sigma.
    """
    if not summaries:
        return ''

    nsig = len(summaries[0].lower)
    parts = sorted({s.component for s in summaries})
    ncomponents = len(parts)

    head = (f'{"param":>10} {"part":>6} {"truth":>12} {"mean":>12} '
            f'{"median":>12} {"sigma":>12} {f"+{nsig}sig":>12} '
            f'{f"-{nsig}sig":>12} {"mu/sig":>8}')
    lines = [head, '-' * len(head)]

    for k, s in enumerate(summaries):
        index = k // ncomponents + 1
        lines.append(
            f'{f"b_sys,{index}":>10} {s.component:>6} {s.truth:>12{fmt}} '
            f'{s.mean:>12{fmt}} {s.median:>12{fmt}} {s.std:>12{fmt}} '
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
    truths : sequence of float or complex, optional
        True amplitudes the chain scatters around.  Complex truths give a
        complex chain, scattered in both parts.
    nsamples : int, optional
        Chain length.  Default 5000.
    sigmas : sequence of float, optional
        Per-parameter scatter.  Default: 0.1 per cent of each truth.
    seed : int, optional
        Seed of the random generator.  Default 0.

    Returns
    -------
    numpy.ndarray
        Samples of shape ``(nsamples, ndim)``, complex if ``truths`` is.
    """
    t = np.asarray(truths)
    s = (1e-3 * np.abs(t) if sigmas is None else np.asarray(sigmas, float))
    rng = np.random.default_rng(seed)

    shape = (nsamples, t.size)
    if np.iscomplexobj(t):
        noise = (rng.standard_normal(shape)
                 + 1.j * rng.standard_normal(shape)) / np.sqrt(2.)
    else:
        noise = rng.standard_normal(shape)

    return t[None, :] + s[None, :] * noise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the demo command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse.  Default: ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: ``save``, ``nsigma``, ``units``, ``ndim``,
        ``components``, ``annotate``.
    """
    parser = argparse.ArgumentParser(
        description='Smoke test of the delta b_sys interval plot on a '
                    'synthetic chain.')
    parser.add_argument('--save', default=None,
                        help='write the demo figure to this path')
    parser.add_argument('--nsigma', type=int, default=3, choices=(1, 2, 3),
                        help='highest credible interval drawn (default 3)')
    parser.add_argument('--statistic', default='intervals',
                        choices=STATISTICS,
                        help='what each point set shows (default intervals)')
    parser.add_argument('--interval-key', default='collapsed',
                        choices=INTERVAL_KEYS, dest='interval_key',
                        help='one key entry for the intervals, or one each '
                             '(default collapsed)')
    parser.add_argument('--units', default='absolute',
                        choices=('absolute', 'sigma'),
                        help="x-axis units (default 'absolute')")
    parser.add_argument('--ndim', type=int, default=4,
                        help='number of synthetic amplitudes (default 4)')
    parser.add_argument('--components', default='real,imag',
                        help="comma-separated reductions to draw "
                             "(default 'real,imag')")
    parser.add_argument('--annotate', action='store_true',
                        help='also print the summary numbers on the figure')
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
    components = [c.strip() for c in args.components.split(',') if c.strip()]
    truths = (np.linspace(4., 20., args.ndim)
              + 1.j * np.linspace(20., 4., args.ndim))
    chain = make_demo_chain(truths, nsamples=20000)

    fig, summaries = plot_delta_bsys(
        chain, truths, components=components, nsigma=args.nsigma,
        statistic=args.statistic, interval_key=args.interval_key,
        units=args.units, annotate=args.annotate,
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
