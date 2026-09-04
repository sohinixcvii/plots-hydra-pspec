#!/usr/bin/env python
"""Blink-comparison corner plots of the systematic amplitudes ``b_sys``.

This is a standalone companion to Figure 6 (``bsys_corner_plot.pdf``) of
``paper_plots_c_v2.ipynb``.  That notebook draws one corner plot from the
sampled systematic amplitudes; this script draws *one corner plot per chain
length* (100k and 250k Gibbs iterations by default) on **identical axes** --
same per-parameter ranges, same histogram bins, same contour levels and the
same y-limits on the diagonal panels -- so that the figures can be blinked
against one another and any change in the posterior is a real change and not a
change of scale.

Nothing in the existing notebooks or modules is imported or modified: the
plotting helper below is a copy of the notebook's ``corner_plot`` with a
``ranges`` argument added.

Configuration lives in the ``CONFIG`` block below; the most common overrides
are also available on the command line.

The 100k and 250k runs live in *different* directories, so each variant carries
its own ``result_dir``; the defaults in the ``CONFIG`` block below point at
both, and ``--variant`` does the same from the command line.  Comparing two
truncations of one chain still works -- give both variants the same directory.

Examples
--------
Draw the default 100k vs 250k comparison and save PDF + PNG + a blink GIF::

    conda run -n py10 python plot_corner_blink.py --save

Name both directories explicitly (``all`` keeps whatever each chain holds)::

    conda run -n py10 python plot_corner_blink.py --save \\
        --variant all /nvme2/.../paper_plots/sim_data/ 100k \\
        --variant all /nvme2/.../paper_plots/250k_run/ 250k

Compare three truncations of the chains in one directory::

    conda run -n py10 python plot_corner_blink.py \\
        --result-dir /path/to/paper_plots/250k_run/ \\
        --nsamples 50000 --nsamples 100000 --nsamples 250000 --save

Check the machinery without any of the real data on disk::

    conda run -n py10 python plot_corner_blink.py --demo
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════

# ── Paths ─────────────────────────────────────────────────────────────────
# The 100k and 250k runs live in separate directories, so each variant below
# carries its own `result_dir`.  RESULT_DIR is only the fallback for a variant
# that names none (and the default of --result-dir).
RESULT_DIR_100K = '/nvme2/scratch/sohini/hydra-pspec-systematic/paper_plots/sim_data/'
RESULT_DIR_250K = '/nvme2/scratch/sohini/hydra-pspec-systematic/paper_plots/250k_run/'
RESULT_DIR = RESULT_DIR_100K
FIG_DIR = '/nvme2/scratch/sohini/hydra-pspec-systematic/Figures'

# ── Cases (runs overlaid inside every figure) ─────────────────────────────
# One entry per run sub-folder of `result_dir`, matching the `run_version_arr`
# / `case_labels` of paper_plots_c_v2.ipynb.  `sys_amps_true` is optional and
# only the first case that supplies it is drawn as the truth lines, exactly as
# the notebook does.  Runs whose b-sys.npy has a different number of modes
# from the first case are skipped with a warning (the corner grid is square).
CASES: List[Dict[str, Any]] = [
    dict(run='low_dl_fr_0', label='Case I',
         sys_amps_true=[1. + 4j, 2. + 3j, 3. + 2j, 4. + 1j]),
    dict(run='high_dl_fr_0', label='Case II'),
    dict(run='low_dl_fr_20', label='Case III'),
    # dict(run='caseiv', label='Combined'),   # twelve modes: needs its own run
]

# ── Variants (one figure per entry; these are what you blink between) ─────
# Each entry is one figure:
#
#   nsamples    iterations to keep, counted from the start of the chain.
#               None (or 'all' on the command line) keeps whatever is there.
#   result_dir  where that variant's runs live; falls back to RESULT_DIR.
#               The 100k and 250k runs are in different directories, so both
#               are named explicitly below.
#   cases/runs  optional per-variant override of CASES, for when the run
#               sub-folders are not named the same way in both directories.
#               `runs` is a list of sub-folder names; `cases` a list of full
#               case dicts.  Labels and truths come from CASES by position
#               when only `runs` is given.
#   label/tag   optional; default to '<count> iterations' and '<count>'.
VARIANTS: List[Dict[str, Any]] = [
    dict(nsamples=100_000, result_dir=RESULT_DIR_100K),
    dict(nsamples=250_000, result_dir=RESULT_DIR_250K),
]

# ── Sample handling ───────────────────────────────────────────────────────
# BURN_MODE 'count'   : drop the first BURN samples   (the notebook's default)
# BURN_MODE 'percent' : drop the first BURN % of each chain, so both variants
#                       are burnt in proportionally
BURN_MODE = 'count'
BURN = 10
THIN = 1

# ── Corner plot appearance ────────────────────────────────────────────────
NSIGMA = 3          # number of contour levels
BINS = 100          # histogram bins per axis (shared, so the blink is fair)
SMOOTH = 2.0        # Gaussian smoothing of the 2D histograms, in bins
RANGE_PAD = 0.05    # fractional padding added to the shared parameter ranges
SHOW_TITLES = True  # per-parameter medians above the diagonal panels
LABELS: Optional[List[str]] = None   # None -> r'$b_{sys,i}$', i = 1 .. ndim
COLORS = ['#1d3557', '#ca6702', '#81babc', '#e63946', '#ff8fba']
TRUTH_COLOR = 'tab:red'
PANEL_SIZE = (5.0, 5.5)   # inches per corner panel -> (20, 22) for ndim = 4

# ── Output ────────────────────────────────────────────────────────────────
SAVE_FIGS = False   # write the figures to FIG_DIR (--save on the command line)
FIG_STEM = 'bsys_corner'
FIG_SUFFIX = '_blink'   # keeps these files clear of the paper figures
FORMATS = ('pdf', 'png')
DPI = 100
MAKE_GIF = True     # also write an animated blink GIF from the PNGs
GIF_MS = 900        # milliseconds per frame

# ═══════════════════════════════════════════════════════════════════════════


def apply_plot_style() -> None:
    """Apply the paper's Matplotlib style (STIX fonts, large default text)."""
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'
    plt.rcParams.update({'font.size': 34})


def format_count(nsamples: int) -> str:
    """Format a sample count as a short tag, e.g. ``250000`` -> ``'250k'``.

    Parameters
    ----------
    nsamples : int
        Number of samples.

    Returns
    -------
    str
        ``'<n>k'`` for exact multiples of 1000, ``'<n>M'`` for exact multiples
        of a million, otherwise the plain integer.
    """
    if nsamples >= 1_000_000 and nsamples % 1_000_000 == 0:
        return f'{nsamples // 1_000_000}M'
    if nsamples >= 1000 and nsamples % 1000 == 0:
        return f'{nsamples // 1000}k'
    return str(nsamples)


def load_bsys(
    run_dir: str,
    nsamples: Optional[int] = None,
    burn: int = BURN,
    burn_mode: str = BURN_MODE,
    thin: int = THIN,
) -> np.ndarray:
    """Load ``b-sys.npy`` from one run and return the sampled amplitudes.

    Parameters
    ----------
    run_dir : str
        Directory of the run, i.e. ``<result_dir>/<run>``.
    nsamples : int, optional
        Keep only the first `nsamples` iterations of the chain.  ``None`` keeps
        the whole chain.  Fewer samples than requested is not an error; the
        number actually read is returned via the array length.
    burn : int
        Burn-in, interpreted according to `burn_mode`.
    burn_mode : {'count', 'percent'}
        ``'count'`` drops the first `burn` samples (what the notebook's
        ``corner_plot`` does); ``'percent'`` drops the first `burn` per cent.
    thin : int
        Keep every `thin`-th sample after burn-in.

    Returns
    -------
    ndarray, shape (Nkept, Nsysmodes)
        Absolute value of the complex systematic amplitudes.

    Raises
    ------
    FileNotFoundError
        If `run_dir` holds no ``b-sys.npy``.
    ValueError
        If `burn_mode` is not recognised, or the burn-in leaves no samples.
    """
    path = os.path.join(run_dir, 'b-sys.npy')
    if not os.path.exists(path):
        raise FileNotFoundError(f'no b-sys.npy in {run_dir}')

    chain = np.load(path, mmap_mode='r')
    if nsamples is not None:
        chain = chain[:int(nsamples)]

    if burn_mode == 'count':
        nburn = int(burn)
    elif burn_mode == 'percent':
        nburn = int(chain.shape[0] * burn / 100.)
    else:
        raise ValueError("burn_mode must be 'count' or 'percent'")

    kept = np.abs(np.asarray(chain[nburn::max(int(thin), 1)]))
    if kept.size == 0:
        raise ValueError(
            f'{path}: burn-in of {nburn} leaves no samples out of '
            f'{chain.shape[0]}'
        )
    return kept


def common_ranges(
    sample_sets: Sequence[np.ndarray],
    pad: float = RANGE_PAD,
) -> List[Tuple[float, float]]:
    """Compute one padded (low, high) range per parameter across all samples.

    Passing the result to every figure is what makes the figures blinkable:
    the axes, and hence the histogram bin edges, are then identical.

    Parameters
    ----------
    sample_sets : sequence of ndarray
        Sample arrays, each of shape ``(Nsamples, ndim)`` with the same `ndim`.
    pad : float
        Fraction of the full span added to each side.

    Returns
    -------
    list of tuple of float
        ``[(lo, hi), ...]``, one per parameter.

    Raises
    ------
    ValueError
        If `sample_sets` is empty or the arrays disagree on `ndim`.
    """
    if len(sample_sets) == 0:
        raise ValueError('sample_sets is empty')
    ndims = {s.shape[1] for s in sample_sets}
    if len(ndims) != 1:
        raise ValueError(f'inconsistent ndim across sample sets: {sorted(ndims)}')

    ndim = ndims.pop()
    ranges = []
    for i in range(ndim):
        lo = min(float(s[:, i].min()) for s in sample_sets)
        hi = max(float(s[:, i].max()) for s in sample_sets)
        span = hi - lo
        if span <= 0:                      # degenerate (constant) parameter
            span = max(abs(hi), 1.0)
        ranges.append((lo - pad * span, hi + pad * span))
    return ranges


def corner_plot(
    samples: np.ndarray,
    labels: Optional[Sequence[str]] = None,
    truths: Optional[Sequence[float]] = None,
    fig: Optional[plt.Figure] = None,
    ranges: Optional[Sequence[Tuple[float, float]]] = None,
    truths_label: str = 'Truth',
    title: Optional[str] = None,
    legend_loc: str = 'best',
    nsigma: int = NSIGMA,
    bins: int = BINS,
    smooth: float = SMOOTH,
    show_titles: bool = SHOW_TITLES,
    color: str = 'C0',
    case_label: Optional[str] = None,
) -> plt.Figure:
    """Draw one case onto a corner figure, in the style of Figure 6.

    A copy of ``corner_plot`` from ``paper_plots_c_v2.ipynb`` with the
    burn/thin arguments removed (burn-in is applied in `load_bsys` instead) and
    a `ranges` argument added so that several figures can share their axes.

    Parameters
    ----------
    samples : ndarray, shape (Nsamples, ndim)
        Samples to plot, already burnt in and thinned.
    labels : sequence of str, optional
        Axis labels, one per parameter.
    truths : sequence of float, optional
        True values, drawn as red cross-hairs.
    fig : matplotlib.figure.Figure, optional
        Existing corner figure to overplot onto; ``None`` creates one.
    ranges : sequence of tuple of float, optional
        ``(lo, hi)`` per parameter.  Use `common_ranges` to share them.
    truths_label : str
        Legend entry for the truth lines.
    title : str, optional
        Figure super-title.
    legend_loc : str
        Matplotlib legend location for the legend in the top-right panel.
    nsigma : int
        Number of contour levels (1, 2 or 3).
    bins : int
        Histogram bins per axis.
    smooth : float
        Gaussian smoothing of the 2D histograms, in bins.
    show_titles : bool
        Show the per-parameter medians above the diagonal panels.
    color : str
        Colour of this case.
    case_label : str, optional
        Legend entry for this case.

    Returns
    -------
    matplotlib.figure.Figure
        The figure, so that further cases can be overplotted onto it.

    Raises
    ------
    ValueError
        If `nsigma` is not 1, 2 or 3.
    """
    if nsigma not in (1, 2, 3):
        raise ValueError('nsigma must be 1, 2, or 3.')

    ndim = samples.shape[1]
    levels = [1 - np.exp(-0.5 * n**2) for n in range(1, nsigma + 1)]

    fig = corner.corner(
        samples,
        labels=labels,
        truths=truths,
        truth_color=TRUTH_COLOR,
        title_kwargs={'fontsize': 30},
        label_kwargs={'fontsize': 35},
        quantiles=[],
        levels=levels,
        fig=fig,
        range=None if ranges is None else list(ranges),
        show_titles=show_titles,
        title_fmt='.3g',
        bins=bins,
        smooth=smooth,
        color=color,
    )
    axes = np.array(fig.axes).reshape(ndim, ndim)

    # ── Clean up tick overlap ────────────────────────────────────────────
    for ii in range(ndim):
        for jj in range(ii + 1):
            ax = axes[ii, jj]
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))
            ax.tick_params(axis='x', labelsize=25, labelrotation=0)
            ax.tick_params(axis='y', labelsize=25, labelrotation=0)
            if ii != jj:  # off-diagonal: also fix y-axis
                ax.yaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=3, prune='both'))
                ax.yaxis.set_label_coords(-0.4, 0.5)  # push y-label left a bit

    # ── Legend, accumulated across calls, in the top-right panel ─────────
    ax0 = axes[0, -1]
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    existing_handles = []
    if ax0.get_legend() is not None:
        existing_handles = list(ax0.get_legend().legend_handles)

    new_handles = []
    if truths is not None:
        new_handles.append(
            Line2D([0], [0], color=TRUTH_COLOR, lw=1.6, label=truths_label))
    if case_label is not None:
        new_handles.append(
            Line2D([0], [0], color=color, lw=1.6, label=case_label))

    all_handles = existing_handles + new_handles
    if all_handles:
        ax0.legend(handles=all_handles, loc=legend_loc, frameon=False)

    if title:
        fig.suptitle(title, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    return fig


def harmonise_diagonals(figs: Sequence[plt.Figure], ndim: int) -> None:
    """Give the diagonal panels of several corner figures a common y-limit.

    ``corner`` scales each marginal histogram to its own peak, so two figures
    drawn from different chain lengths can show the same distribution at
    different heights.  This levels them, which is what the off-diagonal
    panels already get from a shared `ranges`.

    Parameters
    ----------
    figs : sequence of matplotlib.figure.Figure
        Corner figures, all with the same `ndim`.
    ndim : int
        Number of parameters.
    """
    for i in range(ndim):
        lo = min(np.array(f.axes).reshape(ndim, ndim)[i, i].get_ylim()[0]
                 for f in figs)
        hi = max(np.array(f.axes).reshape(ndim, ndim)[i, i].get_ylim()[1]
                 for f in figs)
        for f in figs:
            np.array(f.axes).reshape(ndim, ndim)[i, i].set_ylim(lo, hi)


def resolve_cases(
    variant: Dict[str, Any],
    cases: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return the cases one variant should read.

    A variant normally reads the same runs as every other one, from its own
    directory.  When the two directories do not name their run sub-folders
    the same way, the variant can override them: ``cases`` replaces the list
    outright, while ``runs`` renames the sub-folders and keeps the labels and
    true amplitudes of `cases`, matched by position.

    Parameters
    ----------
    variant : dict
        Variant, possibly holding ``cases`` or ``runs``.
    cases : sequence of dict
        The default cases.

    Returns
    -------
    list of dict
        Cases to read for this variant.

    Raises
    ------
    ValueError
        If `variant` sets both ``cases`` and ``runs``, or if ``runs`` is
        longer than `cases` (there would be no label to pair the extras with).
    """
    if variant.get('cases') is not None and variant.get('runs') is not None:
        raise ValueError("give a variant either 'cases' or 'runs', not both")

    if variant.get('cases') is not None:
        return [dict(c) for c in variant['cases']]

    runs = variant.get('runs')
    if runs is None:
        return [dict(c) for c in cases]

    if len(runs) > len(cases):
        raise ValueError(
            f"variant lists {len(runs)} runs but only {len(cases)} cases are "
            f"configured; give the variant full 'cases' dicts instead"
        )
    resolved = []
    for run, case in zip(runs, cases):
        merged = dict(case)
        merged['run'] = run
        resolved.append(merged)
    return resolved


def read_variant(
    variant: Dict[str, Any],
    cases: Sequence[Dict[str, Any]],
    result_dir: str,
    burn: int = BURN,
    burn_mode: str = BURN_MODE,
    thin: int = THIN,
) -> Dict[str, Any]:
    """Read every case of one variant (one chain length) from disk.

    Each variant has its own results directory, so the chain lengths being
    compared do not have to be two truncations of one run: the 100k and 250k
    runs live in different directories and are read from wherever the variant
    says.  A variant may also override which runs are read, for when the two
    directories do not use the same sub-folder names.

    Cases whose chains have a different number of systematic modes from the
    first case are skipped with a warning, since one corner figure can only
    show a single square grid.

    Parameters
    ----------
    variant : dict
        ``{'nsamples': int or None, 'result_dir': str (optional),
        'runs': list of str (optional), 'cases': list of dict (optional),
        'label': str (optional), 'tag': str (optional)}``.  ``nsamples`` of
        ``None`` keeps the whole chain.
    cases : sequence of dict
        ``{'run': str, 'label': str (optional),
        'sys_amps_true': sequence (optional)}``.  Used unless the variant
        overrides it.
    result_dir : str
        Default results directory, used when the variant sets none.
    burn, burn_mode, thin
        Passed through to `load_bsys`.

    Returns
    -------
    dict
        ``{'label', 'nsamples', 'tag', 'result_dir', 'samples',
        'case_labels', 'truths'}`` where ``samples`` is a list of
        ``(Nkept, ndim)`` arrays and ``nsamples`` is the number actually read
        (the shortest chain), not the number asked for.
    """
    nsamples = variant.get('nsamples')
    nsamples = None if nsamples is None else int(nsamples)
    rdir = variant.get('result_dir') or result_dir
    cases = resolve_cases(variant, cases)

    samples, labels, truths = [], [], None
    ndim = None
    for case in cases:
        run_dir = os.path.join(rdir, case['run'])
        try:
            chain = load_bsys(run_dir, nsamples, burn, burn_mode, thin)
        except (FileNotFoundError, ValueError) as exc:
            print(f'  SKIP {case["run"]}: {exc}')
            continue

        if ndim is None:
            ndim = chain.shape[1]
        elif chain.shape[1] != ndim:
            print(f'  SKIP {case["run"]}: {chain.shape[1]} systematic modes, '
                  f'but the figure is a {ndim}x{ndim} grid')
            continue

        navail = np.load(os.path.join(run_dir, 'b-sys.npy'),
                         mmap_mode='r').shape[0]
        if nsamples is not None and navail < nsamples:
            print(f'  NOTE {case["run"]}: only {navail:,} samples available '
                  f'(asked for {nsamples:,})')

        samples.append(chain)
        labels.append(case.get('label', case['run']))
        if truths is None and case.get('sys_amps_true') is not None:
            truths = list(np.abs(np.asarray(case['sys_amps_true'])))
            if len(truths) != ndim:
                print(f'  NOTE {case["run"]}: {len(truths)} true amplitudes '
                      f'but {ndim} modes; truths not drawn')
                truths = None

    # What was actually read, which is what the figure should be called: the
    # shortest chain of the variant, before burn-in and thinning.
    if samples:
        nread = min(s.shape[0] for s in samples) * max(int(thin), 1)
        nread += int(burn) if burn_mode == 'count' else 0
    else:
        nread = nsamples or 0
    # A tag given by the caller names the variant everywhere, so that reading
    # a whole 250k directory can still be labelled "250k" rather than by the
    # exact number of samples that happen to be on disk.
    count = variant.get('tag') or format_count(
        nsamples if nsamples is not None else nread)

    return dict(
        label=variant.get('label', f'{count} iterations'),
        nsamples=nread,
        tag=count,
        result_dir=rdir,
        samples=samples,
        case_labels=labels,
        truths=truths,
    )


def build_figures(
    variants_data: Sequence[Dict[str, Any]],
    labels: Optional[Sequence[str]] = None,
    colors: Sequence[str] = tuple(COLORS),
    panel_size: Tuple[float, float] = PANEL_SIZE,
    show_titles: bool = SHOW_TITLES,
    add_suptitle: bool = True,
) -> List[plt.Figure]:
    """Draw one corner figure per variant, all sharing their axes.

    Parameters
    ----------
    variants_data : sequence of dict
        Output of `read_variant`, one entry per chain length.
    labels : sequence of str, optional
        Parameter labels; ``None`` uses ``$b_{sys,i}$``.
    colors : sequence of str
        Case colours, cycled.
    panel_size : tuple of float
        Width and height in inches of one corner panel.
    show_titles : bool
        Show the per-parameter medians above the diagonal panels.
    add_suptitle : bool
        Put the variant label at the top of each figure.

    Returns
    -------
    list of matplotlib.figure.Figure
        One figure per variant, in the order given.

    Raises
    ------
    ValueError
        If no variant has any samples.
    """
    all_samples = [s for v in variants_data for s in v['samples']]
    if not all_samples:
        raise ValueError('no samples were read for any variant')

    ndim = all_samples[0].shape[1]
    ranges = common_ranges(all_samples)
    if labels is None:
        labels = [rf'$b_{{sys,{i + 1}}}$' for i in range(ndim)]

    figs = []
    for vdata in variants_data:
        fig, axes = plt.subplots(
            ndim, ndim,
            figsize=(panel_size[0] * ndim, panel_size[1] * ndim),
        )
        for ax in np.asarray(axes).flatten():
            ax.tick_params(length=8)

        for i, chain in enumerate(vdata['samples']):
            corner_plot(
                chain,
                labels=list(labels),
                truths=vdata['truths'] if i == 0 else None,
                fig=fig,
                ranges=ranges,
                color=colors[i % len(colors)],
                show_titles=show_titles,
                case_label=vdata['case_labels'][i],
            )
        if add_suptitle:
            fig.suptitle(vdata['label'], y=1.0)
        figs.append(fig)

    harmonise_diagonals(figs, ndim)
    return figs


def common_bbox(figs: Sequence[plt.Figure], pad: float = 0.1):
    """Return the union of the tight bounding boxes of several figures.

    Saving every figure with the *same* bounding box is what keeps the output
    files pixel-identical in size.  ``bbox_inches='tight'`` alone does not:
    it crops to whatever each figure happens to draw, and the medians printed
    above the diagonal panels are wider in one variant than in another, so the
    images would no longer register when blinked.

    Parameters
    ----------
    figs : sequence of matplotlib.figure.Figure
        Figures to measure.  They are drawn once, to get a renderer.
    pad : float
        Padding in inches added on every side.

    Returns
    -------
    matplotlib.transforms.Bbox
        Bounding box in inches, covering every figure.
    """
    from matplotlib.transforms import Bbox

    x0 = y0 = np.inf
    x1 = y1 = -np.inf
    for fig in figs:
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        x0, y0 = min(x0, bb.x0), min(y0, bb.y0)
        x1, y1 = max(x1, bb.x1), max(y1, bb.y1)
    return Bbox.from_extents(x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def save_blink_gif(png_paths: Sequence[str], out_path: str,
                   duration_ms: int = GIF_MS) -> Optional[str]:
    """Combine PNG frames into an animated GIF that blinks between them.

    Parameters
    ----------
    png_paths : sequence of str
        Frames, in order.
    out_path : str
        GIF to write.
    duration_ms : int
        Milliseconds each frame is shown.

    Returns
    -------
    str or None
        `out_path`, or ``None`` if Pillow is unavailable or fewer than two
        frames were given.
    """
    if len(png_paths) < 2:
        return None
    try:
        from PIL import Image
    except ImportError:
        print('  (Pillow not available, no blink GIF written)')
        return None

    frames = [Image.open(p).convert('P', palette=Image.ADAPTIVE)
              for p in png_paths]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    return out_path


def save_figures(
    figs: Sequence[plt.Figure],
    variants_data: Sequence[Dict[str, Any]],
    fig_dir: str,
    stem: str = FIG_STEM,
    suffix: str = FIG_SUFFIX,
    formats: Sequence[str] = FORMATS,
    dpi: int = DPI,
    make_gif: bool = MAKE_GIF,
) -> List[str]:
    """Write every figure to `fig_dir`, plus an optional blink GIF.

    Parameters
    ----------
    figs : sequence of matplotlib.figure.Figure
        Figures, aligned with `variants_data`.
    variants_data : sequence of dict
        Output of `read_variant`; the ``tag`` keys name the files.
    fig_dir : str
        Output directory; created if it does not exist.
    stem, suffix : str
        File names are ``<stem>_<tag><suffix>.<ext>``.
    formats : sequence of str
        File extensions to write.
    dpi : int
        Raster resolution.
    make_gif : bool
        Also combine the PNGs into ``<stem>_blink.gif``.

    Returns
    -------
    list of str
        Paths written.
    """
    os.makedirs(fig_dir, exist_ok=True)
    bbox = common_bbox(figs)
    written, pngs = [], []
    for fig, vdata in zip(figs, variants_data):
        for ext in formats:
            path = os.path.join(fig_dir, f'{stem}_{vdata["tag"]}{suffix}.{ext}')
            fig.savefig(path, bbox_inches=bbox, dpi=dpi)
            written.append(path)
            if ext == 'png':
                pngs.append(path)
            print(f'  saved {path}')

    if make_gif and pngs:
        gif = save_blink_gif(pngs, os.path.join(fig_dir, f'{stem}{suffix}.gif'))
        if gif is not None:
            written.append(gif)
            print(f'  saved {gif}')
    return written


def make_demo_data(
    root: str,
    cases: Sequence[Dict[str, Any]],
    nsamples: int,
    seed: int = 0,
) -> str:
    """Write small synthetic ``b-sys.npy`` chains, for smoke-testing.

    The chains drift slightly with iteration, so the 100k and 250k figures
    genuinely differ and the blink can be inspected without the real runs.

    Parameters
    ----------
    root : str
        Directory to create the run sub-folders in.
    cases : sequence of dict
        Cases to fabricate; only ``run`` and ``sys_amps_true`` are used.
    nsamples : int
        Chain length to write.
    seed : int
        Seed for the random draws.

    Returns
    -------
    str
        `root`, for use as a ``result_dir``.
    """
    rng = np.random.default_rng(seed)
    truth = np.asarray(cases[0].get(
        'sys_amps_true', [1. + 4j, 2. + 3j, 3. + 2j, 4. + 1j]))
    ndim = truth.size

    for icase, case in enumerate(cases):
        run_dir = os.path.join(root, case['run'])
        os.makedirs(run_dir, exist_ok=True)
        offset = 0.15 * icase
        drift = np.linspace(0., 0.25, nsamples)[:, None]
        chain = (truth[None, :] * (1. + offset)
                 + drift
                 + 0.3 * (rng.standard_normal((nsamples, ndim))
                          + 1j * rng.standard_normal((nsamples, ndim))))
        np.save(os.path.join(run_dir, 'b-sys.npy'), chain)
    return root


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments; ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--result-dir', default=RESULT_DIR,
                   help='directory holding the run sub-folders')
    p.add_argument('--fig-dir', default=FIG_DIR,
                   help='output directory for the figures')
    p.add_argument('--runs', nargs='+', default=None,
                   help='run sub-folders to overlay (default: the CASES block)')
    p.add_argument('--nsamples', type=int, action='append', default=None,
                   metavar='N',
                   help='chain length of one variant, read from --result-dir; '
                        'repeat for each figure.  Use --variant instead when '
                        'the variants live in different directories')
    p.add_argument('--variant', nargs='+', action='append', default=None,
                   metavar=('N DIR', 'TAG'),
                   help='one variant read from its own directory: N is the '
                        'chain length (or "all" for the whole chain), DIR the '
                        'results directory, TAG the optional file-name tag.  '
                        'Repeat once per figure, e.g. '
                        '--variant 100000 /path/sim_data/ '
                        '--variant 250000 /path/250k_run/')
    p.add_argument('--burn', type=int, default=BURN,
                   help=f'burn-in, in {BURN_MODE} (default: {BURN})')
    p.add_argument('--burn-mode', choices=('count', 'percent'),
                   default=BURN_MODE, help='how --burn is interpreted')
    p.add_argument('--thin', type=int, default=THIN,
                   help='keep every THIN-th sample')
    p.add_argument('--no-titles', action='store_true',
                   help='drop the medians above the diagonal panels, which '
                        'move between variants')
    p.add_argument('--dpi', type=int, default=DPI, help='raster resolution')
    p.add_argument('--save', dest='save', action='store_true',
                   default=SAVE_FIGS, help='write the figures to --fig-dir')
    p.add_argument('--no-save', dest='save', action='store_false',
                   help='do not write any files')
    p.add_argument('--no-gif', dest='gif', action='store_false',
                   default=MAKE_GIF, help='do not write the blink GIF')
    p.add_argument('--show', action='store_true',
                   help='open the figures in a window')
    p.add_argument('--demo', action='store_true',
                   help='run on small synthetic chains in a temporary '
                        'directory, for testing without the real data')
    return p.parse_args(argv)


def variants_from_args(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Turn the command line into a list of variants.

    ``--variant N DIR [TAG]`` is the form to use when the chain lengths live in
    different directories, which is how the 100k and 250k runs are stored;
    ``--nsamples N`` is the shorthand for several lengths of one directory.
    Neither given falls back to the `VARIANTS` block.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, holding ``variant`` and ``nsamples``.

    Returns
    -------
    list of dict
        Variants, ready for `read_variant`.

    Raises
    ------
    ValueError
        If ``--variant`` and ``--nsamples`` are combined, if a ``--variant``
        does not have two or three fields, or if its chain length is neither
        an integer nor ``'all'``.
    """
    if args.variant and args.nsamples:
        raise ValueError('give either --variant or --nsamples, not both')

    if args.nsamples:
        return [dict(nsamples=n) for n in args.nsamples]

    if not args.variant:
        return [dict(v) for v in VARIANTS]

    variants = []
    for fields in args.variant:
        if not 2 <= len(fields) <= 3:
            raise ValueError(
                f'--variant takes "N DIR [TAG]", got {" ".join(fields)!r}')
        count, rdir = fields[0], fields[1]
        if count.lower() == 'all':
            nsamples = None
        else:
            try:
                nsamples = int(count)
            except ValueError:
                raise ValueError(
                    f'--variant: {count!r} is not a chain length or "all"'
                ) from None
        variant: Dict[str, Any] = dict(nsamples=nsamples, result_dir=rdir)
        if len(fields) == 3:
            variant['tag'] = fields[2]
        variants.append(variant)
    return variants


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build and optionally save the blink-comparison corner plots.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments; ``None`` uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status: 0 on success, 1 if nothing could be read.
    """
    args = parse_args(argv)
    apply_plot_style()

    cases = ([dict(run=r) for r in args.runs] if args.runs else CASES)
    try:
        variants = variants_from_args(args)
    except ValueError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    result_dir = args.result_dir
    fig_dir = args.fig_dir
    tmpdir = None
    if args.demo:
        # Two separate directories, as the 100k and 250k runs really are.
        tmpdir = tempfile.TemporaryDirectory(prefix='corner_blink_demo_')
        dir_short = make_demo_data(os.path.join(tmpdir.name, 'short_run'),
                                   cases[:3], nsamples=1000)
        dir_long = make_demo_data(os.path.join(tmpdir.name, 'long_run'),
                                  cases[:3], nsamples=3000)
        result_dir = dir_short
        variants = [dict(nsamples=None, result_dir=dir_short, tag='1k'),
                    dict(nsamples=None, result_dir=dir_long, tag='3k')]
        if fig_dir == FIG_DIR:      # no --fig-dir given: keep the demo self-contained
            fig_dir = os.path.join(tmpdir.name, 'figs')
        args.save = True
        print(f'DEMO: synthetic chains in {dir_short} and {dir_long}')
        print(f'DEMO: figures in {fig_dir}')

    try:
        variants_data = []
        for variant in variants:
            asked = variant.get('nsamples')
            print(f'Reading {format_count(asked) if asked else "all"} samples '
                  f'from {variant.get("result_dir") or result_dir}')
            vdata = read_variant(variant, cases, result_dir,
                                 args.burn, args.burn_mode, args.thin)
            print(f'  {len(vdata["samples"])} case(s), '
                  f'{[s.shape for s in vdata["samples"]]}')
            variants_data.append(vdata)

        try:
            figs = build_figures(variants_data,
                                 show_titles=not args.no_titles)
        except ValueError as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 1

        if args.save:
            save_figures(figs, variants_data, fig_dir, dpi=args.dpi,
                         make_gif=args.gif)
        else:
            print('  (--save not given, nothing written)')

        if args.show:
            plt.show()
        else:
            for fig in figs:
                plt.close(fig)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
