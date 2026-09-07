#!/usr/bin/env python
"""Tighten the padding of a finished ``corner`` figure.

Figure 6 of ``paper_plots_c_v2.ipynb`` (``bsys_corner_plot.pdf``) is drawn by
the notebook's ``corner_plot`` helper, which ends with ``fig.tight_layout()``.
On a corner grid that is what produces the large padding: ``corner.corner``
lays the grid out itself, and ``tight_layout`` then re-measures every panel
including its tick labels, pushes the outer margins in and blows the gaps
between the panels out.  With 35 pt parameter labels, 25 pt tick labels and
interior panels that still carry numbers, the margins grow further, and with
three cases overlaid ``tight_layout`` runs three times over.

The single-case notebook fixed this inside its own ``corner_plot``: it hides
every tick label except the bottom row and left column, pins the parameter
labels with ``set_label_coords`` instead of relying on padding, and replaces
``tight_layout`` with an explicit ``subplots_adjust``.  This module does the
same thing from the outside, as a post-processing pass, so nothing about how
the samples are drawn has to change:

>>> import corner_padding as cp
>>> cp.tighten_corner(fig, ndim=4)                    # doctest: +SKIP

Call it once, after the last ``corner_plot`` call of the cell and before
``plt.savefig`` -- ``subplots_adjust`` runs last and so wins over the
``tight_layout`` calls inside the helper.  The maths, the samples, the contour
levels and the titles are untouched: only tick visibility, tick locators, label
positions and the figure margins are set.

Run the module directly for a smoke test on a synthetic corner figure::

    conda run -n py10 python corner_padding.py
    conda run -n py10 python corner_padding.py --save /tmp/corner_tight.pdf
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence, Tuple

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ── Configuration ──────────────────────────────────────────────────────────

# Margins of the panel grid, in figure coordinates.  These replace whatever
# `tight_layout` worked out; they leave room for the outer labels and nothing
# more.
MARGINS = dict(left=0.16, right=0.98, bottom=0.16, top=0.97)

# Gaps between panels, as a fraction of a panel.  `corner` defaults to 0.05;
# `tight_layout` typically inflates this to several tenths.
WSPACE = 0.10
HSPACE = 0.10

# Where the outer parameter labels sit, in axes coordinates.  Explicit
# positions are far more reliable here than `labelpad`.  These suit the 4 x 4
# grid of Figure 6 at 20 x 22 inches; a 12-parameter grid with one tick per
# panel wants roughly -0.65, as in the single-case notebook.
XLABEL_Y = -0.38
YLABEL_X = -0.42


# ── Axes handling ──────────────────────────────────────────────────────────

def corner_axes(fig: plt.Figure, ndim: int) -> np.ndarray:
    """The ``ndim`` x ``ndim`` panel grid of a corner figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure a corner plot was drawn into.
    ndim : int
        Number of parameters, i.e. the side of the grid.

    Returns
    -------
    numpy.ndarray
        Object array of axes, shape ``(ndim, ndim)``.

    Raises
    ------
    ValueError
        If ``ndim`` is not positive, or the figure holds fewer than
        ``ndim**2`` axes.

    Notes
    -----
    Only the first ``ndim**2`` axes are taken, so colorbars or legend axes
    added after the corner grid are ignored.
    """
    if ndim < 1:
        raise ValueError(f'ndim must be >= 1, got {ndim}')

    axes = fig.axes
    if len(axes) < ndim * ndim:
        raise ValueError(f'figure holds {len(axes)} axes, need {ndim * ndim} '
                         f'for a {ndim}x{ndim} corner grid')

    return np.asarray(axes[:ndim * ndim], dtype=object).reshape(ndim, ndim)


def infer_ndim(fig: plt.Figure) -> int:
    """Side of the corner grid implied by the number of axes in ``fig``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure a corner plot was drawn into.

    Returns
    -------
    int
        Largest ``n`` with ``n**2`` axes available.

    Raises
    ------
    ValueError
        If the figure holds no axes.
    """
    if not fig.axes:
        raise ValueError('figure holds no axes')

    return int(np.floor(np.sqrt(len(fig.axes))))


# ── Ticks and labels ───────────────────────────────────────────────────────

def _make_formatter(decimals: Optional[int]) -> mticker.Formatter:
    """Tick formatter with a fixed number of decimals, or a plain one.

    Parameters
    ----------
    decimals : int or None
        Decimal places to force; ``None`` gives a non-scientific
        ``ScalarFormatter``.

    Returns
    -------
    matplotlib.ticker.Formatter
        Formatter for the outer tick labels.
    """
    if decimals is not None:
        return mticker.FormatStrFormatter(f'%.{decimals}f')

    formatter = mticker.ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    return formatter


def hide_all_tick_labels(axes: np.ndarray) -> None:
    """Hide every tick label and offset text of a corner grid.

    Parameters
    ----------
    axes : numpy.ndarray
        ``(ndim, ndim)`` grid from `corner_axes`.

    Notes
    -----
    Interior tick numbers are what force the panels apart: hiding them first
    stops ``corner``'s own defaults from leaking through, and the outer row and
    column are then switched back on by `show_outer_tick_labels`.
    """
    for ax in axes.ravel():
        ax.tick_params(axis='x', labelbottom=False, labeltop=False)
        ax.tick_params(axis='y', labelleft=False, labelright=False)
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)


def show_outer_tick_labels(
    axes: np.ndarray,
    tick_fontsize: float = 25.,
    max_n_ticks: int = 3,
    tick_decimals: Optional[int] = None,
    xtick_rotation: float = 35.,
) -> None:
    """Put tick numbers back on the bottom row and left column only.

    Parameters
    ----------
    axes : numpy.ndarray
        ``(ndim, ndim)`` grid from `corner_axes`.
    tick_fontsize : float, optional
        Font size of the tick labels.  Default 25, the notebook's size.
    max_n_ticks : int, optional
        Candidate ticks per panel before edge pruning.  Default 3.
    tick_decimals : int or None, optional
        Fixed decimal places on the outer labels.  Default None (plain
        non-scientific formatting).
    xtick_rotation : float, optional
        Rotation of the bottom-row labels, in degrees.  Default 35.

    Notes
    -----
    ``prune='both'`` drops the ticks nearest each panel edge, which is what
    stops neighbouring panels' numbers from colliding once the gaps are
    narrowed.
    """
    ndim = axes.shape[0]

    for i in range(ndim):
        for j in range(i + 1):
            ax = axes[i, j]
            ax.tick_params(axis='both', which='major', labelsize=tick_fontsize,
                           length=6, width=1.3, pad=7, direction='out')

            if i == ndim - 1:
                ax.xaxis.set_major_locator(
                    mticker.MaxNLocator(nbins=max_n_ticks, prune='both'))
                ax.xaxis.set_major_formatter(_make_formatter(tick_decimals))
                ax.tick_params(axis='x', labelbottom=True)
                plt.setp(ax.get_xticklabels(), rotation=xtick_rotation,
                         ha='right', rotation_mode='anchor')

            if j == 0 and i > 0:
                ax.yaxis.set_major_locator(
                    mticker.MaxNLocator(nbins=max_n_ticks, prune='both'))
                ax.yaxis.set_major_formatter(_make_formatter(tick_decimals))
                ax.tick_params(axis='y', labelleft=True)
                plt.setp(ax.get_yticklabels(), rotation=0, ha='right',
                         va='center')


def place_parameter_labels(
    axes: np.ndarray,
    label_fontsize: float = 35.,
    xlabel_y: float = XLABEL_Y,
    ylabel_x: float = YLABEL_X,
) -> None:
    """Pin the outer parameter labels at fixed axes coordinates.

    Parameters
    ----------
    axes : numpy.ndarray
        ``(ndim, ndim)`` grid from `corner_axes`.
    label_fontsize : float, optional
        Font size of the parameter labels.  Default 35, the notebook's size.
    xlabel_y : float, optional
        Vertical position of the bottom labels, in axes coordinates.
    ylabel_x : float, optional
        Horizontal position of the left labels, in axes coordinates.

    Notes
    -----
    Fixed coordinates keep the labels clear of the tick numbers without the
    outer margin having to grow, which is what ``labelpad`` would force.
    """
    ndim = axes.shape[0]

    for j in range(ndim):
        ax = axes[-1, j]
        ax.xaxis.label.set_fontsize(label_fontsize)
        ax.xaxis.set_label_coords(0.5, xlabel_y)

    for i in range(1, ndim):
        ax = axes[i, 0]
        ax.yaxis.label.set_fontsize(label_fontsize)
        ax.yaxis.set_label_coords(ylabel_x, 0.5)


def place_diagonal_titles(
    axes: np.ndarray,
    title_fontsize: float = 30.,
    position: Tuple[float, float] = (0.7, 1.03),
) -> None:
    """Shrink and reposition the ``median +hi -lo`` titles on the diagonal.

    Parameters
    ----------
    axes : numpy.ndarray
        ``(ndim, ndim)`` grid from `corner_axes`.
    title_fontsize : float, optional
        Font size of the diagonal titles.  Default 30, the notebook's size.
    position : tuple of float, optional
        Title position in axes coordinates.  Default ``(0.7, 1.03)``.

    Notes
    -----
    The title strings are physically wider than their panels, so they set the
    row spacing unless they are placed explicitly.
    """
    for i in range(axes.shape[0]):
        ax = axes[i, i]
        ax.title.set_fontsize(title_fontsize)
        ax.title.set_position(position)


# ── Entry point ────────────────────────────────────────────────────────────

def tighten_corner(
    fig: plt.Figure,
    ndim: Optional[int] = None,
    label_fontsize: float = 35.,
    tick_fontsize: float = 25.,
    title_fontsize: float = 30.,
    max_n_ticks: int = 3,
    tick_decimals: Optional[int] = None,
    xtick_rotation: float = 35.,
    xlabel_y: float = XLABEL_Y,
    ylabel_x: float = YLABEL_X,
    margins: Optional[dict] = None,
    wspace: float = WSPACE,
    hspace: float = HSPACE,
) -> plt.Figure:
    """Remove the padding a ``tight_layout`` pass left on a corner figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Finished corner figure, after every case has been drawn into it.
    ndim : int, optional
        Number of parameters.  Default: inferred from the number of axes.
    label_fontsize, tick_fontsize, title_fontsize : float, optional
        Font sizes of the parameter labels, tick numbers and diagonal titles.
        Defaults 35, 25 and 30, matching ``paper_plots_c_v2.ipynb``.
    max_n_ticks : int, optional
        Candidate ticks per panel before edge pruning.  Default 3.
    tick_decimals : int or None, optional
        Fixed decimal places on the outer tick labels.  Default None.
    xtick_rotation : float, optional
        Rotation of the bottom-row tick labels.  Default 35 degrees.
    xlabel_y, ylabel_x : float, optional
        Positions of the outer parameter labels, in axes coordinates.
    margins : dict, optional
        ``left``/``right``/``bottom``/``top`` for ``subplots_adjust``.
        Default `MARGINS`.
    wspace, hspace : float, optional
        Gaps between panels.  Defaults `WSPACE` and `HSPACE` (0.10).

    Returns
    -------
    matplotlib.figure.Figure
        The same figure, adjusted in place.

    Notes
    -----
    Nothing here touches the samples, the contour levels, the histograms or
    the title text: only tick visibility, tick locators, label placement and
    the figure margins.  Call it after the last ``corner_plot`` call and
    before ``savefig``.
    """
    axes = corner_axes(fig, infer_ndim(fig) if ndim is None else ndim)

    hide_all_tick_labels(axes)
    show_outer_tick_labels(axes, tick_fontsize=tick_fontsize,
                           max_n_ticks=max_n_ticks,
                           tick_decimals=tick_decimals,
                           xtick_rotation=xtick_rotation)
    place_parameter_labels(axes, label_fontsize=label_fontsize,
                           xlabel_y=xlabel_y, ylabel_x=ylabel_x)
    place_diagonal_titles(axes, title_fontsize=title_fontsize)

    # Last word on the layout: this is what replaces the helper's
    # `fig.tight_layout()`, which is where the padding comes from.
    fig.subplots_adjust(**(MARGINS if margins is None else margins),
                        wspace=wspace, hspace=hspace)

    return fig


# ── Demo / smoke test ──────────────────────────────────────────────────────

def make_demo_figure(
    ndim: int = 4,
    nsamples: int = 2000,
    ncases: int = 2,
    seed: int = 0,
) -> plt.Figure:
    """Corner figure built the way the notebook builds Figure 6.

    Parameters
    ----------
    ndim : int, optional
        Number of parameters.  Default 4.
    nsamples : int, optional
        Samples per case.  Default 2000.
    ncases : int, optional
        Cases overlaid in the one figure.  Default 2.
    seed : int, optional
        Seed of the random generator.  Default 0.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with the padding problem in it: every case ends with a
        ``tight_layout`` call, as the notebook's helper does.

    Raises
    ------
    ImportError
        If ``corner`` is not installed.
    """
    import corner

    rng = np.random.default_rng(seed)
    fig, _ = plt.subplots(ndim, ndim, figsize=(20, 22))
    labels = [rf'$b_{{sys,{i}}}$' for i in range(1, ndim + 1)]

    for case in range(ncases):
        samples = (np.arange(1., ndim + 1.)[None, :]
                   + (0.1 + 0.02 * case) * rng.standard_normal((nsamples, ndim)))
        corner.corner(samples, labels=labels, fig=fig, bins=40, smooth=2.0,
                      show_titles=True, title_fmt='.3g',
                      color=f'C{case}',
                      title_kwargs={'fontsize': 30},
                      label_kwargs={'fontsize': 35})
        fig.tight_layout()          # the notebook helper's last line

    return fig


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the demo command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse.  Default: ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments: ``save``, ``ndim``, ``ncases``.
    """
    parser = argparse.ArgumentParser(
        description='Smoke test of the corner-plot padding fix.')
    parser.add_argument('--save', default=None,
                        help='write the tightened demo figure to this path')
    parser.add_argument('--ndim', type=int, default=4,
                        help='number of parameters (default 4)')
    parser.add_argument('--ncases', type=int, default=2,
                        help='cases overlaid in the figure (default 2)')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Draw a demo corner figure, tighten it and report the margins.

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
    fig = make_demo_figure(ndim=args.ndim, ncases=args.ncases)
    before = fig.subplotpars
    print(f'before: left={before.left:.3f} right={before.right:.3f} '
          f'bottom={before.bottom:.3f} top={before.top:.3f} '
          f'wspace={before.wspace:.3f} hspace={before.hspace:.3f}')

    tighten_corner(fig, ndim=args.ndim)
    after = fig.subplotpars
    print(f'after:  left={after.left:.3f} right={after.right:.3f} '
          f'bottom={after.bottom:.3f} top={after.top:.3f} '
          f'wspace={after.wspace:.3f} hspace={after.hspace:.3f}')

    if args.save:
        fig.savefig(args.save, bbox_inches='tight', dpi=100)
        print(f'wrote {args.save}')

    plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
