"""Tests for `corner_padding.py`.

Everything here runs on synthetic corner figures, so no run outputs are
needed.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import corner_padding as cp


NDIM = 3


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def fig():
    """Corner figure built the way the notebook builds Figure 6."""
    figure = cp.make_demo_figure(ndim=NDIM, nsamples=400, ncases=2)
    yield figure
    plt.close(figure)


# ── corner_axes / infer_ndim ───────────────────────────────────────────────

def test_corner_axes_shape(fig):
    axes = cp.corner_axes(fig, NDIM)
    assert axes.shape == (NDIM, NDIM)
    assert axes[0, 0] is fig.axes[0]


def test_corner_axes_ignores_extra_axes(fig):
    fig.add_axes([0.9, 0.9, 0.05, 0.05])      # a colorbar-like extra
    assert cp.corner_axes(fig, NDIM).shape == (NDIM, NDIM)


def test_corner_axes_rejects_too_large_ndim(fig):
    with pytest.raises(ValueError):
        cp.corner_axes(fig, NDIM + 2)


def test_corner_axes_rejects_bad_ndim(fig):
    with pytest.raises(ValueError):
        cp.corner_axes(fig, 0)


def test_infer_ndim(fig):
    assert cp.infer_ndim(fig) == NDIM


def test_infer_ndim_rejects_empty_figure():
    empty = plt.figure()
    try:
        with pytest.raises(ValueError):
            cp.infer_ndim(empty)
    finally:
        plt.close(empty)


# ── Tick labels ────────────────────────────────────────────────────────────

def test_hide_all_tick_labels(fig):
    axes = cp.corner_axes(fig, NDIM)
    cp.hide_all_tick_labels(axes)
    for ax in axes.ravel():
        assert not any(t.get_visible() for t in ax.get_xticklabels())
        assert not any(t.get_visible() for t in ax.get_yticklabels())


def test_only_outer_panels_keep_numbers(fig):
    cp.tighten_corner(fig, ndim=NDIM)
    axes = cp.corner_axes(fig, NDIM)
    fig.canvas.draw()

    def has_numbers(ax, axis):
        labels = getattr(ax, f'get_{axis}ticklabels')()
        return any(t.get_visible() and t.get_text() for t in labels)

    assert has_numbers(axes[-1, 0], 'x')          # bottom row
    assert has_numbers(axes[-1, 0], 'y')          # left column
    assert not has_numbers(axes[1, 1], 'x')       # interior diagonal
    assert not has_numbers(axes[-1, -1], 'y')     # diagonal, not left column


def test_bottom_labels_are_rotated(fig):
    cp.tighten_corner(fig, ndim=NDIM, xtick_rotation=35.)
    axes = cp.corner_axes(fig, NDIM)
    fig.canvas.draw()
    rotations = {t.get_rotation() for t in axes[-1, 0].get_xticklabels()}
    assert rotations == {35.}


def test_tick_decimals_are_applied(fig):
    cp.tighten_corner(fig, ndim=NDIM, tick_decimals=2)
    axes = cp.corner_axes(fig, NDIM)
    fig.canvas.draw()
    texts = [t.get_text() for t in axes[-1, 0].get_xticklabels()
             if t.get_text()]
    assert texts and all(len(t.split('.')[-1]) == 2 for t in texts)


def test_tick_count_is_capped(fig):
    cp.tighten_corner(fig, ndim=NDIM, max_n_ticks=2)
    axes = cp.corner_axes(fig, NDIM)
    fig.canvas.draw()
    shown = [t for t in axes[-1, 0].get_xticklabels() if t.get_text()]
    assert len(shown) <= 3          # nbins=2 plus pruning


# ── Labels and titles ──────────────────────────────────────────────────────

def test_parameter_labels_are_pinned(fig):
    cp.tighten_corner(fig, ndim=NDIM, xlabel_y=-0.4, ylabel_x=-0.45)
    axes = cp.corner_axes(fig, NDIM)
    assert axes[-1, 0].xaxis.get_label().get_position()[1] == pytest.approx(-0.4)
    assert axes[1, 0].yaxis.get_label().get_position()[0] == pytest.approx(-0.45)


def test_parameter_label_font_sizes(fig):
    cp.tighten_corner(fig, ndim=NDIM, label_fontsize=21.)
    axes = cp.corner_axes(fig, NDIM)
    assert axes[-1, 1].xaxis.label.get_fontsize() == 21.
    assert axes[1, 0].yaxis.label.get_fontsize() == 21.


def test_diagonal_titles_are_resized_and_kept(fig):
    before = [cp.corner_axes(fig, NDIM)[i, i].get_title() for i in range(NDIM)]
    cp.tighten_corner(fig, ndim=NDIM, title_fontsize=17.)
    axes = cp.corner_axes(fig, NDIM)
    after = [axes[i, i].get_title() for i in range(NDIM)]
    assert after == before                       # text untouched
    assert all(axes[i, i].title.get_fontsize() == 17. for i in range(NDIM))


# ── tighten_corner ─────────────────────────────────────────────────────────

def test_tighten_corner_sets_the_margins(fig):
    cp.tighten_corner(fig, ndim=NDIM)
    pars = fig.subplotpars
    assert pars.left == pytest.approx(cp.MARGINS['left'])
    assert pars.right == pytest.approx(cp.MARGINS['right'])
    assert pars.bottom == pytest.approx(cp.MARGINS['bottom'])
    assert pars.top == pytest.approx(cp.MARGINS['top'])
    assert pars.wspace == pytest.approx(cp.WSPACE)
    assert pars.hspace == pytest.approx(cp.HSPACE)


def test_tighten_corner_shrinks_the_gaps(fig):
    before = fig.subplotpars.hspace
    cp.tighten_corner(fig, ndim=NDIM)
    assert fig.subplotpars.hspace < before


def test_tighten_corner_accepts_custom_margins(fig):
    cp.tighten_corner(fig, ndim=NDIM,
                      margins=dict(left=0.2, right=0.9, bottom=0.2, top=0.9),
                      wspace=0.02, hspace=0.03)
    pars = fig.subplotpars
    assert (pars.left, pars.right) == pytest.approx((0.2, 0.9))
    assert pars.wspace == pytest.approx(0.02)


def test_tighten_corner_infers_ndim(fig):
    out = cp.tighten_corner(fig)
    assert out is fig
    assert fig.subplotpars.wspace == pytest.approx(cp.WSPACE)


def test_tighten_corner_leaves_the_data_alone(fig):
    axes = cp.corner_axes(fig, NDIM)
    before = [len(ax.patches) + len(ax.lines) + len(ax.collections)
              for ax in axes.ravel()]
    xlims = [ax.get_xlim() for ax in axes.ravel()]

    cp.tighten_corner(fig, ndim=NDIM)

    after = [len(ax.patches) + len(ax.lines) + len(ax.collections)
             for ax in axes.ravel()]
    assert after == before
    assert [ax.get_xlim() for ax in axes.ravel()] == xlims


def test_tighten_corner_saves_smaller_output(fig, tmp_path):
    loose = tmp_path / 'loose.png'
    tight = tmp_path / 'tight.png'
    fig.savefig(loose, bbox_inches='tight', dpi=40)
    cp.tighten_corner(fig, ndim=NDIM)
    fig.savefig(tight, bbox_inches='tight', dpi=40)
    assert loose.exists() and tight.exists()


# ── Demo entry point ───────────────────────────────────────────────────────

def test_make_demo_figure_grid_size():
    figure = cp.make_demo_figure(ndim=2, nsamples=200, ncases=1)
    try:
        assert len(figure.axes) >= 4
    finally:
        plt.close(figure)


def test_main_runs_and_writes(tmp_path):
    out = tmp_path / 'corner.png'
    assert cp.main(['--ndim', '2', '--ncases', '1', '--save', str(out)]) == 0
    assert out.exists()


def test_main_without_save():
    assert cp.main(['--ndim', '2', '--ncases', '1']) == 0
