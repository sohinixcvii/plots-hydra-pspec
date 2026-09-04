"""Tests for `plot_corner_blink.py`.

Everything here runs on small synthetic chains, so no run outputs are needed.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_corner_blink as pcb


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def cases():
    """Three synthetic cases with four systematic modes each."""
    return [
        dict(run='run_a', label='Case A',
             sys_amps_true=[1. + 4j, 2. + 3j, 3. + 2j, 4. + 1j]),
        dict(run='run_b', label='Case B'),
        dict(run='run_c', label='Case C'),
    ]


@pytest.fixture
def demo_dir(tmp_path, cases):
    """Directory of synthetic ``b-sys.npy`` chains."""
    return pcb.make_demo_data(str(tmp_path / 'runs'), cases, nsamples=400)


# ── format_count ───────────────────────────────────────────────────────────

@pytest.mark.parametrize('n, expected', [
    (100_000, '100k'), (250_000, '250k'), (2_000_000, '2M'), (1234, '1234'),
])
def test_format_count(n, expected):
    assert pcb.format_count(n) == expected


# ── load_bsys ──────────────────────────────────────────────────────────────

def test_load_bsys_shape_and_abs(demo_dir):
    chain = pcb.load_bsys(os.path.join(demo_dir, 'run_a'), burn=0)
    assert chain.shape == (400, 4)
    assert np.all(chain >= 0.)          # amplitudes, not complex samples


def test_load_bsys_truncates_and_burns(demo_dir):
    chain = pcb.load_bsys(os.path.join(demo_dir, 'run_a'), nsamples=100,
                          burn=10, burn_mode='count')
    assert chain.shape[0] == 90


def test_load_bsys_percent_burn(demo_dir):
    chain = pcb.load_bsys(os.path.join(demo_dir, 'run_a'), nsamples=200,
                          burn=10, burn_mode='percent')
    assert chain.shape[0] == 180


def test_load_bsys_thins(demo_dir):
    chain = pcb.load_bsys(os.path.join(demo_dir, 'run_a'), nsamples=100,
                          burn=0, thin=5)
    assert chain.shape[0] == 20


def test_load_bsys_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        pcb.load_bsys(str(tmp_path))


def test_load_bsys_bad_burn_mode(demo_dir):
    with pytest.raises(ValueError):
        pcb.load_bsys(os.path.join(demo_dir, 'run_a'), burn_mode='half')


def test_load_bsys_burn_eats_chain(demo_dir):
    with pytest.raises(ValueError):
        pcb.load_bsys(os.path.join(demo_dir, 'run_a'), nsamples=10, burn=50)


# ── common_ranges ──────────────────────────────────────────────────────────

def test_common_ranges_spans_all_sets():
    a = np.array([[0.], [1.]])
    b = np.array([[-1.], [3.]])
    (lo, hi), = pcb.common_ranges([a, b], pad=0.)
    assert (lo, hi) == (-1., 3.)


def test_common_ranges_pads():
    a = np.array([[0.], [1.]])
    (lo, hi), = pcb.common_ranges([a], pad=0.1)
    assert lo == pytest.approx(-0.1)
    assert hi == pytest.approx(1.1)


def test_common_ranges_constant_parameter():
    a = np.full((5, 1), 2.)
    (lo, hi), = pcb.common_ranges([a], pad=0.05)
    assert hi > lo                       # no zero-width axis


def test_common_ranges_rejects_mismatched_ndim():
    with pytest.raises(ValueError):
        pcb.common_ranges([np.zeros((3, 2)), np.zeros((3, 4))])


def test_common_ranges_rejects_empty():
    with pytest.raises(ValueError):
        pcb.common_ranges([])


# ── corner_plot ────────────────────────────────────────────────────────────

def test_corner_plot_uses_given_ranges():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((500, 2))
    ranges = [(-6., 6.), (-7., 7.)]
    fig = pcb.corner_plot(samples, ranges=ranges, bins=20, smooth=None)
    axes = np.array(fig.axes).reshape(2, 2)
    assert axes[0, 0].get_xlim() == pytest.approx(ranges[0])
    assert axes[1, 0].get_ylim() == pytest.approx(ranges[1])
    matplotlib.pyplot.close(fig)


def test_corner_plot_rejects_bad_nsigma():
    with pytest.raises(ValueError):
        pcb.corner_plot(np.zeros((10, 2)), nsigma=4)


# ── harmonise_diagonals ────────────────────────────────────────────────────

def test_harmonise_diagonals_levels_ylims():
    import matplotlib.pyplot as plt

    figs = []
    for scale in (1., 5.):
        fig, axes = plt.subplots(2, 2)
        for i in range(2):
            axes[i, i].set_ylim(0., scale * (i + 1))
        figs.append(fig)

    pcb.harmonise_diagonals(figs, 2)
    for i in range(2):
        ylims = [np.array(f.axes).reshape(2, 2)[i, i].get_ylim() for f in figs]
        assert ylims[0] == ylims[1]
        assert ylims[0][1] == pytest.approx(5. * (i + 1))
    for fig in figs:
        plt.close(fig)


# ── read_variant ───────────────────────────────────────────────────────────

def test_read_variant_reads_every_case(demo_dir, cases):
    vdata = pcb.read_variant(dict(nsamples=200), cases, demo_dir, burn=0)
    assert len(vdata['samples']) == 3
    assert vdata['tag'] == '200'
    assert vdata['case_labels'] == ['Case A', 'Case B', 'Case C']
    assert len(vdata['truths']) == 4
    assert all(s.shape == (200, 4) for s in vdata['samples'])


def test_read_variant_skips_missing_run(demo_dir, cases):
    vdata = pcb.read_variant(dict(nsamples=100),
                             cases + [dict(run='not_a_run')], demo_dir, burn=0)
    assert len(vdata['samples']) == 3


def test_read_variant_skips_mismatched_ndim(tmp_path, demo_dir, cases):
    odd = os.path.join(demo_dir, 'run_odd')
    os.makedirs(odd, exist_ok=True)
    np.save(os.path.join(odd, 'b-sys.npy'), np.ones((100, 2), dtype=complex))

    vdata = pcb.read_variant(dict(nsamples=100),
                             cases + [dict(run='run_odd')], demo_dir, burn=0)
    assert len(vdata['samples']) == 3


def test_read_variant_honours_variant_result_dir(demo_dir, cases):
    vdata = pcb.read_variant(dict(nsamples=50, result_dir=demo_dir),
                             cases, '/nowhere/at/all', burn=0)
    assert len(vdata['samples']) == 3


# ── build_figures ──────────────────────────────────────────────────────────

def test_build_figures_shares_all_axes(demo_dir, cases):
    import matplotlib.pyplot as plt

    variants_data = [
        pcb.read_variant(dict(nsamples=n), cases, demo_dir, burn=0)
        for n in (100, 400)
    ]
    figs = pcb.build_figures(variants_data, show_titles=False)
    assert len(figs) == 2

    ndim = variants_data[0]['samples'][0].shape[1]
    ax0 = np.array(figs[0].axes).reshape(ndim, ndim)
    ax1 = np.array(figs[1].axes).reshape(ndim, ndim)
    for i in range(ndim):
        for j in range(i + 1):
            assert ax0[i, j].get_xlim() == pytest.approx(ax1[i, j].get_xlim())
            assert ax0[i, j].get_ylim() == pytest.approx(ax1[i, j].get_ylim())
    for fig in figs:
        plt.close(fig)


def test_build_figures_without_samples_raises():
    with pytest.raises(ValueError):
        pcb.build_figures([dict(label='x', nsamples=1, tag='1', samples=[],
                                case_labels=[], truths=None)])


# ── save_figures ───────────────────────────────────────────────────────────

def test_save_figures_writes_identically_sized_files(tmp_path, demo_dir, cases):
    import matplotlib.pyplot as plt
    from PIL import Image

    variants_data = [
        pcb.read_variant(dict(nsamples=n), cases, demo_dir, burn=0)
        for n in (100, 400)
    ]
    figs = pcb.build_figures(variants_data, show_titles=True)
    written = pcb.save_figures(figs, variants_data, str(tmp_path / 'figs'),
                               formats=('png',), dpi=30)

    pngs = [p for p in written if p.endswith('.png')]
    assert len(pngs) == 2
    sizes = {Image.open(p).size for p in pngs}
    assert len(sizes) == 1, f'frames differ in size: {sizes}'
    assert any(p.endswith('.gif') for p in written)
    for fig in figs:
        plt.close(fig)


def test_save_blink_gif_needs_two_frames(tmp_path):
    assert pcb.save_blink_gif(['one.png'], str(tmp_path / 'x.gif')) is None


# ── main ───────────────────────────────────────────────────────────────────

def test_main_demo_runs(tmp_path):
    assert pcb.main(['--demo', '--fig-dir', str(tmp_path / 'figs')]) == 0
    written = sorted(os.listdir(tmp_path / 'figs'))
    assert 'bsys_corner_blink.gif' in written


def test_main_reports_no_data(tmp_path, capsys):
    assert pcb.main(['--result-dir', str(tmp_path), '--runs', 'nope',
                     '--no-save']) == 1
