"""Tests for `plot_delta_bsys.py`.

Everything here runs on small synthetic chains, so no run outputs are needed.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_delta_bsys as pdb


# ── Fixtures ───────────────────────────────────────────────────────────────

TRUTHS = np.array([12., 20.4, 5.])


@pytest.fixture
def chain():
    """Synthetic chain of three amplitudes, scattered about `TRUTHS`."""
    return pdb.make_demo_chain(TRUTHS, nsamples=4000, seed=42)


COMPLEX_TRUTHS = np.array([12. + 5j, 4. + 20j])


@pytest.fixture
def complex_chain():
    """Synthetic complex chain, scattered in both parts."""
    return pdb.make_demo_chain(COMPLEX_TRUTHS, nsamples=4000, seed=7)


@pytest.fixture
def summaries(chain):
    """Summaries of the synthetic chain."""
    return pdb.summarise_delta(pdb.delta_samples(chain, TRUTHS),
                               truths=TRUTHS)


# ── sigma_quantiles ────────────────────────────────────────────────────────

def test_sigma_quantiles_match_corner_plot():
    (lo1, hi1), (lo2, hi2), (lo3, hi3) = pdb.sigma_quantiles(3)
    assert lo1 == pytest.approx(0.15865, abs=1e-5)
    assert hi1 == pytest.approx(0.84135, abs=1e-5)
    assert lo2 == pytest.approx(0.02275, abs=1e-5)
    assert hi2 == pytest.approx(0.97725, abs=1e-5)
    assert lo3 == pytest.approx(0.00135, abs=1e-5)
    assert hi3 == pytest.approx(0.99865, abs=1e-5)


def test_sigma_quantiles_length():
    assert len(pdb.sigma_quantiles(1)) == 1
    assert len(pdb.sigma_quantiles(2)) == 2


def test_sigma_quantiles_rejects_other_levels():
    with pytest.raises(ValueError):
        pdb.sigma_quantiles(4)


# ── delta_samples ──────────────────────────────────────────────────────────

def test_delta_samples_subtracts_truth(chain):
    deltas = pdb.delta_samples(chain, TRUTHS)
    assert deltas.shape == chain.shape
    assert np.allclose(deltas, chain - TRUTHS[None, :])


def test_delta_samples_burns_and_thins(chain):
    deltas = pdb.delta_samples(chain, TRUTHS, burn=100, thin=4)
    assert deltas.shape == ((chain.shape[0] - 100 + 3) // 4, chain.shape[1])
    assert np.allclose(deltas[0], chain[100] - TRUTHS)


def test_delta_samples_rejects_complex():
    with pytest.raises(TypeError):
        pdb.delta_samples(np.zeros((10, 2), dtype=complex), [0., 0.])


def test_delta_samples_rejects_wrong_truths(chain):
    with pytest.raises(ValueError):
        pdb.delta_samples(chain, TRUTHS[:2])


def test_delta_samples_rejects_1d():
    with pytest.raises(ValueError):
        pdb.delta_samples(np.zeros(10), [0.])


def test_delta_samples_rejects_bad_thin(chain):
    with pytest.raises(ValueError):
        pdb.delta_samples(chain, TRUTHS, thin=0)


def test_delta_samples_burn_eats_chain(chain):
    with pytest.raises(ValueError):
        pdb.delta_samples(chain, TRUTHS, burn=chain.shape[0])


# ── summarise_delta ────────────────────────────────────────────────────────

def test_summarise_delta_statistics():
    rng = np.random.default_rng(1)
    samples = 3. + 0.5 * rng.standard_normal((200000, 1))
    summary, = pdb.summarise_delta(pdb.delta_samples(samples, [3.]))

    assert summary.mean == pytest.approx(0., abs=0.01)
    assert summary.median == pytest.approx(0., abs=0.01)
    assert summary.std == pytest.approx(0.5, rel=0.02)
    # 1 sigma interval of a Gaussian is +/- one standard deviation.
    assert summary.plus[0] == pytest.approx(0.5, rel=0.03)
    assert summary.minus[0] == pytest.approx(0.5, rel=0.03)
    assert summary.plus[2] == pytest.approx(1.5, rel=0.05)


def test_summarise_delta_orders_intervals(summaries):
    for s in summaries:
        assert s.lower[2] <= s.lower[1] <= s.lower[0] <= s.median
        assert s.median <= s.upper[0] <= s.upper[1] <= s.upper[2]


def test_summarise_delta_keeps_truths_and_labels(summaries):
    assert [s.truth for s in summaries] == pytest.approx(TRUTHS)
    assert [s.label for s in summaries] == pdb.default_labels(3)
    assert all(s.nsamples == 4000 for s in summaries)


def test_summarise_delta_respects_nsigma(chain):
    summaries = pdb.summarise_delta(pdb.delta_samples(chain, TRUTHS), nsigma=1)
    assert all(len(s.lower) == 1 for s in summaries)


def test_summarise_delta_pull_is_mean_over_sigma(summaries):
    for s in summaries:
        assert s.pull == pytest.approx(s.mean / s.std)


def test_summarise_delta_pull_zero_for_delta_function():
    summary, = pdb.summarise_delta(np.zeros((10, 1)))
    assert summary.pull == 0.


def test_summarise_delta_rejects_bad_labels(chain):
    with pytest.raises(ValueError):
        pdb.summarise_delta(pdb.delta_samples(chain, TRUTHS), labels=['a'])


def test_summarise_delta_rejects_bad_truths(chain):
    with pytest.raises(ValueError):
        pdb.summarise_delta(pdb.delta_samples(chain, TRUTHS), truths=[0.])


def test_summarise_delta_rejects_1d():
    with pytest.raises(ValueError):
        pdb.summarise_delta(np.zeros(10))


# ── Labels and number formatting ───────────────────────────────────────────

def test_default_labels():
    assert pdb.default_labels(2) == [r'$b_{sys,1}$', r'$b_{sys,2}$']


@pytest.mark.parametrize('value, expected', [
    (0.0124, 0), (12.5, 0), (0.0005, -4), (1.2e-6, -6), (4.5e5, 5), (0., 0),
])
def test_common_exponent(value, expected):
    assert pdb.common_exponent(value) == expected


@pytest.mark.parametrize('value, sig, expected', [
    (0.0124, 3, 4), (1.24, 3, 2), (123., 3, 0), (0., 3, 2),
])
def test_decimals_for(value, sig, expected):
    assert pdb.decimals_for(value, sig) == expected


def test_format_triplet_uses_deviation_precision():
    assert pdb.format_triplet(0.001, 0.0124, 0.0121) == \
        '0.0010^{+0.0124}_{-0.0121}'


def test_format_triplet_keeps_a_tiny_centre_visible():
    text = pdb.format_triplet(-2.82e-5, 0.0124, 0.0121)
    assert text.startswith('-0.00003')      # not a bare '-0.0000'


def test_format_triplet_factors_out_a_common_power():
    text = pdb.format_triplet(1e-6, 4e-6, 3e-6)
    assert r'\times 10^{-6}' in text
    assert 'e-06' not in text               # no double superscript


def test_format_mean_sigma_shares_one_power():
    text = pdb.format_mean_sigma(1e-6, 4e-6)
    assert text.count(r'\times 10^{-6}') == 2


def test_format_triplet_accepts_an_explicit_exponent():
    text = pdb.format_triplet(1e-5, 2e-4, 2e-4, exponent=-3)
    assert r'\times 10^{-3}' in text


def test_shared_exponent_covers_similar_rows(summaries):
    exponent = pdb.shared_exponent(summaries)
    assert exponent is not None
    texts = [pdb.format_summary(s, exponent=exponent) for s in summaries]
    assert len({r'\times' in t for t in texts}) == 1     # one convention


def test_shared_exponent_gives_up_on_wide_spreads():
    wide = pdb.summarise_delta(np.hstack([
        1e-6 * np.random.default_rng(0).standard_normal((2000, 1)),
        1e+2 * np.random.default_rng(1).standard_normal((2000, 1)),
    ]))
    assert pdb.shared_exponent(wide) is None


def test_shared_exponent_of_a_degenerate_chain():
    assert pdb.shared_exponent(pdb.summarise_delta(np.zeros((10, 1)))) == 0


def test_format_summary_has_two_lines(summaries):
    text = pdb.format_summary(summaries[0])
    first, second = text.split('\n')
    assert r'\Delta' in first and r'3\sigma' in first
    assert r'\mu' in second and r'\sigma' in second


def test_format_summary_honours_nsigma(summaries):
    assert r'1\sigma' in pdb.format_summary(summaries[0], nsigma=1)


def test_format_summary_rejects_unavailable_nsigma(chain):
    summary, = pdb.summarise_delta(
        pdb.delta_samples(chain, TRUTHS)[:, :1], nsigma=1)
    with pytest.raises(ValueError):
        pdb.format_summary(summary, nsigma=3)


def test_summary_text_has_one_line_per_parameter(summaries):
    lines = pdb.summary_text(summaries).splitlines()
    assert len(lines) == len(summaries) + 2      # header + rule + rows
    assert 'b_sys,3' in lines[-1]


# ── Scaling and axis limits ────────────────────────────────────────────────

def test_scale_factors_absolute(summaries):
    assert np.allclose(pdb._scale_factors(summaries, 'absolute'), 1.)


def test_scale_factors_sigma(summaries):
    scales = pdb._scale_factors(summaries, 'sigma')
    assert np.allclose(scales, [s.std for s in summaries])


def test_scale_factors_sigma_survives_zero_width():
    summary, = pdb.summarise_delta(np.zeros((10, 1)))
    assert pdb._scale_factors([summary], 'sigma')[0] == 1.


def test_scale_factors_rejects_unknown_units(summaries):
    with pytest.raises(ValueError):
        pdb._scale_factors(summaries, 'percent')


def test_axis_limit_covers_every_interval(summaries):
    scales = pdb._scale_factors(summaries, 'absolute')
    limit = pdb._axis_limit(summaries, scales, pad=0.)
    widest = max(max(abs(v) for v in s.lower + s.upper) for s in summaries)
    assert limit >= widest


def test_axis_limit_never_zero():
    summary, = pdb.summarise_delta(np.zeros((10, 1)))
    assert pdb._axis_limit([summary], np.ones(1)) > 0.


# ── plot_delta_bsys ────────────────────────────────────────────────────────

def test_plot_returns_figure_and_summaries(chain):
    fig, summaries = pdb.plot_delta_bsys(chain, TRUTHS)
    try:
        assert len(summaries) == 3
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)


def test_plot_axis_is_centred_on_zero(chain):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS)
    try:
        lo, hi = fig.axes[0].get_xlim()
        assert lo == pytest.approx(-hi)
        assert lo < 0. < hi
    finally:
        plt.close(fig)


def test_plot_draws_the_zero_line(chain):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS)
    try:
        # errorbar leaves some lines with object-dtype data; cast first.
        verticals = [ln for ln in fig.axes[0].lines
                     if np.allclose(np.asarray(ln.get_xdata(), dtype=float), 0.)
                     and ln.get_linestyle() == '--']
        assert verticals, 'no line through zero'
    finally:
        plt.close(fig)


def test_plot_one_row_per_parameter(chain):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS)
    try:
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == pdb.default_labels(3)
    finally:
        plt.close(fig)


def test_plot_rows_run_top_to_bottom(chain):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS)
    try:
        bottom, top = fig.axes[0].get_ylim()
        assert bottom > top          # inverted: parameter 1 at the top
    finally:
        plt.close(fig)


def test_plot_is_free_of_text_by_default(chain):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS)
    try:
        assert not fig.axes[0].texts
    finally:
        plt.close(fig)


def test_plot_annotates_every_row_on_request(chain):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS, annotate=True)
    try:
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert sum(r'\Delta' in t for t in texts) == 3
    finally:
        plt.close(fig)


def test_plot_annotates_every_component(complex_chain):
    fig, _ = pdb.plot_delta_bsys(complex_chain, COMPLEX_TRUTHS, annotate=True)
    try:
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert sum(r'\Delta' in t for t in texts) == 2 * len(COMPLEX_TRUTHS)
    finally:
        plt.close(fig)


def test_plot_sigma_units_label_and_limits(chain):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS, units='sigma')
    try:
        assert r'\sigma' in fig.axes[0].get_xlabel()
        _, hi = fig.axes[0].get_xlim()
        assert 3. < hi < 6.          # 3 sigma intervals plus padding
    finally:
        plt.close(fig)


def test_plot_accepts_custom_labels_and_nsigma(chain):
    fig, summaries = pdb.plot_delta_bsys(
        chain, TRUTHS, labels=['a', 'b', 'c'], nsigma=1)
    try:
        assert [s.label for s in summaries] == ['a', 'b', 'c']
        assert all(len(s.lower) == 1 for s in summaries)
    finally:
        plt.close(fig)


def test_plot_draws_into_a_given_axes(chain):
    fig, ax = plt.subplots()
    try:
        out, _ = pdb.plot_delta_bsys(chain, TRUTHS, ax=ax)
        assert out is fig
    finally:
        plt.close(fig)


def test_plot_handles_a_single_parameter(chain):
    fig, summaries = pdb.plot_delta_bsys(chain[:, :1], TRUTHS[:1])
    try:
        assert len(summaries) == 1
    finally:
        plt.close(fig)


def test_plot_survives_a_short_palette(chain):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS, colors=['#000000'])
    try:
        assert fig.axes[0].get_xlim()[1] > 0.
    finally:
        plt.close(fig)


def test_plot_saves_a_file(chain, tmp_path):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS)
    out = tmp_path / 'delta_bsys.png'
    try:
        fig.savefig(out, bbox_inches='tight', dpi=50)
    finally:
        plt.close(fig)
    assert out.exists() and out.stat().st_size > 0


# ── Components ─────────────────────────────────────────────────────────────

def test_component_chains_splits_a_complex_chain(complex_chain):
    chains = pdb.component_chains(complex_chain, COMPLEX_TRUTHS)
    assert [name for name, _, _ in chains] == ['real', 'imag']

    (_, re_samples, re_truths), (_, im_samples, im_truths) = chains
    assert np.allclose(re_samples, complex_chain.real)
    assert np.allclose(im_samples, complex_chain.imag)
    assert np.allclose(re_truths, COMPLEX_TRUTHS.real)
    assert np.allclose(im_truths, COMPLEX_TRUTHS.imag)
    assert not np.iscomplexobj(re_samples)


def test_component_chains_leaves_a_real_chain_alone(chain):
    (name, samples, truths), = pdb.component_chains(chain, TRUTHS)
    assert name == 'value'
    assert np.allclose(samples, chain)
    assert np.allclose(truths, TRUTHS)


def test_component_chains_honours_an_explicit_choice(complex_chain):
    chains = pdb.component_chains(complex_chain, COMPLEX_TRUTHS,
                                  components=['abs'])
    (name, samples, truths), = chains
    assert name == 'abs'
    assert np.allclose(samples, np.abs(complex_chain))
    assert np.allclose(truths, np.abs(COMPLEX_TRUTHS))


def test_component_chains_rejects_unknown_components(complex_chain):
    with pytest.raises(ValueError):
        pdb.component_chains(complex_chain, COMPLEX_TRUTHS,
                             components=['modulus'])


def test_component_chains_rejects_empty_components(complex_chain):
    with pytest.raises(ValueError):
        pdb.component_chains(complex_chain, COMPLEX_TRUTHS, components=[])


def test_component_chains_rejects_wrong_truths(complex_chain):
    with pytest.raises(ValueError):
        pdb.component_chains(complex_chain, COMPLEX_TRUTHS[:1])


def test_component_offsets_are_symmetric():
    offsets = pdb.component_offsets(2, 1.0)
    assert offsets[0] == pytest.approx(-offsets[1])
    assert offsets[0] < 0. < offsets[1]


def test_component_offsets_of_one_component():
    assert pdb.component_offsets(1, 1.0) == pytest.approx([0.])


def test_complex_chain_gives_two_point_sets_per_row(complex_chain):
    fig, summaries = pdb.plot_delta_bsys(complex_chain, COMPLEX_TRUTHS)
    try:
        assert len(summaries) == 2 * len(COMPLEX_TRUTHS)
        # parameter-major: real then imag within each parameter
        assert [s.component for s in summaries] == ['real', 'imag'] * 2
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert labels == pdb.default_labels(len(COMPLEX_TRUTHS))
    finally:
        plt.close(fig)


def test_complex_components_are_drawn_off_the_row_centre(complex_chain):
    fig, _ = pdb.plot_delta_bsys(complex_chain, COMPLEX_TRUTHS,
                                 row_height=1.0)
    try:
        bars = [ln for ln in fig.axes[0].lines
                if len(ln.get_xdata()) == 2 and ln.get_linewidth() > 5.]
        ys = sorted({round(float(ln.get_ydata()[0]), 6) for ln in bars})
        assert len(ys) == 2 * len(COMPLEX_TRUTHS)       # no two share a line
        assert 0. not in ys                             # offset from centre
    finally:
        plt.close(fig)


def test_complex_components_use_different_colours(complex_chain):
    fig, _ = pdb.plot_delta_bsys(complex_chain, COMPLEX_TRUTHS)
    try:
        bars = [ln for ln in fig.axes[0].lines
                if len(ln.get_xdata()) == 2 and ln.get_linewidth() > 5.]
        colours = {ln.get_color() for ln in bars}
        assert colours == {pdb.PAPER_COLORS[0], pdb.PAPER_COLORS[1]}
    finally:
        plt.close(fig)


def test_legend_names_both_components(complex_chain):
    fig, _ = pdb.plot_delta_bsys(complex_chain, COMPLEX_TRUTHS)
    try:
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert pdb.COMPONENT_LABELS['real'] in labels
        assert pdb.COMPONENT_LABELS['imag'] in labels
    finally:
        plt.close(fig)


def test_legend_omits_component_entries_for_a_real_chain(chain):
    fig, _ = pdb.plot_delta_bsys(chain, TRUTHS)
    try:
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels[0] == 'Truth'
        assert all(not t.startswith(r'$\mathrm{') for t in labels)
    finally:
        plt.close(fig)


def test_summary_text_names_the_components(complex_chain):
    _, summaries = pdb.plot_delta_bsys(complex_chain, COMPLEX_TRUTHS)
    plt.close('all')
    lines = pdb.summary_text(summaries).splitlines()
    assert len(lines) == len(summaries) + 2
    assert 'real' in lines[2] and 'imag' in lines[3]
    assert 'b_sys,2' in lines[-1]


def test_summary_text_of_nothing():
    assert pdb.summary_text([]) == ''


# ── Demo entry point ───────────────────────────────────────────────────────

def test_make_demo_chain_complex_truths_give_a_complex_chain():
    chain = pdb.make_demo_chain([1. + 1j, 2. - 3j], nsamples=300)
    assert np.iscomplexobj(chain)
    assert chain.shape == (300, 2)


def test_make_demo_chain_shape_and_scatter():
    chain = pdb.make_demo_chain([1., 2.], nsamples=500, sigmas=[0.1, 0.2])
    assert chain.shape == (500, 2)
    assert chain[:, 0].std() == pytest.approx(0.1, rel=0.2)


def test_main_runs_and_writes(tmp_path):
    out = tmp_path / 'demo.pdf'
    assert pdb.main(['--save', str(out), '--ndim', '2', '--nsigma', '2']) == 0
    assert out.exists()


def test_main_without_save():
    assert pdb.main(['--ndim', '2', '--units', 'sigma']) == 0
