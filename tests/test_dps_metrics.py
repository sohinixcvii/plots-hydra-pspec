"""Tests for `dps_metrics.py`.

Everything here runs on small synthetic chains, so no run outputs are needed.
"""

import json
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dps_metrics as dm


# ── Fixtures ───────────────────────────────────────────────────────────────

NDELAYS = 60


@pytest.fixture
def reference():
    """Synthetic control run."""
    return dm.make_demo_run('Reference', width_scale=1.0, bias_sigma=0.5, seed=1)


@pytest.fixture
def wider():
    """Twice the posterior width, same offset in units of sigma."""
    return dm.make_demo_run('Wider', width_scale=2.0, bias_sigma=0.5, seed=2)


@pytest.fixture
def biased():
    """Same posterior width, a much larger offset."""
    return dm.make_demo_run('Biased', width_scale=1.0, bias_sigma=3.0, seed=3)


@pytest.fixture
def run_dir(tmp_path, reference):
    """A directory holding the three `.npy` files `load_run` expects."""
    d = tmp_path / 'low_dl_fr_20'
    d.mkdir()
    np.save(d / dm.CHAIN_FILE, reference.ps_chain)
    np.save(d / dm.LNPOST_FILE, reference.ln_post)
    # eor_true is (Ntimes, Nfreqs) visibilities, not a spectrum.
    rng = np.random.default_rng(11)
    eor = (rng.normal(size=(dm.NTIMES_DEFAULT, NDELAYS))
           + 1j * rng.normal(size=(dm.NTIMES_DEFAULT, NDELAYS)))
    np.save(d / dm.EOR_TRUE_FILE, eor)
    return d


# ── delay_power_spectrum ───────────────────────────────────────────────────

def test_delay_power_spectrum_shape():
    s = np.random.default_rng(0).normal(size=(80, NDELAYS)).astype(complex)
    ps = dm.delay_power_spectrum(s)
    assert ps.shape == (NDELAYS,)


def test_delay_power_spectrum_is_real_and_positive():
    s = np.random.default_rng(0).normal(size=(80, NDELAYS)).astype(complex)
    ps = dm.delay_power_spectrum(s)
    assert np.isrealobj(ps)
    assert np.all(ps > 0)


def test_delay_power_spectrum_matches_notebook_calc_ps():
    """Reproduces the notebook helper term for term."""
    s = np.random.default_rng(5).normal(size=(80, NDELAYS)).astype(complex)
    sk = np.fft.ifftshift(s, axes=(1,))
    sk = np.fft.fftn(sk, axes=(1,))
    sk = np.fft.fftshift(sk, axes=(1,))
    expected = np.mean(sk * sk.conj(), axis=0).real / NDELAYS
    assert dm.delay_power_spectrum(s) == pytest.approx(expected)


# ── foreground_mask ────────────────────────────────────────────────────────

def test_foreground_mask_drops_notebook_bins():
    keep = dm.foreground_mask(NDELAYS)
    assert not keep[27:34].any()
    assert keep.sum() == NDELAYS - 7


def test_foreground_mask_default_matches_notebook_rm():
    """The mask is the notebook's `rm = np.arange(27, 34)`, inverted."""
    rm = np.arange(27, 34)
    expected = np.ones(NDELAYS, dtype=bool)
    expected[rm] = False
    assert np.array_equal(dm.foreground_mask(NDELAYS), expected)


def test_foreground_mask_custom_range():
    keep = dm.foreground_mask(NDELAYS, (10, 12))
    assert keep.sum() == NDELAYS - 2
    assert not keep[10] and not keep[11] and keep[12]


def test_foreground_mask_rejects_out_of_range():
    with pytest.raises(ValueError, match='not a valid range'):
        dm.foreground_mask(NDELAYS, (0, NDELAYS + 1))


def test_foreground_mask_rejects_empty_result():
    with pytest.raises(ValueError, match='every delay bin'):
        dm.foreground_mask(NDELAYS, (0, NDELAYS))


# ── mask_from_delays ───────────────────────────────────────────────────────

def test_mask_from_delays_keeps_high_delays():
    delays = np.linspace(-1000, 1000, NDELAYS)
    keep = dm.mask_from_delays(delays, 350.0)
    assert np.all(np.abs(delays[keep]) > 350.0)
    assert np.all(np.abs(delays[~keep]) <= 350.0)


def test_mask_from_delays_rejects_empty_result():
    with pytest.raises(ValueError, match='every delay bin'):
        dm.mask_from_delays(np.linspace(-100, 100, 10), 1000.0)


# ── point_estimate ─────────────────────────────────────────────────────────

def test_point_estimate_mean(reference):
    got = dm.point_estimate(reference.ps_chain, estimator='mean')
    assert got == pytest.approx(reference.ps_chain.mean(axis=0))


def test_point_estimate_median(reference):
    got = dm.point_estimate(reference.ps_chain, estimator='median')
    assert got == pytest.approx(np.median(reference.ps_chain, axis=0))


def test_point_estimate_weighted_matches_notebook(reference):
    """`np.average(ps_sample, weights=ln_post)`, the Figure 7 estimator."""
    with pytest.warns(RuntimeWarning):
        got = dm.point_estimate(
            reference.ps_chain, reference.ln_post, estimator='weighted',
        )
    expected = np.average(
        reference.ps_chain, weights=reference.ln_post, axis=0,
    )
    assert got == pytest.approx(expected)


def test_point_estimate_weighted_warns_on_negative_ln_post(reference):
    """Log-posterior weights are negative; the caller is told."""
    with pytest.warns(RuntimeWarning, match='negative values'):
        dm.point_estimate(
            reference.ps_chain, reference.ln_post, estimator='weighted',
        )


def test_point_estimate_weighted_silent_on_positive_weights(reference):
    """Genuine posterior weights raise nothing."""
    weights = np.abs(reference.ln_post)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        dm.point_estimate(reference.ps_chain, weights, estimator='weighted')
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_point_estimate_weighted_needs_ln_post(reference):
    with pytest.raises(ValueError, match='needs ln_post'):
        dm.point_estimate(reference.ps_chain, None, estimator='weighted')


def test_point_estimate_rejects_mismatched_ln_post(reference):
    with pytest.raises(ValueError, match='samples'):
        dm.point_estimate(
            reference.ps_chain, reference.ln_post[:10], estimator='weighted',
        )


def test_point_estimate_rejects_unknown_estimator(reference):
    with pytest.raises(ValueError, match='must be'):
        dm.point_estimate(reference.ps_chain, estimator='mode')


# ── posterior_summary ──────────────────────────────────────────────────────

def test_posterior_summary_restricted_to_kept_bins(reference):
    post = dm.posterior_summary(reference, estimator='mean')
    assert post.mean.shape == (NDELAYS - 7,)
    assert post.std.shape == (NDELAYS - 7,)


def test_posterior_summary_width_is_upper_minus_lower(reference):
    post = dm.posterior_summary(reference, estimator='mean')
    assert post.width == pytest.approx(post.upper - post.lower)
    assert np.all(post.width > 0)


def test_posterior_summary_credible_interval_brackets_the_mean(reference):
    post = dm.posterior_summary(reference, estimator='mean')
    assert np.all(post.lower < post.mean)
    assert np.all(post.mean < post.upper)


def test_posterior_summary_percentiles_match_notebook(reference):
    """The 95% bounds are the notebook's 2.5 / 97.5 percentiles."""
    post = dm.posterior_summary(reference, estimator='mean', conf_interval=95.0)
    keep = dm.foreground_mask(NDELAYS)
    assert post.upper == pytest.approx(
        np.percentile(reference.ps_chain, 97.5, axis=0)[keep]
    )
    assert post.lower == pytest.approx(
        np.percentile(reference.ps_chain, 2.5, axis=0)[keep]
    )


def test_posterior_summary_wider_interval_is_wider(reference):
    narrow = dm.posterior_summary(reference, estimator='mean', conf_interval=68.0)
    wide = dm.posterior_summary(reference, estimator='mean', conf_interval=95.0)
    assert np.all(wide.width > narrow.width)


def test_posterior_summary_rejects_1d_chain(reference):
    bad = dm.RunData('bad', reference.ps_chain[0], reference.ps_true)
    with pytest.raises(ValueError, match='2-D'):
        dm.posterior_summary(bad, estimator='mean')


def test_posterior_summary_rejects_mismatched_truth(reference):
    bad = dm.RunData('bad', reference.ps_chain, reference.ps_true[:10])
    with pytest.raises(ValueError, match='bins'):
        dm.posterior_summary(bad, estimator='mean')


def test_posterior_summary_rejects_bad_conf_interval(reference):
    with pytest.raises(ValueError, match='conf_interval'):
        dm.posterior_summary(reference, estimator='mean', conf_interval=100.0)


def test_posterior_summary_rejects_mismatched_mask(reference):
    with pytest.raises(ValueError, match='mask has'):
        dm.posterior_summary(
            reference, mask=np.ones(10, dtype=bool), estimator='mean',
        )


# ── normalised_deviation ───────────────────────────────────────────────────

def test_normalised_deviation_sign_is_recovered_minus_true(reference):
    """z > 0 where the posterior mean sits above the truth."""
    keep = dm.foreground_mask(NDELAYS)
    post = dm.posterior_summary(reference, keep, estimator='mean')
    z = dm.normalised_deviation(post, reference.ps_true, keep)
    above = post.mean > reference.ps_true[keep]
    assert np.all(z[above] > 0)
    assert np.all(z[~above] <= 0)


def test_normalised_deviation_scales_with_sigma(reference):
    keep = dm.foreground_mask(NDELAYS)
    post = dm.posterior_summary(reference, keep, estimator='mean')
    z = dm.normalised_deviation(post, reference.ps_true, keep)
    expected = (post.mean - reference.ps_true[keep]) / post.std
    assert z == pytest.approx(expected)


def test_normalised_deviation_accepts_pre_masked_truth(reference):
    keep = dm.foreground_mask(NDELAYS)
    post = dm.posterior_summary(reference, keep, estimator='mean')
    with_mask = dm.normalised_deviation(post, reference.ps_true, keep)
    without = dm.normalised_deviation(post, reference.ps_true[keep])
    assert with_mask == pytest.approx(without)


def test_normalised_deviation_rejects_length_mismatch(reference):
    post = dm.posterior_summary(reference, estimator='mean')
    with pytest.raises(ValueError, match='kept bins'):
        dm.normalised_deviation(post, reference.ps_true[:5])


# ── case_metrics ───────────────────────────────────────────────────────────

def test_case_metrics_counts_kept_bins(reference):
    metrics, _, z = dm.case_metrics(reference, estimator='mean')
    assert metrics.n_bins == NDELAYS - 7
    assert z.shape == (NDELAYS - 7,)


def test_case_metrics_beyond_thresholds_are_consistent(reference):
    metrics, _, z = dm.case_metrics(reference, estimator='mean')
    assert metrics.n_beyond['2'] == int(np.sum(np.abs(z) > 2))
    assert metrics.n_beyond['3'] == int(np.sum(np.abs(z) > 3))
    assert metrics.n_beyond['2'] >= metrics.n_beyond['3']


def test_case_metrics_reduced_chi2_is_mean_z_squared(reference):
    metrics, _, z = dm.case_metrics(reference, estimator='mean')
    assert metrics.reduced_chi2 == pytest.approx(np.mean(z ** 2))


def test_case_metrics_recovers_known_bias(biased):
    """A run built at 3 sigma reports a median |z| near 3."""
    metrics, _, _ = dm.case_metrics(biased, estimator='mean')
    assert metrics.median_abs_z == pytest.approx(3.0, rel=0.2)


def test_case_metrics_unbiased_run_has_small_z():
    unbiased = dm.make_demo_run('Unbiased', bias_sigma=0.0, seed=4)
    metrics, _, _ = dm.case_metrics(unbiased, estimator='mean')
    assert metrics.median_abs_z < 0.3


def test_case_metrics_frac_dev_is_positive(reference):
    metrics, _, _ = dm.case_metrics(reference, estimator='mean')
    assert metrics.median_frac_dev > 0


# ── verdict ────────────────────────────────────────────────────────────────

def test_verdict_mild_when_only_width_grows(reference, wider):
    ref, _, _ = dm.case_metrics(reference, estimator='mean')
    tgt, _, _ = dm.case_metrics(wider, estimator='mean')
    assert dm.verdict(ref, tgt) == 'mildly degraded'


def test_verdict_significant_when_accuracy_degrades(reference, biased):
    ref, _, _ = dm.case_metrics(reference, estimator='mean')
    tgt, _, _ = dm.case_metrics(biased, estimator='mean')
    assert dm.verdict(ref, tgt) == 'significantly degraded'


def test_verdict_tolerance_is_respected(reference, biased):
    """A tolerance wide enough to swallow the bias downgrades the verdict.

    Not all the way to neutral: shifting a lognormal posterior also widens it,
    so `biased` has genuinely wider intervals than `reference`.
    """
    ref, _, _ = dm.case_metrics(reference, estimator='mean')
    tgt, _, _ = dm.case_metrics(biased, estimator='mean')
    assert dm.verdict(ref, tgt) == 'significantly degraded'
    assert dm.verdict(ref, tgt, tolerance=100.0) == 'mildly degraded'


def test_verdict_neutral_when_nothing_moves(reference):
    """Equal runs are not a mild degradation."""
    ref, _, _ = dm.case_metrics(reference, estimator='mean')
    assert dm.verdict(ref, ref) == 'not measurably degraded'


def test_verdict_neutral_for_a_flat_width_ratio(reference, biased):
    """The bias is what decides; a flat width ratio cannot make it mild."""
    ref, _, _ = dm.case_metrics(reference, estimator='mean')
    tgt, _, _ = dm.case_metrics(biased, estimator='mean')
    assert dm.verdict(ref, tgt, tolerance=100.0, width_ratio=1.0) == \
        'not measurably degraded'


def test_verdict_uses_an_explicit_width_ratio(reference):
    ref, _, _ = dm.case_metrics(reference, estimator='mean')
    assert dm.verdict(ref, ref, width_ratio=2.0) == 'mildly degraded'


def test_verdict_width_tolerance_is_respected(reference):
    ref, _, _ = dm.case_metrics(reference, estimator='mean')
    assert dm.verdict(ref, ref, width_ratio=1.05) == 'not measurably degraded'
    assert dm.verdict(ref, ref, width_ratio=1.05, width_tolerance=0.01) == \
        'mildly degraded'


# ── compare_runs ───────────────────────────────────────────────────────────

def test_compare_runs_width_ratio_tracks_the_widening(reference, wider):
    comp = dm.compare_runs(reference, wider, estimator='mean')
    assert comp.width_ratio_median > 1.8
    assert comp.std_ratio_median > 1.8


def test_compare_runs_width_ratio_is_one_against_itself(reference):
    comp = dm.compare_runs(reference, reference, estimator='mean')
    assert comp.width_ratio_median == pytest.approx(1.0)
    assert comp.abs_z_ratio == pytest.approx(1.0)
    assert comp.verdict == 'not measurably degraded'


def test_compare_runs_abs_z_ratio_flat_for_equal_bias(reference, wider):
    """Twice the width and twice the offset leaves |z| where it was."""
    comp = dm.compare_runs(reference, wider, estimator='mean')
    assert comp.abs_z_ratio == pytest.approx(1.0, rel=0.25)


def test_compare_runs_range_brackets_the_median(reference, wider):
    comp = dm.compare_runs(reference, wider, estimator='mean')
    lo, hi = comp.width_ratio_range
    assert lo <= comp.width_ratio_median <= hi


def test_compare_runs_carries_labels(reference, wider):
    comp = dm.compare_runs(reference, wider, estimator='mean')
    assert comp.reference.label == 'Reference'
    assert comp.target.label == 'Wider'


def test_compare_runs_rejects_different_bin_counts(reference):
    other = dm.make_demo_run('Other', ndelays=40, seed=6)
    with pytest.raises(ValueError, match='different numbers of delay bins'):
        dm.compare_runs(reference, other, estimator='mean')


def test_compare_runs_honours_a_custom_mask(reference, wider):
    mask = dm.foreground_mask(NDELAYS, (20, 40))
    comp = dm.compare_runs(reference, wider, mask=mask, estimator='mean')
    assert comp.n_bins == NDELAYS - 20


# ── Reporting ──────────────────────────────────────────────────────────────

def test_summary_text_names_both_runs(reference, wider):
    text = dm.summary_text(dm.compare_runs(reference, wider, estimator='mean'))
    assert 'Reference' in text
    assert 'Wider' in text


def test_summary_text_lists_every_metric(reference, wider):
    text = dm.summary_text(dm.compare_runs(reference, wider, estimator='mean'))
    for row in ('median CI width', 'median |z|', 'reduced chi^2',
                'median frac. deviation', 'bins with |z| > 2', 'verdict'):
        assert row in text


def test_summary_text_columns_are_aligned(reference, wider):
    text = dm.summary_text(dm.compare_runs(reference, wider, estimator='mean'))
    lines = text.splitlines()
    assert len(set(len(line) for line in lines[:-1])) == 1


def test_paper_sentence_carries_the_numbers(reference, wider):
    comp = dm.compare_runs(reference, wider, estimator='mean')
    sentence = dm.paper_sentence(comp)
    assert f'{comp.width_ratio_median:.1f}' in sentence
    assert comp.verdict in sentence
    assert str(comp.n_bins) in sentence


def test_paper_sentence_switches_on_the_verdict(reference, wider, biased):
    mild = dm.paper_sentence(dm.compare_runs(reference, wider, estimator='mean'))
    severe = dm.paper_sentence(dm.compare_runs(reference, biased, estimator='mean'))
    assert 'predominantly one of precision' in mild
    assert 'more biased' in severe
    assert 'predominantly one of precision' not in severe


def test_paper_sentence_neutral_claims_no_degradation(reference):
    """An unchanged run must not be written up as a degradation."""
    sentence = dm.paper_sentence(
        dm.compare_runs(reference, reference, estimator='mean')
    )
    assert 'essentially indistinguishable' in sentence
    assert 'degraded' not in sentence
    assert 'widen by a median factor' not in sentence


def test_to_dict_is_json_serialisable(reference, wider):
    comp = dm.compare_runs(reference, wider, estimator='mean')
    payload = json.dumps(dm.to_dict(comp))
    assert 'width_ratio_median' in payload
    assert json.loads(payload)['reference']['label'] == 'Reference'


# ── load_run ───────────────────────────────────────────────────────────────

def test_load_run_reads_the_three_files(run_dir):
    run = dm.load_run(str(run_dir), nfreqs=NDELAYS)
    assert run.ps_chain.ndim == 2
    assert run.ps_true.shape == (NDELAYS,)
    assert run.ln_post.shape[0] == run.ps_chain.shape[0]


def test_load_run_labels_from_directory_name(run_dir):
    assert dm.load_run(str(run_dir), nfreqs=NDELAYS).label == 'low_dl_fr_20'


def test_load_run_honours_an_explicit_label(run_dir):
    run = dm.load_run(str(run_dir), label='Case III', nfreqs=NDELAYS)
    assert run.label == 'Case III'


def test_load_run_discards_burn_in(run_dir, reference):
    n = reference.ps_chain.shape[0]
    run = dm.load_run(str(run_dir), nburn_pc=10.0, nfreqs=NDELAYS)
    assert run.ps_chain.shape[0] == n - int(n * 0.1)


def test_load_run_zero_burn_in_keeps_everything(run_dir, reference):
    run = dm.load_run(str(run_dir), nburn_pc=0.0, nfreqs=NDELAYS)
    assert run.ps_chain.shape[0] == reference.ps_chain.shape[0]


def test_load_run_trims_to_niter(run_dir):
    run = dm.load_run(str(run_dir), niter=1000, nburn_pc=10.0, nfreqs=NDELAYS)
    assert run.ps_chain.shape[0] == 900


def test_load_run_rejects_too_many_samples(run_dir):
    with pytest.raises(ValueError, match='only'):
        dm.load_run(str(run_dir), niter=10 ** 7, nfreqs=NDELAYS)


def test_load_run_rejects_bad_burn_in(run_dir):
    with pytest.raises(ValueError, match='nburn_pc'):
        dm.load_run(str(run_dir), nburn_pc=100.0, nfreqs=NDELAYS)


def test_load_run_reports_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match='no such run directory'):
        dm.load_run(str(tmp_path / 'absent'))


def test_load_run_reports_a_missing_file(tmp_path):
    d = tmp_path / 'incomplete'
    d.mkdir()
    np.save(d / dm.CHAIN_FILE, np.ones((10, NDELAYS)))
    with pytest.raises(FileNotFoundError, match=dm.LNPOST_FILE):
        dm.load_run(str(d))


# ── Command line ───────────────────────────────────────────────────────────

def test_parse_run_spec_with_label():
    assert dm._parse_run_spec('Case III=/a/b') == ('Case III', '/a/b')


def test_parse_run_spec_without_label():
    assert dm._parse_run_spec('/a/b') == (None, '/a/b')


def test_parse_exclude():
    assert dm._parse_exclude('27:34') == (27, 34)


def test_parse_exclude_rejects_junk():
    import argparse
    with pytest.raises(argparse.ArgumentTypeError):
        dm._parse_exclude('27-34')
    with pytest.raises(argparse.ArgumentTypeError):
        dm._parse_exclude('a:b')


def test_parser_defaults_match_the_notebook():
    args = dm.build_parser().parse_args([])
    assert args.conf == dm.CONF_INTERVAL_DEFAULT
    assert args.burn_pc == dm.NBURN_PC_DEFAULT
    assert args.exclude == dm.FG_EXCLUDE_DEFAULT
    assert args.estimator == 'weighted'


def test_selftest_reports_a_mild_degradation():
    comp = dm.selftest()
    assert comp.verdict == 'mildly degraded'
    assert comp.width_ratio_median > 1.8


def test_main_selftest_prints_the_table(capsys):
    assert dm.main(['--selftest']) == 0
    out = capsys.readouterr().out
    assert 'median |z|' in out
    assert 'Paper sentence:' in out


def test_main_writes_json(tmp_path, capsys):
    path = tmp_path / 'metrics.json'
    assert dm.main(['--selftest', '--json', str(path)]) == 0
    payload = json.loads(path.read_text())
    assert 'width_ratio_median' in payload
    assert payload['target']['label'] == 'Target'


def test_main_runs_on_real_directories(run_dir, tmp_path, capsys):
    """The two-directory path, on synthetic run outputs."""
    second = tmp_path / 'caseiv'
    second.mkdir()
    for name in (dm.CHAIN_FILE, dm.LNPOST_FILE, dm.EOR_TRUE_FILE):
        np.save(second / name, np.load(run_dir / name))
    code = dm.main([
        '--reference', f'Case III={run_dir}',
        '--target', f'Combined={second}',
        '--nfreqs', str(NDELAYS),
        '--estimator', 'mean',
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert 'Case III' in out and 'Combined' in out


# ── _render_table ──────────────────────────────────────────────────────────

def test_render_table_aligns_columns():
    out = dm._render_table(('a', 'bbbb'), [('cc', 'd'), ('e', 'ffff')])
    lines = out.splitlines()
    assert len(set(len(l) for l in lines)) == 1


def test_render_table_adds_a_footer():
    out = dm._render_table(('a',), [('b',)], footer='verdict: fine')
    assert out.splitlines()[-1] == 'verdict: fine'


def test_render_table_rejects_a_short_row():
    with pytest.raises(ValueError, match='cells'):
        dm._render_table(('a', 'b'), [('only',)])


def test_render_table_handles_no_rows():
    assert 'header' in dm._render_table(('header',), [])


# ── b_sys spread ───────────────────────────────────────────────────────────

@pytest.fixture
def bsys_plain():
    """Four independent complex amplitudes."""
    rng = np.random.default_rng(3)
    return rng.normal(size=(2000, 4)) + 1j * rng.normal(size=(2000, 4))


@pytest.fixture
def bsys_partnered(bsys_plain):
    """Twelve amplitudes; the first is degenerate with the last."""
    rng = np.random.default_rng(4)
    chain = np.concatenate(
        [bsys_plain, rng.normal(size=(2000, 8))
         + 1j * rng.normal(size=(2000, 8))], axis=1)
    shared = rng.normal(size=2000) * 3.0 + 1j * rng.normal(size=2000) * 3.0
    chain[:, 0] += shared
    chain[:, 11] -= shared
    return chain


def test_bsys_spread_one_record_per_parameter(bsys_plain):
    out = dm.bsys_spread(bsys_plain)
    assert len(out) == 4
    assert [s.label for s in out] == [f'b_sys,{i}' for i in range(1, 5)]


def test_bsys_spread_total_combines_the_parts(bsys_plain):
    s = dm.bsys_spread(bsys_plain)[0]
    assert s.std_total == pytest.approx(np.hypot(s.std_real, s.std_imag))


def test_bsys_spread_accepts_labels(bsys_plain):
    out = dm.bsys_spread(bsys_plain, labels=list('wxyz'))
    assert [s.label for s in out] == list('wxyz')


def test_bsys_spread_rejects_wrong_label_count(bsys_plain):
    with pytest.raises(ValueError, match='labels for'):
        dm.bsys_spread(bsys_plain, labels=['only'])


def test_bsys_spread_rejects_1d(bsys_plain):
    with pytest.raises(ValueError, match='2-D'):
        dm.bsys_spread(bsys_plain[:, 0])


def test_compare_bsys_spread_detects_the_inflated_parameter(
        bsys_plain, bsys_partnered):
    """The signature the combined-case argument predicts."""
    table = dm.compare_bsys_spread(bsys_plain, bsys_partnered)
    rows = [l for l in table.splitlines() if l.startswith('b_sys,')]
    assert 'wider' in rows[0]
    assert all('unchanged' in r for r in rows[1:])


def test_compare_bsys_spread_flat_against_itself(bsys_plain):
    table = dm.compare_bsys_spread(bsys_plain, bsys_plain)
    assert 'wider' not in table
    assert table.count('unchanged') == 4


def test_compare_bsys_spread_matches_by_position(bsys_plain, bsys_partnered):
    """Only the parameters the two runs share are compared."""
    table = dm.compare_bsys_spread(bsys_plain, bsys_partnered)
    assert len([l for l in table.splitlines() if l.startswith('b_sys,')]) == 4


def test_compare_bsys_spread_honours_explicit_indices(
        bsys_plain, bsys_partnered):
    table = dm.compare_bsys_spread(
        bsys_plain, bsys_partnered, indices=[(0, 11)])
    rows = [l for l in table.splitlines() if l.startswith('b_sys,')]
    assert len(rows) == 1
    assert 'b_sys,1 -> b_sys,12' in rows[0]


def test_compare_bsys_spread_rejects_a_bad_index(bsys_plain, bsys_partnered):
    with pytest.raises(ValueError, match='no parameter'):
        dm.compare_bsys_spread(bsys_plain, bsys_partnered, indices=[(9, 0)])


def test_bsys_correlation_finds_the_partner(bsys_partnered):
    r = dm.bsys_correlation(bsys_partnered, 0, 11)
    assert r['max_abs'] > 0.8
    assert r['real'] < 0            # built anti-correlated


def test_bsys_correlation_near_zero_for_independent(bsys_partnered):
    r = dm.bsys_correlation(bsys_partnered, 1, 2)
    assert r['max_abs'] < 0.2


def test_bsys_correlation_rejects_self_pair(bsys_partnered):
    with pytest.raises(ValueError, match='different parameters'):
        dm.bsys_correlation(bsys_partnered, 3, 3)


def test_bsys_correlation_rejects_bad_index(bsys_partnered):
    with pytest.raises(ValueError, match='no parameter'):
        dm.bsys_correlation(bsys_partnered, 0, 99)


# ── Sky residuals ──────────────────────────────────────────────────────────

def _make_sky_run(root, name, resid_scale, nsamples=60, nt=8, nf=12, nfg=4):
    """A run directory carrying the arrays `sky_residual_rms` reads."""
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    eor_true = (rng.normal(size=(nt, nf)) + 1j * rng.normal(size=(nt, nf))) * 0.1
    fgmodes = rng.normal(size=(nf, nfg))
    fg_amps_true = rng.normal(size=(nt, nfg))
    fg_true = (fgmodes @ fg_amps_true.T).T
    np.save(d / dm.EOR_TRUE_FILE, eor_true)
    np.save(d / dm.FG_TRUE_FILE, fg_true)
    np.save(d / dm.FG_MODES_FILE, fgmodes)
    np.save(d / dm.EOR_GCR_FILE, eor_true[None] + resid_scale * 0.02 * (
        rng.normal(size=(nsamples, nt, nf))
        + 1j * rng.normal(size=(nsamples, nt, nf))))
    np.save(d / dm.FG_AMPS_FILE,
            fg_amps_true[None] + resid_scale * 0.01
            * rng.normal(size=(nsamples, nt, nfg)))
    return d


@pytest.fixture
def sky_runs(tmp_path):
    """Three runs whose residuals grow with a known scale factor."""
    return {
        name: _make_sky_run(tmp_path / 'sky', name, scale)
        for name, scale in (('quiet', 1.0), ('mid', 2.0), ('loud', 4.0))
    }


def _kw():
    return dict(ntimes=8, nfreqs=12, nfgmodes=4, stride=1, nburn_pc=0.0)


def test_sky_residual_rms_returns_positive_scales(sky_runs):
    r = dm.sky_residual_rms(str(sky_runs['quiet']), **_kw())
    assert r.rms_residual > 0 and r.rms_sky > 0 and r.rms_eor > 0


def test_sky_residual_rms_grows_with_the_scatter(sky_runs):
    """A noisier chain leaves a larger residual after averaging."""
    vals = [dm.sky_residual_rms(str(sky_runs[n]), **_kw()).rms_residual
            for n in ('quiet', 'mid', 'loud')]
    assert vals[0] < vals[1] < vals[2]


def test_sky_residual_ratios_are_consistent(sky_runs):
    r = dm.sky_residual_rms(str(sky_runs['mid']), **_kw())
    assert r.fractional == pytest.approx(r.rms_residual / r.rms_sky)
    assert r.relative_to_eor == pytest.approx(r.rms_residual / r.rms_eor)


def test_sky_residual_labels_from_directory(sky_runs):
    assert dm.sky_residual_rms(str(sky_runs['quiet']), **_kw()).label == 'quiet'


def test_sky_residual_honours_an_explicit_label(sky_runs):
    r = dm.sky_residual_rms(str(sky_runs['quiet']), label='Case I', **_kw())
    assert r.label == 'Case I'


def test_sky_residual_stride_reduces_the_sample_count(sky_runs):
    kw = _kw()
    kw['stride'] = 4
    r = dm.sky_residual_rms(str(sky_runs['quiet']), **kw)
    assert r.nsamples == 15


def test_sky_residual_rejects_bad_stride(sky_runs):
    kw = _kw()
    kw['stride'] = 0
    with pytest.raises(ValueError, match='stride'):
        dm.sky_residual_rms(str(sky_runs['quiet']), **kw)


def test_sky_residual_rejects_too_many_samples(sky_runs):
    kw = _kw()
    with pytest.raises(ValueError, match='only'):
        dm.sky_residual_rms(str(sky_runs['quiet']), niter=10**6, **kw)


def test_sky_residual_reports_a_missing_file(tmp_path):
    d = tmp_path / 'bare'
    d.mkdir()
    with pytest.raises(FileNotFoundError, match=dm.EOR_GCR_FILE):
        dm.sky_residual_rms(str(d))


def test_sky_residual_reports_a_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match='no such run directory'):
        dm.sky_residual_rms(str(tmp_path / 'absent'))


def test_sky_residual_table_lists_every_run(sky_runs):
    rs = [dm.sky_residual_rms(str(sky_runs[n]), label=n, **_kw())
          for n in ('quiet', 'mid', 'loud')]
    table = dm.sky_residual_table(rs)
    for n in ('quiet', 'mid', 'loud'):
        assert n in table
    for col in ('RMS residual', 'resid/sky', 'resid/EoR', 'samples'):
        assert col in table


def test_sky_residual_table_handles_no_runs():
    assert dm.sky_residual_table([]) == 'no runs'


# ── Command line: the new tasks ────────────────────────────────────────────

def test_parse_pair_is_one_based():
    assert dm._parse_pair('1,9') == (0, 8)


def test_parse_pair_rejects_junk():
    import argparse
    for bad in ('1-9', 'a,b', '1', '0,3'):
        with pytest.raises(argparse.ArgumentTypeError):
            dm._parse_pair(bad)


def test_main_sky_task_prints_a_table(sky_runs, capsys):
    code = dm.main([
        '--task', 'sky',
        '--runs', f'Case I={sky_runs["quiet"]}', f'Case II={sky_runs["mid"]}',
        '--ntimes', '8', '--nfreqs', '12', '--nfgmodes', '4',
        '--stride', '1', '--burn-pc', '0',
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert 'Case I' in out and 'Case II' in out and 'RMS residual' in out


def test_main_sky_task_needs_runs(capsys):
    assert dm.main(['--task', 'sky']) == 2
    assert '--runs' in capsys.readouterr().out


def test_main_bsys_task_needs_two_runs(capsys):
    assert dm.main(['--task', 'bsys']) == 2
    assert '--reference' in capsys.readouterr().out


def test_main_bsys_task_reports_spread_and_correlation(tmp_path, capsys):
    rng = np.random.default_rng(9)
    ref_d, tgt_d = tmp_path / 'ref', tmp_path / 'tgt'
    ref_d.mkdir(); tgt_d.mkdir()
    ref = rng.normal(size=(500, 4)) + 1j * rng.normal(size=(500, 4))
    tgt = rng.normal(size=(500, 12)) + 1j * rng.normal(size=(500, 12))
    shared = rng.normal(size=500) * 3.0 + 1j * rng.normal(size=500) * 3.0
    tgt[:, 0] += shared
    tgt[:, 11] -= shared
    np.save(ref_d / dm.BSYS_FILE, ref)
    np.save(tgt_d / dm.BSYS_FILE, tgt)

    code = dm.main([
        '--task', 'bsys', '--reference', f'Case I={ref_d}',
        '--target', f'Combined={tgt_d}', '--burn-pc', '0', '--pair', '1,12',
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert 'wider' in out
    assert 'Partner correlation' in out


def test_load_bsys_discards_burn_in(tmp_path):
    d = tmp_path / 'r'
    d.mkdir()
    chain = np.arange(200).reshape(100, 2).astype(complex)
    np.save(d / dm.BSYS_FILE, chain)
    assert dm.load_bsys(str(d), nburn_pc=10.0).shape[0] == 90


def test_load_bsys_reports_a_missing_file(tmp_path):
    d = tmp_path / 'empty'
    d.mkdir()
    with pytest.raises(FileNotFoundError, match=dm.BSYS_FILE):
        dm.load_bsys(str(d))
