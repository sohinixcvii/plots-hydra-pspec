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
    ref, _, _ = dm.case_metrics(reference, estimator='mean')
    tgt, _, _ = dm.case_metrics(biased, estimator='mean')
    assert dm.verdict(ref, tgt, tolerance=100.0) == 'mildly degraded'


# ── compare_runs ───────────────────────────────────────────────────────────

def test_compare_runs_width_ratio_tracks_the_widening(reference, wider):
    comp = dm.compare_runs(reference, wider, estimator='mean')
    assert comp.width_ratio_median > 1.8
    assert comp.std_ratio_median > 1.8


def test_compare_runs_width_ratio_is_one_against_itself(reference):
    comp = dm.compare_runs(reference, reference, estimator='mean')
    assert comp.width_ratio_median == pytest.approx(1.0)
    assert comp.abs_z_ratio == pytest.approx(1.0)
    assert comp.verdict == 'mildly degraded'


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
