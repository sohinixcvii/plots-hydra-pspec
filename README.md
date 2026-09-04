# plots-hydra-pspec-systematic

Plotting scripts and helper libraries for the **hydra-pspec-systematic** paper.
The analysis characterises instrumental systematics in 21 cm power-spectrum
estimation using the Gibbs-sampling framework implemented in
[hydra-pspec](https://github.com/HydraRadio/hydra-pspec).

---

## Repository layout

```
.
├── paper_plots_c.py            # Main script — generates all paper figures
├── paper_plots_c.ipynb         # Notebook version (source of paper_plots_c.py)
├── paper_plots_c_v2.ipynb      # Optimised notebook (recommended; see below)
├── paper_plots_c_v2_nfgmodes.ipynb  # Same analysis, cases differ in Nfgmodes
├── paper_plots_c_v2_single_case.ipynb # Single-run version of the v2 notebook
├── convergence_tests.ipynb     # Convergence diagnostics, 1+ cases (see below)
├── plot_corner_blink.py        # Blink-comparison corner plots, 100k vs 250k (see below)
├── plotting_functions.py       # Waterfall-plot helpers used by paper_plots_c.py
├── plot-test-data-results.py   # Standalone EoR delay power spectrum plotter (CLI)
├── plot_speed_up.py            # MPI speed-up / scaling plotter (CLI)
├── checking_result_files.ipynb # Notebook for sanity-checking result files
├── 100k_runs_paper_plots.ipynb # Notebook for 100k-iteration run plots
├── Figures_sim_data/           # Output directory for generated figures
│   ├── systematics_in_vis_space.pdf
│   └── test_cases_in_dlfr.pdf
├── tests/                      # pytest suite
│   └── test_plot_corner_blink.py
└── plotting_codes/             # Shared utility library
    ├── functions.py            # Fourier transforms, covariance helpers
    ├── plotting_functions.py   # Extended waterfall + diagnostic plot helpers
    ├── paper_plots.py          # SUPERSEDED — old plotting script (kept for reference)
    └── paper_plots.ipynb       # Notebook companion to the superseded script
```

---

## Main paper figure script: `paper_plots_c.py`

Generates Figures 1, 3–11 of the paper and saves them to `fig_dir` as PDF.
Console output (including tqdm progress bars) is also mirrored to
`paper_plots_c_output.txt`.

### Configuration

Edit the **Configuration** block near the top of `paper_plots_c.py`
(or the equivalent cell in the notebooks):

| Variable | Description |
|---|---|
| `result_dir` | Directory containing per-run subdirectories with `.npy` output files |
| `parent_dir` | Root of the `hydra-pspec-systematic` repository (for raw data files) |
| `fig_dir` | Output directory for generated figures |
| `run_version_arr` | List of run subdirectory names, one per test case |
| `nm_list_arr` | Systematic mode (delay-index, fringe-rate-index) pairs, one list per test case |
| `case_idx` | Which test case to use for single-case figures (0, 1, or 2) |
| `Niter` | Number of MCMC samples |
| `Nburn` | Burn-in samples to discard before computing statistics |
| `conf_interval` | Confidence interval (%) for posterior credible regions |

**Expected result directory layout** — each entry in `run_version_arr` must be a
subdirectory of `result_dir` containing:

```
<run_version>/
├── fg_true.npy          # True foreground visibilities  (Ntimes, Nfreqs)
├── eor_true.npy         # True EoR visibilities         (Ntimes, Nfreqs)
├── gain_true.npy        # True systematic gain model    (Ntimes, Nfreqs)
├── data_true.npy        # Total data                    (Ntimes, Nfreqs)
├── fgmodes.npy          # Foreground basis matrix       (Nfreqs, Nfgmodes)
├── fg-amps.npy          # Sampled FG amplitudes         (Niter, Nfgmodes)
├── gcr-eor.npy          # Sampled EoR visibilities      (Niter, Ntimes, Nfreqs)
├── b-sys.npy            # Sampled systematic amplitudes (Niter, Nsys_modes)
├── dps-eor.npy          # Sampled EoR delay power spec. (Niter, Nfreqs)
└── ln-post.npy          # Log-posterior per iteration   (Niter,)
```

### Usage

```bash
# Recommended: capture all output including tqdm
python paper_plots_c.py 2>&1 | tee paper_plots_c_output.txt

# Or simply:
python paper_plots_c.py
```

### Outputs

| Figure | File | Description |
|---|---|---|
| 1 | `systematics_in_vis_space.pdf` | Corrupted vs clean visibilities (2×2 grid) |
| 3 | `test_cases_in_dlfr.pdf` | All three test cases in delay–fringe-rate space |
| 4 | `data_vs_sky_2.pdf` | Data, sky mean, systematics mean for all cases (3×3 grid) |
| 5 | `result_waterfalls.pdf` | True / mean / std / residual sky (single case, 2×2) |
| 5 v2 | `result_waterfalls_v2.pdf` | Same but all three cases (4 rows × 3 columns) |
| 6 | `sys_result_dlfr_waterfalls.pdf` | Systematics recovery (single case, 2×2) |
| 6 v2 | `sys_result_dlfr_waterfalls_v2.pdf` | Same but all three cases (4 rows × 3 columns) |
| 7 | `errors_components.pdf` | Fractional errors by component (single case) |
| 7 v2 | `errors_components_v2.pdf` | Same but all three cases (3 rows × 1 column) |
| 8 | `bsys_corner_plot.pdf` | Corner plot of systematic amplitudes |
| 9 | `sys_frac_error.pdf` | Systematic amplitude recovery (imaginary part) |
| 10 | `delay_power_spectrum_combined.pdf` | Delay power spectra for all cases (full + zoomed) |
| 11 | `fg_sys_corr.pdf` | Foreground–systematics Pearson correlation (all cases) |

Table 1 (ESS and autocorrelation times) is printed to stdout.

---

## Optimised notebook: `paper_plots_c_v2.ipynb`

**Recommended over `paper_plots_c.ipynb` and `paper_plots_c.py`** for interactive
use when memory is a concern.

Key improvements over v1:

| Aspect | v1 | v2 |
|---|---|---|
| Memory per run | ~144 GB (3 × [Niter, Ntimes, Nfreqs] complex64) | ~144 MB (online accumulators) |
| Gibbs iteration passes | 3 separate passes | Single pass |
| Per-case statistics | Full sample arrays stored | Welford online mean/variance |
| Pearson correlation | Full arrays stored | Sum-of-products online estimator |
| Figure 11 correctness | Used single shared `sys_modes_operator` | Per-case operator (bug fixed) |

The v2 notebook also adds Figures 5 v2, 6 v2, and 7 v2 (all-cases multi-panel versions of the single-case figures).

---

## Varying the foreground model: `paper_plots_c_v2_nfgmodes.ipynb`

Runs every analysis step of `paper_plots_c_v2.ipynb`, but the cases are defined
differently.  In `paper_plots_c_v2.ipynb` the three cases differ in **where** the
systematic modes sit in delay–fringe-rate space (`nm_list_arr`).  Here all cases
share the **same** systematic-mode locations — the Case II values,
`[(10, 0), (11, 0), (12, 0), (13, 0)]` — and differ only in the number of
foreground modes `Nfgmodes` given to the Gibbs sampler.

| Aspect | `paper_plots_c_v2.ipynb` | `paper_plots_c_v2_nfgmodes.ipynb` |
|---|---|---|
| Cases distinguished by | systematic-mode location | `Nfgmodes` |
| `nm_list` | one list per case | one shared list (Case II) |
| `sys_modes_operator` | rebuilt per case | built once, shared |
| Number of cases | hard-coded 3 | `Ncases = len(run_version_arr)`, any number ≥ 2 |
| Sample loops | one single-case loop + one all-cases loop | one loop; single-case values index the per-case lists |
| Sample statistics | full `[Niter, Ntimes, Nfreqs]` arrays | Welford mean/variance + centred co-moment Pearson |
| Large `.npy` sample files | read fully into memory | memory-mapped (`USE_MMAP`) |

### Configuration

Additional/changed settings relative to `paper_plots_c_v2.ipynb`:

| Variable | Description |
|---|---|
| `run_version_arr` | One run sub-directory per `Nfgmodes` case (≥ 2). The **Available runs** cell lists the sub-directories of `result_dir` with the `Nfgmodes` recorded in each `fgmodes.npy`. |
| `Nfgmodes_arr` | Foreground modes per case. `None` = take it from that run's `fgmodes.npy` (recommended); an explicit smaller integer uses only the leading modes. |
| `nm_list` | Single shared systematic-mode list, used for every case. |
| `SAVE_FIGS` | Write PDFs to `fig_dir` (created if missing). |
| `USE_MMAP` | Memory-map `gcr-eor.npy` and `fg-amps.npy` instead of loading them. |
| `fg_amp_time_idx` | LST index of the FG amplitudes used for the parameter correlation matrix. |

### Outputs

All figures are written to `fig_dir` with an `_nfg` suffix so they never
overwrite the main paper figures.

| Figure | File | Description |
|---|---|---|
| 1 | `data_components_nfg.pdf` | True EoR / FG / systematics in visibility and delay–fringe-rate space (shared by all cases) |
| 2 | `data_in_dlfr_nfg.pdf` | True sky, true data, and the mean **recovered** data model per `Nfgmodes` |
| 3 | `result_waterfalls_nfg.pdf` | True / mean / std / residual sky, one column per case |
| 4 | `sys_result_dlfr_waterfalls_nfg.pdf` | Systematics recovery, one column per case |
| 5 | `errors_components_nfg.pdf` | Component power spectra and EoR residuals, all cases overlaid |
| 6 | `bsys_corner_plot_nfg.pdf` | Corner plot of `b_sys`, all cases overplotted |
| 7 | `delay_power_spectrum_combined_nfg.pdf` | Delay power spectra, all cases (full + zoom on the systematic delays) |
| 8 | `fg_sys_corr_nfg.pdf` | Foreground–systematics Pearson correlation, one panel per case |
| 9 | `correlation_matrix_nfg.pdf` | `<a_fg, b_sys>` correlation matrices (the FG block grows with `Nfgmodes`) |

| C1 | `convergence_traces_nfg.pdf` | Log-posterior traces and running means of `Re b_sys,1` and `P_EoR` at the first systematic delay |
| C2 | `convergence_acf_ess_nfg.pdf` | Autocorrelation function of `b_sys` and worst efficiency per monitored block |
| C3 | `convergence_ess_dps_nfg.pdf` | EoR delay-power-spectrum efficiency, delay bin by delay bin |

Table 1 (ESS and autocorrelation times) and the convergence tables are printed
to stdout.

### Convergence diagnostics

> These diagnostics also live on their own in
> [`convergence_tests.ipynb`](#convergence-tests-convergence_testsipynb), which
> runs them for any set of runs — including the cases of
> `paper_plots_c_v2.ipynb` and a single run — without the expensive
> delay/fringe-rate sample loop.  Use that notebook when convergence is the
> question; the section here is kept so the `Nfgmodes` notebook stays
> self-contained.

The section after Table 1 answers *which `Nfgmodes` case did the Gibbs sampler
converge best?*  The sampler produces one chain per run, so the statistics used
are the single-chain ones:

| Statistic | What it measures | "Converged" |
|---|---|---|
| `tau` (integrated autocorrelation time, `emcee.autocorr`, Sokal windowing) | Iterations needed to forget the previous sample | `N > 50 tau` (reported as `reliable`) |
| `ESS = N/tau`, efficiency `ESS/N` | Independent draws the chain is worth | Larger is better; `ESS/N` is the fair comparison when the cases ran for different `Niter` |
| Rank-normalised split-Rhat (Vehtari et al. 2021) | Drift/sticking, by comparing `CONV_NSPLIT` consecutive segments of the one chain as if they were independent chains | `<~ 1.01` (`CONV_RHAT_OK`) |
| Geweke *z* | Whether the first 10 % and last 50 % of the chain sample the same distribution | `|z| < 2` (`CONV_Z_OK`) |
| `MCSE = sd/sqrt(ESS)` | Sampling noise left on a posterior mean | Small compared with the posterior scatter |
| Raftery-Lewis `M`, `N`, `I` (Raftery & Lewis 1992) | Iterations needed to estimate the `RL_Q` quantile within `±RL_R` with probability `RL_S`: required burn-in `M`, required total `M+N`, and the dependence factor `I = (M+N)/Nmin` | `M+N` at or below the iterations actually run; `I ~ 1` is as good as independent, `I > 5` flags strong dependence |

Monitored blocks: `ln_post`, `b_sys`, `P_EoR` in the delay bins occupied by the
systematics, and `a_fg` at `fg_amp_time_idx` (`CONV_INCLUDE_FG`).  Complex
parameters are split into real and imaginary scalar chains.  Each case is scored
by its **worst** monitored scalar, and the cases are ranked on three criteria —
efficiency (worst `ESS/N`), mixing (worst split-Rhat) and stationarity (fraction
of scalars failing Geweke) — whose mean rank picks the best-converged case.

Additional settings, all in the estimator cell:

| Variable | Default | Description |
|---|---|---|
| `CONV_NSPLIT` | 4 | Segments used by split-Rhat |
| `CONV_BURN` | `Nburn` | Leading samples dropped before the diagnostics |
| `CONV_GEWEKE_FIRST` / `CONV_GEWEKE_LAST` | 0.1 / 0.5 | Geweke window fractions |
| `CONV_INCLUDE_FG` | `True` | Include the foreground-amplitude chains (extra I/O) |
| `CONV_RHAT_OK` / `CONV_Z_OK` | 1.01 / 2.0 | Pass thresholds used by the summary counts |

A separate table reports `|bias|/sd` and `MCSE/sd` for every systematic
amplitude, to keep *convergence* (a property of the chain) apart from *accuracy*
(a property of the model): a case can mix perfectly and still be biased because
its `Nfgmodes` cannot separate foregrounds from systematics.

These statistics only ever *detect* non-convergence — they cannot prove it.  With
one chain per run they are blind to modes the sampler never visited, and blocked
Gibbs updates correlate all parameter blocks, so they are most useful for
ranking the cases against each other.

### Notes

- Because the true sky, the true gain and the systematic-mode locations are the
  same for every case, all *input* quantities are case-independent; only
  sample-derived quantities differ.  Figure 2 was re-cast accordingly (it showed
  the true data once per case in `paper_plots_c_v2.ipynb`, which would be three
  identical panels here).
- The notebook self-tests its online estimators against the direct
  all-samples-in-memory calculation before using them, and cross-checks that
  every run really does share the same `eor_true`, `fg_true` and `gain_true`.
  The convergence estimators are likewise self-tested against an AR(1) chain of
  known `tau`, an independent chain and a deliberately drifting one.
- Pearson *r* is invariant under a constant offset, so the foreground–gain
  correlation is accumulated against `DLFR(Hb_sys)` rather than
  `DLFR(1 + Hb_sys) = dlfr_ones + DLFR(Hb_sys)`.  The two are equal in exact
  arithmetic, but `dlfr_ones` is a spike of `Ntimes*Nfreqs` at the
  zero-delay/zero-fringe-rate pixel, so adding it first destroys the sample
  scatter in that pixel.

---

## Convergence tests: `convergence_tests.ipynb`

The convergence diagnostics of the `paper_plots_c_v2*` notebooks, split out into
one self-contained notebook that handles **one or more cases**: with a single
case it is an independent assessment of that run, with several it also ranks
them against each other.

It reads only the small chains — `ln-post.npy`, `b-sys.npy`, `dps-eor.npy` and
one LST slice of `fg-amps.npy` — so it needs none of the delay/fringe-rate
sample loop of the paper notebooks and finishes in a couple of minutes.  The
mathematics is unchanged from `paper_plots_c_v2_nfgmodes.ipynb`; only the
configuration was generalised.

### Configuration

Everything lives in the configuration cell.  A "case" is one run directory, so
the cases of all three source notebooks are expressible in the same list:

```python
cases = [
    dict(run='low_dl_fr_0',            # sub-folder of result_dir          (required)
         label=r'Case I',              # legend / table label; default: run
         nm_list=[(3, 0), (4, 0), (5, 0), (6, 0)],   # systematic modes (n, m)
         dl_inds=[3, 4, 5, 6],         # delay indices; default: [n for (n, m) in nm_list]
         sys_amps_true=[1.+4j, 2.+3j, 3.+2j, 4.+1j],  # needed only for Table C5 / Figure C6
         Nfgmodes=None),               # None reads it from that run's fgmodes.npy
]
```

`dl_inds` is kept separate from `nm_list` because the paper notebooks do not
always set it to `n` (Case III of `paper_plots_c_v2.ipynb` uses `[3, 3, 3, 3]`
for modes `(3, 20) … (6, 20)`).  Give it explicitly whenever the run does.

Cases may differ in every one of these: number of systematic modes, systematic
delays, true amplitudes and number of foreground modes.  The comparison figures
take the union where the cases disagree (Figure C6's amplitude axis, Figure C3's
shaded delay band) and drop the parts that would be meaningless (Table C3 is not
built for a single case; Table C5 and Figure C6 are skipped for any case with no
`sys_amps_true`).

| Variable | Default | Description |
|---|---|---|
| `result_dir` / `parent_dir` / `fig_dir` | — | Run outputs, repo root, figure output |
| `table_dir` | `fig_dir/tables` | Markdown/LaTeX/CSV/HTML copies of every table |
| `fig_suffix` | `'_conv'` | Appended to every PDF name, so nothing clashes with the paper figures |
| `SAVE_FIGS` / `SAVE_TABLES` | `True` | Write the PDFs / the table exports |
| `USE_MMAP` | `True` | Memory-map `fg-amps.npy` (only one LST slice is read) |
| `Niter` | 100000 | Iterations to use, clipped per case to what is on disk |
| `fg_amp_time_idx` | `-3` | LST index of the monitored foreground amplitudes |
| `CONV_BURN_PC` | 10 | Burn-in as a percentage of the samples available per case |
| `CONV_BURN_ABS` | `None` | Absolute burn-in in samples; overrides `CONV_BURN_PC` |
| `CONV_NSPLIT` | 4 | Segments used by split-Rhat |
| `CONV_GEWEKE_FIRST` / `CONV_GEWEKE_LAST` | 0.1 / 0.5 | Geweke window fractions |
| `CONV_INCLUDE_FG` | `True` | Include the foreground-amplitude chains (extra I/O) |
| `CONV_RHAT_OK` / `CONV_Z_OK` | 1.01 / 2.0 | Pass thresholds used by the summary counts |
| `TRACE_WINDOW` | `None` | `(start, stop)` within the post-burn-in chain for Figure C1; `None` = all of it |
| `TRACE_RUNNING_MEAN` | `False` | `False` plots raw traces (as in the `Nfgmodes` notebook), `True` plots running means |
| `RL_Q` / `RL_R` / `RL_S` | 0.025 / 0.005 / 0.95 | Raftery-Lewis: quantile to estimate, accuracy on it, probability |
| `RL_EPS` | 0.001 | Raftery-Lewis: convergence tolerance of the two-state chain |
| `RL_MAX_THIN` | 200 | Raftery-Lewis: largest thinning interval tried |

### The statistics

| Statistic | What it measures | "Converged" |
|---|---|---|
| `tau` (integrated autocorrelation time, `emcee.autocorr`, Sokal windowing) | Iterations needed to forget the previous sample | `N > 50 tau` (reported as `reliable`) |
| `ESS = N/tau`, efficiency `ESS/N` | Independent draws the chain is worth | Larger is better; `ESS/N` is the fair comparison when the cases ran for different `Niter` |
| Rank-normalised split-Rhat (Vehtari et al. 2021) | Drift/sticking, by comparing `CONV_NSPLIT` consecutive segments of the one chain as if they were independent chains | `<~ 1.01` (`CONV_RHAT_OK`) |
| Geweke *z* | Whether the first 10 % and last 50 % of the chain sample the same distribution | `\|z\| < 2` (`CONV_Z_OK`) |
| `MCSE = sd/sqrt(ESS)` | Sampling noise left on a posterior mean | Small compared with the posterior scatter |

Monitored blocks: `ln_post`, `b_sys`, `P_EoR` in the delay bins occupied by that
case's systematics, and `a_fg` at `fg_amp_time_idx` (`CONV_INCLUDE_FG`).  Complex
parameters are split into real and imaginary scalar chains.  Each case is scored
by its **worst** monitored scalar, and — with more than one case — the cases are
ranked on three criteria: efficiency (worst `ESS/N`), mixing (worst split-Rhat)
and stationarity (fraction of scalars failing Geweke), whose mean rank picks the
best-converged case.

The estimators are self-tested before use, against an AR(1) chain of known
`tau`, an independent chain and a deliberately drifting one.

These statistics only ever *detect* non-convergence — they cannot prove it.  With
one chain per run they are blind to modes the sampler never visited, and blocked
Gibbs updates correlate all parameter blocks, so they are most useful for ranking
the cases against each other.

### Raftery-Lewis run-length diagnostic

The diagnostics above ask *has this chain converged?*  Raftery & Lewis (1992)
asks *how long would this chain have to be?*  It fixes an estimation target —
the `RL_Q`-th posterior quantile, to within `±RL_R` with probability `RL_S` —
and returns the burn-in `M` and run length `N` needed to hit it, plus the
dependence factor `I = (M+N)/Nmin`, where `Nmin` is what independent draws
would need (3746 for the default settings).

The method reduces the chain to the indicator `Z_t = 1{theta_t <= u_q}`, thins
it until a two-state **first-order** Markov chain fits better than a
second-order one (by BIC), and reads `M` and `N` off that chain's transition
probabilities.  `q`, `r` and `s` are arguments rather than constants because
they *are* the question being asked: a tail quantile or a tighter tolerance
needs a longer run.

Unlike every other table in the notebook, this section reads the **full raw
chains including burn-in** — from the same files and the same `cases`
configuration as the loader cell.  It has to: the point is to estimate the
burn-in a run needed, which is then compared against the `CONV_BURN` actually
applied.  Tables C6 and C7 put required `M` and `M+N` directly next to the
burn-in and iteration count the run really used, so "was this run long enough"
is readable off one column (`required / run`, at or below 1 = yes).

**Why it is implemented in the notebook rather than imported.**  There is no
library route: `pymc` 5.x and `arviz` do not contain the diagnostic (the
PyMC2-era `pymc.raftery_lewis` is gone, and modern PyMC delegates its
statistics to ArviZ, which never had it); `rpy2` + `coda::raftery.diag` would
need a full R installation, which is not present; and no maintained pure-Python
implementation exists on PyPI.  It is therefore written out directly from
Raftery, A. E. & Lewis, S. M. (1992), "One long run with diagnostics:
Implementation strategies for Markov chain Monte Carlo", *Statistical Science*
**7**(4), 493-497, cited in the function docstring, and self-tested against an
AR(1) chain, an independent chain, and the `Nmin` value published for the
default settings.

Caveat: the diagnostic addresses **one quantile at a time** and says nothing
about the rest of the posterior; like the others it is a necessary, not a
sufficient, check.

### Outputs

Figures, written to `fig_dir` with `fig_suffix` appended to each name:

| File | Content |
|---|---|
| `ess_tau_bsys*.pdf` | Figure T1 — `tau` and ESS per systematic amplitude |
| `convergence_dashboard*.pdf` | Figure C0 — efficiency / split-Rhat / Geweke heat maps, case x parameter block |
| `bias_mcse*.pdf` | Figure C6 — `\|bias\|/sd` against `MCSE/sd` per systematic amplitude |
| `convergence_traces*.pdf` | Figure C1 — `ln L`, `Re b_sys,1` and `P_EoR` traces (or running means) |
| `convergence_acf_ess*.pdf` | Figure C2 — `b_sys` autocorrelation and per-block efficiency |
| `convergence_ess_dps*.pdf` | Figure C3 — EoR delay-power-spectrum efficiency per delay bin |
| `raftery_lewis*.pdf` | Figure C4 — Raftery-Lewis required run length against the run actually done, and the dependence factor |

Tables, shown inline and exported to `table_dir` as `.md`, `.tex`, `.csv` and
`.html`:

| Name | Content |
|---|---|
| `table1_ess_bsys` | Table 1 — `tau` / ESS of the systematic amplitudes (full chain, no burn-in) |
| `tableC1_convergence_blocks` | Table C1 — diagnostics per case and parameter block |
| `tableC2_convergence_summary` | Table C2 — one row per case, summarised by its worst scalar |
| `tableC3_convergence_ranking` | Table C3 — ranking of the cases (skipped for a single case) |
| `tableC4_slowest_scalar` | Table C4 — the bottleneck parameter of each case |
| `tableC5_bias_vs_mcse` | Table C5 — accuracy: `\|bias\|/sd` and `MCSE/sd` per amplitude |
| `tableC6_raftery_lewis_blocks` | Table C6 — Raftery-Lewis required run length per case and block, against what was run |
| `tableC7_raftery_lewis_worst` | Table C7 — the parameter that sets each case's required run length |

`Table 1` and the Raftery-Lewis tables (C6, C7) are computed on the full chain;
`Table 1` because that is what the paper notebooks quote, C6 and C7 because the
diagnostic has to see the burn-in in order to estimate it.  Everything else
works on the post-burn-in chain.

### Relation to the paper notebooks

`convergence_tests.ipynb` does not replace anything: the `paper_plots_c_v2*`
notebooks keep their own Table 1 and (in the `Nfgmodes` notebook) their
convergence section, so they remain self-contained.  This notebook is where to
add or re-run convergence work, because it needs no sample loop and is not tied
to one set of cases.  `fig_suffix` defaults to `_conv` so its output never
overwrites the paper figures.

---

## Blink-comparison corner plots: `plot_corner_blink.py`

Standalone companion to Figure 6 (`bsys_corner_plot.pdf`) of
`paper_plots_c_v2.ipynb`.  That notebook draws **one** corner plot of the
sampled systematic amplitudes `b_sys`; this script draws **one corner plot per
chain length** — 100k and 250k Gibbs iterations by default — on identical axes,
so the figures can be blinked against one another and any change in the
posterior is a real change rather than a change of scale.

It imports nothing from the notebooks and modifies nothing: its `corner_plot`
is a copy of the notebook's with a `ranges` argument added.

### What is held fixed between the figures

| | How |
|---|---|
| Parameter ranges | `common_ranges()` takes the union over every variant and case, plus 5 % padding, and passes it to `corner.corner(range=...)` |
| Histogram bin edges | follow from the shared ranges and a shared `BINS` |
| Diagonal y-limits | `harmonise_diagonals()` levels them after drawing (`corner` scales each marginal to its own peak) |
| Contour levels, smoothing, colours | shared configuration |
| Image size on disk | `common_bbox()` saves every figure with the same bounding box, so the frames register pixel-for-pixel — `bbox_inches='tight'` alone would crop each figure to its own title widths |

### Configuration

Everything lives in the `CONFIG` block at the top of the file.

| Setting | Meaning |
|---|---|
| `RESULT_DIR_100K`, `RESULT_DIR_250K` | the two results directories the runs actually live in (`paper_plots/sim_data/` and `paper_plots/250k_run/`); `RESULT_DIR` is only the fallback for a variant that names neither |
| `FIG_DIR` | where the figures are written |
| `CASES` | runs overlaid **inside** each figure — `run`, optional `label`, optional `sys_amps_true` (truth lines are drawn from the first case that has them, as in the notebook) |
| `VARIANTS` | one figure per entry, i.e. what you blink between: `nsamples` (`None` = whole chain), `result_dir`, and optionally `label`, `tag`, and `runs` / `cases` |
| `BURN_MODE`, `BURN`, `THIN` | `'count'` drops the first `BURN` samples (the notebook default); `'percent'` drops the first `BURN` % of each chain, so both variants are burnt in proportionally |
| `NSIGMA`, `BINS`, `SMOOTH`, `RANGE_PAD` | contour levels, shared binning, smoothing, range padding |
| `FIG_STEM`, `FIG_SUFFIX`, `FORMATS`, `DPI` | output naming — `bsys_corner_100k_blink.pdf`, `..._250k_blink.pdf`, … ; `_blink` keeps them clear of the paper figures |
| `MAKE_GIF`, `GIF_MS` | also write `bsys_corner_blink.gif`, alternating between the frames |

### Runs in different directories

The 100k and 250k runs are separate runs in separate directories, so **each
variant carries its own `result_dir`** and the defaults point at both.  Nothing
is shared between the variants except the plot limits.

```python
VARIANTS = [
    dict(nsamples=100_000, result_dir=RESULT_DIR_100K),
    dict(nsamples=250_000, result_dir=RESULT_DIR_250K),
]
```

- `nsamples=None` (or `all` on the command line) reads whatever the directory
  holds, which is usually what you want when the directory *is* the 100k or
  250k run.  Give a `tag` (`'100k'`, `'250k'`) and it names both the files and
  the figure title, whatever the exact sample count turns out to be.
- Comparing two truncations of one chain still works: give both variants the
  same `result_dir` and different `nsamples`.
- If the two directories do not use the same run sub-folder names, a variant
  can override them with `runs=['...', ...]` (labels and true amplitudes are
  taken from `CASES` by position) or with full `cases=[dict(...), ...]`.

Cases whose `b-sys.npy` has a different number of systematic modes from the
first case are skipped with a warning, since one corner figure is a single
square grid.

### Usage

```bash
# Default 100k vs 250k comparison, one directory each; PDF + PNG + blink GIF
conda run -n py10 python plot_corner_blink.py --save

# The same, with both directories named on the command line.
# --variant takes "N DIR [TAG]"; N may be "all" for the whole chain.
conda run -n py10 python plot_corner_blink.py --save \
    --variant all /nvme2/scratch/sohini/hydra-pspec-systematic/paper_plots/sim_data/ 100k \
    --variant all /nvme2/scratch/sohini/hydra-pspec-systematic/paper_plots/250k_run/ 250k

# Three truncations of the chains in one directory (--nsamples uses --result-dir)
conda run -n py10 python plot_corner_blink.py \
    --result-dir /path/to/paper_plots/250k_run/ \
    --nsamples 50000 --nsamples 100000 --nsamples 250000 --save

# Proportional burn-in instead of the notebook's fixed 10 samples
conda run -n py10 python plot_corner_blink.py --burn-mode percent --burn 10 --save

# Drop the medians above the diagonals (they move between variants)
conda run -n py10 python plot_corner_blink.py --no-titles --save

# Check the machinery with synthetic chains, no run outputs needed
conda run -n py10 python plot_corner_blink.py --demo
```

From a notebook, the pieces can also be used directly:

```python
from plot_corner_blink import (read_variant, build_figures, CASES,
                               RESULT_DIR_100K, RESULT_DIR_250K)

variants = [dict(nsamples=None, result_dir=RESULT_DIR_100K, tag='100k'),
            dict(nsamples=None, result_dir=RESULT_DIR_250K, tag='250k')]
data = [read_variant(v, CASES, RESULT_DIR_100K) for v in variants]
figs = build_figures(data)
```

### Outputs

| File | Contents |
|---|---|
| `bsys_corner_<tag>_blink.pdf` / `.png` | one corner plot per chain length, all cases overlaid, shared axes |
| `bsys_corner_blink.gif` | the PNG frames alternating, for blinking in a browser |

### Tests

```bash
conda run -n py10 pytest tests/ -v
```

The suite runs entirely on synthetic chains (`make_demo_data`), so it needs
none of the run outputs.  It checks the sample handling, the shared ranges, the
levelled diagonals, the two-directory path (per-variant `result_dir`, `runs`
overrides and `--variant` parsing) and that the saved frames come out the same
size.

---

## Utility: `plot-test-data-results.py`

Plots the EoR delay power spectrum recovered from a single hydra-pspec run
against the true signal.

```bash
python plot-test-data-results.py \
    --vis-eor   path/to/vis-eor.uvh5 \
    --res-dir   path/to/results/0-1/ \
    --conf-interval 95 \
    --Nburn     0
```

---

## Utility: `plot_speed_up.py`

Plots MPI scaling (speed-up vs baselines/rank, and total time vs number of
ranks) from JSON timing files produced by a hydra-pspec run.

```bash
# Combine timing files from multiple run subdirectories and plot
python plot_speed_up.py --results_dir path/to/runs/

# Plot from an already-combined summary file
python plot_speed_up.py --summary_file path/to/combined_timings.json
```

---

## `plotting_functions.py` (root)

Used directly by `paper_plots_c.py` and the notebooks.

| Function | Returns | Description |
|---|---|---|
| `plot_waterfalls(data, freqs, times, ...)` | `(cax, data_fr_dly)` | FFT to delay–fringe-rate, then plot |
| `plot_waterfalls_from_dlfr(data_dlfr, freqs, times, ...)` | `cax` | Plot pre-transformed data |

Both functions accept `windows`, `mode`, `vmin`/`vmax`, `cmap`, `dynamic_range`,
`limit_drng`, `baseline`, `horizon_color`, `colorbar_flag`, `fontsize`, and
`labelsize` kwargs.

---

## `plotting_codes/functions.py`

Shared numerical utilities.

| Function | Description |
|---|---|
| `data_dly_fr(data, freqs, times, ...)` | 2D FFT: freq–time → delay–fringe-rate |
| `data_fr_dly_to_t_f(data_fr_dly, ...)` | Inverse of the above |
| `fourier_mode_2d(freqs_Hz, times_sec, modes)` | 2D orthonormal Fourier basis functions |
| `sys_modes(freqs_Hz, times_sec, modes)` | Systematic mode operator (Nfreqs×Ntimes, Nmodes) |
| `fourier_operator(n, unitary=True)` | Dense DFT matrix (fftshift convention) |
| `covariance_from_pspec(ps, fourier_op)` | Delay PS → frequency–frequency covariance |
| `form_pseudo_stokes_vis(uvd, ...)` | UVData XX+YY → pseudo-Stokes I |

---

## `plotting_codes/plotting_functions.py`

Extended waterfall and diagnostic helpers (more feature-rich than root
`plotting_functions.py`; used by supplementary analysis notebooks).

| Function | Description |
|---|---|
| `plot_waterfalls(...)` | Same as root version plus `fontsize`/`labelsize` kwargs |
| `plot_waterfalls_from_dlfr(...)` | Same as root version plus `fontsize`/`labelsize` kwargs |
| `plot_inputs(...)` | 2×4 grid: real/imag visibility components + systematics vector |
| `plot_matrices(...)` | H operator + 2×4 covariance matrix grid |
| `plot_results(...)` | Six comparison figures (vis, systematics, EoR, FG, sky, sys vector) |
| `master_plotter(...)` | Generic 3-row × N-column grid (real/imag/abs per dataset) |
| `plot_dps(...)` | Four DPS figures: true vs recovered, residuals, z-scores, error bars |

---

## Dependencies

```
numpy
matplotlib
scipy
astropy
pyuvdata
uvtools
hera_sim
hydra_pspec     # from ../hydra-pspec-systematic/ (or installed via pip install -e)
emcee
corner
cmcrameri
tqdm
pillow          # blink GIF in plot_corner_blink.py (ships with matplotlib)
pytest          # tests/ only
```

Install `hydra-pspec-systematic` in development mode (from its repo root) so
that `sys.path.append('../hydra-pspec-systematic/')` in `paper_plots_c.py`
resolves correctly, or adjust the path to your local checkout.
