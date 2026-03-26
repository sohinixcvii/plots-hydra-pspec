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
├── plotting_functions.py       # Waterfall-plot helpers used by paper_plots_c.py
├── plot-test-data-results.py   # Standalone EoR delay power spectrum plotter (CLI)
├── plot_speed_up.py            # MPI speed-up / scaling plotter (CLI)
├── checking_result_files.ipynb # Notebook for sanity-checking result files
├── 100k_runs_paper_plots.ipynb # Notebook for 100k-iteration run plots
├── Figures_sim_data/           # Output directory for generated figures
│   ├── systematics_in_vis_space.pdf
│   └── test_cases_in_dlfr.pdf
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
```

Install `hydra-pspec-systematic` in development mode (from its repo root) so
that `sys.path.append('../hydra-pspec-systematic/')` in `paper_plots_c.py`
resolves correctly, or adjust the path to your local checkout.
