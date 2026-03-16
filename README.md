# plots-hydra-pspec-systematic

Plotting scripts and helper libraries for the **hydra-pspec-systematic** paper.
The analysis characterises instrumental systematics in 21 cm power-spectrum
estimation using the Gibbs-sampling framework implemented in
[hydra-pspec](https://github.com/HydraRadio/hydra-pspec).

---

## Repository layout

```
.
├── paper_plots_c.py           # Main script — generates all paper figures
├── paper_plots_c.ipynb        # Notebook version (source of paper_plots_c.py)
├── plotting_functions.py      # Waterfall-plot helpers used by paper_plots_c.py
├── plot-test-data-results.py  # Standalone EoR delay power spectrum plotter (CLI)
├── plot_speed_up.py           # MPI speed-up / scaling plotter (CLI)
├── checking_result_files.ipynb# Notebook for sanity-checking result files
├── 100k_runs_paper_plots.ipynb# Notebook for 100k-iteration run plots
└── plotting_codes/            # Shared utility library
    ├── functions.py           # Fourier transforms, covariance helpers
    ├── plotting_functions.py  # Extended waterfall + diagnostic plot helpers
    ├── paper_plots.py         # SUPERSEDED — old plotting script (kept for reference)
    └── paper_plots.ipynb      # Notebook companion to the superseded script
```

---

## Main paper figure script: `paper_plots_c.py`

Generates Figures 1, 3–11 of the paper and saves them to `fig_dir` as PDF.
Console output (including tqdm progress bars) is also mirrored to
`paper_plots_c_output.txt`.

### Configuration

Edit the **Configuration** block near the top of `paper_plots_c.py`:

| Variable | Description |
|---|---|
| `result_dir` | Directory containing per-run subdirectories with `.npy` output files |
| `parent_dir` | Root of the `hydra-pspec-systematic` repository (for raw data files) |
| `fig_dir` | Output directory for generated figures |
| `run_version_arr` | List of run subdirectory names (one per test case) |
| `nm_list_arr` | Systematic mode (delay, fringe-rate) index pairs per test case |
| `case_idx` | Which test case to use for single-case figures (0, 1, or 2) |
| `Niter` | Number of MCMC samples |
| `Nburn` | Burn-in samples to discard |
| `conf_interval` | Confidence interval (%) for posterior credible regions |

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
| 1 | `systematics_in_vis_space.pdf` | Corrupted vs clean visibilities |
| 3 | `test_cases_in_dlfr.pdf` | All three test cases in delay–fringe-rate space |
| 4 | `data_vs_sky_2.pdf` | Data, sky mean, systematics mean for all cases |
| 5 | `result_waterfalls.pdf` | True/mean/std/residual sky (single case) |
| 6 | `sys_result_dlfr_waterfalls.pdf` | Systematics recovery (single case) |
| 7 | `errors_components.pdf` | Fractional errors by component |
| 8 | `bsys_corner_plot.pdf` | Corner plot of systematic amplitudes |
| 9 | `sys_frac_error.pdf` | Systematic amplitude recovery |
| 10 | `delay_power_spectrum_combined.pdf` | Delay power spectra for all cases |
| 11 | `fg_sys_corr.pdf` | Foreground–systematics Pearson correlation |

Table 1 (ESS and autocorrelation times) is printed to stdout.

---

## Utility: `plot-test-data-results.py`

Plots the EoR delay power spectrum recovered from a single hydra-pspec run
against the true signal.

```bash
python plot-test-data-results.py \
    --vis-eor path/to/vis-eor.uvh5 \
    --res-dir path/to/results/0-1/ \
    --conf-interval 95 \
    --Nburn 0
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

Used directly by `paper_plots_c.py`.

| Function | Description |
|---|---|
| `plot_waterfalls(data, freqs, times, ...)` | FFT to delay–fringe-rate, then plot |
| `plot_waterfalls_from_dlfr(data_dlfr, freqs, times, ...)` | Plot pre-transformed data |

---

## `plotting_codes/functions.py`

Shared numerical utilities.

| Function | Description |
|---|---|
| `data_dly_fr(data, freqs, times, ...)` | 2D FFT: time–freq → delay–fringe-rate |
| `data_fr_dly_to_t_f(data_fr_dly, ...)` | Inverse of the above |
| `fourier_mode_2d(freqs_Hz, times_sec, modes)` | 2D Fourier basis functions |
| `sys_modes(freqs_Hz, times_sec, modes)` | Systematic mode operator matrix |
| `fourier_operator(n, unitary=True)` | Dense DFT matrix |
| `covariance_from_pspec(ps, fourier_op)` | Delay PS → frequency–frequency covariance |
| `form_pseudo_stokes_vis(uvd, ...)` | XX+YY → pseudo-Stokes I |

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
hydra_pspec          # from ../hydra-pspec-systematic/
emcee
corner
cmcrameri
tqdm
```

Install the `hydra-pspec-systematic` package in development mode so that the
`sys.path.append('../hydra-pspec-systematic/')` line in `paper_plots_c.py`
resolves correctly, or adjust the path to your local checkout.
