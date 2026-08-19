# Changelog

---

## 2026-08-19 — Notebook for varying the number of foreground modes

### `paper_plots_c_v2_nfgmodes.ipynb` (new file)

Reproduces every analysis step of `paper_plots_c_v2.ipynb` for cases that share
the **same** systematic-mode locations (the Case II values,
`[(10, 0), (11, 0), (12, 0), (13, 0)]`) and differ only in `Nfgmodes`.

**Case handling**
- `nm_list` is a single shared list; `sys_modes_operator`, `delta_g_true` and
  the true sky/gain/data are built once instead of per case.
- `Ncases = len(run_version_arr)` throughout — every figure grid, colour list,
  hatch list and loop takes any number of cases ≥ 2 instead of a hard-coded 3.
- `Nfgmodes_arr` gives the foreground modes per case; `None` entries are filled
  in from each run's `fgmodes.npy`.  `fgmodes` and `fg-amps` are sliced to that
  value consistently everywhere (v2 mixed the full and sliced bases between the
  power spectra and the visibility model).
- New **Available runs** cell lists the sub-directories of `result_dir` with the
  `Nfgmodes` in each `fgmodes.npy` and any missing result files.
- New validation cell resolves `Nfgmodes_arr`, warns on duplicate values, and
  builds the per-case labels/colours.
- Cross-checks that every run really does share the same `eor_true`, `fg_true`
  and `gain_true`, and that `gain_true == 1 + H b_sys_true`.

**Efficiency / correctness**
- The v2 single-case loop and all-cases loop are merged into **one** pass per
  case; the single-case figures index the per-case lists instead of re-running
  the samples.
- Statistics come from `OnlineMoments` (Welford) and `OnlinePearson` (centred
  co-moment).  v2's cell-15 declared online accumulators but still allocated
  four `[Niter, Ntimes, Nfreqs]` arrays (`fgd_all`, `sysd_all`, `sky_d_arr`,
  `dg_d_arr`) and computed from those — 76.8 GB each at `Niter = 100000`.
- `gcr-eor.npy` and `fg-amps.npy` are memory-mapped (`USE_MMAP`).
- The Pearson accumulator uses the centred co-moment update rather than
  "sum of squares minus square of sums", which cancels catastrophically here.
- The foreground–gain correlation is accumulated against `DLFR(Hb_sys)` instead
  of `dlfr_ones + DLFR(Hb_sys)`.  Pearson *r* is invariant under the constant,
  and `dlfr_ones` is a spike of `Ntimes*Nfreqs` at the zero-delay/zero-fringe-rate
  pixel that otherwise swamps the sample scatter there.
- Helper functions are defined **before** the derived-parameters cell.  In
  `paper_plots_c_v2.ipynb`, cell-6 calls `compute_dlfr` (defined in cell-8) and
  uses `times_jd` (assigned later in cell-6 itself), so that cell cannot be run
  top-to-bottom in a fresh kernel.
- The EoR residual panel divides by the **per-delay-bin** sample scatter; v2
  divided by a single global scalar `np.std(ps_sample)` while labelling the axis
  as a per-bin z-score.
- Notebook self-tests `OnlineMoments`/`OnlinePearson` against the direct
  calculation before using them.

**Figures**
- Nine figures, all suffixed `_nfg` and written to `fig_dir` (default
  `.../Figures/nfgmodes`) so they never overwrite the main paper figures.
  `SAVE_FIGS` controls writing; the saves that were commented out in v2
  (`errors_components`, `delay_power_spectrum_combined`, `fg_sys_corr`) are
  active here.
- Figure 2 re-cast: v2 showed the true data once per case, which distinguished
  the cases only because they had different systematic-mode locations.  It now
  shows the shared true sky and true data followed by the mean **recovered**
  data model per `Nfgmodes`.
- Figure 5 (component errors) and Figure 7 (delay power spectra) overlay all
  cases; the shared systematic-delay band and lines are drawn once, and the
  zoom window is derived from `nm_list` instead of being hard-coded.
- Figure 9 uses per-case `param_sizes = [Nfgmodes_i, Nsysmodes]` (v2 hard-coded
  `[10, 4]`), and restores the global font size afterwards.

### `README.md`

- Added `paper_plots_c_v2_nfgmodes.ipynb` to the repository layout and a section
  describing its case definition, configuration, outputs and caveats.

---

## 2026-03-26 — Documentation update

### `README.md`

**Updated**
- Added `paper_plots_c_v2.ipynb` to the repository layout table.
- Added **Optimised notebook** section describing v1 vs v2 memory and
  correctness differences (144 GB → 144 MB; Welford/online-Pearson statistics;
  per-case `sys_modes_operator` fix).
- Expanded **Outputs** table to include v2 figures (5 v2, 6 v2, 7 v2).
- Added **Expected result directory layout** block listing all required `.npy`
  files and their shapes.
- Added `fontsize`/`labelsize` kwargs to `plotting_functions.py` function table.
- Added `plotting_codes/plotting_functions.py` function table with all diagnostic
  functions (`plot_inputs`, `plot_matrices`, `plot_results`, `master_plotter`,
  `plot_dps`).
- Added `Figures_sim_data/` to the repository layout tree.

### `CHANGES.md`

- Added this entry.

---

## 2026-03-16 — Efficiency improvements to plotting notebook

### `paper_plots_c_v2.ipynb` (new file)

Refactored version of `paper_plots_c.ipynb` with major performance and
correctness improvements.

**Efficiency**
- Gibbs iteration loop (cell-18) collapsed from 3 separate passes into a
  single pass over `Niter`; only the three delay-fringe-rate arrays
  (`sky_dlfr_arr`, `fg_dlfr_arr`, `delta_g_dlfr_arr`) are retained in memory.
- All-cases loop (cell-20) uses **Welford online mean/variance** and **online
  Pearson correlation** so the full sample arrays never need to be stored.
- `lsts_sec`, `dlfr` (lambda wrapping `data_dly_fr` with Blackman–Harris
  window), and `dlfr_ones` (= `dlfr(ones)`) are computed once in cell-6 and
  reused throughout; avoids redundant FFT calls on constant arrays.

**Bug fixes**
- Old cell-43 incorrectly used the single `sys_modes_operator` (built for
  `case_idx`) when plotting all three cases. Each case now uses its own
  operator, stored in `corr_maps` and looked up by index.

---

## 2026-03-12 — Code cleanup and waterfall function fixes

### `plotting_codes/plotting_functions.py` — waterfall functions

**Added**
- `fontsize` and `labelsize` keyword arguments to `plot_waterfalls` so callers
  can control axis-label and tick-label sizes without monkey-patching rcParams.

### `plotting_functions.py` (root)

**Removed**
- Dead imports: `os`, `scipy.stats`, `pyuvdata.UVData`, `pathlib.Path`,
  `matplotlib.colors.LogNorm/Normalize`.
- Commented-out `hera_sim` import block.
- Unused module-level variable `op_dir = 'paper_plots/'`.
- Large commented-out conditional block (the original 2×2 waterfall grid
  logic, ~25 lines) in `plot_waterfalls`.
- Commented-out tick-params grid options.
- Redundant `j = 1 / column = j % 2 / row = j // 2` pattern (these were
  always constant; replaced with direct values).

**Fixed**
- `if ax == None:` → `if ax is None:` (identity test for objects, not
  equality) in both functions.
- Added missing `from astropy import constants` (required when `baseline` is
  passed and the geometric horizon is computed).
- `cbar_label` logic in `plot_waterfalls_from_dlfr`: added an `else` branch
  so `mode='abs'` with no caller-supplied label falls back to
  `"Amplitude [Jy Hz s]"` rather than silently passing `None` to
  `colorbar.set_label`.

**Improved**
- Full NumPy-style docstrings on both functions.
- Simplified the dynamic-range clipping logic to a one-liner (`clip_drng`
  reflects that this function always plots the delay–fringe-rate quadrant).

### `plotting_codes/plotting_functions.py` — library cleanup

**Removed**
- Dead imports: `scipy.stats`, `pathlib.Path`, `matplotlib.colors.LogNorm/Normalize`.
  (`os`, `UVData`, `Path`, `LogNorm`, `Normalize` are kept where actually
  used by `plot_inputs`, `plot_matrices`, `plot_results`, `plot_dps`, and
  `master_plotter`.)
- Commented-out `hera_sim` import block.
- Redundant `j / column / row` constants in `plot_waterfalls` and
  `plot_waterfalls_from_dlfr` (same fix as above).
- No-op assignments `vmin = vmin` / `vmax = vmax` in `plot_waterfalls`.

**Fixed**
- `if ax == None:` → `if ax is None:` in both waterfall functions.
- Added `from astropy import constants` for baseline/horizon computation.
- `os.path.isdir(...) == False` → `os.makedirs(..., exist_ok=True)` in
  `plot_inputs`, `plot_matrices`, `plot_results` (more Pythonic and avoids
  TOCTOU race).

**Improved**
- Added missing docstring to `plot_waterfalls_from_dlfr`.
- Full docstrings on all functions.

### `plotting_codes/functions.py`

**Removed**
- Commented-out `cost_fn` function block (~10 lines).
- Unused imports: `sys`, `Quantity` (from `astropy.units`),
  `waterfall` (from `uvtools.plot`).
- Debug `# print(nf, nt)` and `# print(kfreq[...], ktime[...])` comments
  in `fourier_mode_2d`.
- Commented-out intermediate FFT lines (`data_fr`, `data_dly`) in
  `data_dly_fr`.
- Redundant `if windows is not None / else` block in `data_dly_fr`;
  `gen_window(None, n)` already returns a boxcar, so both branches were
  identical.

**Fixed**
- Corrected the docstring of `data_dly_fr`, which was copied from the old
  waterfall function and incorrectly described "a 2×2 grid of plots".
- Replaced `%` string formatting with f-strings in assertion messages in
  `fourier_mode_2d`.

**Improved**
- Full NumPy-style docstrings on all functions.
- Consistent reshape style: `[:, None]` / `[None, :]` for broadcasting.

### `plotting_codes/paper_plots.py`

- Added a deprecation notice noting it is superseded by `../paper_plots_c.py`
  and contains hardcoded paths.

### New files

- **`.gitignore`** — standard exclusions: `__pycache__/`, `*.pyc`,
  `.ipynb_checkpoints/`, `paper_plots_c_output.txt`, `.DS_Store`.
- **`README.md`** — covers repository layout, how to run `paper_plots_c.py`,
  configuration variables, output file list, CLI tool usage, and dependencies.
- **`CHANGES.md`** — this file.

---

## 2026-03-11 — Notebook-to-script conversion

### `paper_plots_c.py` (new file)

`paper_plots_c.ipynb` converted to a standalone Python script (856 lines).
Produces all paper figures (Figures 1, 3–11) from the command line without
requiring a Jupyter kernel.

---

## 2026-03-01 — Removed redundant notebooks and scripts

**Deleted**
- `clean_paper_plots.ipynb` — superseded by `paper_plots_c.ipynb`.
- `paper_plots.ipynb` — old monolithic notebook.
- `paper_plots.py` — old script derived from the above.
- `paper_plots_andromeda.ipynb` — machine-specific notebook with hardcoded
  Andromeda paths.

**Updated**
- `100k_runs_paper_plots.ipynb` — brought up to date with current data layout.
- `paper_plots_c.ipynb` — extended with additional figure cells.
