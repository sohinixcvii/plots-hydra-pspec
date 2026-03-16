# Cleanup Changes

Changes made during the 2026-03-12 cleanup pass.

---

## `plotting_functions.py` (root)

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

---

## `plotting_codes/plotting_functions.py`

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

---

## `plotting_codes/functions.py`

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

---

## `plotting_codes/paper_plots.py`

- Added a deprecation notice at the top of the file noting that it is
  superseded by `../paper_plots_c.py` and contains hardcoded paths.

---

## New files

### `.gitignore`
Added standard exclusions: `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`,
`paper_plots_c_output.txt`, `.DS_Store`.

### `README.md`
New self-explanatory README covering: repository layout, how to run
`paper_plots_c.py`, configuration variables, output file list, CLI tool
usage, and the dependency list.

### `CHANGES.md`
This file.
