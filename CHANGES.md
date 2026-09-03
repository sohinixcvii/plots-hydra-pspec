# Changelog

---

## 2026-09-03 — Raftery-Lewis run-length diagnostic in `convergence_tests.ipynb`

Added the Raftery-Lewis diagnostic alongside the existing trace / split-Rhat /
ESS sections.  Where those ask *has this chain converged?*, this one asks *how
long would this chain have to be?* — and answers it against the run that was
actually done.  Eleven cells inserted before the table-export section; the
notebook goes from 43 cells to 55.

**Why it is implemented rather than imported**

No library route exists in this environment, which was checked rather than
assumed:

- `pymc` is not installed, and the function no longer exists in it anyway.
  Checking the current release (5.26.0) rather than older references: `raftery`
  appears nowhere in the distribution, and `pymc/stats/__init__.py` delegates to
  ArviZ.  The PyMC2-era `pymc.raftery_lewis` is gone, not relocated.
- `arviz` is not installed, and 0.23.4 contains no `raftery` either.
- `rpy2` is not installed and **there is no R on this machine at all** (nothing
  on PATH, `/usr/local/bin`, `/opt/homebrew/bin` or `/Library/Frameworks`), so
  `coda::raftery.diag` would mean adding a whole R stack as a dependency of a
  plotting repo.
- No maintained pure-Python implementation exists: a PyPI search for "raftery
  lewis" returns zero hits.  The nearest candidate, `mcmc-diagnostics`
  ("Python analogue of the R coda package"), provides only ESS and a
  Cramér-von Mises test — no Raftery-Lewis — is a single 0.1 release from
  February 2020, and its own "coda" path just shells out to R via rpy2.
  `pymc-extras` and `pymcmcstat` were also checked: no `raftery` in either.

So it is written out directly from Raftery, A. E. & Lewis, S. M. (1992), "One
long run with diagnostics: Implementation strategies for Markov chain Monte
Carlo", *Statistical Science* **7**(4), 493-497, cited in the docstring.  No new
dependency: numpy and scipy only, which the notebook already used.

**Added**

- `raftery_lewis_nmin(q, r, s)` — iterations needed for independent draws.
- `_bic_first_vs_second_order(z)` — BIC test of first- against second-order
  binary Markov dependence, used to choose the thinning interval.
- `raftery_lewis(chain, q, r, s, eps, max_thin)` — the diagnostic itself,
  returning required burn-in `M`, run length `N`, total, `Nmin`, dependence
  factor `I = (M+N)/Nmin`, the thinning `k`, and an `ok`/`note` pair reporting
  why a chain could not be scored.  `q`, `r` and `s` are arguments with the
  standard defaults (0.025 / 0.005 / 0.95), not hard-coded values, because they
  are the question being asked.
- `RL_Q`, `RL_R`, `RL_S`, `RL_EPS`, `RL_MAX_THIN` in the configuration cell,
  next to the existing `CONV_*` settings.
- **Table C6** — required `M` and `M+N` per case and parameter block, each
  block summarised by its most demanding scalar, printed next to the burn-in
  and iteration count the run actually used, with a `required / run` column
  (≤ 1 = the run was long enough) and a verdict of
  long enough / burn-in short / too short.
- **Table C7** — the single scalar that sets each case's requirement.
- **Figure C4** (`raftery_lewis*.pdf`) — required iterations per block with each
  run length drawn as a dashed line (bars above it were not satisfied) and the
  `Nmin` floor marked; dependence factor `I` alongside, against the `I = 1` and
  `I = 5` references.

**Applied to the same chains as the existing cells** — the same monitored
blocks (`ln_post`, `b_sys`, `P_EoR` at the systematic delays, `a_fg`), the same
`real_columns` split of complex parameters, the same `cases` configuration and
the same files.  The one deliberate difference is that this section reads the
**full raw chains including burn-in**: the diagnostic estimates the burn-in a
run needed, which is only meaningful if it can see the start of the chain, and
it is what makes the required-vs-actual burn-in comparison possible.

**Testing**

- `Nmin` checked against the values published for the standard settings: 3746
  for `q=0.025, r=0.005, s=0.95`, and 38415 for `q=0.5` — both exact.
- An independent chain gives thinning `k=1` and `I = 1.00`; AR(1) chains with
  rho = 0.5 / 0.8 / 0.95 give monotonically increasing requirements
  (3748 → 8292 → 18988 → 56380 iterations, `I` = 1.0 → 15.1).
- `r` and `s` verified to move the answer in the right direction; constant,
  non-finite and too-short chains are reported via `ok`/`note` rather than
  raising.  These checks are in the notebook as a self-test cell, in the same
  style as the existing estimator self-tests.
- Whole notebook executed end to end with `jupyter nbconvert --execute` in three
  configurations (four cases with mixed mode counts at 30k samples; a single
  case; four cases at 4k samples, i.e. barely above `Nmin`).  All clean.  On the
  synthetic runs the recovered dependence factor tracked the injected AR(1)
  correlation exactly (rho = 0.6 → I = 2.5, 0.7 → 3.6, 0.85 → 6.4, 0.9 → 10.0),
  and the most autocorrelated case was correctly flagged "too short" at 30k.
- Runtime is ~5 ms per 100k-sample scalar, so under a second for a full
  four-case notebook.

**Unchanged**

- Every existing cell of `convergence_tests.ipynb`, including the outputs from
  the user's run against the real 100k-sample data, and the notebook's
  kernelspec.  The new cells were inserted in place rather than by regenerating
  the notebook.
- `plotting_codes/tables.py` — Tables C6 and C7 use its existing API.

---

## 2026-09-02 — convergence tests split into `convergence_tests.ipynb`

The convergence diagnostics of the `paper_plots_c_v2*` notebooks now also exist
as a standalone notebook that handles **one or more cases**, either as an
independent assessment of a single run or as a comparison and ranking of
several.  43 cells, 21 of them code.

**No mathematical logic was changed.**  Every estimator (`real_columns`,
`tau_int`, `split_rhat`, `geweke_z`, `convergence_table`, `_nanagg`), its
self-test, and every table and figure body was carried over from
`paper_plots_c_v2_nfgmodes.ipynb` unchanged.  What changed is the configuration
around them.

**Added**

- `convergence_tests.ipynb`, with the sections: estimators and their self-test,
  per-case diagnostics, Table 1 + Figure T1 (`tau`/ESS of `b_sys`), Table C1 +
  Figure C0 (dashboard), Tables C2/C3/C4 + the verdict callout, Table C5 +
  Figure C6 (bias against MCSE), Figures C1/C2/C3 (traces, autocorrelation,
  per-delay efficiency), and the table export.
- A `cases` list that describes each run by `run`, `label`, `nm_list`,
  `dl_inds`, `sys_amps_true` and `Nfgmodes`.  This covers the case definitions
  of all three source notebooks: the differing `nm_list_arr` / `dl_inds` of
  `paper_plots_c_v2.ipynb`, the differing `Nfgmodes` of
  `paper_plots_c_v2_nfgmodes.ipynb`, and the single run of
  `paper_plots_c_v2_single_case.ipynb`.
- A validation cell that resolves the defaults, reads `Nfgmodes` from each run's
  `fgmodes.npy`, and checks `nm_list` / `dl_inds` / `sys_amps_true` against the
  width of `b-sys.npy` before any diagnostic runs.
- `fig_suffix` (default `_conv`), so the figures never overwrite the paper ones.

**Generalised** (the reason the notebook works for any number of cases)

- Per-case systematic modes.  `Nsysmodes`, `dl_inds` and `sys_amps_true` were
  single shared values in the `Nfgmodes` notebook; they are now per-case arrays,
  so cases with different numbers of systematic modes can be compared.  Figure
  C6 plots the union of the amplitudes over all cases and simply leaves a gap
  where a case has no such mode.  Figure C3 shades the union of the systematic
  delays, and Figure C1 names each case's own first systematic delay in the
  legend.
- Per-case burn-in.  `CONV_BURN_PC` (default 10 %) with an absolute
  `CONV_BURN_ABS` override, replacing the single `CONV_BURN = Nburn`; the three
  source notebooks expressed burn-in in three different ways (10000 samples,
  `10` as a percent, and `int(Niter * Nburn_pc / 100)`).
- Single-case handling.  Table C3 (the ranking) is not built for one case, and
  the verdict callout drops the "best-converged case" wording.  Table C5 and
  Figure C6 are skipped for any case with no `sys_amps_true`.
- Table 1 accepts cases with different numbers of systematic amplitudes.
- Figure C1's plotted range is `TRACE_WINDOW` (default: the whole post-burn-in
  chain) instead of the hard-coded `20000:80000`, which assumed a 100k run, and
  the x-axis is now the absolute Gibbs-iteration number. `TRACE_RUNNING_MEAN`
  (default `False`) selects raw traces — what the `Nfgmodes` notebook actually
  plots — or the running means its markdown describes.

**Dropped from the imports**

- `pyuvdata`, `hera_sim`, `uvtools`, `corner`, `cmcrameri`, `tqdm`, `scipy.fft`
  and the `hydra_pspec` / `plotting_codes.functions` / `plotting_functions`
  helpers.  None of the diagnostics need the delay/fringe-rate machinery, so the
  notebook reads only `ln-post.npy`, `b-sys.npy`, `dps-eor.npy` and one LST
  slice of `fg-amps.npy` — no sample loop, a couple of minutes end to end.

**Unchanged**

- The `paper_plots_c_v2*` notebooks.  They keep their own Table 1 and, in the
  `Nfgmodes` notebook, their convergence section, so they stay self-contained;
  the README cross-references the new notebook from there.
- `plotting_codes/tables.py`.  The new notebook uses its existing API
  (`tau_ess`, `ess_table`, `plot_ess_tau`, `Table`/`Col`/`show`, `verdict`, the
  `flag_*` helpers, `annotated_heatmap`, `quality_cmap`, `callout`,
  `save_tables`) without modification.

**Testing**

Executed end to end with `jupyter nbconvert --execute` against synthetic runs
built from AR(1) chains of known `tau` centred on known true amplitudes, in four
configurations: four cases with mixed systematic-mode counts (4/4/4/12) and
foreground-mode counts (8/8/10/12); a single case; all cases with no
`sys_amps_true`; and `CONV_BURN_ABS` / `CONV_INCLUDE_FG=False` /
`TRACE_WINDOW` / `TRACE_RUNNING_MEAN=True` together.  All four ran clean, and
the recovered `tau` ordering and posterior means matched the injected values.

---

## 2026-08-24 — `paper_plots_c_v2_single_case.ipynb` cleaned up

The notebook was a copy of the three-case `paper_plots_c_v2.ipynb` with the case
loop removed, so it still carried the multi-case scaffolding and a second copy
of half its figures.  It is now a self-contained single-run notebook: 54 cells
down to 41.

**Removed**

- The duplicate `astropy` cache print (it appeared twice).
- The one-element `*_all` lists (`sky_true_dlfr_all`, `corr_maps`, …).  The
  statistics they wrapped are now plain arrays computed in the processing cell:
  `std_sky_dlfr`, `res_sky_dlfr`, `res_sys_dlfr`, `corr_map`.
- **Figures 5 v2, 6 v2 and 7 v2.**  With one case they plotted exactly the same
  arrays as Figures 5, 6 and 7, only stacked in one column instead of 2x2.
- `ps_data` and `ps_del_g`, two `calc_ps` calls per Gibbs iteration whose
  results were never used, together with the unused sample means
  (`eor_pspec_avg`, `fg_pspec_avg`, `sys_pspec_avg`, `data_pspec_avg`,
  `delg_pspec_avg`, `b_sys_mean`) — half the work in the 100k-iteration loop.
- Unused loads in the Figure 3 cell (`vis_eor`, `vis_sky`, `lst_eor`, `lst_sky`,
  `data_true_dlfr_full`) and ~20 unused imports (`emcee`, `hera_sim`, `scipy`,
  `cmcrameri`, the duplicate `import corner`, …).
- The trailing empty cell.

**Fixed**

- `case_idx` labelled the run "Case I" while `run_version` was `8fg_caseii_100k`
  and `nm_list` was Case II's modes.  `case_idx` and the three-element
  `run_version_arr` / `nm_list_arr` / `dl_inds` are gone; the run is now
  described by `run_version`, `case_label`, `nm_list` and `dl_ind`, with the
  other two cases kept as comments.
- `Nburn = 10  # burn-in (%)` was used as a sample index, so the delay power
  spectrum discarded 10 samples out of 100000 rather than 10 %.  It is now
  `Nburn_pc = 10` with `Nburn = int(Niter * Nburn_pc / 100)`, matching
  `paper_plots_c_v2_nfgmodes.ipynb`.  **This changes Figure 10.**
- The Figure 3 cell overwrote `eor_true`, `fg_true` and `sys_model_true` with
  different arrays than the processing cell uses, so re-running it silently
  invalidated Figures 4-11.  Those variables are now `*_f3`, and the cell reuses
  `sys_modes_operator` instead of rebuilding the operator.
- The sample loop mixed `fgmodes` (all modes) into `fg_vis` but
  `fgmodes[:, :Nfgmodes]` into `ps_fg`.  `fgmodes` is now truncated once at load
  and both use it; the amplitudes are truncated to match.
- `ps_fg` / `ps_sys` were allocated with `ps_sample.shape` but only filled for
  `Niter` rows, leaving uninitialised memory in Figure 7 if the chain on disk is
  longer than `Niter`.  They are now `(Niter, Nfreqs)` and the DPS cell slices
  `[Nburn:Niter]`.
- Figure 10 recomputed `calc_ps(n.T)` twice; it reuses `n_pspec`.
- The Table 1 / Figure T1 text still spoke of comparing cases.

**Refactored — the sample loop no longer holds the chain in memory**

- `sky_dlfr_arr`, `fg_dlfr_arr` and `delta_g_dlfr_arr` were
  `[Niter, Ntimes, Nfreqs]` complex arrays: 7.7 GB each at Niter = 100000, and
  `gcr-eor.npy` was loaded whole on top of them, ~32 GB in all.  The means,
  standard deviations and the FG-systematics Pearson map are now accumulated
  online with Welford updates, and `gcr-eor.npy` / `fg-amps.npy` are opened
  with `mmap_mode='r'` (`USE_MMAP`, as in `paper_plots_c_v2_nfgmodes.ipynb`).
  Peak memory is now ~100 MB, dominated by the per-iteration power spectra that
  Figure 7 needs in full.
- The accumulated statistics agree with the array-based ones to ~1e-14
  (float64 round-off).  The one exception is the Pearson map, where the online
  form is the *more* accurate of the two: the old code correlated
  `DLFR(1 + dg)`, whose constant `dlfr_ones` term swamps the per-sample
  fluctuation and costs ~8 significant figures to cancellation.  Pearson r is
  invariant to that constant, so the loop accumulates `DLFR(dg)` instead.

---

## 2026-08-24 — Presentable diagnostic tables and figures

The ESS / autocorrelation-time results and the whole convergence section used to
be printed with bare `print(array)` calls.  They are now rendered as styled
tables with an explicit colour code, each with a companion figure, so the output
can be pasted straight into an email, Slack or the paper.

### `plotting_codes/tables.py` (new file)

Shared, pandas-free rendering layer used by every plotting notebook.

- `Table` / `Col` — a table that renders itself as styled HTML (inline CSS only,
  so a copy-paste keeps its formatting), Markdown, a LaTeX `tabular` and CSV.
  Per-cell colour flags (green / amber / red) come from the column's `flag`.
- `ess_table(chains, ...)` — builds and displays the tau / ESS table for one or
  more chains; `plot_ess_tau(table)` is its companion figure (tau and ESS as
  grouped bars, with the ESS quality bands shaded).
- `tau_ess(chain)` — tau and ESS with **tau floored at one sample**.  Sokal
  windowing can return tau < 1, even negative, for an effectively independent
  chain, which previously produced negative "effective sample sizes" (see
  below).  ESS <= N now holds by construction.
- `annotated_heatmap`, `quality_cmap`, `callout`, `verdict`, the `flag_*`
  helpers, and `save_tables(dir)` which writes every table built in the session
  as `.md`, `.tex`, `.csv` and `.html`.
- Thresholds in one place: `ESS_GOOD = 1000`, `ESS_WARN = 400` (Vehtari et al.
  2021), `RHAT_OK = 1.01`, `GEWEKE_OK = 2`.

### `paper_plots_c_v2_nfgmodes.ipynb`

- **Table 1** — tau / ESS per systematic amplitude as a coloured table, plus
  **Figure T1** (`ess_tau_bsys_nfg.pdf`): tau and ESS as grouped bars.
- **Table C1** — per-case, per-block convergence diagnostics with a verdict
  column, plus **Figure C0** (`convergence_dashboard_nfg.pdf`): worst ESS/N,
  split-Rhat and |Geweke z| as annotated heat maps.
- **Tables C2 / C3 / C4** — convergence summary per case, the ranking (1 = best)
  and the bottleneck parameter of each case, with the take-home sentence in a
  coloured verdict box.
- **Table C5** + **Figure C6** (`bias_mcse_nfg.pdf`) — |bias|/sigma against
  MCSE/sigma for every systematic amplitude (model error vs chain-length error).
- **Bug fix**: `tau_int` now goes through `tbl.tau_ess`, so tau is floored at one
  sample.  Before this, blocks whose chains are effectively independent
  (`ln_post`, `P_EoR_sys`, `a_fg`) returned tau < 0 and hence *negative* ESS —
  which propagated into "minESS/N = -6.69" in the summary table and into the
  ranking, so the previously reported best-converged case was decided by
  meaningless numbers.  Re-run the notebook to get corrected values.
- New settings `SAVE_TABLES` and `table_dir` (default `fig_dir/tables`), plus an
  export cell that writes every table in all four formats.

### `paper_plots_c_v2.ipynb`, `paper_plots_c.ipynb`, `100k_runs_paper_plots.ipynb`, `paper_plots_c_v2_single_case.ipynb`

Table 1 replaced by `tbl.ess_table(...)` plus the `ess_tau_bsys.pdf` figure and
a `tbl.save_tables(fig_dir + '/tables')` export.  Same numbers, readable layout.

---

## 2026-08-20 — Convergence diagnostics in `paper_plots_c_v2_nfgmodes.ipynb`

New section after Table 1 that answers *for which `Nfgmodes` case did the Gibbs
sampler converge best?*  Purely additive: no existing cell was modified, and the
existing Table 1 (`emcee` ESS of `b_sys`) is left in place.

**New cells**
- **Convergence estimators** — `real_columns` (splits complex chains into real
  and imaginary scalar chains), `tau_int` (integrated autocorrelation time via
  `emcee.autocorr`, Sokal windowing, `tol=0` so short chains return a number
  rather than raising), `split_rhat` (rank-normalised split-Rhat, Vehtari et al.
  2021, over `CONV_NSPLIT` consecutive segments of the single chain),
  `geweke_z` (autocorrelation-corrected stationarity z-score) and
  `convergence_table` (tau, ESS, ESS/N, MCSE, sd, Rhat, z, `reliable = N > 50 tau`
  per scalar).  New settings `CONV_NSPLIT`, `CONV_BURN`, `CONV_GEWEKE_FIRST`,
  `CONV_GEWEKE_LAST`, `CONV_INCLUDE_FG`, `CONV_RHAT_OK`, `CONV_Z_OK`.
- **Self-test** — mirrors the existing online-estimator self-test: `tau_int`
  against an AR(1) chain with the exact `tau = (1+rho)/(1-rho)`, and
  `split_rhat`/`geweke_z` against an independent chain (must pass) and a
  drifting chain (must fail).
- **Per-case diagnostics** — monitors `ln_post`, `b_sys`, `P_EoR` in the delay
  bins occupied by the systematics, and `a_fg` at `fg_amp_time_idx`.  Reuses
  `ln_post_all` / `ps_sample_all` from the all-cases loop and reloads only
  `b-sys.npy` plus a single-LST slice of `fg-amps.npy`, so the section is cheap
  relative to the sample loop.  Also stores `ess_dps_all` (ESS per delay bin).
- **Summary and ranking** — per-case table (worst/median `ESS/N`, worst `ESS`,
  worst tau, worst Rhat, number of scalars with `Rhat > 1.01`, worst `|z|`,
  fraction failing `|z| < 2`) and a rank over three criteria — efficiency,
  mixing, stationarity — whose mean picks the best-converged case.  Scoring uses
  each case's **worst** monitored scalar, and `ESS/N` rather than raw `ESS` so
  cases with different `Niter` compare fairly.  Prints the slowest-mixing scalar
  per case and warns when even the best case has `Rhat > CONV_RHAT_OK`.
- **Convergence is not accuracy** — per-`b_sys`-component `|bias|/sd` (model
  property) alongside `MCSE/sd` and `ESS` (chain property).
- **Figure C1** `convergence_traces_nfg.pdf` — log-posterior traces (each case
  relative to its own median, since the absolute value depends on `Nfgmodes`)
  and running means of `Re b_sys,1` and of `P_EoR` at the first systematic delay.
- **Figure C2** `convergence_acf_ess_nfg.pdf` — `b_sys` autocorrelation function
  (log lag axis; the mixing times differ by orders of magnitude) and a grouped
  bar chart of the worst `ESS/N` per monitored block.
- **Figure C3** `convergence_ess_dps_nfg.pdf` — `ESS/N` of the sampled EoR delay
  power spectrum per delay bin, with the systematic delays marked.

**Validation**

The data (`/nvme2/scratch/...`) is not reachable from the machine the section was
written on, so all cells were executed against synthetic AR(1) chains with known
correlation lengths and a deliberate drift (a fake `result_dir` with `b-sys.npy`
and `fg-amps.npy` of the right shapes).  The ranking recovers the intended order,
the drifting case is flagged by both split-Rhat and Geweke, and all three figures
render.  The section still needs one run against the real chains.

**Documentation**
- `README.md`: new "Convergence diagnostics" subsection (statistic table,
  settings table, caveats) and Figures C1–C3 added to the outputs table.

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
