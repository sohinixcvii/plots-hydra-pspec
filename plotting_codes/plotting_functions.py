import matplotlib.pyplot as plt
import numpy as np
from astropy import constants, units
from matplotlib.colors import LogNorm, Normalize
from uvtools.dspec import gen_window
from uvtools.utils import FFT, fourier_freqs
from uvtools.plot import waterfall
from pyuvdata import UVData
from pathlib import Path
import scipy.stats as sci_st
import os


def plot_inputs(eor_vis, fg_vis, vis, vis_sys, bsys_test, y_test):
    """Plot input visibilities and systematics vectors as diagnostic grids."""
    input_plot_dir = output_dir_path + 'input_plots/'
    os.makedirs(input_plot_dir, exist_ok=True)

    # 2×4 grid: real (top row) and imaginary (bottom row) parts of each visibility
    fig, ax = plt.subplots(2, 4, figsize=(56, 14))

    im = ax[0, 0].imshow(eor_vis.real, aspect='auto')
    ax[0, 0].set_title("EoR visibilities")
    ax[0, 0].set_ylabel("Real Part")
    plt.colorbar(im)

    im = ax[0, 1].imshow(fg_vis.real, aspect='auto')
    ax[0, 1].set_title("FG Visibilities")
    plt.colorbar(im)

    im = ax[0, 2].imshow(vis.real, aspect='auto')
    ax[0, 2].set_title("Clean sky Visibilities")
    plt.colorbar(im)

    im = ax[0, 3].imshow(vis_sys.real, aspect='auto')
    ax[0, 3].set_title("Visibilities with systematics")
    plt.colorbar(im)

    im = ax[1, 0].imshow(eor_vis.imag, aspect='auto')
    ax[1, 0].set_ylabel("Imaginary Part")
    plt.colorbar(im)

    im = ax[1, 1].imshow(fg_vis.imag, aspect='auto')
    plt.colorbar(im)

    im = ax[1, 2].imshow(vis.imag, aspect='auto')
    plt.colorbar(im)

    im = ax[1, 3].imshow(vis_sys.imag, aspect='auto')
    plt.colorbar(im)

    fig.suptitle("Input visibilities", ha='center', va='top')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(input_plot_dir + 'Input_visibility_plots.png', bbox_inches='tight', dpi=300)

    # 1×4 grid: systematics vector (1D) and matrix (2D)
    fig, ax = plt.subplots(1, 4, figsize=(21, 7))

    ax[0].plot(bsys_test.real, marker='.')
    ax[0].set_title("Real part")

    ax[1].plot(bsys_test.imag, marker='.')
    ax[1].set_title("Imaginary part")

    im = ax[2].matshow(y_test.real, aspect='auto')
    ax[2].set_title("Real part")
    plt.colorbar(im)

    im = ax[3].matshow(y_test.imag, aspect='auto')
    ax[3].set_title("Imaginary part")
    plt.colorbar(im)

    fig.suptitle("The systematics vector", ha='center', va='top')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(input_plot_dir + 'Input_systematics_plots.png', bbox_inches='tight', dpi=300)


def plot_matrices(h_j, nm_list, signal_S, fgmodes, N_cov, Ninv):
    """Plot the H operator and all covariance/inverse matrices."""
    input_plot_dir = output_dir_path + 'input_plots/'
    os.makedirs(input_plot_dir, exist_ok=True)

    # H operator: real and imaginary parts
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    nm_ticks = [str(nm) for nm in nm_list]
    tick_locs = np.linspace(0, len(nm_list) - 1, len(nm_list))

    im0 = ax[0].matshow(h_j.real, origin='lower', aspect='auto')
    ax[0].set_title('Real part')
    ax[0].set_xlabel('Modes')
    ax[0].set_xticks(tick_locs, labels=nm_ticks)
    plt.colorbar(im0)

    im1 = ax[1].matshow(h_j.imag, origin='lower', aspect='auto')
    ax[1].set_title('Imaginary part')
    ax[1].set_xlabel('Modes')
    ax[1].set_xticks(tick_locs, labels=nm_ticks)
    plt.colorbar(im1)

    fig.suptitle("H operator", ha='center', va='bottom', fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(input_plot_dir + 'H_operator.png', bbox_inches='tight', dpi=300)

    # Covariance matrices: 2×4 grid (real top, imaginary bottom)
    fig, ax = plt.subplots(2, 4, figsize=(45, 18))

    im0 = ax[0, 0].imshow(signal_S.real, origin='lower', cmap='PuRd')
    ax[0, 0].set_title("EoR cov", fontsize=20)
    ax[0, 0].set_ylabel("Real part", fontsize=20)
    plt.colorbar(im0)

    im1 = ax[0, 1].imshow(fgmodes.real, origin='lower', cmap='PuRd')
    ax[0, 1].set_title("Foreground covariance", fontsize=20)
    plt.colorbar(im1)

    im2 = ax[0, 2].imshow(N_cov.real, origin='lower', cmap='PuRd')
    ax[0, 2].set_title("Noise covariance", fontsize=20)
    plt.colorbar(im2)

    im3 = ax[0, 3].imshow(Ninv.real, origin='lower', cmap='PuRd')
    ax[0, 3].set_title("Noise cov inverse", fontsize=20)
    plt.colorbar(im3)

    im0 = ax[1, 0].imshow(signal_S.imag, origin='lower', cmap='PuRd')
    ax[1, 0].set_ylabel("Imaginary part", fontsize=20)
    plt.colorbar(im0)

    im1 = ax[1, 1].imshow(fgmodes.imag, origin='lower', cmap='PuRd')
    plt.colorbar(im1)

    im2 = ax[1, 2].imshow(N_cov.imag, origin='lower', cmap='PuRd')
    plt.colorbar(im2)

    im3 = ax[1, 3].imshow(Ninv.imag, origin='lower', cmap='PuRd')
    plt.colorbar(im3)

    fig.suptitle("Plotting all the matrices from file", ha='center', va='bottom', fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.savefig(input_plot_dir + 'matrices_from_file.png', bbox_inches='tight', dpi=300)


def plot_results(vis, vis_sys, model, y_test, sys_model, eor_vis, signal_cr, fg_sol, fg_test, bsys_test, b_sys):
    """Plot solver results against ground truth for all model components."""
    result_plot_dir = output_dir_path + 'result_plots/'
    os.makedirs(result_plot_dir, exist_ok=True)

    # Visibility comparison: simulated, solved, residuals
    fig, ax = plt.subplots(2, 3, figsize=(27, 15))

    im0 = ax[0, 0].imshow((vis_sys).real, origin='lower', cmap='PuRd')
    ax[0, 0].set_title('Visibilities (simulated)')
    ax[0, 0].set_ylabel('Real Part')
    plt.colorbar(im0)

    im1 = ax[0, 1].imshow(model.real, origin='lower', cmap='PuRd')
    ax[0, 1].set_title('Visibilities (solved)')
    plt.colorbar(im1)

    im2 = ax[0, 2].imshow((model - vis_sys).real, origin='lower', cmap='PuRd')
    ax[0, 2].set_title('Residuals')
    plt.colorbar(im2)

    im0 = ax[1, 0].imshow((vis_sys).imag, origin='lower', cmap='PuRd')
    ax[1, 0].set_ylabel('Imaginary Part')
    plt.colorbar(im0)

    im1 = ax[1, 1].imshow(model.imag, origin='lower', cmap='PuRd')
    plt.colorbar(im1)

    im2 = ax[1, 2].imshow((model - vis_sys).imag, origin='lower', cmap='PuRd')
    plt.colorbar(im2)

    fig.suptitle("Comparison of test and solved visibilities", ha='center', va='bottom', fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(result_plot_dir + 'visibilities_test_vs_solved.png', bbox_inches='tight', dpi=300)

    # Systematics comparison
    fig, ax = plt.subplots(2, 3, figsize=(27, 15))

    im3 = ax[0, 0].imshow(y_test.real, origin='lower', cmap='PuRd')
    ax[0, 0].set_title('Systematics (simulated)')
    ax[0, 0].set_ylabel('Real Part')
    plt.colorbar(im3)

    im4 = ax[0, 1].imshow(sys_model.real, origin='lower', cmap='PuRd')
    ax[0, 1].set_title('Systematics (solved)')
    plt.colorbar(im4)

    im5 = ax[0, 2].imshow((sys_model - y_test).real, origin='lower', cmap='PuRd')
    ax[0, 2].set_title('Residuals')
    plt.colorbar(im5)

    im3 = ax[1, 0].imshow(y_test.imag, origin='lower', cmap='PuRd')
    ax[1, 0].set_ylabel('Imaginary Part')
    plt.colorbar(im3)

    im4 = ax[1, 1].imshow(sys_model.imag, origin='lower', cmap='PuRd')
    plt.colorbar(im4)

    im5 = ax[1, 2].imshow((sys_model - y_test).imag, origin='lower', cmap='PuRd')
    plt.colorbar(im5)

    fig.suptitle("Comparison of systematics data and solution", ha='center', va='bottom', fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(result_plot_dir + 'systematics_test_vs_solved.png', bbox_inches='tight', dpi=300)

    # EoR comparison
    fig, ax = plt.subplots(2, 3, figsize=(28, 14))

    im0 = ax[0, 0].imshow(eor_vis.real)
    plt.colorbar(im0)
    ax[0, 0].set_ylabel("Real part")
    ax[0, 0].set_title("EoR test")

    im1 = ax[0, 1].imshow(signal_cr.real)
    plt.colorbar(im1)
    ax[0, 1].set_title("EoR solved")

    im1 = ax[0, 2].imshow(signal_cr.real - eor_vis.real)
    plt.colorbar(im1)
    ax[0, 2].set_title("Residuals")

    im2 = ax[1, 0].imshow(eor_vis.imag)
    plt.colorbar(im2)
    ax[1, 0].set_ylabel("Imaginary part")

    im3 = ax[1, 1].imshow(signal_cr.imag)
    plt.colorbar(im3)

    im3 = ax[1, 2].imshow(signal_cr.imag - eor_vis.imag)
    plt.colorbar(im3)

    fig.suptitle("Comparison of EoR test and solution", ha='center', va='bottom', fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(result_plot_dir + 'eor_test_vs_solved.png', bbox_inches='tight', dpi=300)

    # Foreground comparison
    fig, ax = plt.subplots(2, 3, figsize=(28, 14))

    im0 = ax[0, 0].matshow(fg_test.real)
    plt.colorbar(im0)
    ax[0, 0].set_ylabel("Real part")
    ax[0, 0].set_title("FG test")

    im1 = ax[0, 1].matshow(fg_sol.real)
    plt.colorbar(im1)
    ax[0, 1].set_title("FG solved")

    im1 = ax[0, 2].matshow(fg_test.real - fg_sol.real)
    plt.colorbar(im1)
    ax[0, 2].set_title("Residuals")

    im2 = ax[1, 0].matshow(fg_test.imag)
    plt.colorbar(im2)
    ax[1, 0].set_ylabel("Imaginary part")

    im3 = ax[1, 1].matshow(fg_sol.imag)
    plt.colorbar(im3)

    im3 = ax[1, 2].matshow(fg_test.imag - fg_sol.imag)
    plt.colorbar(im3)

    fig.suptitle("Comparison of FG test and solution", ha='center', va='bottom', fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(result_plot_dir + 'FG_test_vs_solved.png', bbox_inches='tight', dpi=300)

    # Full sky model comparison
    sky_sol = signal_cr + fg_sol

    fig, ax = plt.subplots(2, 3, figsize=(28, 14))

    im0 = ax[0, 0].imshow(vis.real)
    plt.colorbar(im0)
    ax[0, 0].set_ylabel("Real part")
    ax[0, 0].set_title("Sky model test")

    im1 = ax[0, 1].imshow(sky_sol.real)
    plt.colorbar(im1)
    ax[0, 1].set_title("Solved sky model")

    im1 = ax[0, 2].imshow(vis.real - sky_sol.real)
    plt.colorbar(im1)
    ax[0, 2].set_title("Residuals")

    im2 = ax[1, 0].imshow(vis.imag)
    plt.colorbar(im2)
    ax[1, 0].set_ylabel("Imaginary part")

    im3 = ax[1, 1].imshow(sky_sol.imag)
    plt.colorbar(im3)

    im3 = ax[1, 2].imshow(vis.imag - sky_sol.imag)
    plt.colorbar(im3)

    fig.suptitle("Comparison of sky model test and solution", ha='center', va='bottom', fontsize=15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(result_plot_dir + 'sky_model_test_vs_solved.png', bbox_inches='tight', dpi=300)

    # Systematics vector: 1D comparison
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    x = np.arange(len(bsys_test))

    ax[0].plot(x, bsys_test.real, 'rx', label='x_true')
    ax[0].plot(x, b_sys.real, 'b.', label='x_solution')
    ax[0].legend()
    ax[0].set_title("Real part")
    ax[0].set_xlabel("Indices")
    ax[0].set_ylabel("Values")

    ax[1].plot(x, bsys_test.imag, 'rx', label='x_true')
    ax[1].plot(x, b_sys.imag, 'b.', label='x_solution')
    ax[1].legend()
    ax[1].set_title("Imaginary part")
    ax[1].set_xlabel("Indices")
    ax[1].set_ylabel("Values")

    fig.suptitle("Comparison of systematics vector: solved and true", ha='center', va='bottom')
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig(result_plot_dir + 'sys_vector_test_vs_solved.png', bbox_inches='tight', dpi=300)


def master_plotter(
    data_sets,
    col_labels=None,
    fig_title='Data comparison',
    plot_type='imshow',
    norm='linear',
    save_flag=True,
    cmap='seismic',
    dir='../paper_plots/',
    imag_flag=True,
    vmin=None,
    vmax=None,
    show=False
):
    """
    Plot a list of 2D complex arrays in a grid, showing real, imaginary, and
    absolute components.

    Parameters
    ----------
    data_sets : list of 2D ndarray
        Complex-valued arrays to plot. Each entry occupies one column.
    col_labels : list of str, optional
        Column headers. Defaults to "Data 0", "Data 1", ...
    fig_title : str, optional
        Figure title and (when saving) filename stem. Default: 'Data comparison'.
    plot_type : {'imshow', 'matshow'}, optional
        Matplotlib function for each subplot. Default: 'imshow'.
    norm : {'linear', 'log'} or Normalize, optional
        Color normalization. Default: 'linear'.
    save_flag : bool, optional
        Save the figure to ``dir``. Default: True.
    cmap : str or Colormap, optional
        Colormap. Default: 'seismic'.
    dir : str, optional
        Output directory. Default: '../paper_plots/'.
    imag_flag : bool, optional
        If True, show real, imaginary, and absolute rows. If False, real only.
    vmin, vmax : float or list of float, optional
        Color scale limits per dataset.
    show : bool, optional
        Call ``plt.show()`` after plotting. Default: False.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    num_sets = len(data_sets)
    col_labels = col_labels or [f"Data {i}" for i in range(num_sets)]

    if len(col_labels) != num_sets:
        raise ValueError("Number of column labels must match number of data sets.")

    if isinstance(norm, str):
        if norm == 'linear':
            norm_fn = None
        elif norm == 'log':
            norm_fn = LogNorm()
        else:
            raise ValueError(f"Unknown norm '{norm}'")
    else:
        norm_fn = norm

    nrows = 3 if imag_flag else 1
    fig, ax = plt.subplots(nrows, num_sets, figsize=(num_sets * 5, nrows * 6), squeeze=False)
    ylabels = ['Real', 'Imaginary', 'Absolute']

    for i in range(num_sets):
        data = data_sets[i]
        vmin_i = vmin[i] if isinstance(vmin, (list, tuple, np.ndarray)) else vmin
        vmax_i = vmax[i] if isinstance(vmax, (list, tuple, np.ndarray)) else vmax
        parts = [np.real(data), np.imag(data), np.abs(data)] if imag_flag else [np.real(data)]
        for j, part in enumerate(parts):
            plot_ax = ax[j, i]
            if plot_type == 'imshow':
                im = plot_ax.imshow(part, origin='lower', cmap=cmap, norm=norm_fn, vmin=vmin_i, vmax=vmax_i)
            elif plot_type == 'matshow':
                im = plot_ax.matshow(part, cmap=cmap, norm=norm_fn, vmin=vmin_i, vmax=vmax_i, aspect='auto')
            else:
                raise ValueError("plot_type must be 'imshow' or 'matshow'")
            if i == 0:
                plot_ax.set_ylabel(ylabels[j], fontsize=16)
            plot_ax.set_title(col_labels[i], fontsize=16)
            cbar = plt.colorbar(im, ax=plot_ax, fraction=0.046, pad=0.04)
            cbar.set_label(label='Amplitudes', fontsize=16)
            cbar.ax.tick_params(which='both')

    fig.suptitle(fig_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_flag:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(dir + fig_title + '.png', bbox_inches='tight', dpi=300, transparent=True)
    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_dps(vis_eor_path, res_dir, dir='../paper_plots/', Nburn=0, conf_interval=95):
    """
    Plot the delay power spectrum (DPS) recovered by hydra_pspec against truth.

    Produces three figures:
      1. True vs recovered DPS with confidence intervals.
      2. Residuals (recovered − true) vs delay.
      3. Z-score of residuals vs delay.
      4. Error bar magnitudes (lower/upper/mean) vs delay.

    Parameters
    ----------
    vis_eor_path : str
        Path to a UVData-compatible file containing the true EoR visibilities.
    res_dir : str
        Directory containing ``dps-eor.npy`` and ``ln-post.npy`` from
        hydra_pspec output.
    dir : str, optional
        Output directory for figures. Default: '../paper_plots/'.
    Nburn : int, optional
        Number of burn-in samples to discard. Default: 0.
    conf_interval : float, optional
        Confidence interval percentage for the recovered DPS. Default: 95.
    """
    uvd = UVData()
    uvd.read(vis_eor_path)
    uvd.conjugate_bls()
    vis_eor = uvd.get_data((0, 1, "xx"))  # shape (Ntimes, Nfreqs)

    freqs = uvd.freq_array * units.Hz
    if uvd.use_future_array_shapes:
        freqs = freqs[0]
    df = freqs[1] - freqs[0]
    Nfreqs = freqs.size

    # True delay power spectrum via FFT of EoR visibilities
    axes = (1,)
    ds_eor_true = np.fft.ifftshift(vis_eor, axes=axes)
    ds_eor_true = np.fft.fftn(ds_eor_true, axes=axes)
    ds_eor_true = np.fft.fftshift(ds_eor_true, axes=axes)
    dps_eor_true = (np.abs(ds_eor_true)**2).mean(axis=0)
    delays = np.fft.fftshift(np.fft.fftfreq(Nfreqs, d=df.to("1/ns")))

    # Load hydra_pspec posterior samples
    dps_eor_hp = np.load(Path(res_dir) / "dps-eor.npy")
    ln_post = np.load(Path(res_dir) / "ln-post.npy")
    if Nburn > 0:
        dps_eor_hp = dps_eor_hp[Nburn:]
        ln_post = ln_post[Nburn:]

    # Posterior-weighted mean and confidence interval
    dps_eor_hp_pwm = np.average(dps_eor_hp, weights=ln_post, axis=0)
    percentile = conf_interval / 2 + 50
    dps_eor_hp_ubound = np.percentile(dps_eor_hp, percentile, axis=0)
    dps_eor_hp_lbound = np.percentile(dps_eor_hp, 100 - percentile, axis=0)
    dps_eor_hp_err = np.vstack((
        dps_eor_hp_pwm - dps_eor_hp_lbound,
        dps_eor_hp_ubound - dps_eor_hp_pwm
    ))

    # Figure 1: true vs recovered DPS
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(delays, dps_eor_true, "k:", label="True")
    ax.errorbar(
        delays, dps_eor_hp_pwm, yerr=np.abs(dps_eor_hp_err),
        color="k", marker="o", capsize=3,
        label=f"Recovered ({conf_interval}% Confidence)"
    )
    ax.legend(loc="upper right")
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"$P(\tau)$ [arb. units]")
    ax.set_title("EoR Delay Power Spectrum Comparison (systematics)")
    ax.set_yscale("log")
    ax.grid()
    fig.tight_layout()
    plt.savefig(dir + 'EoR_DPS_comparison.png', bbox_inches='tight', dpi=300)

    # Figure 2: residuals vs delay
    res = dps_eor_hp_pwm - dps_eor_true
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.errorbar(delays, res, yerr=0.68 * np.abs(dps_eor_hp_err), marker="o", capsize=3)
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"Data - true dps")
    ax.set_title("Residuals vs delays")
    ax.grid()
    fig.tight_layout()
    plt.savefig(dir + 'EoR_DPS_res_vs_delays.png', bbox_inches='tight', dpi=300)

    # Figure 3: z-score of residuals
    z_sc = sci_st.zscore(res)
    sig = np.std(dps_eor_hp_err)
    yerr_z = dps_eor_hp_err / sig

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.errorbar(delays, z_sc, yerr=np.abs(yerr_z), marker="o",
                markerfacecolor='blue', capsize=3, ecolor='blue')
    ax.set_xlabel(r"$\tau$ [ns]")
    ax.set_ylabel(r"Z score")
    ax.set_title("Z score vs delays")
    ax.set_ylim(-5, 5)
    ax.grid()
    fig.tight_layout()
    plt.savefig(dir + 'EoR_DPS__Score_vs_delays.png', bbox_inches='tight', dpi=300)

    # Figure 4: error bar lower/upper limits and their mean
    fig, ax = plt.subplots(figsize=(24, 5))
    ax.plot(delays, dps_eor_hp_err[0, :], marker="o", label='Lower limit', ls='dotted')
    ax.plot(delays, dps_eor_hp_err[1, :], marker="o", label='Upper limit', ls='dotted')
    ax.plot(delays, np.mean(dps_eor_hp_err, axis=0), marker="o", label='Mean', c='k')
    ax.set_title("Error bar means and upper-lower limits")
    ax.grid()
    ax.legend()
    plt.savefig(dir + 'EoR_DPS_Error_bar_mins_limits.png', bbox_inches='tight', dpi=300, transparent=True)


def plot_waterfalls(data, freqs, times, windows=None, mode='log', fig=None, ax=None, xlabel=None,
                    vmin=None, vmax=None, cmap='inferno', dynamic_range=None, limit_drng='all',
                    baseline=None, horizon_color='magenta', plot_limits=None, colorbar_flag=True,
                    freq_window_kwargs=None, time_window_kwargs=None,
                    fontsize=None, labelsize=None):
    """
    Plot visibility data in the delay–fringe-rate domain.

    Applies a 2D Fourier transform (time → fringe-rate, frequency → delay) to
    the input visibility and displays the result as a waterfall image.

    Parameters
    ----------
    data : ndarray, shape=(Ntimes, Nfreqs)
        Visibility data in units of Jy.
    freqs : ndarray, shape=(Nfreqs,)
        Observed frequencies in Hz.
    times : ndarray, shape=(Ntimes,)
        Observed times in JD.
    windows : str or sequence, optional
        Taper window(s); passed to ``uvtools.dspec.gen_window``. Default: boxcar.
    mode : str, optional
        Plot mode for ``uvtools.plot.waterfall``. Default: 'log'.
    fig, ax : optional
        Figure/axes to plot into; created if not provided.
    xlabel : str, optional
        Label for the x-axis. Default: "Delay [ns]".
    vmin, vmax : float, optional
        Color scale limits.
    cmap : str or Colormap, optional
        Colormap. Default: 'inferno'.
    dynamic_range : float, optional
        Orders of magnitude for the color range when ``mode='log'``.
    limit_drng : str or sequence, optional
        Domains for dynamic-range clipping. Default: 'all'.
    baseline : float or array-like, optional
        Baseline length in metres for geometric horizon lines.
    horizon_color : str or tuple, optional
        Color for horizon lines. Default: 'magenta'.
    plot_limits : dict, optional
        Axis limit overrides, e.g. ``{'delay': (-500, 500)}``.
    colorbar_flag : bool, optional
        Whether to draw a colorbar. Default: True.
    freq_window_kwargs : dict, optional
        Extra kwargs for the frequency taper ``gen_window`` call.
    time_window_kwargs : dict, optional
        Extra kwargs for the time taper ``gen_window`` call.
    fontsize : float, optional
        Font size for axis and colorbar labels.
    labelsize : float, optional
        Font size for tick labels.

    Returns
    -------
    cax : matplotlib.image.AxesImage
        Return value of the waterfall call.
    data_fr_dly : ndarray
        The delay–fringe-rate transformed data.
    """
    freq_window_kwargs = freq_window_kwargs or {}
    time_window_kwargs = time_window_kwargs or {}

    # Build taper windows and reshape for broadcasting
    time_window = gen_window(windows, times.size, **time_window_kwargs)[:, None]
    freq_window = gen_window(windows, freqs.size, **freq_window_kwargs)[None, :]

    # 2D Fourier transform: time → fringe-rate, frequency → delay
    data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)

    fringe_rates = fourier_freqs(times * units.day.to('s')) * 1e3  # mHz
    dlys = fourier_freqs(freqs) * 1e9  # ns

    if baseline is not None:
        horizon = np.linalg.norm(baseline) / constants.c.value * 1e9  # ns

    if ax is None:
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.subplots(1, 1)

    ax.set_facecolor('white')
    ax.tick_params(direction='out', length=6, width=2, colors='black', labelsize=labelsize)

    if xlabel is None:
        xlabel = "Delay [ns]"
    ax.set_xlabel(xlabel, fontsize=fontsize, color='black')
    ax.set_ylabel("Fringe Rate [mHz]", fontsize=fontsize, color='black')

    extent = (dlys.min(), dlys.max(), fringe_rates.max(), fringe_rates.min())
    xlimits, ylimits = extent[:2], extent[2:]
    if plot_limits is not None:
        xlimits = plot_limits.get("delay", xlimits)
        ylimits = plot_limits.get("fringe_rate", ylimits)

    # Dynamic range clipping: active for the delay and fringe-rate axes
    if limit_drng == 'all':
        limit_drng = ("freq", "time", "delay", "fringe_rate")
    clip_drng = ("freq" in limit_drng) or ("fringe_rate" in limit_drng)

    if clip_drng and dynamic_range is not None and mode == 'log':
        if vmax is None:
            vmax = np.log10(np.abs(data_fr_dly)).max()
        if vmin is None:
            vmin = vmax - dynamic_range
    elif not clip_drng:
        vmin, vmax = None, None

    vis_label = r"$\log_{10}|\tilde{V}(\tau, f)|$ [Jy Hz s]"
    cbar_label = vis_label if mode == 'log' else "Phase [rad]"

    fig.sca(ax)
    cax = waterfall(data_fr_dly, extent=extent, mode=mode, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    if baseline is not None:
        ax.axvline(horizon, color=horizon_color, ls='--')
        ax.axvline(-horizon, color=horizon_color, ls='--')

    if colorbar_flag:
        cb = plt.colorbar(cax)
        cb.set_label(cbar_label, c='black', fontsize=fontsize)
        cb.ax.tick_params(axis='y', which='both', color='black', labelcolor='black', labelsize=labelsize)

    return cax, data_fr_dly


def plot_waterfalls_from_dlfr(data_dlfr, freqs, times, mode='log', fig=None, ax=None, xlabel=None,
                               vmin=None, vmax=None, cmap='inferno', dynamic_range=None, limit_drng='all',
                               baseline=None, horizon_color='magenta', plot_limits=None, colorbar_flag=True,
                               fontsize=None, labelsize=None):
    """
    Plot pre-computed delay–fringe-rate data as a waterfall image.

    Equivalent to ``plot_waterfalls`` but accepts data already in the
    delay–fringe-rate domain, skipping the FFT step.

    Parameters
    ----------
    data_dlfr : ndarray, shape=(Ntimes, Nfreqs)
        Visibility data in delay–fringe-rate space.
    freqs : ndarray, shape=(Nfreqs,)
        Observed frequencies in Hz (used to derive the delay axis).
    times : ndarray, shape=(Ntimes,)
        Observed times in JD (used to derive the fringe-rate axis).
    mode : str, optional
        Plot mode for ``uvtools.plot.waterfall``. Default: 'log'.
    fig, ax : optional
        Figure/axes to plot into; created if not provided.
    xlabel : str, optional
        Label for the x-axis. Default: "Delay [ns]".
    vmin, vmax : float, optional
        Color scale limits.
    cmap : str or Colormap, optional
        Colormap. Default: 'inferno'.
    dynamic_range : float, optional
        Orders of magnitude for the color range when ``mode='log'``.
    limit_drng : str or sequence, optional
        Domains for dynamic-range clipping. Default: 'all'.
    baseline : float or array-like, optional
        Baseline length in metres for geometric horizon lines.
    horizon_color : str or tuple, optional
        Color for horizon lines. Default: 'magenta'.
    plot_limits : dict, optional
        Axis limit overrides, e.g. ``{'delay': (-500, 500)}``.
    colorbar_flag : bool, optional
        Whether to draw a colorbar. Default: True.
    fontsize : float, optional
        Font size for axis and colorbar labels.
    labelsize : float, optional
        Font size for tick labels.

    Returns
    -------
    cax : matplotlib.image.AxesImage
        Return value of the waterfall call.
    """
    fringe_rates = fourier_freqs(times * units.day.to('s')) * 1e3  # mHz
    dlys = fourier_freqs(freqs) * 1e9  # ns

    if baseline is not None:
        horizon = np.linalg.norm(baseline) / constants.c.value * 1e9  # ns

    if ax is None:
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.subplots(1, 1)

    ax.set_facecolor('white')
    ax.tick_params(direction='out', length=6, width=2, colors='black', labelsize=labelsize)

    if xlabel is None:
        xlabel = "Delay [ns]"
    ax.set_xlabel(xlabel, fontsize=fontsize, color='black')
    ax.set_ylabel("Fringe Rate [mHz]", fontsize=fontsize, color='black')

    extent = (dlys.min(), dlys.max(), fringe_rates.max(), fringe_rates.min())
    xlimits, ylimits = extent[:2], extent[2:]
    if plot_limits is not None:
        xlimits = plot_limits.get("delay", xlimits)
        ylimits = plot_limits.get("fringe_rate", ylimits)

    # Dynamic range clipping
    if limit_drng == 'all':
        limit_drng = ("freq", "time", "delay", "fringe_rate")
    clip_drng = ("freq" in limit_drng) or ("fringe_rate" in limit_drng)

    if clip_drng and dynamic_range is not None and mode == 'log':
        if vmax is None:
            vmax = np.log10(np.abs(data_dlfr)).max()
        if vmin is None:
            vmin = vmax - dynamic_range
    elif not clip_drng:
        vmin, vmax = None, None

    vis_label = r"$\log_{10}|\tilde{V}(\tau, f)|$ [Jy Hz s]"
    cbar_label = vis_label if mode == 'log' else "Phase [rad]"

    fig.sca(ax)
    cax = waterfall(data_dlfr, extent=extent, mode=mode, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    if baseline is not None:
        ax.axvline(horizon, color=horizon_color, ls='--')
        ax.axvline(-horizon, color=horizon_color, ls='--')

    if colorbar_flag:
        cb = plt.colorbar(cax)
        cb.set_label(cbar_label, c='black', fontsize=fontsize)
        cb.ax.tick_params(axis='y', which='both', color='black', labelcolor='black', labelsize=labelsize)

    return cax
