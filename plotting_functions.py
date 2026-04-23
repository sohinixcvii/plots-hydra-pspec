import matplotlib.pyplot as plt
import numpy as np
from astropy import constants, units
from uvtools.dspec import gen_window
from uvtools.utils import FFT, fourier_freqs
from uvtools.plot import waterfall


def plot_waterfalls(data, freqs, times, windows=None, mode='log', fig=None, ax=None, xlabel=None,
                    vmin=None, vmax=None, cmap='inferno', dynamic_range=None, limit_drng='all',
                    baseline=None, horizon_color='magenta', plot_limits=None, colorbar_flag=True,
                    ylab_flag=True,
                    freq_window_kwargs=None, time_window_kwargs=None):
    """
    Plot visibility data in the delay–fringe-rate domain.

    Applies a 2D Fourier transform (time → fringe-rate, frequency → delay) to
    the input visibility data and displays the result as a waterfall image.

    Parameters
    ----------
    data : ndarray, shape=(Ntimes, Nfreqs)
        Visibility data in units of Jy.
    freqs : ndarray, shape=(Nfreqs,)
        Observed frequencies in Hz.
    times : ndarray, shape=(Ntimes,)
        Observed times in JD.
    windows : str or sequence of str, optional
        Taper window(s) for the Fourier transforms; passed to
        ``uvtools.dspec.gen_window``. A single string applies the same taper
        along both axes. Default is no taper (boxcar).
    mode : str, optional
        Plot mode passed to ``uvtools.plot.waterfall``. One of 'log', 'abs',
        'real', 'imag', or 'phs'. Default: 'log'.
    fig : matplotlib.figure.Figure, optional
        Figure to plot into. Created if not provided.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. Created if not provided.
    xlabel : str, optional
        Label for the x-axis. Default: "Delay [ns]".
    vmin, vmax : float, optional
        Color scale limits. Computed from data if not provided.
    cmap : str or Colormap, optional
        Colormap. Default: 'inferno'.
    dynamic_range : float, optional
        Orders of magnitude for the color range when ``mode='log'``.
        Sets ``vmin = vmax - dynamic_range``.
    limit_drng : str or sequence of str, optional
        Which domains to apply the dynamic range limit to. Options:
        'freq', 'time', 'delay', 'fringe_rate', or 'all'. Default: 'all'.
    baseline : float or array-like, optional
        Baseline length in metres. If given, geometric horizon lines are
        overlaid on the delay axis.
    horizon_color : str or tuple, optional
        Color for the horizon lines. Default: 'magenta'.
    plot_limits : dict, optional
        Axis limits keyed by domain name, e.g. ``{'delay': (-500, 500)}``.
        Delay in ns, fringe-rate in mHz.
    colorbar_flag : bool, optional
        Whether to draw a colorbar. Default: True.
    freq_window_kwargs : dict, optional
        Extra kwargs forwarded to ``gen_window`` for the frequency taper.
    time_window_kwargs : dict, optional
        Extra kwargs forwarded to ``gen_window`` for the time taper.

    Returns
    -------
    cax : matplotlib.image.AxesImage
        Return value of the waterfall call (useful for external colorbars).
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
    ax.tick_params(direction='out', length=6, width=2, colors='black')

    if xlabel is None:
        xlabel = "Delay [ns]"
    ax.set_xlabel(xlabel, color='black')
    if ylab_flag:
        ax.set_ylabel("Fringe Rate [mHz]", color='black')

    # Axis limits: use plot_limits overrides or the full extent
    extent = (dlys.min(), dlys.max(), fringe_rates.max(), fringe_rates.min())
    xlimits, ylimits = extent[:2], extent[2:]
    if plot_limits is not None:
        xlimits = plot_limits.get("delay", xlimits)
        ylimits = plot_limits.get("fringe_rate", ylimits)

    # Dynamic range clipping: active when plotting delay or fringe-rate axes
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
    cax = waterfall(data_fr_dly, extent=extent, mode=mode, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    if baseline is not None:
        ax.axvline(horizon, color=horizon_color, ls='--')
        ax.axvline(-horizon, color=horizon_color, ls='--')

    if colorbar_flag:
        cb = plt.colorbar(cax)
        cb.set_label(cbar_label, c='black')
        cb.ax.tick_params(axis='y', which='both', color='black', labelcolor='black')

    return cax, data_fr_dly


def plot_waterfalls_from_dlfr(data_dlfr, freqs, times, mode='log', fig=None, ax=None, xlabel=None,
                               vmin=None, vmax=None, cmap='inferno', dynamic_range=None, limit_drng='all',
                               ylab_flag=True,
                               baseline=None, horizon_color='magenta', plot_limits=None,
                               colorbar_flag=True, cbar_label=None):
    """
    Plot pre-computed delay–fringe-rate data as a waterfall image.

    Equivalent to ``plot_waterfalls`` but accepts data that has already been
    transformed to the delay–fringe-rate domain, skipping the FFT step.

    Parameters
    ----------
    data_dlfr : ndarray, shape=(Ntimes, Nfreqs)
        Visibility data already in delay–fringe-rate space.
    freqs : ndarray, shape=(Nfreqs,)
        Observed frequencies in Hz (used to derive the delay axis).
    times : ndarray, shape=(Ntimes,)
        Observed times in JD (used to derive the fringe-rate axis).
    mode : str, optional
        Plot mode passed to ``uvtools.plot.waterfall``. Default: 'log'.
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
        Domains for which to apply dynamic-range clipping. Default: 'all'.
    baseline : float or array-like, optional
        Baseline length in metres for geometric horizon lines.
    horizon_color : str or tuple, optional
        Color for horizon lines. Default: 'magenta'.
    plot_limits : dict, optional
        Axis limit overrides, e.g. ``{'delay': (-500, 500)}``.
    colorbar_flag : bool, optional
        Whether to draw a colorbar. Default: True.
    cbar_label : str, optional
        Override for the colorbar label. For ``mode='log'``, the default
        label is the standard visibility units string. For ``mode='abs'``,
        the default is "Amplitude [Jy Hz s]".

    Returns
    -------
    cax : matplotlib.image.AxesImage
        Return value of the waterfall call.
    """
    fringe_rates = fourier_freqs(times * units.day.to('s')) * 1e3  # mHz
    dlys = fourier_freqs(freqs) * 1e9  # ns
    jd = int(np.floor(times[0]))

    if baseline is not None:
        horizon = np.linalg.norm(baseline) / constants.c.value * 1e9  # ns

    if ax is None:
        fig = plt.figure(figsize=(10, 10), facecolor='white')
        ax = fig.subplots(1, 1)

    ax.set_facecolor('white')
    ax.tick_params(direction='out', length=6, width=2, colors='black')

    if xlabel is None:
        xlabel = "Delay [ns]"
    ax.set_xlabel(xlabel, color='black')
    if ylab_flag:
        ax.set_ylabel("Fringe Rate [mHz]", color='black')

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

    # Determine colorbar label
    vis_label = r"$\log_{10}|\tilde{V}(\tau, f)|$ [Jy Hz s]"
    if cbar_label is None:
        if mode == 'log':
            cbar_label = vis_label
        elif mode == 'phs':
            cbar_label = "Phase [rad]"
        else:
            cbar_label = "Amplitude [Jy Hz s]"
    elif mode == 'log':
        # Append units label to caller-supplied prefix
        cbar_label = cbar_label + ' ' + vis_label

    fig.sca(ax)
    cax = waterfall(data_dlfr, extent=extent, mode=mode, vmin=vmin, vmax=vmax, cmap=cmap, origin='lower')
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)

    if baseline is not None:
        ax.axvline(horizon, color=horizon_color, ls='--')
        ax.axvline(-horizon, color=horizon_color, ls='--')

    if colorbar_flag:
        cb = plt.colorbar(cax)
        cb.set_label(cbar_label, c='black')
        cb.ax.tick_params(axis='y', which='both', color='black', labelcolor='black')

    return cax
