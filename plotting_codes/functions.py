import numpy as np
import matplotlib.pyplot as plt
import os
from pyuvdata import UVData
from uvtools.dspec import gen_window
from uvtools.utils import FFT, fourier_freqs
from astropy import units
import pyuvdata.utils as uvutils


def form_pseudo_stokes_vis(uvd, convention=1.0):
    """
    Form pseudo-Stokes I visibilities from XX and YY polarizations.

    Parameters
    ----------
    uvd : pyuvdata.UVData
        UVData object containing XX and YY polarization visibilities.
    convention : float, optional
        Scaling factor such that pI = convention * (XX + YY). Default: 1.0.

    Returns
    -------
    uvd : pyuvdata.UVData
        UVData object with pI stored in the XX slot.
    """
    assert isinstance(uvd, UVData), "uvd must be a pyuvdata.UVData object."

    if uvutils.polstr2num("pI") not in uvd.polarization_array:
        xx_pol_num = uvutils.polstr2num("xx")
        yy_pol_num = uvutils.polstr2num("yy")
        xpol_ind = np.where(uvd.polarization_array == xx_pol_num)[0]
        ypol_ind = np.where(uvd.polarization_array == yy_pol_num)[0]
        uvd.data_array[..., xpol_ind] += uvd.data_array[..., ypol_ind]
        uvd.data_array *= convention
        uvd.select(polarizations=["xx"])

    return uvd


def fourier_mode_2d(freqs_Hz, times_sec, modes, box=None):
    """
    Construct a set of 2D Fourier basis functions over a frequency–time grid.

    Each mode is specified by an integer pair (nf, nt), where nf and nt are
    the frequency and time wavenumber indices (as returned by
    ``numpy.fft.fftfreq``). The resulting basis functions are orthonormal on
    the discrete grid.

    Parameters
    ----------
    freqs_Hz : array_like, shape=(Nfreqs,)
        Frequency array in Hz. Must be evenly spaced.
    times_sec : array_like, shape=(Ntimes,)
        Time array in seconds. Must be evenly spaced.
    modes : list of (int, int)
        List of (nf, nt) wavenumber index pairs to construct.
    box : tuple of tuple, optional
        Not implemented. Reserved for future box-region selection.

    Returns
    -------
    basis_fns : ndarray, shape=(Nmodes, Nfreqs, Ntimes), complex
        Orthonormal 2D Fourier basis functions.
    kfreq : ndarray, shape=(Nfreqs,)
        Fourier wavenumbers along the frequency axis, in ns (delay).
    ktime : ndarray, shape=(Ntimes,)
        Fourier wavenumbers along the time axis, in mHz (fringe-rate).
    """
    Nfreqs, Ntimes = freqs_Hz.size, times_sec.size

    dfreq = freqs_Hz[1] - freqs_Hz[0]
    dtime = times_sec[1] - times_sec[0]

    # FFT wavenumbers (in physical units: delay [ns] and fringe-rate [mHz])
    kfreq = np.fft.fftfreq(Nfreqs, d=dfreq)   # seconds (delay)
    ktime = np.fft.fftfreq(Ntimes, d=dtime)   # Hz (fringe-rate)

    # Integer mode indices for validation
    nfreq = (np.fft.fftfreq(Nfreqs) * Nfreqs).astype(int)
    ntime = (np.fft.fftfreq(Ntimes) * Ntimes).astype(int)

    # Frequency and time axes relative to the first sample (for the exponent)
    f = freqs_Hz - freqs_Hz[0]
    t = times_sec - times_sec[0]

    basis_fns = np.zeros((len(modes), Nfreqs, Ntimes), dtype=np.complex128)
    for i, mode in enumerate(modes):
        nf, nt = mode
        assert isinstance(nf, int), "modes must only contain pairs of integers"
        assert isinstance(nt, int), "modes must only contain pairs of integers"
        assert nf in nfreq, (
            f"Delay mode nf={nf} not in available range ({nfreq.min()} -- {nfreq.max()})."
        )
        assert nt in ntime, (
            f"Fringe rate mode nt={nt} not in available range ({ntime.min()} -- {ntime.max()})."
        )

        idx_f = np.where(nfreq == nf)[0][0]
        idx_t = np.where(ntime == nt)[0][0]

        basis_fns[i] = (
            np.exp(2.0 * np.pi * 1j * (kfreq[idx_f] * f[:, np.newaxis]
                                        + ktime[idx_t] * t[np.newaxis, :]))
            / np.sqrt(Nfreqs * Ntimes)
        )

    return basis_fns, kfreq * 1e9, ktime * 1e3  # convert to ns and mHz


def sys_modes(freqs_Hz, times_sec, modes):
    """
    Construct the systematic mode operator (a 2D Fourier basis matrix).

    Calls ``fourier_mode_2d`` and reshapes the result into a
    (Nfreqs*Ntimes, Nmodes) operator suitable for matrix multiplication.

    Parameters
    ----------
    freqs_Hz : array_like, shape=(Nfreqs,)
        Frequency array in Hz.
    times_sec : array_like, shape=(Ntimes,)
        Time array in seconds.
    modes : list of (int, int)
        Wavenumber index pairs; see ``fourier_mode_2d``.

    Returns
    -------
    op : ndarray, shape=(Nfreqs*Ntimes, Nmodes), complex
        Systematic mode operator.
    """
    u, kfreq, ktime = fourier_mode_2d(freqs_Hz=freqs_Hz,
                                       times_sec=times_sec,
                                       modes=modes)
    return u.reshape((u.shape[0], -1)).T


def data_dly_fr(data, freqs, times, windows=None,
                freq_window_kwargs=None, time_window_kwargs=None):
    """
    Transform visibility data from time–frequency to delay–fringe-rate space.

    Applies optional taper windows and then a 2D FFT:
      - along the time axis  → fringe-rate
      - along the frequency axis → delay

    Parameters
    ----------
    data : ndarray, shape=(Ntimes, Nfreqs)
        Visibility data in units of Jy.
    freqs : ndarray, shape=(Nfreqs,)
        Observed frequencies in Hz.
    times : ndarray, shape=(Ntimes,)
        Observed times in JD.
    windows : str or sequence of str, optional
        Taper window(s) passed to ``uvtools.dspec.gen_window``.
        Default: no taper (boxcar).
    freq_window_kwargs : dict, optional
        Extra kwargs for the frequency taper ``gen_window`` call.
    time_window_kwargs : dict, optional
        Extra kwargs for the time taper ``gen_window`` call.

    Returns
    -------
    data_fr_dly : ndarray, shape=(Ntimes, Nfreqs), complex
        Data in delay–fringe-rate space.
    """
    freq_window_kwargs = freq_window_kwargs or {}
    time_window_kwargs = time_window_kwargs or {}

    # gen_window handles windows=None gracefully (returns a boxcar)
    time_window = gen_window(windows, times.size, **time_window_kwargs)[:, None]
    freq_window = gen_window(windows, freqs.size, **freq_window_kwargs)[None, :]

    # Two-step 2D FFT: first along time (→ fringe-rate), then along frequency (→ delay)
    data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)

    return data_fr_dly


def fourier_operator(n, unitary=True):
    """
    Build a dense DFT matrix of size n×n.

    Applying this matrix to a data vector is equivalent to:

    .. code-block:: python

        x_fft = numpy.fft.ifftshift(x)
        x_fft = numpy.fft.fft(x_fft)
        x_fft = numpy.fft.fftshift(x_fft)

    Parameters
    ----------
    n : int
        Size of the data (and matrix side length).
    unitary : bool, optional
        If True, the matrix is normalised so that F†F = I. Default: True.

    Returns
    -------
    fourier_op : ndarray, shape=(n, n), complex
        DFT matrix.
    """
    norm = np.sqrt(n) if unitary else 1.0

    i_x = (np.arange(n) - n // 2).reshape(1, -1)
    i_k = (np.arange(n) - n // 2).reshape(-1, 1)

    return np.exp(-2 * np.pi * 1j * (i_k * i_x / n)) / norm


def covariance_from_pspec(ps, fourier_op):
    """
    Convert a sampled delay power spectrum into a frequency–frequency
    covariance matrix for the next Gibbs iteration.

    Parameters
    ----------
    ps : ndarray, shape=(Nfreqs,)
        Power spectrum values in delay space.
    fourier_op : ndarray, shape=(Nfreqs, Nfreqs), complex
        DFT matrix from ``fourier_operator``.

    Returns
    -------
    C : ndarray, shape=(Nfreqs, Nfreqs), complex
        Frequency–frequency covariance matrix.
    """
    Nfreqs = ps.size
    Csigfft = np.zeros((Nfreqs, Nfreqs), dtype=complex)
    Csigfft[np.diag_indices(Nfreqs)] = ps
    return fourier_op.T.conj() @ Csigfft @ fourier_op


def data_fr_dly_to_t_f(data_fr_dly, freqs, times,
                        windows=None, freq_window_kwargs=None, time_window_kwargs=None,
                        eps=1e-12):
    """
    Invert the forward delay–fringe-rate transform.

    Reverses the operation performed by ``data_dly_fr``:
        data_fr_dly = FFT(FFT(data * time_window, axis=0) * freq_window, axis=1)

    Parameters
    ----------
    data_fr_dly : ndarray, shape=(Ntimes, Nfreqs), complex
        Data in fringe-rate / delay space.
    freqs : ndarray, shape=(Nfreqs,)
        Frequencies (used to rebuild the frequency window).
    times : ndarray, shape=(Ntimes,)
        Times (used to rebuild the time window).
    windows : str or sequence or None, optional
        Same argument used in the forward step.
    freq_window_kwargs, time_window_kwargs : dict or None, optional
        Same kwargs used in the forward step.
    eps : float, optional
        Floor for safe division when un-tapering near-zero window values.
        Default: 1e-12.

    Returns
    -------
    data : ndarray, shape=(Ntimes, Nfreqs), complex
        Reconstructed visibilities in time–frequency space.
    """
    freq_window_kwargs = freq_window_kwargs or {}
    time_window_kwargs = time_window_kwargs or {}

    time_window = gen_window(windows, times.size, **time_window_kwargs)[:, None]
    freq_window = gen_window(windows, freqs.size, **freq_window_kwargs)[None, :]

    # Inverse along frequency axis (delay → frequency).
    # uvtools FFT = fftshift(fft(x)), so its inverse is ifft(ifftshift(x)).
    x = np.fft.ifft(np.fft.ifftshift(data_fr_dly, axes=1), axis=1)
    x = x / np.where(np.abs(freq_window) > eps, freq_window, 1.0)

    # Inverse along time axis (fringe-rate → time)
    data = np.fft.ifft(np.fft.ifftshift(x, axes=0), axis=0)
    data = data / np.where(np.abs(time_window) > eps, time_window, 1.0)

    return data
