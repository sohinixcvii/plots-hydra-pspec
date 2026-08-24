"""
Presentation helpers for the MCMC diagnostic tables of the plotting notebooks.

The notebooks used to print their ESS / autocorrelation-time results with bare
``print(array)`` calls, which are hard to read and impossible to paste into a
message or a paper draft.  Everything here turns the same numbers into

* a styled HTML table rendered in the notebook (inline CSS only, so that a
  copy-paste into email, Slack or Google Docs keeps its formatting),
* a Markdown copy of the same table (pastes into GitHub / Slack / Notion),
* a LaTeX ``tabular`` (pastes into the paper), and
* a CSV copy,

together with a colour code (green / amber / red) that tells a reader who does
not know the conventions whether a number is acceptable.

Nothing in here depends on pandas; only numpy is required, and IPython is used
only if it happens to be available.

Typical use in a notebook::

    from plotting_codes.tables import ess_table, plot_ess_tau, save_tables

    tbl = ess_table({lab: chain for lab, chain in zip(labels, chains)},
                    n_samples=Niter, title='Table 1 — sampling efficiency')
    fig = plot_ess_tau(tbl)
    save_tables(fig_dir + '/tables')
"""

from __future__ import annotations

import csv
import io
import os

import numpy as np

# ── Conventions ──────────────────────────────────────────────────────────────
# ESS thresholds follow Vehtari et al. (2021): ESS > 400 is the minimum for a
# trustworthy posterior mean/quantile, and a few thousand is comfortable.
ESS_GOOD = 1000.0   # green  at or above this
ESS_WARN = 400.0    # amber  between ESS_WARN and ESS_GOOD, red below
RHAT_OK = 1.01      # rank-normalised split-Rhat: converged at or below
RHAT_WARN = 1.05
GEWEKE_OK = 2.0     # |z|: stationary at or below
GEWEKE_WARN = 3.0
TAU_RELIABLE = 50.0  # the tau estimate itself needs N > 50 tau

# ── Palette (matches the paper figures: teal / amber / berry) ────────────────
C_HEAD = '#0b3d4a'      # header background
C_HEAD_TXT = '#ffffff'
C_ZEBRA = '#f6f7f8'
C_GRID = '#d5d9dd'
C_GOOD = '#e3f2ec'      # pale teal-green
C_GOOD_TXT = '#12604a'
C_WARN = '#fdf1dc'      # pale amber
C_WARN_TXT = '#8a5300'
C_BAD = '#fbe4e8'       # pale berry
C_BAD_TXT = '#9e1e38'

_FLAG_COLORS = {
    'good': (C_GOOD, C_GOOD_TXT),
    'warn': (C_WARN, C_WARN_TXT),
    'bad': (C_BAD, C_BAD_TXT),
    None: (None, None),
}

# Registry of every table built since the kernel started, so that a single
# `save_tables()` call at the end of a notebook writes all of them out.
TABLES: "dict[str, Table]" = {}


# ─────────────────────────────────────────────────────────────────────────────
# Generic table
# ─────────────────────────────────────────────────────────────────────────────
class Col:
    """One column of a `Table`.

    Parameters
    ----------
    key : str
        Key looked up in each row dict.
    header : str
        Column heading (plain text; use `unit` for the unit line).
    fmt : str or callable
        ``str.format`` spec applied to the value, or a callable value -> str.
    align : {'left', 'right', 'center'}
    flag : callable or None
        ``value -> 'good' | 'warn' | 'bad' | None``; controls the cell colour.
    """

    def __init__(self, key, header, fmt='{:.4g}', align='right', flag=None):
        self.key = key
        self.header = header
        self.fmt = fmt
        self.align = align
        self.flag = flag

    def text(self, value):
        if value is None:
            return '--'
        if isinstance(value, float) and not np.isfinite(value):
            return '--'
        if callable(self.fmt):
            return self.fmt(value)
        if isinstance(value, str):
            return value
        return self.fmt.format(value)

    def flag_of(self, value):
        if self.flag is None or value is None:
            return None
        try:
            if isinstance(value, float) and not np.isfinite(value):
                return None
            return self.flag(value)
        except (TypeError, ValueError):
            return None


def flag_ge(good, warn):
    """Larger is better."""
    def f(v):
        return 'good' if v >= good else ('warn' if v >= warn else 'bad')
    return f


def flag_le(good, warn):
    """Smaller is better."""
    def f(v):
        return 'good' if v <= good else ('warn' if v <= warn else 'bad')
    return f


def flag_bool(v):
    return 'good' if v else 'bad'


class Table:
    """A rendered diagnostic table.

    Holds the rows and columns and knows how to write itself as HTML (for the
    notebook), Markdown, LaTeX and CSV.  Displaying it in a notebook shows the
    HTML version; ``print(tbl)`` shows the Markdown version.
    """

    def __init__(self, rows, cols, title=None, subtitle=None, notes=(),
                 name=None, group_key=None, register=True):
        self.rows = list(rows)
        self.cols = list(cols)
        self.title = title
        self.subtitle = subtitle
        self.notes = list(notes)
        self.group_key = group_key      # start a new visual block when it changes
        self.name = name or _slug(title or 'table')
        if register:
            TABLES[self.name] = self

    # -- cell access ---------------------------------------------------------
    def cell(self, row, col):
        return col.text(row.get(col.key)), col.flag_of(row.get(col.key))

    def as_matrix(self):
        """Header row plus formatted body rows, as plain strings."""
        head = [c.header for c in self.cols]
        body = [[self.cell(r, c)[0] for c in self.cols] for r in self.rows]
        return head, body

    # -- renderers -----------------------------------------------------------
    def to_html(self, markdown_block=True, font_size='13px'):
        pad = '6px 12px'
        out = [
            f'<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,'
            f'sans-serif;font-size:{font_size};color:#1b1f23;margin:8px 0 22px 0;">'
        ]
        if self.title:
            out.append(
                f'<div style="font-size:15px;font-weight:650;margin-bottom:2px;">'
                f'{_esc(self.title)}</div>')
        if self.subtitle:
            out.append(
                f'<div style="color:#57606a;margin-bottom:8px;max-width:62em;'
                f'line-height:1.45;">{_esc(self.subtitle)}</div>')
        out.append('<table style="border-collapse:collapse;border-spacing:0;">')

        # header
        out.append(f'<tr style="background:{C_HEAD};color:{C_HEAD_TXT};">')
        for c in self.cols:
            out.append(
                f'<th style="padding:{pad};text-align:{c.align};font-weight:600;'
                f'white-space:nowrap;border:1px solid {C_HEAD};">'
                f'{_esc(c.header)}</th>')
        out.append('</tr>')

        # body
        prev_group = object()
        for i, r in enumerate(self.rows):
            group = r.get(self.group_key) if self.group_key else None
            new_group = self.group_key is not None and group != prev_group
            prev_group = group
            top = f'border-top:2px solid {C_HEAD};' if (new_group and i) else ''
            zebra = C_ZEBRA if (i % 2) else '#ffffff'
            out.append(f'<tr style="background:{zebra};">')
            for c in self.cols:
                txt, flag = self.cell(r, c)
                bg, fg = _FLAG_COLORS[flag]
                style = (f'padding:{pad};text-align:{c.align};'
                         f'border:1px solid {C_GRID};{top}'
                         f'white-space:nowrap;font-variant-numeric:tabular-nums;')
                if bg:
                    style += f'background:{bg};color:{fg};font-weight:600;'
                out.append(f'<td style="{style}">{_esc(txt)}</td>')
            out.append('</tr>')
        out.append('</table>')

        for n in self.notes:
            out.append(
                f'<div style="color:#57606a;font-size:12px;margin-top:6px;'
                f'max-width:62em;line-height:1.5;">{n}</div>')

        if markdown_block:
            md = _esc(self.to_markdown())
            out.append(
                '<details style="margin-top:8px;"><summary style="cursor:pointer;'
                'color:#57606a;font-size:12px;">Markdown / LaTeX source (click to '
                'copy)</summary>'
                f'<pre style="font-size:12px;background:{C_ZEBRA};padding:10px;'
                f'border:1px solid {C_GRID};overflow-x:auto;">{md}\n\n'
                f'{_esc(self.to_latex())}</pre></details>')
        out.append('</div>')
        return ''.join(out)

    def to_markdown(self):
        head, body = self.as_matrix()
        widths = [max(len(head[j]), *(len(r[j]) for r in body)) if body
                  else len(head[j]) for j in range(len(self.cols))]
        aligns = [c.align for c in self.cols]

        def line(cells):
            return '| ' + ' | '.join(
                cells[j].rjust(widths[j]) if aligns[j] == 'right'
                else cells[j].ljust(widths[j])
                for j in range(len(cells))) + ' |'

        sep = '| ' + ' | '.join(
            ('-' * (widths[j] - 1) + ':') if aligns[j] == 'right'
            else ('-' * widths[j])
            for j in range(len(self.cols))) + ' |'
        parts = []
        if self.title:
            parts.append(f'**{self.title}**\n')
        parts += [line(head), sep] + [line(r) for r in body]
        for n in self.notes:
            parts.append('')
            parts.append(_strip_tags(n))
        return '\n'.join(parts)

    def to_latex(self, caption=None, label=None):
        head, body = self.as_matrix()
        spec = ''.join({'right': 'r', 'left': 'l', 'center': 'c'}[c.align]
                       for c in self.cols)
        esc = _latex_escape
        lines = [r'\begin{table}[t]', r'\centering',
                 r'\begin{tabular}{' + spec + '}', r'\hline']
        lines.append(' & '.join(esc(h) for h in head) + r' \\')
        lines.append(r'\hline')
        lines += [' & '.join(esc(c) for c in r) + r' \\' for r in body]
        lines += [r'\hline', r'\end{tabular}']
        cap = caption or self.title
        if cap:
            lines.append(r'\caption{' + esc(cap) + '}')
        if label:
            lines.append(r'\label{' + label + '}')
        lines.append(r'\end{table}')
        return '\n'.join(lines)

    def to_csv(self):
        head, body = self.as_matrix()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(head)
        w.writerows(body)
        return buf.getvalue()

    def to_text(self):
        """Fixed-width plain text (used when IPython is unavailable)."""
        return self.to_markdown().replace('|', ' ').replace('-:', '--')

    # -- notebook integration ------------------------------------------------
    def _repr_html_(self):
        return self.to_html()

    def __str__(self):
        return self.to_markdown()

    def column(self, key):
        """The raw (unformatted) values of one column, as an array."""
        return np.array([r.get(key, np.nan) for r in self.rows])


def show(table):
    """Display `table` in the notebook (HTML if possible, text otherwise)."""
    try:
        from IPython.display import HTML, display
        display(HTML(table.to_html()))
    except Exception:
        print(table.to_text())
    return table


def callout(text, kind='good', title=None, show_it=True):
    """
    A coloured verdict box (the take-home sentence under a table).

    `kind` is 'good', 'warn' or 'bad' and picks the same palette as the table
    cells.  Returns the HTML string; displays it as well unless `show_it`.
    """
    bg, fg = _FLAG_COLORS.get(kind, (C_ZEBRA, '#1b1f23'))
    bar = {'good': C_GOOD_TXT, 'warn': C_WARN_TXT, 'bad': C_BAD_TXT}.get(kind, '#57606a')
    head = (f'<div style="font-weight:700;margin-bottom:4px;">{title}</div>'
            if title else '')
    html = (f'<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,'
            f'sans-serif;font-size:13.5px;background:{bg};color:{fg};'
            f'border-left:6px solid {bar};padding:10px 14px;margin:6px 0 20px 0;'
            f'max-width:62em;line-height:1.5;">{head}{text}</div>')
    if show_it:
        try:
            from IPython.display import HTML, display
            display(HTML(html))
        except Exception:
            print(_strip_tags(html))
    return html


def save_tables(directory, tables=None, formats=('md', 'tex', 'csv', 'html'),
                verbose=True):
    """Write every registered table to `directory` in the given formats."""
    os.makedirs(directory, exist_ok=True)
    tables = TABLES if tables is None else tables
    written = []
    for name, t in tables.items():
        for ext in formats:
            text = {'md': t.to_markdown, 'tex': t.to_latex,
                    'csv': t.to_csv, 'html': t.to_html}[ext]()
            path = os.path.join(directory, f'{name}.{ext}')
            with open(path, 'w') as fh:
                fh.write(text)
            written.append(path)
    if verbose:
        print(f'Wrote {len(written)} files for {len(tables)} tables in {directory}')
        for name in tables:
            print(f'  {name}.{{{",".join(formats)}}}')
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
def tau_ess(chain, c=5.0, tol=0, tau_floor=1.0):
    """
    Integrated autocorrelation time and effective sample size per column.

    Sokal's automatic windowing (``emcee.autocorr.integrated_time``) can return
    tau < 1 -- even a negative value -- for a chain that is already effectively
    independent, because the windowed sum of a noisy autocorrelation function
    can go negative.  Such a value is not physical and, left alone, produces
    negative or absurdly large "effective sample sizes".  tau is therefore
    floored at `tau_floor` = 1 sample, i.e. ESS is capped at N.

    Parameters
    ----------
    chain : array_like, shape (N,) or (N, Ncol)
        Samples.  Complex input is passed through to emcee, which then returns
        one tau per column combining the real and imaginary parts.
    c, tol : float
        Passed to ``emcee.autocorr.integrated_time``.  ``tol=0`` never raises;
        reliability (N > 50 tau) is reported separately.
    tau_floor : float
        Lower bound applied to tau.

    Returns
    -------
    tau, ess : np.ndarray, shape (Ncol,)
    """
    import emcee

    x = np.asarray(chain)
    x = x[:, None] if x.ndim == 1 else x.reshape(x.shape[0], -1)
    n = x.shape[0]
    tau = np.full(x.shape[1], np.nan)
    good = np.asarray(np.abs(x).std(axis=0)) > 0
    if good.any():
        tau[good] = np.atleast_1d(emcee.autocorr.integrated_time(
            x[:, good], c=c, tol=tol, quiet=True, has_walkers=False)).real
    tau = np.where(np.isfinite(tau), np.maximum(tau, tau_floor), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        ess = n / tau
    return tau, ess


def ess_columns(n_samples=None):
    """The standard tau / ESS column set."""
    cols = [
        Col('case', 'Case', fmt='{}', align='left'),
        Col('param', 'Parameter', fmt='{}', align='left'),
        Col('n', 'N samples', fmt='{:,.0f}'),
        Col('tau', 'τ [samples]', fmt='{:,.1f}'),
        Col('ess', 'ESS', fmt='{:,.0f}', flag=flag_ge(ESS_GOOD, ESS_WARN)),
        Col('ess_frac', 'ESS / N', fmt='{:.3f}'),
        Col('reliable', f'N > {TAU_RELIABLE:.0f}τ',
            fmt=lambda v: 'yes' if v else 'NO', flag=flag_bool),
    ]
    if n_samples is not None:
        cols = [c for c in cols if c.key != 'n']
    return cols


def ess_table(chains, n_samples=None, param_names=None, title=None,
              subtitle=None, name=None, notes=None, c=5.0, tol=0,
              show_table=True):
    """
    Build (and display) the tau / ESS table for one or more chains.

    Parameters
    ----------
    chains : dict {case label -> array (N,) or (N, Nparam)}
        One entry per case; the array is the chain of the monitored parameter
        block.  Complex chains give one tau per complex parameter.
    n_samples : int or dict or None
        Number of samples used per case.  ``None`` takes it from each chain.
    param_names : list of str or None
        Names of the columns of each chain (default ``b_sys,1 ...``).
    title, subtitle, notes, name : str
        Presentation.
    show_table : bool
        Display the table immediately (notebook use).

    Returns
    -------
    Table
    """
    rows = []
    for case, chain in chains.items():
        x = np.asarray(chain)
        x = x[:, None] if x.ndim == 1 else x.reshape(x.shape[0], -1)
        n = x.shape[0] if n_samples is None else (
            n_samples[case] if isinstance(n_samples, dict) else n_samples)
        tau, ess = tau_ess(x[:n], c=c, tol=tol)
        names = param_names or [
            (f'b_sys,{k + 1}' if x.shape[1] > 1 else 'parameter')
            for k in range(x.shape[1])]
        names = [_plain(nm) for nm in names]
        for k in range(x.shape[1]):
            rows.append(dict(case=case, param=names[k], n=n, tau=tau[k],
                             ess=ess[k], ess_frac=ess[k] / n,
                             reliable=bool(n > TAU_RELIABLE * tau[k])))

    default_notes = [
        f'<b>ESS</b> is the number of independent samples the chain is worth: '
        f'ESS = N / τ. Colour code: '
        f'{_chip("green", C_GOOD, C_GOOD_TXT)} ESS &ge; {ESS_GOOD:,.0f} '
        f'(comfortable), {_chip("amber", C_WARN, C_WARN_TXT)} '
        f'{ESS_WARN:,.0f} &ndash; {ESS_GOOD:,.0f} (usable), '
        f'{_chip("red", C_BAD, C_BAD_TXT)} below {ESS_WARN:,.0f} '
        f'(posterior means and 95% intervals are not yet trustworthy).',
        f'<b>τ</b> is the integrated autocorrelation time: the number of '
        f'Gibbs iterations the sampler needs to forget its previous sample. '
        f'The last column checks N &gt; {TAU_RELIABLE:.0f}τ, the regime in '
        f'which the tau estimate itself is trustworthy.',
    ]
    t = Table(rows, ess_columns(), title=title, subtitle=subtitle,
              notes=default_notes if notes is None else notes,
              name=name, group_key='case')
    return show(t) if show_table else t


def plot_ess_tau(table, colors=None, figsize=(20, 8), ess_ref=(ESS_WARN, ESS_GOOD),
                 title=None, param_key='param', case_key='case'):
    """
    Companion figure for an `ess_table`: tau (left) and ESS (right) as grouped
    bars, one group per parameter and one bar per case.

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    cases, params = [], []
    for r in table.rows:
        if r[case_key] not in cases:
            cases.append(r[case_key])
        if r[param_key] not in params:
            params.append(r[param_key])
    lut = {(r[case_key], r[param_key]): r for r in table.rows}

    if colors is None:
        base = ['#1d3557', '#ca6702', '#81babc', '#e63946', '#ff8fba']
        colors = [base[i % len(base)] for i in range(len(cases))]

    x = np.arange(len(params), dtype=float)
    width = 0.8 / max(len(cases), 1)
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for j, case in enumerate(cases):
        off = (j - (len(cases) - 1) / 2) * width
        tau = [lut.get((case, p), {}).get('tau', np.nan) for p in params]
        ess = [lut.get((case, p), {}).get('ess', np.nan) for p in params]
        axs[0].bar(x + off, tau, width, color=colors[j], label=str(case),
                   edgecolor='black', linewidth=0.8)
        axs[1].bar(x + off, ess, width, color=colors[j], label=str(case),
                   edgecolor='black', linewidth=0.8)

    for ax, ylab in zip(axs, [r'$\tau$ [samples]  (lower is better)',
                              r'ESS $= N/\tau$  (higher is better)']):
        ax.set_xticks(x)
        ax.set_xticklabels(params, rotation=0)
        ax.set_ylabel(ylab)
        ax.set_yscale('log')
        ax.grid(axis='y', ls=':', alpha=0.5)
        ax.set_axisbelow(True)

    # Shade the ESS panel by the quality bands, so the verdict is readable
    # without knowing the thresholds.
    lo, hi = axs[1].get_ylim()
    bands = [(lo, ess_ref[0], C_BAD, f'ESS < {ess_ref[0]:,.0f}: unreliable'),
             (ess_ref[0], ess_ref[1], C_WARN,
              f'{ess_ref[0]:,.0f}-{ess_ref[1]:,.0f}: usable'),
             (ess_ref[1], hi, C_GOOD, f'ESS > {ess_ref[1]:,.0f}: comfortable')]
    for y0, y1, colr, lab in bands:
        # Clip each band to the visible range: with all-good or all-bad data
        # some bands fall outside the axes entirely, and an unclipped label
        # would drag the axes limits with it.
        y0c, y1c = max(y0, lo), min(y1, hi)
        if y1c <= y0c:
            continue
        axs[1].axhspan(y0c, y1c, color=colr, zorder=0)
        axs[1].text(len(params) - 0.42, np.sqrt(y0c * y1c), lab,
                    va='center', ha='right', color='0.25', fontsize='small',
                    zorder=3,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none',
                              alpha=0.8))
    for ref in ess_ref:
        if lo < ref < hi:
            axs[1].axhline(ref, color='0.45', ls='--', lw=1.5, zorder=0.6)
    axs[1].set_ylim(lo, hi)

    axs[0].legend(ncol=min(len(cases), 3), fontsize='small')
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return fig


# ── Ready-made flags for the standard MCMC diagnostics ───────────────────────
flag_ess = flag_ge(ESS_GOOD, ESS_WARN)
flag_ess_frac = flag_ge(0.05, 0.01)          # efficiency: 5% good, 1% usable
flag_rhat = flag_le(RHAT_OK, RHAT_WARN)      # split-Rhat: <= 1.01 converged
flag_geweke = flag_le(GEWEKE_OK, GEWEKE_WARN)  # |z|: <= 2 stationary
flag_bias = flag_le(1.0, 3.0)                # |bias| / sd, in posterior sigmas
flag_mcse = flag_le(0.05, 0.10)              # MCSE / sd: Monte Carlo noise floor


def verdict(rhat, ess, z=None):
    """One-word verdict combining the three standard diagnostics."""
    bad = (np.isfinite(rhat) and rhat > RHAT_WARN) or \
          (np.isfinite(ess) and ess < ESS_WARN) or \
          (z is not None and np.isfinite(z) and z > GEWEKE_WARN)
    if bad:
        return 'not converged'
    warn = (np.isfinite(rhat) and rhat > RHAT_OK) or \
           (np.isfinite(ess) and ess < ESS_GOOD) or \
           (z is not None and np.isfinite(z) and z > GEWEKE_OK)
    return 'marginal' if warn else 'converged'


VERDICT_FLAG = {'converged': 'good', 'marginal': 'warn', 'not converged': 'bad'}


def flag_verdict(v):
    return VERDICT_FLAG.get(v)


def quality_cmap(reverse=False):
    """
    Berry -> amber -> teal colour map matching the red / amber / green cell
    colours of the tables, so a figure and its table read the same way.
    ``reverse=True`` for quantities where *smaller* is better (Rhat, |z|).
    """
    from matplotlib.colors import LinearSegmentedColormap

    colours = ['#8f1b32', '#d4881f', '#f2efe9', '#2f8f74', '#0f5a45']
    if reverse:
        colours = colours[::-1]
    return LinearSegmentedColormap.from_list('quality', colours)


def annotated_heatmap(ax, matrix, row_labels, col_labels, fmt='{:.3g}',
                      cmap=None, vmin=None, vmax=None, norm=None, title=None,
                      cbar_label=None, fontsize=None, title_fontsize=None,
                      text_threshold=0.55):
    """
    A small labelled heat map: one cell per (row, column) with its value
    printed on top.  Used for the case x parameter-block convergence
    dashboards, where the numbers matter as much as the colours.

    Returns the AxesImage.
    """
    import matplotlib.pyplot as plt

    M = np.asarray(matrix, dtype=float)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                   aspect='auto')
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which='minor', color='white', lw=2)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=0)

    rgba = im.cmap(im.norm(M))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.text(j, i, '--', ha='center', va='center', color='0.4',
                        fontsize=fontsize)
                continue
            lum = 0.299 * rgba[i, j, 0] + 0.587 * rgba[i, j, 1] + 0.114 * rgba[i, j, 2]
            ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center',
                    color='white' if lum < text_threshold else 'black',
                    fontsize=fontsize)
    if title:
        ax.set_title(title, pad=12, fontsize=title_fontsize)
    if cbar_label:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label)
    return im


# ─────────────────────────────────────────────────────────────────────────────
# small utilities
# ─────────────────────────────────────────────────────────────────────────────
def _esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;'))


def _strip_tags(s):
    import re
    s = re.sub(r'<[^>]+>', '', str(s))
    return (s.replace('&ge;', '>=').replace('&le;', '<=')
            .replace('&ndash;', '-').replace('&amp;', '&')
            .replace('&lt;', '<').replace('&gt;', '>'))


def plain_label(s):
    """A LaTeX label rendered as plain text for table cells.

    Plain strings (no ``$``) are returned untouched, so ``b_sys,1`` keeps its
    underscore; only maths markup is unwrapped.
    """
    out = str(s)
    if '$' not in out:
        return out
    for a, b in (('$', ''), ('\\mathcal{L}', 'L'), ('\\mathrm', ''),
                 ('\\rm', ''), ('\\ln', 'ln'), ('{', ''), ('}', ''),
                 ('\\', '')):
        out = out.replace(a, b)
    return ' '.join(out.split())


_plain = plain_label   # internal alias


def _latex_escape(s):
    out = str(s)
    for a, b in (('\\', r'\textbackslash '), ('&', r'\&'), ('%', r'\%'),
                 ('_', r'\_'), ('#', r'\#'), ('$', r'\$'),
                 ('<', r'$<$'), ('>', r'$>$'),
                 ('τ', r'$\tau$'), ('≥', r'$\ge$'),
                 ('≤', r'$\le$'), ('–', '--')):
        out = out.replace(a, b)
    return out


def _chip(label, bg, fg):
    return (f'<span style="background:{bg};color:{fg};padding:1px 6px;'
            f'border-radius:3px;font-weight:600;">{label}</span>')


def _slug(s):
    import re
    return re.sub(r'[^a-z0-9]+', '_', str(s).lower()).strip('_')[:60]
