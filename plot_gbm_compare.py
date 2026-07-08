"""Paper-ready figure and LaTeX table for the GBM method comparison."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'experiments'))
from common.reporting import apply_style, save_pdf, save_latex

CSV     = Path(__file__).resolve().parent / 'results' / 'gbm' / 'gbm_hardcoded_compare.csv'
RESULTS = Path(__file__).resolve().parent / 'results' / 'gbm'
RESULTS.mkdir(parents=True, exist_ok=True)

METHOD_DISPLAY = {
    'analytical_IBP':        'Analytical Hessian, IBP',
    'DJ_IBP':                'Double Jacobian, IBP',
    'hardcoded_IBP':         'MLP-IBP (no auto_LiRPA)',
    'DJ_CROWN-IBP':          'Double Jacobian, CROWN-IBP',
    'analytical_CROWN-IBP':  'Analytical Hessian, CROWN-IBP',
}
METHOD_COLOR = {
    'analytical_IBP':        'tab:green',
    'DJ_IBP':                'tab:blue',
    'hardcoded_IBP':         'tab:orange',
    'DJ_CROWN-IBP':          'tab:blue',
    'analytical_CROWN-IBP':  'tab:green',
}
METHOD_LS = {
    'analytical_IBP':        '-',
    'DJ_IBP':                '-',
    'hardcoded_IBP':         '-',
    'DJ_CROWN-IBP':          '--',
    'analytical_CROWN-IBP':  '--',
}
METHOD_ORDER = [
    'analytical_IBP', 'DJ_IBP', 'hardcoded_IBP',
    'DJ_CROWN-IBP', 'analytical_CROWN-IBP',
]

# ---- load data ----
df = pd.read_csv(CSV)
df_chk = df[df['violations'].notna()].copy()
df_chk['violations'] = df_chk['violations'].astype(int)
df_chk['wall_s']     = df_chk['wall_s'].astype(float)
df_chk['seed']       = df_chk['seed'].astype(int)

df_sum = df[df['converged'].notna()].copy()
df_sum['seed']             = df_sum['seed'].astype(int)
df_sum['total_elapsed_s']  = df_sum['total_elapsed_s'].astype(float)
df_sum['compile_s']        = pd.to_numeric(df_sum['compile_s'], errors='coerce').fillna(0.0)
df_sum['converged']        = df_sum['converged'].astype(str).map(
    lambda v: True if v == 'True' else (False if v == 'False' else None))

conv = (df_sum[df_sum['converged'] == True][['seed', 'method', 'checkpoint_epoch']]
        .rename(columns={'checkpoint_epoch': 'conv_epoch'}))

compile_map = {(row['seed'], row['method']): row['compile_s']
               for _, row in df_sum.iterrows()}

# ---- aggregation helpers (parameterised by seed subset and method subset) ----
T_MAX = df_chk['wall_s'].max() * 1.02


def make_epoch_agg(seeds, methods):
    all_epochs = sorted(df_chk[df_chk['seed'].isin(seeds) &
                                df_chk['method'].isin(methods)
                                ]['checkpoint_epoch'].unique())
    records = []
    for (seed, method), grp in df_chk[df_chk['seed'].isin(seeds) &
                                       df_chk['method'].isin(methods)
                                       ].groupby(['seed', 'method']):
        ep2viol = dict(zip(grp['checkpoint_epoch'], grp['violations']))
        row = conv[(conv['seed'] == seed) & (conv['method'] == method)]
        conv_ep = int(row['conv_epoch'].values[0]) if len(row) else None
        for ep in all_epochs:
            if ep in ep2viol:
                records.append({'seed': seed, 'method': method, 'epoch': ep,
                                'violations': ep2viol[ep]})
            elif conv_ep is not None and ep > conv_ep:
                records.append({'seed': seed, 'method': method, 'epoch': ep,
                                'violations': 0})
    df_filled = pd.DataFrame(records)
    return (df_filled.groupby(['method', 'epoch'])['violations']
            .agg(mean='mean', std='std').reset_index())


def make_time_agg(seeds, methods, subtract_compile):
    t_grid = np.linspace(0, T_MAX, 400)
    out = {}
    for method in methods:
        sub = df_chk[df_chk['method'] == method & df_chk['seed'].isin(seeds)] \
            if False else df_chk[(df_chk['method'] == method) & (df_chk['seed'].isin(seeds))]
        seed_interps = []
        for seed, grp in sub.groupby('seed'):
            cs = compile_map.get((seed, method), 0.0)
            ts = grp.sort_values('wall_s')['wall_s'].values
            vs = grp.sort_values('wall_s')['violations'].values.astype(float)
            if subtract_compile:
                ts = ts - cs
            row = conv[(conv['seed'] == seed) & (conv['method'] == method)]
            if len(row):
                ts = np.append(ts, T_MAX)
                vs = np.append(vs, 0.0)
            interped = np.where(t_grid < ts[0], np.nan,
                                np.interp(t_grid, ts, vs, right=vs[-1]))
            seed_interps.append(interped)
        if seed_interps:
            arr = np.stack(seed_interps)
            out[method] = (t_grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0))
    return out


def draw_epoch_panel(ax, agg_epoch, methods):
    for method in methods:
        sub = agg_epoch[agg_epoch['method'] == method].sort_values('epoch')
        if sub.empty:
            continue
        color = METHOD_COLOR[method]
        ls    = METHOD_LS[method]
        y     = sub['mean'].clip(lower=0.5)
        yerr  = sub['std'].fillna(0.0)
        ax.semilogy(sub['epoch'], y, color=color, ls=ls)
        ax.fill_between(sub['epoch'],
                        (y - yerr).clip(lower=0.5),
                        (y + yerr).clip(lower=0.5),
                        color=color, alpha=0.15)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Decrease Violations')
    ax.set_xlim(0, 20000)
    ax.set_ylim(bottom=0.4)


def draw_time_panel(ax, agg, xlabel):
    for method, (t, mean, std) in agg.items():
        color = METHOD_COLOR[method]
        ls    = METHOD_LS[method]
        valid = ~np.isnan(mean)
        y    = np.clip(mean[valid], 0.5, None)
        y_lo = np.clip(mean[valid] - std[valid], 0.5, None)
        y_hi = np.clip(mean[valid] + std[valid], 0.5, None)
        ax.semilogy(t[valid], y, color=color, ls=ls)
        ax.fill_between(t[valid], y_lo, y_hi, color=color, alpha=0.15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Decrease Violations')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0.4)


def make_legend_handles(methods):
    colors_seen = dict()
    for m in methods:
        colors_seen[METHOD_COLOR[m]] = m
    color_label = {
        'tab:green':  'Analytical Hessian',
        'tab:blue':   'Double Jacobian',
        'tab:orange': 'MLP-IBP (no auto_LiRPA)',
    }
    color_handles = [mlines.Line2D([], [], color=c, lw=1.5,
                                   label=color_label[c])
                     for c in colors_seen if c in color_label]
    styles_seen = sorted({METHOD_LS[m] for m in methods})
    style_label = {'-': 'IBP', '--': 'CROWN-IBP'}
    style_handles = [mlines.Line2D([], [], color='black', ls=s, lw=1.5,
                                   label=style_label[s])
                     for s in styles_seen if s in style_label]
    return color_handles + style_handles


def save_violations_fig(seeds, methods, path):
    agg_ep  = make_epoch_agg(seeds, methods)
    agg_w   = make_time_agg(seeds, methods, subtract_compile=False)
    agg_wnc = make_time_agg(seeds, methods, subtract_compile=True)
    apply_style()
    plt.rcParams['figure.figsize'] = (13.0, 3.0)
    fig, axes = plt.subplots(1, 3)
    draw_epoch_panel(axes[0], agg_ep, methods)
    draw_time_panel(axes[1], agg_w,   'Wall Time (s)')
    draw_time_panel(axes[2], agg_wnc, 'Wall Time Excl. Compilation (s)')
    handles = make_legend_handles(methods)
    fig.legend(handles=handles, ncol=len(handles),
               loc='lower center', bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    save_pdf(fig, path)


_original_seeds = {98540, 77709, 5193, 98048, 50058}
_all_seeds      = set(df_chk['seed'].unique())
_focused        = ['DJ_IBP', 'hardcoded_IBP', 'DJ_CROWN-IBP']

# 5 original seeds, all methods (includes analytical hessian)
save_violations_fig(_original_seeds, METHOD_ORDER,
                    RESULTS / 'gbm_violations_5seeds.pdf')

# all 10 seeds, focused 3 methods only (fair comparison, equal seed counts)
save_violations_fig(_all_seeds, _focused,
                    RESULTS / 'gbm_violations_10seeds.pdf')


# ---- LaTeX table ----
rows = []
for method in METHOD_ORDER:
    sub_conv = conv[conv['method'] == method]
    n_seeds  = df_sum[df_sum['method'] == method]['seed'].nunique()
    n_conv   = len(sub_conv)

    if n_conv > 0:
        ep     = sub_conv['conv_epoch'] + 1
        ep_str = rf'${ep.mean()/1000:.1f}k \pm {ep.std()/1000:.1f}k$'
        t      = df_sum[(df_sum['method'] == method) & (df_sum['converged'] == True)]['total_elapsed_s']
        t_str  = rf'${t.mean():.0f} \pm {t.std():.0f}$\,s'
    else:
        ep_str = '---'
        t_str  = '---'

    rows.append({
        'Method':      METHOD_DISPLAY[method],
        'Conv.':       rf'{n_conv}/{n_seeds}',
        'Epoch':       ep_str,
        'Wall-clock':  t_str,
    })

df_table = pd.DataFrame(rows)
table_path = RESULTS / 'gbm_comparison_table.tex'
save_latex(
    df_table,
    table_path,
    caption=(r'GBM supermartingale certificate: convergence over 5 seeds '
             r'($m=200$, max refinement depth~2). '
             r'Epoch and wall-clock are mean~$\pm$~std over converged seeds only.'),
    label='tab:gbm_comparison',
    index=False,
)

latex = table_path.read_text()
latex = latex.replace(r'\begin{table}', r'\begin{table}[H]' + '\n\\centering')
table_path.write_text(latex)


# ---- per-seed plots: one panel per seed, original seeds on top row ----
_original = {98540, 77709, 5193, 98048, 50058}
all_seeds_unsorted = df_chk['seed'].unique()
all_seeds = (sorted(s for s in all_seeds_unsorted if s in _original) +
             sorted(s for s in all_seeds_unsorted if s not in _original))

_all_epochs_sorted = sorted(df_chk['checkpoint_epoch'].unique())
EP_MAX = max(_all_epochs_sorted)

_color_handles = [
    mlines.Line2D([], [], color='tab:green',  lw=1.5, label='Analytical Hessian'),
    mlines.Line2D([], [], color='tab:blue',   lw=1.5, label='Double Jacobian'),
    mlines.Line2D([], [], color='tab:orange', lw=1.5, label='MLP-IBP (no auto_LiRPA)'),
]
_style_handles = [
    mlines.Line2D([], [], color='black', ls='-',  lw=1.5, label='IBP'),
    mlines.Line2D([], [], color='black', ls='--', lw=1.5, label='CROWN-IBP'),
]


def save_perseed_fig(metric, xlabel, path):
    """metric: 'epoch', 'wall_time', or 'wall_time_nc'."""
    n_seeds = len(all_seeds)
    n_cols  = (n_seeds + 1) // 2
    apply_style()
    plt.rcParams['figure.figsize'] = (n_cols * 2.8, 6.0)
    fig, axes = plt.subplots(2, n_cols, sharey=True)
    axes_flat = axes.flatten()

    for ax, seed in zip(axes_flat, all_seeds):
        for method in METHOD_ORDER:
            color = METHOD_COLOR[method]
            ls    = METHOD_LS[method]
            grp   = df_chk[(df_chk['method'] == method) & (df_chk['seed'] == seed)]
            if grp.empty:
                continue
            vs = grp.sort_values('wall_s')['violations'].values.astype(float)

            if metric == 'epoch':
                xs = grp.sort_values('wall_s')['checkpoint_epoch'].values.astype(float)
                x_end = EP_MAX
            else:
                xs = grp.sort_values('wall_s')['wall_s'].values
                if metric == 'wall_time_nc':
                    cs = compile_map.get((seed, method), 0.0)
                    xs = xs - cs
                x_end = T_MAX

            row = conv[(conv['seed'] == seed) & (conv['method'] == method)]
            if len(row):
                xs = np.append(xs, x_end)
                vs = np.append(vs, 0.5)
            ax.semilogy(xs, np.clip(vs, 0.5, None), color=color, ls=ls, linewidth=1.3)

        ax.set_title(f'Seed {seed}', fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, EP_MAX if metric == 'epoch' else T_MAX)
        ax.set_ylim(bottom=0.4)

    for ax in axes_flat[n_seeds:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Decrease Violations')

    fig.legend(handles=_color_handles + _style_handles, ncol=5,
               loc='lower center', bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    save_pdf(fig, path)


save_perseed_fig('epoch',        'Training Epoch',                    RESULTS / 'gbm_epoch_all_seeds.pdf')
save_perseed_fig('wall_time',    'Wall Time (s)',                     RESULTS / 'gbm_wall_time_all_seeds.pdf')
save_perseed_fig('wall_time_nc', 'Wall Time Excl. Compilation (s)',   RESULTS / 'gbm_wall_time_nc_all_seeds.pdf')
