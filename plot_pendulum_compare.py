"""Paper-ready figure and LaTeX table for the inverted pendulum method comparison."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / 'experiments'))
from common.reporting import apply_style, save_pdf, save_latex

CSV     = Path(__file__).resolve().parent / 'results' / 'pendulum' / 'pendulum_compare.csv'
RESULTS = Path(__file__).resolve().parent / 'results' / 'pendulum'
RESULTS.mkdir(parents=True, exist_ok=True)

METHOD_DISPLAY = {
    'analytical_IBP':        'Analytical Hessian, IBP',
    'DJ_IBP':                'Double Jacobian, IBP',
    'hardcoded_IBP':         'Hardcoded IBP',
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

df = pd.read_csv(CSV)
df_chk = df[df['violations'].notna()].copy()
df_chk['violations'] = df_chk['violations'].astype(int)
df_sum = df[df['converged'].notna()].copy()
df_sum['elapsed_s'] = df_sum['elapsed_s'].astype(float)
df_sum['converged'] = df_sum['converged'].astype(str).map({'True': True, 'False': False})

conv = (df_sum[df_sum['converged'] == True][['seed', 'method', 'checkpoint_epoch']]
        .rename(columns={'checkpoint_epoch': 'conv_epoch'}))

all_epochs = sorted(df_chk['checkpoint_epoch'].unique())
records = []
for (seed, method), grp in df_chk.groupby(['seed', 'method']):
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
agg = (df_filled.groupby(['method', 'epoch'])['violations']
       .agg(mean='mean', std='std')
       .reset_index())

apply_style()
plt.rcParams['figure.figsize'] = (5.0, 2.8)
fig, ax = plt.subplots()

for method in METHOD_ORDER:
    sub = agg[agg['method'] == method].sort_values('epoch')
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

color_handles = [
    mlines.Line2D([], [], color='tab:green', lw=1.5, label='Analytical Hessian'),
    mlines.Line2D([], [], color='tab:blue',  lw=1.5, label='Double Jacobian'),
]
style_handles = [
    mlines.Line2D([], [], color='black', ls='-',  lw=1.5, label='IBP'),
    mlines.Line2D([], [], color='black', ls='--', lw=1.5, label='CROWN-IBP'),
]
ax.legend(handles=color_handles + style_handles, ncol=2, fontsize=6.5,
          loc='upper right')

save_pdf(fig, RESULTS / 'pendulum_violations_vs_epoch.pdf')

rows = []
for method in METHOD_ORDER:
    sub_conv = conv[conv['method'] == method]
    n_seeds  = df_sum[df_sum['method'] == method]['seed'].nunique()
    n_conv   = len(sub_conv)

    if n_conv > 0:
        ep     = sub_conv['conv_epoch'] + 1
        ep_std = ep.std()
        ep_str = (rf'${ep.mean()/1000:.1f}k \pm {ep_std/1000:.1f}k$'
                  if not np.isnan(ep_std) else rf'${ep.mean()/1000:.1f}k$')
        t      = df_sum[(df_sum['method'] == method) & (df_sum['converged'] == True)]['elapsed_s']
        t_std  = t.std()
        t_str  = (rf'${t.mean():.0f} \pm {t_std:.0f}$\,s'
                  if not np.isnan(t_std) else rf'${t.mean():.0f}$\,s')
    else:
        ep_str = '---'
        t_str  = '---'

    rows.append({
        'Method':     METHOD_DISPLAY[method],
        'Conv.':      rf'{n_conv}/{n_seeds}',
        'Epoch':      ep_str,
        'Wall-clock': t_str,
    })

df_table = pd.DataFrame(rows)
table_path = RESULTS / 'pendulum_comparison_table.tex'
save_latex(
    df_table,
    table_path,
    caption=(r'Inverted pendulum supermartingale certificate: convergence over 5 seeds '
             r'($m=200$, max refinement depth~2). Hardcoded IBP bounds sin and the '
             r'neural policy via interval arithmetic with no auto\_LiRPA. '
             r'Epoch and wall-clock are mean~$\pm$~std over converged seeds only.'),
    label='tab:pendulum_comparison',
    index=False,
)

latex = table_path.read_text()
latex = latex.replace(r'\begin{table}', r'\begin{table}[H]' + '\n\\centering')
table_path.write_text(latex)
