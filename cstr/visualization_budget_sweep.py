import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# -- Global style --------------------------------------------------------------
plt.rcParams.update({
    'figure.dpi':          600,
    'savefig.dpi':         600,
    'font.family':         'DejaVu Sans',
    'font.size':           14,
    'axes.titlesize':      16,
    'axes.labelsize':      15,
    'axes.titleweight':    'bold',
    'axes.labelweight':    'bold',
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'axes.grid':           True,
    'grid.alpha':          0.20,
    'grid.linestyle':      '--',
    'grid.linewidth':      0.6,
    'lines.linewidth':     2.8,
    'legend.fontsize':     13,
    'legend.framealpha':   0.92,
    'legend.edgecolor':    '#cccccc',
    'legend.frameon':      True,
    'xtick.labelsize':     13,
    'ytick.labelsize':     13,
    'xtick.major.size':    6,
    'ytick.major.size':    6,
    'xtick.major.width':   1.4,
    'ytick.major.width':   1.4,
    'xtick.minor.size':    3,
    'ytick.minor.size':    3,
    'figure.constrained_layout.use': False,
})

# -- Method look-up tables (support all historic + current keys) ---------------
METHOD_COLOR = {
    'sequential_al':    '#1f77b4',
    'random_random':    '#d62728',
    'sequential_random':'#ff7f0e',
    'full_al':          '#1f77b4',
    'warm_random':      '#2ca02c',
    'al_random':        '#9467bd',
    'random_al':        '#8c564b',
}
METHOD_LABEL = {
    'sequential_al':    'Sequential AL (Our Work)',
    'random_random':    'Random-Random',
    'sequential_random':'Sequential Random',
    'full_al':          'Full AL (Our Work)',
    'warm_random':      'Warm-Random',
    'al_random':        'AL Data + Rand. Coll.',
    'random_al':        'Rand. Data + AL Coll.',
}
METHOD_MARKER = {
    'sequential_al':    'o',
    'random_random':    's',
    'sequential_random':'^',
    'full_al':          'o',
    'warm_random':      'D',
    'al_random':        'v',
    'random_al':        'P',
}
METHOD_DASH = {
    'sequential_al':    '-',
    'random_random':    '--',
    'sequential_random':'-.',
    'full_al':          '-',
    'warm_random':      ':',
}

OUTPUT_NAMES = ['Ca', 'T', 'Tc', 'h']
OUTPUT_UNITS = ['mol/L', 'K', 'K', 'm']
OUTPUT_COLOR = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']


# -- Helpers -------------------------------------------------------------------
def _primary_al_key(results):
    """Return the AL method key present in results."""
    sample = next(iter(results.values()))
    for k in ('sequential_al', 'full_al'):
        if k in sample:
            return k
    return list(sample.keys())[0]


def _agg(results, metric='test_mse'):
    methods = list(next(iter(results.values())).keys())
    out = {}
    for method in methods:
        budgets, means, stds, allv = [], [], [], []
        for b in sorted(results.keys()):
            vals = [r[metric] for r in results[b].get(method, [])
                    if metric in r and r[metric] == r[metric]]
            if vals:
                budgets.append(b)
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
                allv.append(vals)
        out[method] = {'budgets': budgets, 'mean': means,
                       'std': stds, 'all': allv}
    return out


def _best_seed(results, budget, method):
    runs = results.get(budget, {}).get(method, [])
    valid = [r for r in runs if 'test_mse' in r and r['test_mse'] == r['test_mse']]
    if not valid:
        return None
    return min(valid, key=lambda r: r['test_mse'])


def _label(m):
    return METHOD_LABEL.get(m, m)

def _color(m):
    return METHOD_COLOR.get(m, '#666666')

def _marker(m):
    return METHOD_MARKER.get(m, 'o')

def _dash(m):
    return METHOD_DASH.get(m, '-')


def _finish(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  [ok] {path}')


# ==============================================================================
# FIG 1 - Sample efficiency (MSE)   ->  Sec. 4.1 Main result
# ==============================================================================
def plot_sample_efficiency(results, out_dir, metric='test_mse'):
    stats  = _agg(results, metric)
    ylabel = 'Test MSE' if metric == 'test_mse' else 'Test MAE'
    fname  = f'sample_efficiency_{metric.split("_")[1]}.png'

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method, st in stats.items():
        if not st['budgets']:
            continue
        b  = np.array(st['budgets'])
        mu = np.array(st['mean'])
        sd = np.array(st['std'])
        lw = 3.2 if 'al' in method and 'random' not in method else 2.2
        zord = 4 if 'al' in method and 'random' not in method else 3
        ax.plot(b, mu,
                color=_color(method), marker=_marker(method),
                linestyle=_dash(method),
                linewidth=lw, markersize=9, zorder=zord,
                label=_label(method))
        ax.fill_between(b, mu - sd, mu + sd,
                        color=_color(method), alpha=0.12, zorder=zord-1)

    ax.set_yscale('log')
    ax.set_xlabel('Number of Labeled Samples (N)')
    ax.set_ylabel(ylabel)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.legend(loc='upper right', handlelength=2.4)
    _finish(fig, out_dir / fname)


# ==============================================================================
# FIG 2 - Per-output sample efficiency   ->  Sec. 4.9 Per-variable breakdown
# ==============================================================================
def plot_per_output_efficiency(results, out_dir):
    al_key = _primary_al_key(results)
    budgets = sorted(results.keys())
    methods = [al_key, 'random_random']

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for oi, (oname, ounit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
        ax = axes[oi]
        for method in methods:
            b_list, mu_list, sd_list = [], [], []
            for b in budgets:
                runs = [r for r in results[b].get(method, [])
                        if 'output_mse' in r and r['output_mse']]
                if runs:
                    vals = [r['output_mse'][oi] for r in runs]
                    b_list.append(b)
                    mu_list.append(np.mean(vals))
                    sd_list.append(np.std(vals))
            if b_list:
                b_a  = np.array(b_list)
                mu_a = np.array(mu_list)
                sd_a = np.array(sd_list)
                lw   = 3.0 if method == al_key else 2.0
                ax.plot(b_a, mu_a, color=_color(method),
                        marker=_marker(method), linestyle=_dash(method),
                        linewidth=lw, markersize=8, label=_label(method))
                ax.fill_between(b_a, mu_a - sd_a, mu_a + sd_a,
                                color=_color(method), alpha=0.12)
        ax.set_yscale('log')
        ax.set_xlabel('Labeled Samples (N)')
        ax.set_ylabel(f'MSE  [{ounit}^2]')
        ax.set_title(f'Output: {oname}', fontweight='bold')
        ax.legend(fontsize=11)

    fig.tight_layout(pad=2.0)
    _finish(fig, out_dir / 'sample_efficiency_per_output.png', tight=False)


# ==============================================================================
# FIG 3 - % improvement vs budget   ->  Sec. 4.1 / Sec. 4.3
# ==============================================================================
def plot_improvement_vs_budget(results, out_dir):
    al_key  = _primary_al_key(results)
    budgets = sorted(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ai, metric in enumerate(['test_mse', 'test_mae']):
        ax    = axes[ai]
        label = 'MSE' if metric == 'test_mse' else 'MAE'
        imps, b_plot = [], []

        for b in budgets:
            rr  = [r[metric] for r in results[b].get('random_random', [])
                   if metric in r and r[metric] == r[metric]]
            al  = [r[metric] for r in results[b].get(al_key, [])
                   if metric in r and r[metric] == r[metric]]
            if rr and al:
                imp = (np.mean(rr) - np.mean(al)) / np.mean(rr) * 100
                imps.append(imp)
                b_plot.append(b)

        colors_bar = ['#1f77b4' if v >= 0 else '#d62728' for v in imps]
        bars = ax.bar(b_plot, imps, color=colors_bar,
                      width=22, edgecolor='white', linewidth=1.2, alpha=0.88)

        for bar, val in zip(bars, imps):
            ypos = val + 1.5 if val >= 0 else val - 3.5
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:+.0f}%', ha='center', va='bottom',
                    fontsize=11, fontweight='bold', color='#333333')

        ax.axhline(0, color='#555555', linewidth=1.2, linestyle='--')
        ax.set_xlabel('Labeled Samples (N)')
        ax.set_ylabel(f'{label} Improvement over Random-Random (%)')
        ax.set_xticks(b_plot)
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    fig.tight_layout()
    _finish(fig, out_dir / 'improvement_vs_budget.png', tight=False)


# ==============================================================================
# FIG 4 - Bar charts at each budget   ->  Sec. 4.1
# ==============================================================================
def plot_budget_bars(results, out_dir):
    al_key  = _primary_al_key(results)
    budgets = sorted(results.keys())
    methods = [m for m in next(iter(results.values())).keys()]
    n_m     = len(methods)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for bi, b in enumerate(budgets):
        ax = axes[bi]
        vals, errs, cols, labs = [], [], [], []
        for m in methods:
            runs = [r['test_mse'] for r in results[b].get(m, [])
                    if 'test_mse' in r and r['test_mse'] == r['test_mse']]
            if runs:
                vals.append(np.mean(runs))
                errs.append(np.std(runs))
                cols.append(_color(m))
                labs.append(_label(m).replace(' (Our Work)', '\n(Our Work)'))

        x = np.arange(len(vals))
        bars = ax.bar(x, vals, yerr=errs, color=cols,
                      capsize=5, edgecolor='white', linewidth=1.0,
                      alpha=0.87, error_kw={'linewidth': 2, 'ecolor': '#444'})
        ax.set_xticks(x)
        ax.set_xticklabels(labs, fontsize=10)
        ax.set_ylabel('Test MSE', fontsize=11)
        ax.set_title(f'N = {b}', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs='auto', numticks=5))

    # Hide unused subplot
    for idx in range(len(budgets), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout(pad=2.2)
    _finish(fig, out_dir / 'budget_bar_charts.png', tight=False)


# ==============================================================================
# FIG 5 - Round-by-round convergence   ->  Sec. 3.6 / Sec. 4.1
# ==============================================================================
def plot_convergence_grid(results, out_dir):
    al_key  = _primary_al_key(results)
    budgets = sorted(results.keys())
    n_b     = len(budgets)
    ncols   = min(4, n_b)
    nrows   = (n_b + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    focus_methods = [al_key, 'random_random']

    for bi, b in enumerate(budgets):
        ax = axes[bi]
        for method in focus_methods:
            run = _best_seed(results, b, method)
            if run is None:
                continue
            hist = run.get('val_mse_history') or run.get('total_loss_history', [])
            if not hist:
                continue
            ax.semilogy(range(len(hist)), hist,
                        color=_color(method), linewidth=2.6,
                        linestyle=_dash(method), label=_label(method))

        ax.set_xlabel('Round / Epoch')
        ax.set_ylabel('Val MSE')
        ax.set_title(f'N = {b}', fontweight='bold')
        ax.legend(fontsize=10)

    for idx in range(len(budgets), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout(pad=2.2)
    _finish(fig, out_dir / 'convergence_grid.png', tight=False)


# ==============================================================================
# FIG 6 - True vs predicted   ->  Sec. 4.9 Generalisation
# ==============================================================================
def plot_true_vs_predicted(results, out_dir):
    al_key   = _primary_al_key(results)
    budgets  = sorted(results.keys())
    focus_b  = budgets[len(budgets) // 2]

    al_run = _best_seed(results, focus_b, al_key)
    rr_run = _best_seed(results, focus_b, 'random_random')
    if al_run is None or 'test_predictions' not in al_run:
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    pairs = [(al_run, 'Sequential AL (Our Work)', 0),
             (rr_run, 'Random-Random', 1)]

    for row, (run, title, ri) in enumerate(pairs):
        if run is None or 'test_predictions' not in run:
            continue
        y_pred = np.array(run['test_predictions'])
        y_true = np.array(run['test_targets'])

        for oi, (oname, ounit) in enumerate(zip(OUTPUT_NAMES, OUTPUT_UNITS)):
            ax  = axes[ri, oi]
            yt  = y_true[:, oi]
            yp  = y_pred[:, oi]
            lim = [min(yt.min(), yp.min()) * 0.98,
                   max(yt.max(), yp.max()) * 1.02]

            ax.scatter(yt, yp, color=_color(al_key if ri == 0 else 'random_random'),
                       alpha=0.45, s=18, edgecolors='none')
            ax.plot(lim, lim, 'k--', linewidth=1.8, alpha=0.7, label='Ideal')

            mse = np.mean((yt - yp)**2)
            ax.text(0.05, 0.93, f'MSE={mse:.2e}',
                    transform=ax.transAxes, fontsize=11,
                    va='top', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.8, edgecolor='#ccc'))

            ax.set_xlabel(f'True {oname}  [{ounit}]')
            ax.set_ylabel(f'Predicted {oname}  [{ounit}]')
            if oi == 0:
                ax.set_title(title, fontweight='bold', fontsize=13)
            ax.set_xlim(lim); ax.set_ylim(lim)

    fig.tight_layout(pad=2.2)
    _finish(fig, out_dir / 'true_vs_predicted_best.png', tight=False)


# ==============================================================================
# FIG 7 - Seed variance (violin)   ->  Sec. 4.1 Statistical significance
# ==============================================================================
def plot_seed_variance(results, out_dir):
    al_key  = _primary_al_key(results)
    budgets = sorted(results.keys())
    methods = list(next(iter(results.values())).keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ai, metric in enumerate(['test_mse', 'test_mae']):
        ax = axes[ai]
        positions, data_list, colors_list = [], [], []
        gap   = len(methods) + 1
        ticks = []
        tick_labs = []

        for bi, b in enumerate(budgets):
            base_pos = bi * gap
            ticks.append(base_pos + (len(methods) - 1) / 2)
            tick_labs.append(str(b))
            for mi, method in enumerate(methods):
                vals = [r[metric] for r in results[b].get(method, [])
                        if metric in r and r[metric] == r[metric]]
                if vals:
                    positions.append(base_pos + mi)
                    data_list.append(vals)
                    colors_list.append(_color(method))

        parts = ax.violinplot(data_list, positions=positions,
                              showmeans=True, showextrema=True, widths=0.7)
        for pc, col in zip(parts['bodies'], colors_list):
            pc.set_facecolor(col)
            pc.set_alpha(0.55)
        for k in ('cmeans', 'cmins', 'cmaxes', 'cbars'):
            if k in parts:
                parts[k].set_color('#333333')
                parts[k].set_linewidth(1.5)

        ax.set_yscale('log')
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labs)
        ax.set_xlabel('Labeled Samples (N)')
        ax.set_ylabel('Test MSE' if metric == 'test_mse' else 'Test MAE')

        legend_elems = [Line2D([0],[0], color=_color(m), lw=5,
                                label=_label(m)) for m in methods]
        ax.legend(handles=legend_elems, fontsize=10, loc='upper right')

    fig.tight_layout()
    _finish(fig, out_dir / 'seed_variance_analysis.png', tight=False)


# ==============================================================================
# FIG 8 - Computational cost   ->  Sec. 4.7
# ==============================================================================
def plot_computational_cost(results, out_dir):
    al_key  = _primary_al_key(results)
    budgets = sorted(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: wall time vs budget
    ax = axes[0]
    for method in next(iter(results.values())).keys():
        b_list, t_list = [], []
        for b in budgets:
            ts = [r['wall_time'] for r in results[b].get(method, [])
                  if 'wall_time' in r]
            if ts:
                b_list.append(b)
                t_list.append(np.mean(ts) / 60)
        if b_list:
            ax.plot(b_list, t_list, color=_color(method),
                    marker=_marker(method), linestyle=_dash(method),
                    linewidth=2.6, markersize=8, label=_label(method))

    ax.set_xlabel('Labeled Samples (N)')
    ax.set_ylabel('Wall Time (min)')
    ax.legend(fontsize=11)

    # Panel 2: MSE per minute (efficiency)
    ax = axes[1]
    for b in budgets:
        al_runs = results[b].get(al_key, [])
        rr_runs = results[b].get('random_random', [])
        al_valid = [r for r in al_runs if 'test_mse' in r and 'wall_time' in r]
        rr_valid = [r for r in rr_runs if 'test_mse' in r and 'wall_time' in r]
        if al_valid and rr_valid:
            al_mse = np.mean([r['test_mse'] for r in al_valid])
            rr_mse = np.mean([r['test_mse'] for r in rr_valid])
            al_t   = np.mean([r['wall_time'] for r in al_valid]) / 60
            rr_t   = np.mean([r['wall_time'] for r in rr_valid]) / 60
            ratio  = (rr_mse / al_mse) if al_mse > 0 else 0
            ax.bar(b - 12, al_mse, width=20,
                   color=_color(al_key), alpha=0.85, label=_label(al_key) if b == budgets[0] else '')
            ax.bar(b + 12, rr_mse, width=20,
                   color=_color('random_random'), alpha=0.85,
                   label=_label('random_random') if b == budgets[0] else '')

    ax.set_yscale('log')
    ax.set_xlabel('Labeled Samples (N)')
    ax.set_ylabel('Test MSE')
    ax.legend(fontsize=11)

    fig.tight_layout()
    _finish(fig, out_dir / 'computational_cost_sweep.png', tight=False)


# ==============================================================================
# FIG 9 - Uncertainty at test time   ->  Sec. 3.2 / Sec. 4.3
# ==============================================================================
def plot_uncertainty_vs_budget(results, out_dir):
    al_key  = _primary_al_key(results)
    budgets = sorted(results.keys())

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method in [al_key, 'random_random']:
        b_list, u_list, u_std = [], [], []
        for b in budgets:
            runs = [r for r in results[b].get(method, [])
                    if 'test_epistemic' in r and r['test_epistemic']]
            if runs:
                unc = [np.mean(r['test_epistemic']) for r in runs]
                b_list.append(b)
                u_list.append(np.mean(unc))
                u_std.append(np.std(unc))
        if b_list:
            b_a  = np.array(b_list)
            mu_a = np.array(u_list)
            sd_a = np.array(u_std)
            ax.plot(b_a, mu_a, color=_color(method),
                    marker=_marker(method), linestyle=_dash(method),
                    linewidth=2.8, markersize=9, label=_label(method))
            ax.fill_between(b_a, mu_a - sd_a, mu_a + sd_a,
                            color=_color(method), alpha=0.12)

    ax.set_yscale('log')
    ax.set_xlabel('Labeled Samples (N)')
    ax.set_ylabel('Mean Epistemic Uncertainty (beta / (alpha nu))')
    ax.legend(loc='upper right')
    _finish(fig, out_dir / 'uncertainty_vs_budget.png')


# ==============================================================================
# FIG 10 - Sequential round progression   ->  Sec. 3.6 Sequential protocol
# ==============================================================================
def plot_round_progression(results, out_dir):
    """Show how MSE evolves round-by-round as data is added sequentially."""
    al_key  = _primary_al_key(results)
    budgets = sorted(results.keys())
    # Pick 3 representative budgets
    sel_budgets = [budgets[0], budgets[len(budgets)//2], budgets[-1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for bi, b in enumerate(sel_budgets):
        ax = axes[bi]
        for method in [al_key, 'sequential_random', 'random_random']:
            runs = results[b].get(method, [])
            valid = [r for r in runs
                     if 'round_mse_history' in r and r['round_mse_history']]
            if not valid:
                valid = [r for r in runs
                         if 'val_mse_history' in r and r['val_mse_history']]
            if not valid:
                continue
            # Average over seeds
            max_len = max(len(r.get('round_mse_history',
                                    r.get('val_mse_history', [])))
                          for r in valid)
            mat = np.full((len(valid), max_len), np.nan)
            for ri, r in enumerate(valid):
                h = r.get('round_mse_history', r.get('val_mse_history', []))
                mat[ri, :len(h)] = h
            mu = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0)
            xs = np.arange(len(mu))
            ax.semilogy(xs, mu, color=_color(method),
                        linestyle=_dash(method), linewidth=2.6,
                        label=_label(method))
            ax.fill_between(xs, mu - sd, mu + sd,
                            color=_color(method), alpha=0.12)

        ax.set_xlabel('Query Round')
        ax.set_ylabel('Test MSE')
        ax.set_title(f'N = {b}', fontweight='bold')
        ax.legend(fontsize=10)

    fig.tight_layout()
    _finish(fig, out_dir / 'round_progression.png', tight=False)


# ==============================================================================
# FIG 11 - Pool structure (PCA)   ->  Sec. 3.7 Biased pool
# ==============================================================================
def plot_pool_structure(results, out_dir):
    """
    We don't have the raw pool here, so we visualise
    the SELECTED POINTS distributions across seeds as a proxy.
    Uses test_predictions spread as a stand-in if pool not available.
    """
    al_key = _primary_al_key(results)
    budget = sorted(results.keys())[2]   # mid budget

    al_run = _best_seed(results, budget, al_key)
    rr_run = _best_seed(results, budget, 'random_random')
    if al_run is None or 'test_predictions' not in al_run:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ai, (run, title) in enumerate([(al_run, 'Sequential AL (Our Work)'),
                                        (rr_run, 'Random-Random')]):
        ax = axes[ai]
        if run is None or 'test_predictions' not in run:
            ax.set_visible(False)
            continue
        pred = np.array(run['test_predictions'])
        true = np.array(run['test_targets'])
        # Plot first two output dims as proxy for distribution coverage
        ax.scatter(true[:, 0], true[:, 1], c='#aaaaaa',
                   alpha=0.35, s=14, label='True', edgecolors='none')
        ax.scatter(pred[:, 0], pred[:, 1],
                   c=_color(al_key if ai == 0 else 'random_random'),
                   alpha=0.55, s=18, label='Predicted', edgecolors='none')
        ax.set_xlabel(f'{OUTPUT_NAMES[0]}  [{OUTPUT_UNITS[0]}]')
        ax.set_ylabel(f'{OUTPUT_NAMES[1]}  [{OUTPUT_UNITS[1]}]')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=11)

    fig.tight_layout()
    _finish(fig, out_dir / 'prediction_coverage.png', tight=False)


# ==============================================================================
# FIG 12 - Paper summary (5-panel)   ->  Main paper figure
# ==============================================================================
def plot_paper_summary(results, out_dir):
    al_key  = _primary_al_key(results)
    stats   = _agg(results, 'test_mse')
    budgets = sorted(results.keys())

    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.34)

    # -- Panel A: Sample efficiency --------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    for method, st in stats.items():
        if not st['budgets']:
            continue
        b  = np.array(st['budgets'])
        mu = np.array(st['mean'])
        sd = np.array(st['std'])
        lw = 3.2 if method == al_key else 2.0
        ax.plot(b, mu, color=_color(method), marker=_marker(method),
                linestyle=_dash(method), linewidth=lw, markersize=8,
                label=_label(method))
        ax.fill_between(b, mu - sd, mu + sd,
                        color=_color(method), alpha=0.12)
    ax.set_yscale('log')
    ax.set_xlabel('Labeled Samples (N)')
    ax.set_ylabel('Test MSE')
    ax.text(0.04, 0.97, '(a)', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')
    ax.legend(fontsize=10, loc='upper right')

    # -- Panel B: % improvement -----------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    imps, b_vals = [], []
    for b in budgets:
        rr = [r['test_mse'] for r in results[b].get('random_random', [])
              if 'test_mse' in r]
        al = [r['test_mse'] for r in results[b].get(al_key, [])
              if 'test_mse' in r]
        if rr and al:
            imps.append((np.mean(rr) - np.mean(al)) / np.mean(rr) * 100)
            b_vals.append(b)
    col_bars = ['#1f77b4' if v >= 0 else '#d62728' for v in imps]
    ax.bar(b_vals, imps, color=col_bars, width=22,
           edgecolor='white', linewidth=1, alpha=0.88)
    for bv, imp in zip(b_vals, imps):
        ax.text(bv, imp + 1.5, f'{imp:.0f}%', ha='center',
                fontsize=11, fontweight='bold', color='#222')
    ax.axhline(0, color='#555', linewidth=1.2, linestyle='--')
    ax.set_xlabel('Labeled Samples (N)')
    ax.set_ylabel('MSE Improvement over RR (%)')
    ax.set_xticks(b_vals)
    ax.text(0.04, 0.97, '(b)', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')

    # -- Panel C: Per-output MSE at mid-budget --------------------------------
    ax    = fig.add_subplot(gs[0, 2])
    mid_b = budgets[len(budgets) // 2]
    x     = np.arange(4)
    for mi, method in enumerate([al_key, 'random_random']):
        runs = [r for r in results[mid_b].get(method, [])
                if 'output_mse' in r and r['output_mse']]
        if runs:
            mu  = np.mean([r['output_mse'] for r in runs], axis=0)
            sd  = np.std( [r['output_mse'] for r in runs], axis=0)
            off = (mi - 0.5) * 0.38
            ax.bar(x + off, mu, 0.36, yerr=sd, color=_color(method),
                   alpha=0.85, capsize=4, edgecolor='white',
                   label=_label(method))
    ax.set_xticks(x)
    ax.set_xticklabels(OUTPUT_NAMES)
    ax.set_yscale('log')
    ax.set_xlabel('Output Variable')
    ax.set_ylabel(f'MSE  (N={mid_b})')
    ax.legend(fontsize=10)
    ax.text(0.04, 0.97, '(c)', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')

    # -- Panel D: True vs predicted for best AL seed --------------------------
    ax    = fig.add_subplot(gs[1, 0])
    focus = budgets[-2]
    run   = _best_seed(results, focus, al_key)
    if run and 'test_predictions' in run:
        yt = np.array(run['test_targets'])[:, 0]
        yp = np.array(run['test_predictions'])[:, 0]
        lim = [min(yt.min(), yp.min()) * 0.98,
               max(yt.max(), yp.max()) * 1.02]
        ax.scatter(yt, yp, color=_color(al_key), alpha=0.45,
                   s=15, edgecolors='none')
        ax.plot(lim, lim, 'k--', linewidth=2, alpha=0.7)
        mse = np.mean((yt - yp)**2)
        ax.text(0.05, 0.93, f'Ca  MSE={mse:.2e}',
                transform=ax.transAxes, fontsize=11, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.85, edgecolor='#ccc'))
        ax.set_xlabel(f'True Ca  [mol/L]')
        ax.set_ylabel(f'Predicted Ca  [mol/L]')
        ax.set_xlim(lim); ax.set_ylim(lim)
    ax.text(0.04, 0.97, '(d)', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')
    ax.set_title(f'Sequential AL - N={focus}', fontweight='bold', fontsize=12)

    # -- Panel E: Seed variance at key budgets ---------------------------------
    ax      = fig.add_subplot(gs[1, 1])
    sel_b   = [budgets[0], budgets[2], budgets[-1]]
    methods = list(next(iter(results.values())).keys())
    positions, data_list, cols_viol = [], [], []
    ticks, tick_labs = [], []
    gap = len(methods) + 1
    for bi, b in enumerate(sel_b):
        base = bi * gap
        ticks.append(base + (len(methods) - 1) / 2)
        tick_labs.append(f'N={b}')
        for mi, m in enumerate(methods):
            vals = [r['test_mse'] for r in results[b].get(m, [])
                    if 'test_mse' in r and r['test_mse'] == r['test_mse']]
            if vals:
                positions.append(base + mi)
                data_list.append(vals)
                cols_viol.append(_color(m))

    if data_list:
        parts = ax.violinplot(data_list, positions=positions,
                              showmeans=True, widths=0.7)
        for pc, col in zip(parts['bodies'], cols_viol):
            pc.set_facecolor(col); pc.set_alpha(0.55)
        for k in ('cmeans', 'cmins', 'cmaxes', 'cbars'):
            if k in parts:
                parts[k].set_color('#333')
                parts[k].set_linewidth(1.5)
    ax.set_yscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labs)
    ax.set_ylabel('Test MSE')
    legend_e = [Line2D([0],[0], color=_color(m), lw=5, label=_label(m))
                for m in methods]
    ax.legend(handles=legend_e, fontsize=9, loc='upper right')
    ax.text(0.04, 0.97, '(e)', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')

    # -- Panel F: Uncertainty vs budget ---------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    for method in [al_key, 'random_random']:
        b_list, u_list = [], []
        for b in budgets:
            runs = [r for r in results[b].get(method, [])
                    if 'test_epistemic' in r and r['test_epistemic']]
            if runs:
                b_list.append(b)
                u_list.append(np.mean([np.mean(r['test_epistemic'])
                                       for r in runs]))
        if b_list:
            ax.plot(b_list, u_list, color=_color(method),
                    marker=_marker(method), linestyle=_dash(method),
                    linewidth=2.8, markersize=8, label=_label(method))
    ax.set_yscale('log')
    ax.set_xlabel('Labeled Samples (N)')
    ax.set_ylabel('Mean Epistemic Uncertainty')
    ax.legend(fontsize=10)
    ax.text(0.04, 0.97, '(f)', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')

    _finish(fig, out_dir / 'paper_summary_figure.png', tight=False)


# ==============================================================================
# MASTER CALLER
# ==============================================================================
def generate_sweep_visualizations(results, cfg):
    import matplotlib.ticker
    globals()['matplotlib'] = matplotlib   # make ticker available inside fns

    out = Path(cfg.get('results_dir', 'sweep_results')) / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    print(f'\nGenerating sweep visualizations -> {out}/')

    fns = [
        ('sample_efficiency_mse.png',       lambda: plot_sample_efficiency(results, out, 'test_mse')),
        ('sample_efficiency_mae.png',        lambda: plot_sample_efficiency(results, out, 'test_mae')),
        ('sample_efficiency_per_output.png', lambda: plot_per_output_efficiency(results, out)),
        ('improvement_vs_budget.png',        lambda: plot_improvement_vs_budget(results, out)),
        ('budget_bar_charts.png',            lambda: plot_budget_bars(results, out)),
        ('convergence_grid.png',             lambda: plot_convergence_grid(results, out)),
        ('true_vs_predicted_best.png',       lambda: plot_true_vs_predicted(results, out)),
        ('seed_variance_analysis.png',       lambda: plot_seed_variance(results, out)),
        ('computational_cost_sweep.png',     lambda: plot_computational_cost(results, out)),
        ('uncertainty_vs_budget.png',        lambda: plot_uncertainty_vs_budget(results, out)),
        ('round_progression.png',            lambda: plot_round_progression(results, out)),
        ('prediction_coverage.png',          lambda: plot_pool_structure(results, out)),
        ('paper_summary_figure.png',         lambda: plot_paper_summary(results, out)),
    ]

    for fname, fn in fns:
        try:
            fn()
        except Exception as e:
            import traceback
            print(f'  [warn]  {fname}: {e}')
            traceback.print_exc()
