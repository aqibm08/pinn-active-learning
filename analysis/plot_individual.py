"""
Generates the individual paper figures from the shipped result JSONs, using
the Wong colour-blind-safe palette defined in plot_style. Output goes to
<repo>/figures (not tracked in git). Run from anywhere:

    python analysis/plot_individual.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from _paths import RESULTS, FIGS, PSA_DATA, add_psa_to_path
from plot_style import (
    apply_paper_style, METHOD_COLORS, METHOD_LABELS, METHOD_MARKERS,
    COL_AL, COL_SEQ_RANDOM, COL_RAND_RANDOM, COL_IC_CORESET,
    COL_PHYSICS, COL_GRADIENT, COL_QBC, COL_ANN_AL, COL_ANN_RR,
    annotate_bar, style_boxplot,
)


apply_paper_style()
OUT = FIGS; OUT.mkdir(parents=True, exist_ok=True)

STEP_DIRS_BIASED = {
    'ads':   RESULTS / 'psa_clustered_pool' / 'ads',
    'blow':  RESULTS / 'psa_clustered_pool' / 'blow',
    'evac':  RESULTS / 'psa_clustered_pool' / 'evac',
    'press': RESULTS / 'psa_clustered_pool' / 'press',
}
STEP_LABEL = {'ads': 'Adsorption', 'blow': 'Blowdown',
              'evac': 'Evacuation', 'press': 'Repressurisation'}


def load_psa(folder):
    f = Path(folder) / 'psa_v7_sweep_final.json'
    if not f.exists(): return None
    d = json.loads(f.read_text())
    out = {}
    for b_str, methods in d.items():
        b = int(b_str)
        out[b] = {m: [r['test_metrics']['mse_total']
                      for r in runs if r.get('test_metrics')]
                  for m, runs in methods.items()}
    return out


def load_cstr(folder=RESULTS / 'cstr'):
    f = Path(folder) / 'sweep_results_final.json'
    d = json.loads(f.read_text())
    out = {}
    for b_str, methods in d.items():
        b = int(b_str)
        out[b] = {m: [r['test_mse'] for r in runs if 'test_mse' in r]
                  for m, runs in methods.items()}
    return out


def pct_improvement(al, base):
    if not al or not base:
        return float('nan')
    return 100.0 * (1.0 - np.mean(al) / np.mean(base))


def save_fig(fig, name):
    out = OUT / f'{name}.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  wrote {out.name}')


# --- Individual: budget curve ---------------------------------------------
def _budget_curve(data, methods, name, ylabel='Test MSE',
                   xlabel='Labeled samples (N)', size=(9, 5.5)):
    fig, ax = plt.subplots(figsize=size)
    budgets = sorted(data.keys())
    for m in methods:
        vals = [(b, np.mean(data[b][m]), np.std(data[b][m], ddof=1))
                for b in budgets if data[b].get(m)]
        if not vals: continue
        bs = [v[0] for v in vals]
        mns = np.array([v[1] for v in vals])
        sds = np.array([v[2] for v in vals])
        n = len(data[bs[0]][m])
        se = sds / np.sqrt(max(1, n))
        low = np.maximum(mns - se, mns * 0.5)
        ax.plot(bs, mns, marker=METHOD_MARKERS[m], color=METHOD_COLORS[m],
                 markeredgecolor='black', markersize=12, markeredgewidth=0.8,
                 label=METHOD_LABELS[m], linewidth=2.6)
        ax.fill_between(bs, low, mns + se,
                         color=METHOD_COLORS[m], alpha=0.22)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    save_fig(fig, name)


def _improvement_bars(data, al_method, baselines, name, xlabel,
                       size=(8, 6.0), ylim=(0, 100)):
    fig, ax = plt.subplots(figsize=size)
    budgets = sorted(data.keys())
    n_bars = len(baselines)
    w = 0.8 / n_bars
    x = np.arange(len(budgets))
    for i, (base, color, label) in enumerate(baselines):
        pcts = [pct_improvement(data[b].get(al_method, []),
                                 data[b].get(base, [])) for b in budgets]
        offset = (i - (n_bars - 1) / 2) * w
        bars = ax.bar(x + offset, pcts, w, color=color, edgecolor='black',
                       linewidth=0.8, label=label)
        for bar, v in zip(bars, pcts):
            if v is None or (isinstance(v, float) and np.isnan(v)): continue
            ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                     f'{v:.0f}%', ha='center', va='bottom',
                     fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel(xlabel); ax.set_ylabel('MSE improvement (%)')
    ax.set_ylim(*ylim)
    ax.axhline(0, color='k', linewidth=0.5)
    # Legend OUTSIDE plot area (top), 2 columns - never overlaps bars
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
               ncol=n_bars, frameon=True, fontsize=12)
    save_fig(fig, name)


def _seed_boxes(data, budget, methods, name, size=(8, 5.5)):
    if budget not in data: return
    fig, ax = plt.subplots(figsize=size)
    box_data = [data[budget][m] for m in methods]
    bp = ax.boxplot(box_data,
                     tick_labels=[METHOD_LABELS[m] for m in methods],
                     patch_artist=True, widths=0.55,
                     medianprops=dict(color='black', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='white',
                                     markeredgecolor='black', markersize=7),
                     showmeans=True)
    style_boxplot(bp, [METHOD_COLORS[m] for m in methods])
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods],
                        rotation=12, ha='right')
    ax.set_ylabel('Test MSE'); ax.set_yscale('log')
    save_fig(fig, name)


# --- CSTR individual figures ----------------------------------------------
def figs_cstr(cstr):
    methods = ['sequential_al', 'sequential_random', 'random_random']
    _budget_curve(cstr, methods, 'cstr_budget_curve')
    _improvement_bars(cstr, 'sequential_al',
        [('sequential_random', COL_SEQ_RANDOM, 'vs Sequential Random'),
         ('random_random',     COL_RAND_RANDOM, 'vs Random-Random')],
        'cstr_improvement_bars', 'Labeled samples (N)',
        ylim=(0, 105))
    _seed_boxes(cstr, max(cstr.keys()), methods, 'cstr_seed_boxes')


# --- PSA per-step individual figures -------------------------------------
def figs_psa(psa_all):
    methods = ['pinn_al', 'pinn_seqrand', 'pinn_oneshot']
    for step in ['ads', 'blow', 'evac', 'press']:
        data = psa_all[step]
        if data is None: continue
        _budget_curve(data, methods, f'psa_{step}_budget_curve',
                       xlabel='Labeled scenarios (N)')
        _improvement_bars(data, 'pinn_al',
            [('pinn_seqrand', COL_SEQ_RANDOM, 'vs Sequential Random'),
             ('pinn_oneshot', COL_RAND_RANDOM, 'vs Random-Random')],
            f'psa_{step}_improvement_bars', 'Labeled scenarios (N)',
            ylim=(0, 100))
        _seed_boxes(data, 60, methods, f'psa_{step}_seed_boxes')


# --- PSA average-across-steps ---------------------------------------------
def fig_psa_avg(psa_all):
    psa_budgets = [12, 30, 60]
    steps = ['ads', 'blow', 'evac', 'press']
    sr = np.zeros((4, 3)); rr = np.zeros((4, 3))
    for i, s in enumerate(steps):
        for j, b in enumerate(psa_budgets):
            d = psa_all[s][b]
            sr[i, j] = pct_improvement(d.get('pinn_al', []),
                                        d.get('pinn_seqrand', []))
            rr[i, j] = pct_improvement(d.get('pinn_al', []),
                                        d.get('pinn_oneshot', []))
    sr_avg, sr_sd = sr.mean(axis=0), sr.std(axis=0, ddof=1)
    rr_avg, rr_sd = rr.mean(axis=0), rr.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(3); w = 0.38
    b1 = ax.bar(x - w/2, sr_avg, w, yerr=sr_sd, color=COL_SEQ_RANDOM,
                 capsize=6, edgecolor='black', linewidth=0.8,
                 label='vs Sequential Random')
    b2 = ax.bar(x + w/2, rr_avg, w, yerr=rr_sd, color=COL_RAND_RANDOM,
                 capsize=6, edgecolor='black', linewidth=0.8,
                 label='vs Random-Random')
    for bars, vals in [(b1, sr_avg), (b2, rr_avg)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()*0.42, v + 3,
                     f'{v:.0f}%', ha='right', va='bottom',
                     fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in psa_budgets])
    ax.set_xlabel('Labeled scenarios (N)'); ax.set_ylabel('MSE improvement (%)')
    ax.set_ylim(0, 100); ax.axhline(0, color='k', linewidth=0.5)
    ax.legend(loc='upper right')
    save_fig(fig, 'psa_avg_4steps_improvement')


# --- Pool comparison (SI) -------------------------------------------------
def figs_pool_compare(uniform, clustered):
    methods_compared = [
        ('pinn_seqrand', COL_SEQ_RANDOM, 'vs Sequential Random'),
        ('pinn_oneshot', COL_RAND_RANDOM, 'vs Random-Random'),
    ]
    _improvement_bars(uniform, 'pinn_al', methods_compared,
                       'si_uniform_pool_gains', 'Labeled scenarios (N)',
                       ylim=(-15, 75))
    _improvement_bars(clustered, 'pinn_al', methods_compared,
                       'si_clustered_pool_gains', 'Labeled scenarios (N)',
                       ylim=(0, 100))


# --- Signal ablation (SI) -------------------------------------------------
def fig_signal_ablation():
    f = RESULTS / 'psa_signal_ablation' / 'ablation_stepads_b30.json'
    if not f.exists():
        print(f'  [skip] {f}'); return
    d = json.loads(f.read_text())
    rows = []
    for variant, runs in d.items():
        if not runs: continue
        ints = [r['internal_mse'] for r in runs]
        rows.append((variant, np.mean(ints),
                      np.std(ints, ddof=1) if len(ints)>1 else 0))
    rows.sort(key=lambda r: r[1])
    labels = {
        'pinn_badge':       'BADGE diversity',
        'pinn_ic_coreset':  'IC-coreset',
        'pinn_v4_mix':      'v4 mixture (4 signals)',
        'pinn_qbc':         'Committee disagree',
        'pinn_physics':     'PDE residual',
        'pinn_gradient':    'Input-grad norm',
        'pinn_random':      'Random (baseline)',
    }
    colors = {
        'pinn_badge':       COL_AL,
        'pinn_ic_coreset':  COL_IC_CORESET,
        'pinn_v4_mix':      '#8c8c8c',
        'pinn_qbc':         COL_QBC,
        'pinn_physics':     COL_PHYSICS,
        'pinn_gradient':    COL_GRADIENT,
        'pinn_random':      COL_RAND_RANDOM,
    }
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(rows))
    means = [r[1] for r in rows]; sds = [r[2] for r in rows]
    cols = [colors.get(r[0], '#888') for r in rows]
    bars = ax.bar(x, means, yerr=sds, capsize=6, color=cols,
                   edgecolor='black', linewidth=0.8)
    for bar, r in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(means)*0.03,
                 f'{r[1]:.2e}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(r[0], r[0]) for r in rows],
                        rotation=20, ha='right')
    ax.set_ylabel('Internal test MSE  (b=30, ads)')
    save_fig(fig, 'si_signal_ablation')


# --- Pool density (Methods) -----------------------------------------------
def fig_pool_density():
    add_psa_to_path()
    from psa_data_utils import (load_step_data, identify_scenarios,
                                  split_scenarios, make_biased_scenario_pool)
    x_all, _ = load_step_data(str(PSA_DATA / 'train_data_ads_merged.mat'), step='ads')
    scens = identify_scenarios(x_all)
    pool_s, _, _ = split_scenarios(scens, 30, 40, seed=42)
    ic_u = np.stack([s['ic_summary'][:2] for s in pool_s])
    clustered = make_biased_scenario_pool(pool_s, 0.70, 42)
    ic_c = np.stack([s['ic_summary'][:2] for s in clustered])

    # Uniform
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ic_u[:, 0], ic_u[:, 1], s=70, alpha=0.65,
                c=COL_AL, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(r'Mean $y_{1,\,\mathrm{IC}}$ (CO$_2$ mole fraction)')
    ax.set_ylabel(r'Mean $P_{\mathrm{IC}}$ (dim.less)')
    save_fig(fig, 'methods_pool_uniform')

    # Clustered
    src_counts = {}
    for s in clustered:
        key = s.get('_replica_of', id(s))
        src_counts[key] = src_counts.get(key, 0) + 1
    sizes = np.array([src_counts.get(s.get('_replica_of', id(s)), 1)
                      for s in clustered]) * 30
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ic_c[:, 0], ic_c[:, 1], s=sizes, alpha=0.55,
                c=COL_RAND_RANDOM, edgecolors='black', linewidth=0.5)
    ax.set_xlabel(r'Mean $y_{1,\,\mathrm{IC}}$ (CO$_2$ mole fraction)')
    ax.set_ylabel(r'Mean $P_{\mathrm{IC}}$ (dim.less)')
    save_fig(fig, 'methods_pool_clustered')


# --- Convergence (SI) -----------------------------------------------------
def fig_convergence(folder=None, budget=60, seed=42):
    f = Path(folder or RESULTS / 'psa_clustered_pool' / 'ads') / f'psa_v7_sweep_budget{budget}.json'
    if not f.exists():
        print(f'  [skip] {f}'); return
    d = json.loads(f.read_text())
    methods = ['pinn_al', 'pinn_seqrand', 'pinn_oneshot']
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in methods:
        for r in d[str(budget)].get(m, []):
            if r.get('seed') == seed:
                h = r.get('val_mse_history', [])
                if not h: continue
                epochs = np.arange(len(h)) * 5
                ax.plot(epochs, h, color=METHOD_COLORS[m], linewidth=2.6,
                         label=METHOD_LABELS[m])
                break
    ax.set_xlabel('Training epoch'); ax.set_ylabel('Validation MSE')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    save_fig(fig, 'si_convergence_b60_seed42')


# --- Main -----------------------------------------------------------------
def main():
    print('Loading data...')
    cstr = load_cstr()
    psa_all = {s: load_psa(d) for s, d in STEP_DIRS_BIASED.items()}
    uniform = load_psa(RESULTS / 'psa_uniform_pool')
    print(f'  Generating individual figures into {OUT}/')
    figs_cstr(cstr)
    figs_psa(psa_all)
    fig_psa_avg(psa_all)
    figs_pool_compare(uniform, psa_all['ads'])
    fig_signal_ablation()
    fig_pool_density()
    fig_convergence()
    print('Done.')


if __name__ == '__main__':
    main()
