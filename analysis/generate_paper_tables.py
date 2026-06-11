"""
Builds the paper's Table 3 and Table 4 (markdown) from the shipped JSONs.

Table 3: headline results per case study, budget and method - mean MSE +/-
std across 5 seeds with paired one-sided Wilcoxon markers.
Table 4: per-output PSA breakdown at b=60 for each cycle step.

Reads results/cstr and results/psa_clustered_pool; writes <repo>/tables.
"""
import json
from pathlib import Path

from _paths import RESULTS, TABLES
import numpy as np
from scipy import stats

OUT = TABLES; OUT.mkdir(exist_ok=True)

STEP_DIRS_BIASED = {
    'ads':   RESULTS / 'psa_clustered_pool' / 'ads',
    'blow':  RESULTS / 'psa_clustered_pool' / 'blow',
    'evac':  RESULTS / 'psa_clustered_pool' / 'evac',
    'press': RESULTS / 'psa_clustered_pool' / 'press',
}
STEP_LABEL = {'ads': 'Adsorption', 'blow': 'Blowdown',
              'evac': 'Evacuation', 'press': 'Repressurisation'}


def load_psa(folder, key='test_metrics'):
    f = Path(folder) / 'psa_v7_sweep_final.json'
    if not f.exists(): return None
    d = json.loads(f.read_text())
    out = {}
    for b_str, methods in d.items():
        b = int(b_str)
        out[b] = {m: [r[key] for r in runs if r.get(key)]
                  for m, runs in methods.items()}
    return out


def load_cstr(folder=RESULTS / 'cstr'):
    f = Path(folder) / 'sweep_results_final.json'
    d = json.loads(f.read_text())
    out = {}
    for b_str, methods in d.items():
        b = int(b_str)
        out[b] = {m: [r for r in runs if 'test_mse' in r]
                  for m, runs in methods.items()}
    return out


def fmt_msd(vals):
    """Format mean +/- std as '1.23e-04 +/- 5.6e-05'."""
    if not vals:
        return 'n/a'
    a = np.array(vals)
    return f'{a.mean():.2e} +/- {a.std(ddof=1):.1e}'


def wilcoxon_p(al_vals, base_vals):
    n = min(len(al_vals), len(base_vals))
    if n < 2: return float('nan')
    try:
        _, p = stats.wilcoxon(al_vals[:n], base_vals[:n], alternative='less')
        return p
    except Exception:
        return float('nan')


def build_table3(cstr, psa_all):
    """Headline results table.  Mean +/- std across 5 seeds, both case studies."""
    lines = []
    lines.append('# Table 3 - Headline test-MSE results (mean +/- std across 5 seeds)')
    lines.append('')
    lines.append('Lower is better.  Annotations: $\\dagger$  Wilcoxon paired one-sided '
                  '$p \\le 0.05$ vs PINN+AL; $\\ddagger$  $p \\le 0.10$.')
    lines.append('')

    # --- CSTR ------------------------------------------------------------
    lines.append('## (a) CSTR')
    lines.append('')
    cstr_budgets = sorted(cstr.keys())
    cstr_methods = ['sequential_al', 'sequential_random', 'random_random']
    cstr_labels = {
        'sequential_al':     'PINN+AL (Our Work)',
        'sequential_random': 'Sequential Random',
        'random_random':     'Random-Random',
    }
    hdr = '| Budget $N$ | ' + ' | '.join(cstr_labels[m] for m in cstr_methods) + ' |'
    sep = '|---|' + '---|' * len(cstr_methods)
    lines.append(hdr); lines.append(sep)
    for b in cstr_budgets:
        row = [str(b)]
        al_mse = [r['test_mse'] for r in cstr[b].get('sequential_al', [])]
        for m in cstr_methods:
            mse_vals = [r['test_mse'] for r in cstr[b].get(m, [])]
            cell = fmt_msd(mse_vals)
            if m != 'sequential_al' and al_mse and mse_vals:
                p = wilcoxon_p(al_mse, mse_vals)
                if p <= 0.05: cell += ' $\\dagger$'
                elif p <= 0.10: cell += ' $\\ddagger$'
            row.append(cell)
        lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')

    # --- PSA (per step) --------------------------------------------------
    lines.append('## (b) PSA - all four cycle steps')
    lines.append('')
    psa_methods = ['pinn_al', 'pinn_seqrand', 'pinn_oneshot']
    psa_labels = {
        'pinn_al':      'PINN+AL (Our Work)',
        'pinn_seqrand': 'Sequential Random',
        'pinn_oneshot': 'Random-Random',
    }
    psa_budgets = [12, 30, 60]
    for step in ['ads', 'blow', 'evac', 'press']:
        lines.append(f'### {STEP_LABEL[step]}')
        lines.append('')
        hdr = '| Budget $N$ | ' + ' | '.join(psa_labels[m] for m in psa_methods) + ' |'
        lines.append(hdr); lines.append(sep)
        for b in psa_budgets:
            row = [str(b)]
            al_mse = [r['mse_total'] for r in psa_all[step][b].get('pinn_al', [])]
            for m in psa_methods:
                mse_vals = [r['mse_total'] for r in psa_all[step][b].get(m, [])]
                cell = fmt_msd(mse_vals)
                if m != 'pinn_al' and al_mse and mse_vals:
                    p = wilcoxon_p(al_mse, mse_vals)
                    if p <= 0.05: cell += ' $\\dagger$'
                    elif p <= 0.10: cell += ' $\\ddagger$'
                row.append(cell)
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    return '\n'.join(lines)


def build_table4(psa_all):
    """Per-output PSA breakdown at b=60."""
    lines = []
    lines.append('# Table 4 - Per-output PSA test-MSE breakdown at $N=60$')
    lines.append('')
    lines.append('Mean across 5 seeds.  Bold = best within row.')
    lines.append('')
    output_cols = ['mse_y1', 'mse_P', 'mse_q1', 'mse_q2']
    output_labels = [r'$y_1$', r'$P$', r'$q_1$', r'$q_2$']
    psa_methods = ['pinn_al', 'pinn_seqrand', 'pinn_oneshot']
    psa_labels = {
        'pinn_al':      'PINN+AL (Our Work)',
        'pinn_seqrand': 'Sequential Random',
        'pinn_oneshot': 'Random-Random',
    }
    hdr = '| Step | Method | ' + ' | '.join(output_labels) + ' | Total |'
    sep = '|---|---|' + '---|' * (len(output_cols) + 1)
    lines.append(hdr); lines.append(sep)
    for step in ['ads', 'blow', 'evac', 'press']:
        b = 60
        if b not in psa_all[step]: continue
        rows_per_method = {}
        for m in psa_methods:
            runs = psa_all[step][b].get(m, [])
            if not runs: continue
            per_out = {c: np.mean([r.get(c, np.nan) for r in runs])
                       for c in output_cols}
            total = np.mean([r['mse_total'] for r in runs])
            rows_per_method[m] = (per_out, total)
        # Find best per output across methods
        bests = {c: min(rpm[0][c] for rpm in rows_per_method.values())
                 for c in output_cols}
        bests['total'] = min(rpm[1] for rpm in rows_per_method.values())
        for m in psa_methods:
            if m not in rows_per_method: continue
            per_out, total = rows_per_method[m]
            cells = [STEP_LABEL[step] if m == psa_methods[0] else '',
                     psa_labels[m]]
            for c in output_cols:
                v = per_out[c]
                fmt = f'{v:.2e}'
                if v == bests[c]:
                    fmt = f'**{fmt}**'
                cells.append(fmt)
            tot_fmt = f'{total:.2e}'
            if total == bests['total']:
                tot_fmt = f'**{tot_fmt}**'
            cells.append(tot_fmt)
            lines.append('| ' + ' | '.join(cells) + ' |')
        # Blank separator row between steps
        if step != 'press':
            lines.append('| | | | | | | |')
    lines.append('')

    return '\n'.join(lines)


def main():
    print('Loading data...')
    cstr = load_cstr()
    psa_all = {s: load_psa(d) for s, d in STEP_DIRS_BIASED.items()}

    t3 = build_table3(cstr, psa_all)
    t4 = build_table4(psa_all)

    (OUT / 'tables_3_4.md').write_text(t3 + '\n\n---\n\n' + t4,
                                            encoding='utf-8')
    print(f'\nWrote {OUT / "tables_3_4.md"}')


if __name__ == '__main__':
    main()
