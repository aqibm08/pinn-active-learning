"""
analyze_all_steps.py - multi-step PSA paper table

Aggregates JSON results from each PSA step's biased-pool sweep into one
paper-ready table.  Reports the headline numbers: PINN+AL improvement %
over each random baseline, per step, per budget, with Wilcoxon p-values.

Usage:
    python analyze_all_steps.py --base-dir . --steps ads blow evac press
"""
import argparse, json
from pathlib import Path

from _paths import RESULTS
import numpy as np
from scipy import stats


STEP_DIRS = {
    'ads':   'psa_clustered_pool/ads',
    'blow':  'psa_clustered_pool/blow',
    'evac':  'psa_clustered_pool/evac',
    'press': 'psa_clustered_pool/press',
}


def extract_internal(runs):
    return {int(r['seed']): float(r['test_metrics']['mse_total'])
            for r in runs if r.get('test_metrics')}


def paired_pct(al_d, base_d):
    common = sorted(set(al_d) & set(base_d))
    if len(common) < 2:
        return None
    al = np.array([al_d[s] for s in common])
    bs = np.array([base_d[s] for s in common])
    rel = (bs - al) / bs
    try:
        _, p = stats.wilcoxon(al, bs, alternative='less')
    except Exception:
        p = np.nan
    return dict(
        mean_pct=100 * float(rel.mean()),
        median_pct=100 * float(np.median(rel)),
        win_rate=float((rel > 0).mean()),
        p_value=float(p),
        n=len(common),
    )


def render_table(all_data, baselines=('pinn_seqrand', 'pinn_oneshot')):
    """All_data: {step: {budget: {method: [runs]}}}."""
    steps = list(all_data.keys())
    budgets = sorted({b for s in all_data.values() for b in s.keys()})

    lines = ['# PSA Multi-Step Headline Table', '']
    lines.append(f"Methods compared:  PINN+AL vs {', '.join(baselines)}")
    lines.append(f"Statistics: mean improvement % over random baselines, "
                  f"Wilcoxon paired p-value (one-sided, AL better)")
    lines.append('')

    # Per baseline table
    for base in baselines:
        lines.append(f'## vs `{base}`')
        lines.append('')
        hdr = '| Step | ' + ' | '.join(f'b={b}' for b in budgets) + ' |'
        sep = '|---|' + '---|' * len(budgets)
        lines.append(hdr)
        lines.append(sep)
        for step in steps:
            row = [step]
            for b in budgets:
                step_b = all_data[step].get(b, {})
                al_d = extract_internal(step_b.get('pinn_al', []))
                base_d = extract_internal(step_b.get(base, []))
                p = paired_pct(al_d, base_d)
                if p is None:
                    row.append('n/a')
                else:
                    sig = ' *' if p['p_value'] < 0.05 else ''
                    row.append(
                        f"{p['mean_pct']:+.0f}% ({p['win_rate']:.0%}, "
                        f"p={p['p_value']:.3f}{sig})")
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    # Raw numbers table (mean +/- std)
    lines.append('## Raw internal-test MSE (mean +/- std across 5 seeds)')
    lines.append('')
    methods_to_show = ['pinn_al', 'pinn_seqrand', 'pinn_oneshot',
                        'pinn_ic_coreset', 'ann_al', 'ann_oneshot']
    for step in steps:
        lines.append(f'### {step}')
        lines.append('')
        hdr = '| Method | ' + ' | '.join(f'b={b}' for b in budgets) + ' |'
        sep = '|---|' + '---|' * len(budgets)
        lines.append(hdr)
        lines.append(sep)
        for m in methods_to_show:
            row = [m]
            for b in budgets:
                runs = all_data[step].get(b, {}).get(m, [])
                vals = [r['test_metrics']['mse_total']
                        for r in runs if r.get('test_metrics')]
                if not vals:
                    row.append('n/a')
                else:
                    mean = np.mean(vals)
                    std = np.std(vals, ddof=1) if len(vals) > 1 else 0
                    row.append(f'{mean:.2e} +/- {std:.1e}')
            lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-dir', default=str(RESULTS))
    p.add_argument('--steps', nargs='+', default=['ads', 'blow', 'evac', 'press'])
    p.add_argument('--out', default='multi_step_summary.md')
    args = p.parse_args()

    base = Path(args.base_dir)
    all_data = {}
    for step in args.steps:
        sub = base / STEP_DIRS.get(step, f'psa_clustered_pool/{step}')
        f = sub / 'psa_v7_sweep_final.json'
        if not f.exists():
            print(f'[skip] {step}: no results at {f}')
            continue
        raw = json.loads(f.read_text())
        all_data[step] = {int(b): m for b, m in raw.items()}
        print(f'[ok]   {step}: {len(all_data[step])} budgets')

    if not all_data:
        print('No data found.')
        raise SystemExit(1)

    md = render_table(all_data)
    Path(args.out).write_text(md, encoding='utf-8')
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
