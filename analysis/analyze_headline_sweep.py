"""
analyze_headline_sweep.py - summary statistics for a finished PSA sweep.

Reads the JSON output of run_psa_budget_sweep_v7.py and produces:
  - markdown summary tables (mean +/- std, median, IQR per method x budget)
  - per-seed paired comparisons (pinn_al vs each baseline)
  - win-rate per baseline (out of N seeds, how often did pinn_al win?)
  - Wilcoxon signed-rank p-values (paired, one-sided)
  - CSV exports for downstream plotting

Usage (from the repo root; the results dir can hold one step or several
step subfolders):
    python analysis/analyze_headline_sweep.py --results-dir results/psa_clustered_pool --out summary.md
"""
import argparse, json, csv
from pathlib import Path
import numpy as np
from scipy import stats


METHOD_LABELS = {
    'pinn_al':         'PINN+AL (BADGE)',
    'pinn_ic_coreset': 'PINN+AL (IC-coreset, model-free)',
    'pinn_seqrand':    'PINN+random (sequential)',
    'pinn_oneshot':    'PINN+random (one-shot)',
    'ann_al':          'ANN+AL (BADGE)',
    'ann_oneshot':     'ANN+random (one-shot)',
}


def load_results(results_dir: Path, step: str = None) -> dict:
    """Return {step: {budget: {method: [seed_runs]}}}.

    If step given, return single step.  Otherwise scan subdirs.
    """
    out = {}
    if step is not None:
        f = results_dir / 'psa_v7_sweep_final.json'
        if f.exists():
            out[step or 'ads'] = json.loads(f.read_text())
            return out
    # All-steps mode: each subdir has one json
    for sub in sorted(results_dir.iterdir()):
        if not sub.is_dir():
            continue
        f = sub / 'psa_v7_sweep_final.json'
        if f.exists():
            out[sub.name] = json.loads(f.read_text())
    return out


def extract_metric(runs, key='lhs_test_metrics'):
    """Extract mse_total from each run for given metric key.  Returns dict
    {seed: mse} so paired comparisons work even if seeds differ in coverage."""
    out = {}
    for r in runs:
        m = r.get(key)
        if m is None:
            continue
        out[int(r['seed'])] = float(m['mse_total'])
    return out


def summarise_method(runs, key):
    """Return summary dict for one method's runs at one budget."""
    vals = list(extract_metric(runs, key).values())
    if not vals:
        return None
    a = np.array(vals)
    return dict(
        mean=float(a.mean()),
        std=float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        median=float(np.median(a)),
        iqr=float(np.percentile(a, 75) - np.percentile(a, 25)),
        n=len(a),
        min=float(a.min()), max=float(a.max()),
        values=a.tolist(),
    )


def paired_comparison(al_runs, base_runs, key, al_label='pinn_al'):
    """Per-seed paired comparison: returns dict with win_rate, mean_improvement,
    p-value (Wilcoxon signed-rank), and per-seed deltas."""
    al_d   = extract_metric(al_runs,   key)
    base_d = extract_metric(base_runs, key)
    common = sorted(set(al_d.keys()) & set(base_d.keys()))
    if len(common) < 2:
        return None
    al_v   = np.array([al_d[s] for s in common])
    base_v = np.array([base_d[s] for s in common])
    delta  = base_v - al_v          # positive = al wins (lower MSE)
    rel    = (base_v - al_v) / base_v   # fractional improvement
    win_rate = float((delta > 0).mean())
    mean_pct = 100.0 * float(rel.mean())
    median_pct = 100.0 * float(np.median(rel))
    # Wilcoxon (paired, non-parametric).  Robust to outliers.
    try:
        stat, p = stats.wilcoxon(al_v, base_v, alternative='less')
    except Exception:
        stat, p = (np.nan, np.nan)
    return dict(
        n_pairs=len(common),
        win_rate=win_rate,
        mean_pct=mean_pct,
        median_pct=median_pct,
        p_value=float(p) if p is not None else None,
        per_seed_delta=delta.tolist(),
        per_seed_rel=rel.tolist(),
        seeds=common,
    )


def fmt_sci(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'n/a'
    return f'{x:.3e}'


def render_summary_md(all_results, out_path: Path):
    """Build markdown summary with mean+/-std + median tables + paired comparisons."""
    lines = ['# PSA PINN+AL Headline Sweep - Analysis', '']

    for step, res in all_results.items():
        budgets = sorted([int(b) for b in res.keys()])
        methods = list(METHOD_LABELS.keys())

        lines.append(f'## Step: `{step}`')
        lines.append('')

        for kind, key in [('LHS-CSS external test', 'lhs_test_metrics'),
                           ('Internal test (same dist)', 'test_metrics')]:
            lines.append(f'### {kind}')
            lines.append('')
            # Mean +/- std
            hdr = '| Budget | ' + ' | '.join(METHOD_LABELS[m] for m in methods if m in next(iter(res.values()), {})) + ' |'
            sep = '|' + '---|' * (len([m for m in methods if m in next(iter(res.values()), {})]) + 1)
            lines.append(hdr)
            lines.append(sep)
            for b in budgets:
                bk = str(b)
                row = [f'{b}']
                for m in methods:
                    if m not in res[bk]: continue
                    s = summarise_method(res[bk][m], key)
                    if s is None:
                        row.append('n/a')
                    else:
                        row.append(f'{s["mean"]:.3e} +/- {s["std"]:.2e}')
                lines.append('| ' + ' | '.join(row) + ' |')
            lines.append('')

            # Per-budget paired pinn_al vs baselines
            lines.append(f'**Paired comparisons (pinn_al vs baseline, per seed):**')
            lines.append('')
            for b in budgets:
                bk = str(b)
                if 'pinn_al' not in res[bk]: continue
                al_runs = res[bk]['pinn_al']
                lines.append(f'_b={b}_  ')
                for base in ['pinn_ic_coreset', 'pinn_seqrand', 'pinn_oneshot',
                              'ann_al', 'ann_oneshot']:
                    if base not in res[bk]: continue
                    cmp = paired_comparison(al_runs, res[bk][base], key)
                    if cmp is None:
                        lines.append(f'- vs {base}: n/a (insufficient pairs)')
                        continue
                    sig = ' *' if (cmp['p_value'] is not None
                                   and cmp['p_value'] < 0.05) else ' '
                    lines.append(
                        f'- vs **{base}**: mean={cmp["mean_pct"]:+.1f}%  '
                        f'median={cmp["median_pct"]:+.1f}%  '
                        f'win_rate={cmp["win_rate"]:.0%}  '
                        f'(n={cmp["n_pairs"]}, Wilcoxon p={cmp["p_value"]:.3f}{sig})')
                lines.append('')
            lines.append('')

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {out_path}')


def export_csv(all_results, out_dir: Path):
    """Per-step, per-method, per-seed CSV for plotting in any tool."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for step, res in all_results.items():
        rows = []
        budgets = sorted([int(b) for b in res.keys()])
        for b in budgets:
            for method, runs in res[str(b)].items():
                for r in runs:
                    rows.append({
                        'step': step, 'budget': b, 'method': method,
                        'seed': r['seed'],
                        'internal_mse': r['test_metrics']['mse_total'] if r.get('test_metrics') else None,
                        'lhs_mse': r['lhs_test_metrics']['mse_total'] if r.get('lhs_test_metrics') else None,
                        'wall_time': r.get('wall_time'),
                    })
        if rows:
            f = out_dir / f'sweep_{step}.csv'
            with open(f, 'w', newline='') as fp:
                w = csv.DictWriter(fp, fieldnames=rows[0].keys())
                w.writeheader(); w.writerows(rows)
            print(f'Wrote {f} ({len(rows)} rows)')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', required=True,
                    help='Directory containing psa_v7_sweep_final.json '
                         '(or step subdirs, each with one)')
    p.add_argument('--step', default=None, help='If results-dir is a single-step dir')
    p.add_argument('--out', default='headline_summary.md')
    p.add_argument('--csv-dir', default='headline_csv')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    all_results = load_results(Path(args.results_dir), step=args.step)
    if not all_results:
        print(f'No results found in {args.results_dir}')
        raise SystemExit(1)
    print(f'Loaded {len(all_results)} step(s): {list(all_results.keys())}')
    render_summary_md(all_results, Path(args.out))
    export_csv(all_results, Path(args.csv_dir))
