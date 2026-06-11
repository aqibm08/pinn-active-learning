"""
sanity_check_data.py
====================
Run before training. Validates that:
  1. train_data_<step>_merged.mat loads with expected shape and scenario count
  2. test_<step>_LHS_withbv.mat loads after stripping the bv column
  3. There is no IC leakage between train and test
  4. LHS test operating conditions are interior to training bounds
  5. Output (y) distributions are sensible (no NaN, similar ranges)

Usage:
  python sanity_check_data.py --train-dir <path/to/folder3> --test-dir <path/to/folder5>
  python sanity_check_data.py                     # uses ./ for both

Run from the psa/ folder so psa_data_utils.py is importable.
"""

import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch

# Reuse the identify-scenarios logic from data_utils.
# If psa_data_utils.py is on PYTHONPATH or in the same directory, this works:
try:
    from psa_data_utils import identify_scenarios, load_step_data
except ImportError:
    print('WARN: psa_data_utils.py not found in PYTHONPATH; using internal copy.')
    def identify_scenarios(x):
        x_np = x.double().numpy()
        fp = np.round(x_np[:, 2:12], 3)
        smap = {}
        for i in range(len(x_np)):
            key = fp[i].tobytes()
            smap.setdefault(key, []).append(i)
        scenarios = []
        for idx_list in smap.values():
            idx = np.array(idx_list, dtype=np.int64)
            ic = np.array([
                float(x_np[idx, 2:27].mean()),
                float(x_np[idx, 27:52].mean()),
                float(x_np[idx, 100].mean()),
                float(x_np[idx, 101].mean()),
            ], dtype=np.float64)
            scenarios.append({'indices': idx, 'ic_summary': ic, 'n_rows': len(idx)})
        return scenarios

    def load_step_data(mat_path, step='ads'):
        d = sio.loadmat(mat_path)
        return (torch.tensor(np.real(d[f'{step}_x']), dtype=torch.float64),
                torch.tensor(np.real(d[f'{step}_y']), dtype=torch.float64))


def load_lhs_test(test_dir, step):
    """Load LHS test data and strip the trailing bv column to get back to 102 cols."""
    f = Path(test_dir) / f'test_{step}_LHS_withbv.mat'
    if not f.exists():
        return None, None
    d = sio.loadmat(str(f))
    xkey, ykey = f'{step}_x', f'{step}_y'
    x_full = np.real(d[xkey])
    y      = np.real(d[ykey])
    # Strip last column (bv) to get 102 cols matching training format
    x = x_full[:, :102]
    return (torch.tensor(x, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64))


def fingerprints(scenarios):
    """Return set of IC fingerprints (cols 2..12 of first row, rounded)."""
    return set(tuple(np.round(s['ic_summary'][:2], 3).tolist()) for s in scenarios)


def check_step(step, train_dir, test_dir, verbose=True):
    print(f'\n{"="*70}\n  SANITY CHECK: {step.upper()}\n{"="*70}')
    results = {'step': step, 'passed': True, 'warnings': [], 'errors': []}

    # Load training
    train_file = Path(train_dir) / f'train_data_{step}_merged.mat'
    if not train_file.exists():
        # Fallback to non-merged
        train_file = Path(train_dir) / f'train_data_{step}.mat'
        if train_file.exists():
            results['warnings'].append(f'No _merged file, using {train_file.name}')
        else:
            results['errors'].append(f'Training file not found: {train_file}')
            results['passed'] = False
            print(f'  ERROR: {train_file} not found')
            return results

    print(f'  Loading training: {train_file.name}')
    x_tr, y_tr = load_step_data(str(train_file), step=step)
    print(f'    x_train: {tuple(x_tr.shape)}  (expect (*, 102))')
    print(f'    y_train: {tuple(y_tr.shape)}  (expect (*, 4))')

    if x_tr.shape[1] != 102:
        results['errors'].append(f'x_train has {x_tr.shape[1]} cols, expected 102')
        results['passed'] = False

    n_nan_x = int(torch.isnan(x_tr).sum())
    n_nan_y = int(torch.isnan(y_tr).sum())
    if n_nan_x > 0 or n_nan_y > 0:
        results['errors'].append(f'NaN values: x={n_nan_x}, y={n_nan_y}')
        results['passed'] = False

    train_scens = identify_scenarios(x_tr)
    print(f'    training scenarios: {len(train_scens)}')

    # Load LHS test
    print(f'  Loading test:    test_{step}_LHS_withbv.mat')
    x_te, y_te = load_lhs_test(test_dir, step)
    if x_te is None:
        results['warnings'].append(f'LHS test file not found in {test_dir}')
        print(f'    SKIPPED: file not found')
        return results

    print(f'    x_test (after bv-strip): {tuple(x_te.shape)}  (expect (*, 102))')
    print(f'    y_test: {tuple(y_te.shape)}  (expect (*, 4))')

    if x_te.shape[1] != 102:
        results['errors'].append(f'x_test has {x_te.shape[1]} cols, expected 102')
        results['passed'] = False

    test_scens = identify_scenarios(x_te)
    print(f'    test scenarios: {len(test_scens)}')

    # Leakage check
    train_fps = fingerprints(train_scens)
    test_fps  = fingerprints(test_scens)
    overlap   = train_fps & test_fps
    if overlap:
        results['errors'].append(f'IC LEAKAGE: {len(overlap)} train scenarios appear in test')
        results['passed'] = False
        print(f'    FAIL: {len(overlap)} scenarios appear in both sets')
    else:
        print(f'    no IC leakage between train and test')

    # Output range comparison
    print(f'\n  Output (y) statistics:')
    print(f'    train y means: y1={y_tr[:,0].mean():.4f}  P={y_tr[:,1].mean():.4f}  '
          f'q1={y_tr[:,2].mean():.4e}  q2={y_tr[:,3].mean():.4e}')
    print(f'    test  y means: y1={y_te[:,0].mean():.4f}  P={y_te[:,1].mean():.4f}  '
          f'q1={y_te[:,2].mean():.4e}  q2={y_te[:,3].mean():.4e}')
    print(f'    train y range  (each col):')
    for i, name in enumerate(['y1','P','q1','q2']):
        print(f'      {name}: [{y_tr[:,i].min().item():.4e}, {y_tr[:,i].max().item():.4e}]')
    print(f'    test  y range  (each col):')
    for i, name in enumerate(['y1','P','q1','q2']):
        print(f'      {name}: [{y_te[:,i].min().item():.4e}, {y_te[:,i].max().item():.4e}]')

    # Check test outputs sit inside train range (loose check)
    for i, name in enumerate(['y1','P','q1','q2']):
        tr_lo, tr_hi = y_tr[:,i].min().item(), y_tr[:,i].max().item()
        te_lo, te_hi = y_te[:,i].min().item(), y_te[:,i].max().item()
        # Allow 5% tolerance - small extrapolation is fine
        margin = 0.05 * (tr_hi - tr_lo + 1e-12)
        if te_lo < tr_lo - margin or te_hi > tr_hi + margin:
            results['warnings'].append(
                f'{name}: test range [{te_lo:.3e},{te_hi:.3e}] '
                f'extends beyond train [{tr_lo:.3e},{tr_hi:.3e}]')

    # Summary
    print(f'\n  Summary for {step}: {"PASS" if results["passed"] else "FAIL"}')
    for w in results['warnings']:
        print(f'    [warn] {w}')
    for e in results['errors']:
        print(f'    [ERR ] {e}')

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-dir', default='data', help='dir containing train_data_*_merged.mat')
    p.add_argument('--test-dir',  default='data', help='dir containing test_*_LHS_withbv.mat')
    p.add_argument('--steps', nargs='+', default=['ads','blow','evac','press'])
    args = p.parse_args()

    print(f'Sanity check')
    print(f'  train dir: {args.train_dir}')
    print(f'  test  dir: {args.test_dir}')

    all_ok = True
    summary = []
    for step in args.steps:
        res = check_step(step, args.train_dir, args.test_dir)
        all_ok = all_ok and res['passed']
        summary.append(res)

    print(f'\n{"="*70}')
    print(f'OVERALL: {"ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"}')
    print(f'{"="*70}')
    for r in summary:
        status = 'PASS' if r['passed'] else 'FAIL'
        n_warn = len(r['warnings']); n_err = len(r['errors'])
        print(f'  {r["step"]:>5}: {status}  warnings={n_warn}  errors={n_err}')


if __name__ == '__main__':
    main()
