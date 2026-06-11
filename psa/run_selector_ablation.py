"""
Single-signal AL ablation driver 

   pinn_random        random selection (baseline)
   pinn_ic_coreset    model-free farthest-first on the [y1, P] IC summary
   pinn_badge         BADGE diversity only
   pinn_qbc           committee disagreement only
   pinn_physics       PDE residual only
   pinn_gradient      input-gradient norm only
   pinn_v4_mix        the v4 four-signal mixture, as a sanity check


Usage:
    python run_selector_ablation.py --budget 30 --seeds 42 123 456 \
        --out-dir ../results/psa_signal_ablation_rerun
"""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch

from psa_pinn_model_v3 import build_model
from psa_data_utils import evaluate_model
from psa_collocation_selector_v3 import PSACollocationSelectorV3
from psa_al_trainer_v3 import PSAActiveLearningTrainerV3
from psa_data_selector_v4 import PSADataSelectorV4
from psa_data_selector_v5 import PSADataSelectorV5

# reuse the sweep runner data prep so handling is identical
from run_psa_budget_sweep_v7 import prepare_data, SWEEP_CONFIG, _save


# Selector configurations to test
SELECTOR_VARIANTS = [
    ('pinn_random',     PSADataSelectorV5, {'mode': 'random'}),
    ('pinn_ic_coreset', PSADataSelectorV5, {'mode': 'ic_coreset'}),
    ('pinn_badge',      PSADataSelectorV5, {'mode': 'badge'}),
    ('pinn_qbc',        PSADataSelectorV5, {'mode': 'qbc'}),
    ('pinn_physics',    PSADataSelectorV5, {'mode': 'physics'}),
    ('pinn_gradient',   PSADataSelectorV5, {'mode': 'gradient'}),
    ('pinn_v4_mix',     PSADataSelectorV4, {}),
]


def run_one(variant_name, selector_class, selector_kwargs,
            budget, cfg, splits, lb, ub, seed, device):
    """Train one PINN model with the given selector."""
    torch.manual_seed(seed); np.random.seed(seed)

    model = build_model(
        budget, lb.numpy(), ub.numpy(), device,
        model_type='pinn',
        pca_components=splits['pca_components'],
        pca_mean=splits['pca_mean'],
        n_committee=cfg['n_committee'], init_seed=seed,
        y_lb=splits['y_lb'], y_ub=splits['y_ub'],
        t_max=splits['t_max'])

    data_sel = selector_class(
        k_select=cfg['k_per_round'], device=device, **selector_kwargs)

    coll_sel = PSACollocationSelectorV3(
        n_candidates=cfg['n_coll_candidates'],
        coll_multiplier=cfg['coll_multiplier'],
        use_manifold=cfg['use_manifold_coll'],
        device=device)

    trainer = PSAActiveLearningTrainerV3(
        model=model, data_selector=data_sel, coll_selector=coll_sel,
        n0=cfg['n_initial'], k_per_round=cfg['k_per_round'], budget=budget,
        epochs_per_round=cfg['epochs_per_round'],
        n_final_epochs=cfg['final_epochs'],
        lambda_pde_max=cfg['lambda_pde_max'],
        warmup_epochs=cfg['warmup_epochs'],
        lam_evid=cfg['lam_evid'],
        lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        device=device, verbose=False,         # less output, this is a sweep
        train_committee=True)

    res = trainer.run(
        splits['pool_scenarios'],
        x_pool=splits['x_pool'],
        x_pool_orig=splits['x_pool_orig'],
        y_pool=splits['y_pool'],
        x_val=splits['x_val'], y_val=splits['y_val'],
        seed=seed)

    res['test_metrics'] = evaluate_model(
        model, splits['x_test'], splits['y_test'], device,
        y_lb=splits['y_lb'], y_ub=splits['y_ub'])
    if splits['x_lhs_test'] is not None:
        res['lhs_test_metrics'] = evaluate_model(
            model, splits['x_lhs_test'], splits['y_lhs_test'], device,
            y_lb=splits['y_lb'], y_ub=splits['y_ub'])
    else:
        res['lhs_test_metrics'] = None

    return res


def run_ablation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={device}  out_dir={out_dir}  budget={args.budget}  '
          f'seeds={args.seeds}')

    cfg = copy.deepcopy(SWEEP_CONFIG)
    cfg['step'] = args.step
    # quick protocol (~5 min per run, otherwise this takes overnight);
    # same epoch counts as the headline sweep
    cfg.update(epochs_per_round=120, final_epochs=300, warmup_epochs=120)

    splits, lb, ub = prepare_data(cfg, args.data_dir)

    variants_to_run = SELECTOR_VARIANTS
    if args.variants:
        keep = set(args.variants)
        variants_to_run = [v for v in SELECTOR_VARIANTS if v[0] in keep]
        print(f'Restricting to variants: {[v[0] for v in variants_to_run]}')

    results = {v[0]: [] for v in variants_to_run}
    t_global = time.time()

    for seed in args.seeds:
        print(f'\n-- seed={seed} --')
        for variant_name, selector_class, selector_kwargs in variants_to_run:
            t0 = time.time()
            try:
                res = run_one(variant_name, selector_class, selector_kwargs,
                               args.budget, cfg, splits, lb, ub, seed, device)
            except Exception as e:
                print(f'  [ERROR] {variant_name}: {e}')
                import traceback; traceback.print_exc()
                continue
            int_mse = res['test_metrics']['mse_total']
            lhs_mse = res['lhs_test_metrics']['mse_total'] \
                      if res['lhs_test_metrics'] else float('nan')
            wt = time.time() - t0
            results[variant_name].append(dict(
                seed=seed, internal_mse=int_mse, lhs_mse=lhs_mse,
                wall_time=wt))
            print(f'  {variant_name:18s}  internal={int_mse:.3e}  '
                  f'LHS={lhs_mse:.3e}  ({wt:.0f}s)')

        _save(results, out_dir / f'ablation_step{args.step}_b{args.budget}.json')

    print(f'\nTotal wall time: {(time.time() - t_global)/60:.1f} min')
    print_table(results, args.budget)


def print_table(results, budget):
    print('\n' + '=' * 76)
    print(f'  SELECTOR ABLATION - budget={budget}')
    print('=' * 76)
    print(f'{"variant":<22}{"int_mean":>12}{"int_std":>11}'
          f'{"LHS_mean":>12}{"LHS_std":>11}{"n":>4}')
    print('-' * 76)
    rows = []
    for name, runs in results.items():
        if not runs:
            rows.append((name, float('nan'), float('nan'),
                          float('nan'), float('nan'), 0))
            continue
        int_v = np.array([r['internal_mse'] for r in runs])
        lhs_v = np.array([r['lhs_mse'] for r in runs])
        rows.append((name, int_v.mean(), int_v.std(),
                     lhs_v.mean(), lhs_v.std(), len(runs)))

    rows.sort(key=lambda r: r[3] if not np.isnan(r[3]) else float('inf'))
    for name, im, isd, lm, lsd, n in rows:
        print(f'{name:<22}{im:>12.3e}{isd:>11.2e}{lm:>12.3e}{lsd:>11.2e}{n:>4}')

    # Compute relative-to-random comparisons
    rand = results.get('pinn_random', [])
    if rand:
        rand_lhs = np.array([r['lhs_mse'] for r in rand])
        rand_lhs_mean = rand_lhs.mean()
        print('\n  vs pinn_random on LHS:')
        for name, runs in results.items():
            if name == 'pinn_random' or not runs:
                continue
            lhs_v = np.array([r['lhs_mse'] for r in runs])
            pct = 100.0 * (1.0 - lhs_v.mean() / rand_lhs_mean)
            sym = '+' if pct > 0 else ''
            print(f'    {name:<22} {sym}{pct:.1f}%')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=int, default=30)
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    p.add_argument('--step', default='ads', choices=['ads', 'blow', 'evac', 'press'])
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir', default='psa_ablation_v5')
    p.add_argument('--variants', nargs='+', default=None,
                    help='Restrict to subset of variants by name')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_ablation(args)
