"""
Main PSA sweep runner: methods x budgets x seeds on one cycle step.

Methods:
  pinn_al          PINN + PCA bottleneck + committee + PDE + BADGE selector
  pinn_ic_coreset  same model, model-free IC-coreset selector
  pinn_seqrand     same model and protocol, random selection each round
  pinn_oneshot     same model, all labels drawn at once (Random-Random)
  ann_al           vanilla ANN + BADGE selector (no PDE loss)
  ann_oneshot      vanilla ANN, one-shot random

The paper's headline runs use quick mode on the clustered pool, e.g.

  python run_psa_budget_sweep_v7.py --quick --biased-pool --step ads \
      --budgets 12 30 60 --seeds 42 123 456 789 1024 \
      --methods pinn_al pinn_seqrand pinn_oneshot \
      --out-dir ../results/psa_clustered_pool_rerun/ads
"""

import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from psa_pinn_model_v3 import (build_model, evidential_loss,
                                fit_pca_for_bottleneck,
                                PSADualHeadPINN, PSAVanillaANN)
from psa_data_utils    import (load_step_data, load_lhs_test_data,
                                compute_bounds, evaluate_model,
                                identify_scenarios, split_scenarios,
                                get_rows, build_pool_tensors,
                                make_biased_scenario_pool,
                                N_STEPS, N_Z_POINTS,
                                compute_y_bounds, normalise_y)
from psa_data_selector_v4         import PSADataSelectorV4, RandomPSADataSelector
from psa_data_selector_v5         import PSADataSelectorV5
from psa_collocation_selector_v3  import PSACollocationSelectorV3
from psa_al_trainer_v3            import PSAActiveLearningTrainerV3

torch.set_default_dtype(torch.float64)


# --- Sweep configuration ------------------------------------------------------
# Tuned for 403-scenario pool. Budgets span 3-37% of pool.
SWEEP_CONFIG = {
    'budgets'          : [12, 30, 60, 100, 150],
    'seeds'            : [42, 123, 456, 789, 1024],
    'methods'          : ['pinn_al', 'pinn_ic_coreset', 'pinn_seqrand',
                          'pinn_oneshot',
                          'ann_al',  'ann_oneshot'],
    'step'             : 'ads',

    # Larger reservations for richer evaluation (still leaves ~333 in pool)
    'n_val_scenarios'  : 30,
    'n_test_scenarios' : 40,

    'n_initial'        : 4,
    'k_per_round'      : 4,         # 4 picks/round -> 12 budget = 1 round, 150 = ~36 rounds
    'epochs_per_round' : 250,
    'final_epochs'     : 1500,
    'warmup_epochs'    : 250,

    'lr'               : 1e-3,
    'weight_decay'     : 1e-4,    # v7: 10x stronger than v5 (1e-5 -> 1e-4)
                                  # for better OOD generalisation. Reduces
                                  # the overfit-to-transient pathology.

    'lambda_pde_max'   : 0.002,   # v7: lowered from 0.005 -> 0.002.
                                  # PDE term should support data fit, not
                                  # dominate it.  Reducing prevents the
                                  # NIG/PDE coupling that drove data loss
                                  # to -1.1 in v5 smoke test.

    'lam_evid'         : 0.05,    # v7: 5x stronger NIG regulariser
                                  # (0.01 -> 0.05) - keeps the evidential
                                  # head honest, prevents alpha*nu explosion.
    'pca_k'            : 8,

    'n_coll_candidates': 1500,
    'coll_multiplier'  : 4,
    'use_manifold_coll': True,

    'n_committee'      : 3,

    # Data file naming
    'train_suffix'     : '_merged',     # train_data_<step>_merged.mat
    'lhs_test_suffix'  : '_LHS_withbv', # test_<step>_LHS_withbv.mat
}


# --- Data loading -------------------------------------------------------------

def prepare_data(cfg, data_dir='.'):
    train_path = Path(data_dir) / f"train_data_{cfg['step']}{cfg['train_suffix']}.mat"
    lhs_path   = Path(data_dir) / f"test_{cfg['step']}{cfg['lhs_test_suffix']}.mat"

    if not train_path.exists():
        raise FileNotFoundError(f'Training data missing: {train_path}')

    # Load training data (h5py / v7.3 .mat)
    x_all, y_all = load_step_data(str(train_path), step=cfg['step'])
    print(f"Loaded {cfg['step']}: x={tuple(x_all.shape)}  y={tuple(y_all.shape)}")

    lb, ub = compute_bounds(x_all)
    scens  = identify_scenarios(x_all)
    print(f'Identified {len(scens)} scenarios')

    pool_s, val_s, test_s = split_scenarios(
        scens, n_val_scen=cfg['n_val_scenarios'],
        n_test_scen=cfg['n_test_scenarios'], seed=42)
    print(f'Split: pool={len(pool_s)}  val={len(val_s)}  internal-test={len(test_s)}')

    # Optional operating-condition-clustered pool (the deployment regime the
    # paper argues for): replicate scenarios near 3 IC nominals so random
    # sampling concentrates in the dense regions while the test set stays
    # uniform. Same construction as the CSTR sweep.
    if cfg.get('biased_pool', False):
        bias_frac = cfg.get('bias_fraction', 0.70)
        print(f'Biasing pool: {bias_frac:.0%} clustered near 3 IC nominals')
        pool_s = make_biased_scenario_pool(pool_s, bias_fraction=bias_frac, seed=42)
        print(f'  pool size after biasing: {len(pool_s)} (was {cfg["n_val_scenarios"] + cfg["n_test_scenarios"]} reserved)')

    x_val,  y_val  = get_rows(val_s,  x_all, y_all)
    x_test, y_test = get_rows(test_s, x_all, y_all)
    x_pool, y_pool, pool_local = build_pool_tensors(pool_s, x_all, y_all)

    # Load EXTERNAL LHS test set
    if lhs_path.exists():
        x_lhs_full, y_lhs_full = load_lhs_test_data(str(lhs_path), step=cfg['step'])
        print(f'External LHS test: x={tuple(x_lhs_full.shape)}  y={tuple(y_lhs_full.shape)}')
        # Identify scenarios in the LHS test, extract equal rows per scenario
        lhs_scens = identify_scenarios(x_lhs_full)
        x_lhs, y_lhs = get_rows(lhs_scens, x_lhs_full, y_lhs_full)
        print(f'  LHS scenarios: {len(lhs_scens)}, rows after extraction: {x_lhs.shape[0]}')
    else:
        print(f'[warn] LHS test not found at {lhs_path} - skipping external eval')
        x_lhs, y_lhs = None, None

    # PCA bottleneck on full training data
    if cfg.get('pca_k', 0) > 0:
        comps, mean = fit_pca_for_bottleneck(x_all, scens, k=cfg['pca_k'])
    else:
        comps, mean = None, None

    # y normalisation
    y_lb, y_ub = compute_y_bounds(y_all)
    y_pool = normalise_y(y_pool, y_lb, y_ub)
    y_val  = normalise_y(y_val,  y_lb, y_ub)

    # t_max so build_model can override ub[:,1] (see model module docstring).
    # Without this the t column normalises to [-1, 247] for ads, far outside Tanh's sweet spot.
    t_max = float(x_all[:, 1].max())
    print(f'  t_max (physical) = {t_max:.2f}s - will override ub[:,1] inside build_model')

    return dict(
        pool_scenarios=pool_local,
        x_pool=x_pool, y_pool=y_pool,
        x_pool_orig=x_pool.clone(),
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,             # internal (same-dist) test
        x_lhs_test=x_lhs, y_lhs_test=y_lhs,       # external (CSS) test
        y_lb=y_lb, y_ub=y_ub, t_max=t_max,
        pca_components=comps, pca_mean=mean,
    ), lb, ub


# --- Sequential method --------------------------------------------------------

def run_sequential(method, budget, cfg, splits, lb, ub, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)

    is_pinn = method.startswith('pinn')
    # Sequential AL variants - selector chosen per method suffix
    is_al           = method.endswith('_al')
    is_ic_coreset   = method.endswith('_ic_coreset')
    is_sequential   = is_al or is_ic_coreset or method.endswith('_seqrand')
    model_type = 'pinn' if is_pinn else 'ann'

    model = build_model(budget, lb.numpy(), ub.numpy(), device,
                         model_type=model_type,
                         pca_components=splits['pca_components'],
                         pca_mean=splits['pca_mean'],
                         n_committee=cfg['n_committee'], init_seed=seed,
                         y_lb=splits['y_lb'], y_ub=splits['y_ub'],
                         t_max=splits['t_max'])

    # Selector wiring:
    #   *_al           -> v5 BADGE (winner from ablation, beats v4 mix on both
    #                    mean LHS and std at b=30)
    #   *_ic_coreset   -> v5 model-free farthest-first on IC summary
    #                    (paper baseline showing diversity alone suffices)
    #   *_seqrand      -> Random selection (fair comparison with same protocol)
    if is_ic_coreset:
        data_sel = PSADataSelectorV5(mode='ic_coreset',
                                      k_select=cfg['k_per_round'], device=device)
    elif is_al:
        data_sel = PSADataSelectorV5(mode='badge',
                                      k_select=cfg['k_per_round'], device=device)
    else:
        data_sel = RandomPSADataSelector(k_select=cfg['k_per_round'], device=device)

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
        lambda_pde_max=cfg['lambda_pde_max'] if is_pinn else 0.0,
        warmup_epochs=cfg['warmup_epochs'],
        lam_evid=cfg['lam_evid'],
        lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        device=device, verbose=True,
        train_committee=is_pinn)

    res = trainer.run(
        splits['pool_scenarios'],
        x_pool=splits['x_pool'],
        x_pool_orig=splits['x_pool_orig'],
        y_pool=splits['y_pool'],
        x_val=splits['x_val'], y_val=splits['y_val'],
        seed=seed)

    # Internal test
    res['test_metrics'] = evaluate_model(
        model, splits['x_test'], splits['y_test'], device,
        y_lb=splits['y_lb'], y_ub=splits['y_ub'])

    # External LHS-CSS test (key generalisation metric)
    if splits['x_lhs_test'] is not None:
        res['lhs_test_metrics'] = evaluate_model(
            model, splits['x_lhs_test'], splits['y_lhs_test'], device,
            y_lb=splits['y_lb'], y_ub=splits['y_ub'])
    else:
        res['lhs_test_metrics'] = None

    res.update(budget=budget, method=method, seed=seed)
    return res


# --- One-shot method ----------------------------------------------------------

def run_oneshot(method, budget, cfg, splits, lb, ub, seed, device):
    torch.manual_seed(seed); np.random.seed(seed)

    pool = splits['pool_scenarios']
    rng  = np.random.default_rng(seed)
    sel  = [pool[i] for i in rng.choice(len(pool), min(budget, len(pool)), replace=False)]
    idx_arr = np.concatenate([s['indices'] for s in sel]).tolist()

    x_lab = splits['x_pool'][idx_arr].double().to(device)
    y_lab = splits['y_pool'][idx_arr].double().to(device)
    x_val = splits['x_val'].double().to(device); y_val = splits['y_val'].double().to(device)
    x_pool_orig = splits['x_pool_orig'].double().to(device)
    x_lab_orig  = x_pool_orig[idx_arr]

    is_pinn = method.startswith('pinn')
    model_type = 'pinn' if is_pinn else 'ann'

    model = build_model(budget, lb.numpy(), ub.numpy(), device,
                         model_type=model_type,
                         pca_components=splits['pca_components'],
                         pca_mean=splits['pca_mean'],
                         n_committee=cfg['n_committee'], init_seed=seed,
                         y_lb=splits['y_lb'], y_ub=splits['y_ub'],
                         t_max=splits['t_max'])

    n_rnd = max(0, (budget - cfg['n_initial'] + cfg['k_per_round'] - 1) // cfg['k_per_round'])
    total = cfg['epochs_per_round'] * (n_rnd + 1) + cfg['final_epochs']

    opt   = optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)

    if is_pinn and cfg['lambda_pde_max'] > 0.0:
        coll_sel = PSACollocationSelectorV3(
            n_candidates=cfg['n_coll_candidates'],
            coll_multiplier=cfg['coll_multiplier'],
            use_manifold=cfg['use_manifold_coll'], device=device)
        x_coll = coll_sel.select(model, x_lab_orig, rng_seed=seed, round_idx=1).to(device)
        if x_coll.shape[0] > 0:
            model.update_residual_scales(x_coll, momentum=0.0)
    else:
        x_coll = torch.empty(0, 102, dtype=torch.float64, device=device)

    warmup = cfg['warmup_epochs'] or cfg['epochs_per_round']
    lam_max = cfg['lambda_pde_max'] if is_pinn else 0.0

    t0 = time.time()
    dl_h=[]; pl_h=[]; vl_h=[]; vm_h=[]; u_h=[]; lam_h=[]
    import torch.nn as nn

    for ep in range(total):
        model.train(); opt.zero_grad()
        if is_pinn:
            d_evid = model.loss_evidential(x_lab, y_lab, lam_evid=cfg['lam_evid'])
            d_det  = model.loss_deterministic_mse(x_lab, y_lab)
            d_loss = d_evid + d_det
        else:
            d_loss = model.loss_mse(x_lab, y_lab)

        lam = (lam_max * min(1.0, ep / max(1, warmup))) if is_pinn else 0.0
        if is_pinn and lam > 0.0 and len(x_coll) > 0:
            p_loss = model.loss_pde(x_coll)
            loss = d_loss + lam * p_loss
            pl_h.append(float(p_loss.item()))
        else:
            loss = d_loss
            pl_h.append(0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); sched.step()
        dl_h.append(float(d_loss.item())); lam_h.append(lam)

        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                pred = model.forward_deterministic(x_val)
                vmse = float(nn.functional.mse_loss(pred, y_val))
                if is_pinn:
                    g, nu, a, b = model.forward_evidential(x_val)
                    vl  = float(evidential_loss(g, nu, a, b, y_val))
                    unc = float((b / (a * nu + 1e-8)).mean())
                else:
                    vl, unc = vmse, 0.0
            vl_h.append(vl); vm_h.append(vmse); u_h.append(unc)
            model.train()

    metrics = evaluate_model(model, splits['x_test'], splits['y_test'], device,
                              y_lb=splits['y_lb'], y_ub=splits['y_ub'])

    if splits['x_lhs_test'] is not None:
        lhs_metrics = evaluate_model(model, splits['x_lhs_test'], splits['y_lhs_test'],
                                      device, y_lb=splits['y_lb'], y_ub=splits['y_ub'])
    else:
        lhs_metrics = None

    return dict(
        test_metrics=metrics, lhs_test_metrics=lhs_metrics,
        data_loss_history=dl_h, pde_loss_history=pl_h, total_loss_history=[],
        val_loss_history=vl_h, val_mse_history=vm_h, val_mae_history=[],
        uncertainty_history=u_h, lambda_pde_history=lam_h,
        val_mse_per_output_history={},
        train_size_history=[budget]*len(dl_h),
        collocation_size_history=[len(x_coll)]*len(dl_h),
        budget=budget, method=method, seed=seed, wall_time=time.time() - t0)


# --- Sweep --------------------------------------------------------------------

def run_sweep(cfg, args):
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = Path(getattr(args, 'out_dir', 'psa_al_results_v7'))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device={device}  out_dir={out_dir}')

    splits, lb, ub = prepare_data(cfg, getattr(args, 'data_dir', '.'))
    results = {}

    for budget in cfg['budgets']:
        if budget > len(splits['pool_scenarios']):
            print(f'[Skip] budget={budget} > pool={len(splits["pool_scenarios"])}')
            continue
        results[budget] = {m: [] for m in cfg['methods']}
        print(f'\n{"="*70}\nBUDGET = {budget} scenarios\n{"="*70}')

        for seed in cfg['seeds']:
            print(f'\n-- seed={seed} --')
            for method in cfg['methods']:
                print(f'\n[{method}]')
                t0 = time.time()
                try:
                    if method.endswith('_oneshot'):
                        res = run_oneshot(method, budget, cfg, splits, lb, ub, seed, device)
                    else:
                        res = run_sequential(method, budget, cfg, splits, lb, ub, seed, device)
                except Exception as e:
                    print(f'  [ERROR] {method}: {e}')
                    import traceback; traceback.print_exc()
                    continue

                results[budget][method].append(dict(
                    method=method, budget=budget, seed=seed,
                    test_metrics=res['test_metrics'],
                    lhs_test_metrics=res.get('lhs_test_metrics'),
                    wall_time=res.get('wall_time', time.time() - t0),
                    data_loss_history=res.get('data_loss_history', []),
                    pde_loss_history=res.get('pde_loss_history', []),
                    val_loss_history=res.get('val_loss_history', []),
                    val_mse_history=res.get('val_mse_history', []),
                    uncertainty_history=res.get('uncertainty_history', []),
                    lambda_pde_history=res.get('lambda_pde_history', []),
                    train_size_history=res.get('train_size_history', []),
                    collocation_size_history=res.get('collocation_size_history', []),
                ))
                lhs_str = ''
                if res.get('lhs_test_metrics'):
                    lhs_str = f'  LHS: {res["lhs_test_metrics"]["mse_total"]:.4e}'
                print(f'  internal: {res["test_metrics"]["mse_total"]:.4e}{lhs_str}  ({time.time()-t0:.0f}s)')

                # Defensive: free GPU memory between runs.  v7 OOM'd at b=100
                # ann_oneshot, suspected leftover allocation accumulation.
                import gc
                del res
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        _save(results, out_dir / f'psa_v7_sweep_budget{budget}.json')

    final = out_dir / 'psa_v7_sweep_final.json'
    _save(results, final)
    print(f'\n[Sweep done] -> {final}')
    _print_table(results, cfg['budgets'], cfg['methods'])
    return results


def _ser(o):
    if isinstance(o, dict): return {str(k): _ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_ser(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, torch.Tensor): return o.tolist()
    return o

def _save(r, p):
    with open(p, 'w') as f: json.dump(_ser(r), f, indent=2)


def _print_table(results, budgets, methods):
    for kind in ('internal test', 'LHS external test'):
        print('\n' + '=' * 100)
        print(f'  {kind.upper()}')
        print('=' * 100)
        hdr = f'{"Budget":>6}  ' + ''.join(f'{m[:14]:>16}' for m in methods)
        print(hdr); print('-' * len(hdr))
        key = 'test_metrics' if kind == 'internal test' else 'lhs_test_metrics'
        for b in budgets:
            if b not in results: continue
            row = f'{b:>6}  '
            for m in methods:
                vs = []
                for r in results[b].get(m, []):
                    metrics = r.get(key)
                    if metrics is not None: vs.append(metrics['mse_total'])
                row += f'{np.mean(vs):>16.4e}' if vs else f'{"n/a":>16}'
            print(row)
        if 'pinn_al' in methods:
            print(f'\n  IMPROVEMENT % on {kind} (pinn_al vs each baseline, mean across budgets):')
            for base in ['pinn_ic_coreset', 'pinn_seqrand', 'pinn_oneshot',
                          'ann_al', 'ann_oneshot']:
                if base not in methods: continue
                imps = []
                for b in budgets:
                    if b not in results: continue
                    al = [r[key]['mse_total'] for r in results[b].get('pinn_al', [])
                          if r.get(key) is not None]
                    bv = [r[key]['mse_total'] for r in results[b].get(base, [])
                          if r.get(key) is not None]
                    if al and bv:
                        imps.append(100.0 * (1.0 - np.mean(al) / np.mean(bv)))
                if imps:
                    print(f'    pinn_al vs {base:>14}: mean={np.mean(imps):+6.1f}%  '
                          f'range [{min(imps):+5.1f}%, {max(imps):+5.1f}%]')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--quick', action='store_true')
    p.add_argument('--step', default='ads', choices=['ads','blow','evac','press'])
    p.add_argument('--all-steps', action='store_true')
    p.add_argument('--budgets', nargs='+', type=int)
    p.add_argument('--seeds',   nargs='+', type=int)
    p.add_argument('--methods', nargs='+')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--out-dir',  default='psa_al_results_v7')
    p.add_argument('--lambda-pde', type=float, default=None)
    p.add_argument('--no-pca-bottleneck', action='store_true')
    p.add_argument('--biased-pool', action='store_true',
                   help='operating-condition-clustered pool (70%% of entries near 3 IC nominals)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = copy.deepcopy(SWEEP_CONFIG)
    cfg['step'] = args.step

    if args.quick:
        cfg.update(budgets=[12, 30], seeds=[42, 123],
                   epochs_per_round=120, final_epochs=300, warmup_epochs=120,
                   methods=['pinn_al', 'pinn_seqrand', 'ann_al'])
        print('[Quick mode]')

    if args.budgets: cfg['budgets'] = args.budgets
    if args.seeds:   cfg['seeds']   = args.seeds
    if args.methods: cfg['methods'] = args.methods
    if args.lambda_pde is not None: cfg['lambda_pde_max'] = args.lambda_pde
    if args.no_pca_bottleneck: cfg['pca_k'] = 0
    if args.biased_pool: cfg['biased_pool'] = True

    if args.all_steps:
        all_results = {}
        out_root = Path(args.out_dir)
        for step in ['ads', 'blow', 'evac', 'press']:
            print(f'\n{"#"*70}\n#  STEP: {step.upper()}\n{"#"*70}')
            scfg = copy.deepcopy(cfg); scfg['step'] = step
            sargs = copy.copy(args); sargs.out_dir = str(out_root / step)
            try:
                all_results[step] = run_sweep(scfg, sargs)
            except Exception as e:
                print(f'\n[ERROR] step {step} failed: {e}')
                import traceback; traceback.print_exc()
                all_results[step] = {}

        print(f'\n{"=" * 100}\nCOMBINED - ALL STEPS\n{"=" * 100}')
        for step, res in all_results.items():
            print(f'\n--- {step} ---')
            _print_table(res, cfg['budgets'], cfg['methods'])

        # Save combined summary
        summary = {}
        for step, res in all_results.items():
            summary[step] = {}
            for b in res:
                summary[step][str(b)] = {}
                for m in cfg['methods']:
                    runs = res[b].get(m, [])
                    summary[step][str(b)][m] = {
                        'internal_mse': [r['test_metrics']['mse_total'] for r in runs],
                        'lhs_mse': [r['lhs_test_metrics']['mse_total']
                                     for r in runs if r.get('lhs_test_metrics')],
                    }
        _save(summary, out_root / 'psa_v7_all_steps_summary.json')
    else:
        run_sweep(cfg, args)
