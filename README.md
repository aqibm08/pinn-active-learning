# Active Learning for Physics-Informed Neural Network Surrogates of Chemical Processes

Code, data, result logs and model checkpoints for the paper

> Active Learning for Physics-Informed Neural Network Surrogates of
> Chemical Processes: Unified Framework

The framework couples a dual-head PINN (a deterministic prediction head plus
an evidential uncertainty head on a shared backbone) with sequential active
learning over a pool of candidate simulations, and is evaluated on two case
studies:

* **CSTR** - a continuous stirred-tank reactor governed by four coupled
  ODEs. 6-D input, 4-D output, point-level selection.
* **PSA** - a four-step pressure swing adsorption cycle (adsorption,
  blowdown, evacuation, repressurization) governed by coupled
  time-dependent PDEs. 102-D input, 4-D output, scenario-level selection,
  one surrogate per cycle step.

## Layout

| folder | contents |
|---|---|
| `psa/` | PSA framework: model, selectors, trainer, sweep runners |
| `psa/data/` | merged simulator exports (.mat, ~91 MB total) |
| `psa/simulator_matlab/` | MATLAB PSA cycle simulator and LHS data pipeline |
| `cstr/` | CSTR framework and sweep runner, incl. the ODE data generator |
| `cstr/data/` | steady-state CSTR dataset (1000 samples) |
| `analysis/` | figure and table generation from the shipped JSONs |
| `results/` | per-run JSON logs for every experiment in the paper, plus sweep stdout logs |
| `checkpoints/` | trained PSA model used for the true-vs-predicted figures |

`figures/` and `tables/` are created on demand by the analysis scripts and
are not tracked.

## Setup

Python 3.10+ with PyTorch. CUDA is used when available; everything also
runs on CPU, just slower.

```
pip install -r requirements.txt
```

## Reproducing the paper figures and tables (no training required)

The headline result JSONs are shipped, so figures and tables regenerate
directly:

```
python analysis/plot_individual.py          # individual figures -> figures/
python analysis/plot_true_vs_pred.py        # true-vs-pred figures (loads checkpoints/)
python analysis/generate_paper_tables.py    # Tables 3 and 4 -> tables/
python analysis/analyze_headline_sweep.py --results-dir results/psa_clustered_pool
python analysis/analyze_all_steps.py        # multi-step summary table
```

## Re-running the experiments

PSA headline sweep, one cycle step at a time (about 5.5 h per step for
5 seeds x 3 budgets x 6 methods on a workstation GPU):

```
cd psa
python run_psa_budget_sweep_v7.py --quick --biased-pool --step ads \
    --budgets 12 30 60 --seeds 42 123 456 789 1024 \
    --methods pinn_al pinn_ic_coreset pinn_seqrand pinn_oneshot ann_al ann_oneshot \
    --out-dir ../results/psa_clustered_pool_rerun/ads
```

`--quick` selects the 120-epoch-per-round / 300-epoch-polish protocol used
for all numbers in the paper. Omit `--biased-pool` to get the uniform-pool
comparison from the supplementary information. `--biased-pool` builds the
operating-condition-clustered pool described in the paper (the flag kept
its working name from development).

Selector ablation (supplementary S.3):

```
cd psa
python run_selector_ablation.py --budget 30 --seeds 42 123 456 \
    --out-dir ../results/psa_signal_ablation_rerun
```

CSTR sweep (about 4.4 h aggregate):

```
cd cstr
python run_budget_sweep.py
```

Data integrity checks (shapes, NaNs, train/test leakage):

```
cd psa
python sanity_check_data.py
```

## Where each paper artifact comes from

| paper artifact | source |
|---|---|
| Table 3a and the CSTR figures | `results/cstr/sweep_results_final.json` |
| Table 3b, Table 4 and the PSA figures | `results/psa_clustered_pool/<step>/psa_v7_sweep_final.json` |
| Uniform-vs-clustered pool comparison (SI S.1) | `results/psa_uniform_pool/psa_v7_sweep_final.json` |
| Selector ablation (SI S.3) | `results/psa_signal_ablation/ablation_stepads_b30.json` |
| Convergence curves (SI S.4) | histories in `results/psa_clustered_pool/ads/psa_v7_sweep_budget60.json` |

Each JSON entry stores, per budget / method / seed: total and per-output
test MSE and MAE, the full training histories (data loss, PDE loss,
validation MSE, lambda schedule, labeled-set growth, collocation-set size)
and wall time. `results/logs/` keeps the raw stdout of the headline runs.

## Citation

The paper is under review; citation details will be added on acceptance.

## License

MIT, see [LICENSE](LICENSE).
