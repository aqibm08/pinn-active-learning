# PSA case study

Surrogates for a four-step pressure swing adsorption cycle (binary CO2/N2
separation on Zeolite-13X, isothermal). Each cycle step (ads, blow, evac,
press) is treated as its own surrogate problem. A *scenario* is one
simulation of a step under one set of LHS-sampled operating conditions and
contributes 625 training rows (25 spatial grid points x 25 time snapshots).

Input layout per row (102 columns): z, t, then the four 25-point initial
profiles (y1, P, q1, q2). Columns 100-101 carry the tail of the q2 profile;
bed voidage (0.37) and CO2 saturation capacity (3.298 mol/kg) are constant
across all scenarios and live inside the PDE residual computation rather
than the input vector. Outputs are (y1, P, q1, q2) at the queried (z, t).

## Files

| file | role |
|---|---|
| `psa_pinn_model_v3.py` | dual-head PINN (K=3 committee + NIG evidential head), frozen PCA bottleneck, PDE residuals |
| `psa_data_utils.py` | .mat loading, scenario identification and split, clustered-pool construction, evaluation |
| `psa_data_selector_v5.py` | single-signal AL selectors: badge, qbc, physics, gradient, ic_coreset, random |
| `psa_data_selector_v4.py` | the four-signal mixture selector, plus scoring helpers shared with v5 |
| `psa_collocation_selector_v3.py` | manifold-aware collocation point generation |
| `psa_al_trainer_v3.py` | sequential AL trainer (one Adam + one cosine schedule across all rounds) |
| `run_psa_budget_sweep_v7.py` | sweep runner: methods x budgets x seeds for one cycle step |
| `run_selector_ablation.py` | single-signal ablation driver (supplementary S.3) |
| `sanity_check_data.py` | data integrity checks (shapes, NaNs, train/test IC leakage) |

The paper's PINN+AL is the v5 selector with `mode='badge'`. Hyperparameters
live in `SWEEP_CONFIG` at the top of the sweep runner; `--quick` switches to
the 120-epoch-per-round protocol used for all reported numbers.

Module names keep the version suffixes they had during development. The
shipped result files reference them (`psa_v7_sweep_*.json`), so renaming
would only break that correspondence.
