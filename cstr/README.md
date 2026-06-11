# CSTR case study

The benchmark problem: a continuous stirred-tank reactor with four coupled
ODE balances (concentration, reactor temperature, jacket temperature,
liquid level). The surrogate maps (Qf, Qc, current state) to the state one
sampling interval ahead. The exogenous inputs are held constant for much
longer than the reactor settling time, so the sampled points sit at or near
steady state and the PINN residual enforces the steady-state (algebraic)
form of the four balances.

| file | role |
|---|---|
| `isothermal_cstr.py` | data generator: SciPy RK45 under randomized piecewise-constant input sequences |
| `main_dual_al_experiment.py` | the model (DualHeadPhysicsPINN), data loading/splitting, and the older 4-way AL-vs-random experiment |
| `dual_al_trainer.py` | trainer for the 4-way experiment (data AL + collocation AL together) |
| `data_point_selector.py` | gradient + physics + diversity scoring with KMeans cluster-then-select |
| `active_learning_core.py` | collocation-point scoring and selection |
| `run_budget_sweep.py` | the paper's budget sweep: sequential AL vs the two random baselines |
| `visualization_budget_sweep.py` | sweep figures |
| `data/` | `SS_data_ndata1000_sampT1_cstr_ForEncDec.mat`, the 1000-sample dataset |
| `checkpoints/` | model states from the 4-way demonstration run (the sweep retrains from scratch) |

Run the sweep from this folder:

```
python run_budget_sweep.py
```

Outputs land in `sweep_results/` (not tracked); the run used in the paper
is shipped at `../results/cstr/sweep_results_final.json`.
