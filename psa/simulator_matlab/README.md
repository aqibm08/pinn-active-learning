# MATLAB PSA simulator and data pipeline

The scripts from the in-house MATLAB project that generated the training
data. The full project is organised in numbered folders
(`1Regular_PSA_code`, `2Collect_different_initial_conditions`,
`3Generate_data_using_initconditions`); the files here are the core pieces
needed to understand the pipeline and to re-run it inside that scaffold.

| file | role |
|---|---|
| `Ze_GA.m`, `Ze_GA_param.m` | single-bed four-step PSA cycle simulator (finite volumes, 25 cells) and its parameterised variant |
| `main_file_LHS.m` | LHS driver: draws designs over 7 operating conditions (step durations, velocities, intermediate and low pressures), runs each for 4 cycles, saves per-step initial-state matrices |
| `run_LHS_pipeline.m` | one-button orchestration: back up, sample, simulate, merge |
| `merge_train_data.m` | concatenates original and LHS-extended runs into `train_data_<step>_merged.mat` |
| `Yinit_temp.mat` | initial state used to warm-start the cycle integration |

Operating-condition bounds are +/-50% of the nominal `Ze_GA.m` values, with
the constraint P_L < P_INT < P_H. The merged exports actually used in the
paper are shipped in `../data`, so nothing here needs to be run to
reproduce the paper's results.
