# PSA data

`train_data_<step>_merged.mat` (MATLAB v7.3 / HDF5): merged simulator
output per cycle step, fields `<step>_x` (N x 102, float64) and `<step>_y`
(N x 4). Scenario counts after cleaning: ads 403, blow 458, evac 491,
press 394. Loaded with `psa_data_utils.load_step_data`.

`test_<step>_LHS_withbv.mat`: auxiliary held-out test scenarios at cyclic
steady state. These have 103 columns; the trailing column is bed voidage
and is stripped on load. The sweep runner evaluates them whenever the files
are present, but they are not part of the metrics reported in the paper.

Outputs y = (y1, P, q1, q2): gas-phase CO2 mole fraction, dimensionless
pressure P/P0, and CO2 / N2 adsorbed-phase loadings.

To regenerate the raw data from scratch, see `../simulator_matlab/`.
