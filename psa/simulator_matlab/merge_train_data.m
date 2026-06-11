% MERGE_TRAIN_DATA  Concatenate original + LHS-extended training data.
%
% Run from inside folder 3Generate_data_using_initconditions, AFTER:
%   1. The original train_data_*.mat files are present (or backed up)
%   2. main_file.m has been re-run with the new (LHS) init_conditions.mat
%      to produce new train_data_*.mat files
%
% This script expects:
%   train_data_<step>_orig.mat   - original (renamed by user before LHS run)
%   train_data_<step>.mat        - new LHS-based output
%
% Produces:
%   train_data_<step>_merged.mat - concatenation, suitable for ANN training

steps = {'ads','press','blow','evac'};

for s = 1:numel(steps)
    step = steps{s};
    origFile = sprintf('train_data_%s_orig.mat',   step);
    newFile  = sprintf('train_data_%s.mat',        step);
    mrgFile  = sprintf('train_data_%s_merged.mat', step);

    if ~isfile(origFile)
        fprintf('[%s] %s not found, skipping merge.\n', step, origFile);
        continue;
    end
    if ~isfile(newFile)
        fprintf('[%s] %s not found, skipping merge.\n', step, newFile);
        continue;
    end

    A = load(origFile);
    B = load(newFile);
    xName = sprintf('%s_x', step);
    yName = sprintf('%s_y', step);

    merged_x = [A.(xName); B.(xName)];
    merged_y = [A.(yName); B.(yName)];

    S = struct();
    S.(xName) = merged_x;
    S.(yName) = merged_y;
    save(mrgFile, '-struct', 'S', '-v7.3');

    fprintf('[%s] orig %d + new %d = merged %d rows -> %s\n', ...
            step, size(A.(xName),1), size(B.(xName),1), size(merged_x,1), mrgFile);
end

fprintf('\nDone. To use merged data for training, rename train_data_<step>_merged.mat\n');
fprintf('to train_data_<step>.mat in folder 4TrainingANNs (after backing up).\n');
