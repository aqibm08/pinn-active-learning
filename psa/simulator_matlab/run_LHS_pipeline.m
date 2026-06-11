% RUN_LHS_PIPELINE  One-button workflow for generating LHS-extended PSA training data.
%
% Run from the project root (same level as folders 1, 2, 3, ...).
%
% This script performs the following steps:
%   1. Backs up the existing train_data_*.mat files in folder 3
%      (renames *.mat to *_orig.mat).
%   2. Backs up the existing init_conditions.mat in folder 2.
%   3. Runs the LHS sampler (main_file_LHS.m) in folder 2 to produce a new
%      init_conditions.mat with ~400 scenarios.
%   4. Copies the new init_conditions.mat to folder 3.
%   5. Runs folder 3's main_file.m to generate new train_data_*.mat.
%   6. Calls merge_train_data.m to produce train_data_*_merged.mat.
%
% At the end you'll have:
%   3Generate_data_using_initconditions/
%     train_data_<step>_orig.mat    (original, untouched)
%     train_data_<step>.mat         (new LHS-only)
%     train_data_<step>_merged.mat  (original + new, for training)

clear all; close all; clc

projectRoot = pwd;
folder1 = fullfile(projectRoot, '1Regular_PSA_code');
folder2 = fullfile(projectRoot, '2Collect_different_initial_conditions');
folder3 = fullfile(projectRoot, '3Generate_data_using_initconditions');

assert(isfolder(folder1), 'Cannot find folder 1Regular_PSA_code under %s', projectRoot);
assert(isfolder(folder2), 'Cannot find folder 2Collect_different_initial_conditions under %s', projectRoot);
assert(isfolder(folder3), 'Cannot find folder 3Generate_data_using_initconditions under %s', projectRoot);

addpath(genpath(projectRoot));

% ---- Step 1: back up folder 3's existing train_data files ----------
steps = {'ads','press','blow','evac'};
for s = 1:numel(steps)
    src = fullfile(folder3, sprintf('train_data_%s.mat', steps{s}));
    dst = fullfile(folder3, sprintf('train_data_%s_orig.mat', steps{s}));
    if isfile(src) && ~isfile(dst)
        copyfile(src, dst);
        fprintf('Backed up %s -> %s\n', steps{s}, dst);
    elseif isfile(dst)
        fprintf('Backup already exists for %s, skipping.\n', steps{s});
    end
end

% ---- Step 2: back up folder 2's existing init_conditions.mat -------
src = fullfile(folder2, 'init_conditions.mat');
dst = fullfile(folder2, 'init_conditions_orig.mat');
if isfile(src) && ~isfile(dst)
    copyfile(src, dst);
    fprintf('Backed up old init_conditions.mat -> init_conditions_orig.mat\n');
end

% ---- Step 3: run LHS sampler in folder 2 ---------------------------
fprintf('\n=== Running LHS sampler ===\n');
cd(folder2);
t1 = tic;
main_file_LHS;
fprintf('Folder 2 LHS run took %.1f sec\n', toc(t1));

% ---- Step 4: copy new init_conditions to folder 3 ------------------
copyfile(fullfile(folder2,'init_conditions.mat'), folder3);
fprintf('Copied new init_conditions.mat to folder 3.\n');

% ---- Step 5: run folder 3's main_file.m ---------------------------
fprintf('\n=== Running folder 3 data generator ===\n');
cd(folder3);
t2 = tic;
main_file;          % the existing (FIXED) main_file with delt_ads:1: fix
fprintf('Folder 3 run took %.1f sec\n', toc(t2));

% ---- Step 6: merge old + new --------------------------------------
fprintf('\n=== Merging old and new train_data files ===\n');
merge_train_data;

% ---- Final summary ------------------------------------------------
fprintf('\n=== PIPELINE COMPLETE ===\n');
for s = 1:numel(steps)
    f = fullfile(folder3, sprintf('train_data_%s_merged.mat', steps{s}));
    if isfile(f)
        info = load(f);
        xname = sprintf('%s_x', steps{s});
        if isfield(info, xname)
            fprintf('  %-5s: %d rows in %s\n', steps{s}, size(info.(xname),1), f);
        end
    end
end

cd(projectRoot);
