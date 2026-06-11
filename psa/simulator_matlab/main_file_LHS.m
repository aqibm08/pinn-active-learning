% MAIN_FILE_LHS  Latin Hypercube Sampling driver for PSA training data.
%
% Replaces folder 2's main_file.m. Uses LHS to draw 100 designs across
% 7 operating conditions (vpress_, vads_, vblow_, vevac_, tads_, PINTc, PLc),
% runs each design for 4 cycles, and saves the initial-state matrices for
% all 4 PSA steps. Checkpoints every 20 designs.
%
% Bounds: +/-50% of the original Ze_GA.m nominal values, with the constraint
%         PLc < PINTc < 1 (low pressure < intermediate pressure < high).
%
% Output: init_conditions.mat with fields
%   Y0_press_mat, Y0_ads_mat, Y0_blow_mat, Y0_evac_mat   [N x 152]
%   opcond_log    [N x 7]   - per-row operating conditions, for traceability
%   opcond_names  {1 x 7}   - column labels for opcond_log
%
% Run from inside folder 2Collect_different_initial_conditions.

clear all; close all; clc

% Make Ze_GA_param visible (it lives in folder 1)
addpath(genpath('..'));

load Yinit_temp.mat                  % loads Y (the cold-bed initial state)

% ---- LHS setup ------------------------------------------------------
nDesigns = 100;
nCycles  = 4;
rng(42);                             % reproducible

% Nominal values (matching Ze_GA.m hardcoded values) and +/-50% bounds.
nominal = struct();
nominal.vpress_ = 0.2;
nominal.vads_   = 0.5;
nominal.vblow_  = 0.05;
nominal.vevac_  = 1.0;
nominal.tads_   = 25;
nominal.PINTc   = 0.2;
nominal.PLc     = 0.1;

names  = {'vpress_','vads_','vblow_','vevac_','tads_','PINTc','PLc'};
lo     = zeros(1,numel(names));
hi     = zeros(1,numel(names));
for k = 1:numel(names)
    lo(k) = 0.5 * nominal.(names{k});
    hi(k) = 1.5 * nominal.(names{k});
end

% Generate LHS samples. Oversample by 3x and reject any that violate
% PLc < PINTc to guarantee we have nDesigns valid points.
nOver = 3 * nDesigns;
U     = lhsdesign(nOver, numel(names), 'criterion','maximin', 'iterations', 20);
designs = lo + U .* (hi - lo);
PINTc_idx = find(strcmp(names,'PINTc'));
PLc_idx   = find(strcmp(names,'PLc'));
keep = designs(:,PLc_idx) < designs(:,PINTc_idx);
designs = designs(keep,:);
if size(designs,1) < nDesigns
    error('LHS rejection left only %d valid designs; widen bounds.', size(designs,1));
end
designs = designs(1:nDesigns,:);

fprintf('LHS: %d designs x %d cycles = %d total simulations\n', ...
        nDesigns, nCycles, nDesigns*nCycles);
fprintf('Bounds (lo .. hi):\n');
for k = 1:numel(names)
    fprintf('  %-8s : %.4g .. %.4g  (nominal %.4g)\n', ...
            names{k}, lo(k), hi(k), nominal.(names{k}));
end

% ---- Storage --------------------------------------------------------
Y0_press_mat = [];
Y0_ads_mat   = [];
Y0_blow_mat  = [];
Y0_evac_mat  = [];
opcond_log   = [];
opcond_names = names;
failures     = 0;

global iteration; iteration = 0;
Yinit = Y;

% ---- Main loop ------------------------------------------------------
tStart = tic;
for i = 1:nDesigns
    % Pack this design's parameters into a struct
    opcond = struct();
    for k = 1:numel(names)
        opcond.(names{k}) = designs(i,k);
    end

    Y_running = Yinit;          % start each design from cold bed
    success_this_design = 0;

    for j = 1:nCycles
        try
            [~, ~, yall1, yall2, yall3, yall4] = Ze_GA_param(Y_running, opcond);

            Y0_press_mat = [Y0_press_mat; yall1(1,:)]; %#ok<AGROW>
            Y0_ads_mat   = [Y0_ads_mat;   yall2(1,:)]; %#ok<AGROW>
            Y0_blow_mat  = [Y0_blow_mat;  yall3(1,:)]; %#ok<AGROW>
            Y0_evac_mat  = [Y0_evac_mat;  yall4(1,:)]; %#ok<AGROW>
            opcond_log   = [opcond_log;   designs(i,:)]; %#ok<AGROW>

            Y_running = yall4(end, 1:150);  % seed next cycle from end of evac
            success_this_design = success_this_design + 1;

        catch ME
            failures = failures + 1;
            fprintf('  design %d cycle %d FAILED: %s\n', i, j, ME.message);
            break;   % don't keep cycling a broken design
        end
    end

    if mod(i, 5) == 0 || i == nDesigns
        elapsed = toc(tStart);
        eta = elapsed/i * (nDesigns - i);
        fprintf('  [%3d/%3d]  rows so far: %d  elapsed %.0fs  ETA %.0fs  fails %d\n', ...
                i, nDesigns, size(Y0_ads_mat,1), elapsed, eta, failures);
    end

    % Checkpoint every 20 designs
    if mod(i, 20) == 0
        save('init_conditions_checkpoint.mat', ...
             'Y0_press_mat', 'Y0_ads_mat', 'Y0_blow_mat', 'Y0_evac_mat', ...
             'opcond_log', 'opcond_names', 'designs', 'i');
        fprintf('  -- checkpoint saved at design %d --\n', i);
    end
end

% ---- Final save -----------------------------------------------------
save('init_conditions.mat', ...
     'Y0_press_mat', 'Y0_ads_mat', 'Y0_blow_mat', 'Y0_evac_mat', ...
     'opcond_log', 'opcond_names', 'designs');

fprintf('\n=== LHS sampling complete ===\n');
fprintf('  Y0_press_mat: %d x %d\n', size(Y0_press_mat));
fprintf('  Y0_ads_mat:   %d x %d\n', size(Y0_ads_mat));
fprintf('  Y0_blow_mat:  %d x %d\n', size(Y0_blow_mat));
fprintf('  Y0_evac_mat:  %d x %d\n', size(Y0_evac_mat));
fprintf('  failures:     %d\n', failures);
fprintf('  total time:   %.1f sec (%.2f hours)\n', toc(tStart), toc(tStart)/3600);
