% Copyright 2015 Xilin Shen and Emily Finn (GPL v2)
% Original CPM Matlab implementation obtained from NITRC.
% The unmodified version is preserved in the first commit of this repo.
% See README for modification details.

% ------------------------------------------------------------------------------
% Purpose
%   Run a CPM permutation test with a fixed p-threshold using
%   leave-one-out cross-validation (LOOCV). This version supports HPC job-array 
%   batching so permutations can be split across tasks to reduce run time
%
% What this script does
%   1) Loads connectivity (all_mats: M×M×N) and behaviour (all_behav: N×1)
%   2) Optionally builds covariates (cov: N×L) from <sample>.csv
%   3) Computes the “true” statistic: r(predicted, observed) via LOOCV
%   4) If running permutations (see below), shuffles labels and recomputes r
%      for a specified slice of iterations, saving a batch file to disk
%
% Run modes
%   - True-only run:
%       no_iterations = 1    % prints metrics/plots for the true model
%   - Permutation testing 
%       no_iterations > 1       
%       Provide environment variables:
%           START_ITER, END_ITER  (inclusive range for this task)
%           RESULTS_DIR           (output folder for .mat batch files)
%           SLURM_ARRAY_TASK_ID   (only used to name a per-task subfolder, optional)
%       HPC: This batched version runs on a single CPU core.
%
% Key parameters 
%   - threshold     : p-value for edge selection (e.g., 0.01).
%   - fs_option     : 1 Pearson, 2 Spearman, 3 Partial Pearson, 4 Partial Spearman.
%   - covariates    : chosen from {'age','sex','gap'}
%   - which_age     : 'scan' or 'bdr' 
%   - no_iterations : number of permutations (>= 1000 recommended for inference)
%   - checks/summary: 'Y'/'N' for normality diagnostics and summary plots on the true run
%
% Inputs 
%   - resultsROI_Condition001.mat : Z (M×M×N); Z(:,:,k) = subject k connectivity.
%   - <sample>.csv                : behaviour (bdr_raw) and optional covariates
%                                   (e.g., age, sex, test–scan gap).
%
% Outputs
%   - Console   : true r and summary metrics/edge counts (if enabled)
%   - Figures   : scatter and boxplot comparing predicted vs observed (if enabled)
%   - Batch file: RESULTS_DIR/perm_batch_<start>_<end>.mat containing:
%       perm_r            [K×1] r values for this batch (K = END_ITER−START_ITER+1)
%       start_iter, end_iter, true_prediction_r
% ------------------------------------------------------------------------------

clear;
clc;
close all;

% ---------- params ----------
threshold     = 0.01;  
fs_option     = 4;    
sample        = 'bdr_6months';              
which_age     = 'scan';
covariates    = {'age','sex'};       % choose from: age, sex, gap
no_iterations = 1000;                % for permutation testing
checks        = 'N';                 % normality diagnostics

set(groot, 'defaultAxesFontSize', 14); 
set(groot, 'defaultTextFontSize', 14);

% ---------- paths ----------
parent   = regexprep(pwd, [filesep 'code$'], '');
data_dir = fullfile(parent, 'data');

load(fullfile(data_dir, 'resultsROI_Condition001.mat'), 'Z');
T = readtable(fullfile(data_dir, [sample '.csv']));   

% ---------- prepare data  ----------
bad_idx   = isnan(T.bdr_raw);
all_mats = Z(:,:,~bad_idx);
all_behav= T.bdr_raw(~bad_idx);
age      = T{~bad_idx, ['age_' which_age]};
sex      = T.sex(~bad_idx) - 1;
gap      = T.gap(~bad_idx);
cov_tbl  = table(age, gap, sex, 'VariableNames', {'age','gap','sex'});
test_cov = cov_tbl{:, covariates};
no_sub   = size(all_mats,3);

% ---------- true stat  ----------

fprintf('\n=== True model ===\n');
true_prediction_r = predict_behavior(all_mats, all_behav, threshold, fs_option, test_cov, checks, 'Y');
fprintf('True r = %.4f\n', true_prediction_r);

% ---------- permutations ----------

if no_iterations >1
    
    fprintf('\n=== Permutations ===\n');

    % ---------- batch slice ----------
    start_iter = str2double(getenv('START_ITER'));
    end_iter   = str2double(getenv('END_ITER'));
    
    job_dir = getenv('RESULTS_DIR');
    
    fprintf('Running permutations %d..%d -> %s\n', start_iter, end_iter, job_dir);

    % ---------- batch permutations ----------

    perm_ids = start_iter:end_iter;
    perm_r   = zeros(length(perm_ids),1);

    for k = 1:length(perm_ids)
        it = perm_ids(k);
        rng(it,"threefry");    % reproducible, unique per iteration across arrays
        idx = randperm(no_sub);
        new_behav = all_behav(idx);
        perm_r(k) = predict_behavior(all_mats, new_behav, threshold, fs_option, test_cov, 'N','N');
    end

    save(fullfile(job_dir, sprintf('perm_batch_%d_%d.mat', start_iter, end_iter)), ...
        'perm_r','start_iter','end_iter','true_prediction_r', ...
        'threshold','which_age','covariates','no_iterations','fs_option','sample');

    fprintf('\nPermutations complete: run merge_perm_batches.m for p-value\n');

end

