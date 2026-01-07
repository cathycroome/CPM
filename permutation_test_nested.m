% Copyright 2015 Xilin Shen and Emily Finn (GPL v2)
% Original CPM Matlab implementation obtained from NITRC.
% The unmodified version is preserved in the first commit of this repo.
% See README for modification details.
%
% -------------------------------------------------------------------------
% Purpose
%   Run CPM with outer LOOCV to predict a behavioural score (e.g., BDR), 
%   optionally assess significance via permutation testing, and (optionally) 
%   tune the p-threshold for edge selection via an inner K-fold CV (nested-CV).
%
% What this script does
%   1) Loads subject-level connectivity matrices Z (M x M x N) and behaviour
%      vector (N x 1). Optionally builds a covariate matrix (N x L).
%   2) Computes the “true” outer-LOOCV statistic: r(predicted, observed).
%      - If threshold is a vector, an inner 5-fold CV selects the threshold
%        within each outer fold from the candidate set {.001, .01, .05}.
%   3) (Optional) Permutation testing: constructs a null by shuffling the
%      behaviour labels and recomputing the full LOOCV pipeline for each
%      permutation (right-tailed test).
%   4) Reports the permutation p-value 
%   5) (Optional) Prints summary metrics, edge counts, and saves scatter/box
%      plots for the true model.
%
% Run modes
%   - True-only run:
%       no_iterations = 1    % prints metrics/plots for the true model
%   - Permutation testing 
%       no_iterations = 1000 (or larger)      
%       HPC: request 32–64 CPUs for permutation testing with inner K-fold CV 
%       Uses a thread-based pool; precomputes label permutations.
%
% Key parameters (set below or via SLURM environment)
%   - threshold     : scalar (e.g., 0.01) or vector (e.g., [0.001, 0.01, 0.05]);
%                     if vector, inner LOO tunes and picks best threshold per fold.
%   - covariates    : choose from {'age','sex','gap'} or {} for none.
%   - which_age     : 'scan' or 'bdr' (column prefix in the sample .csv).
%   - no_iterations : number of permutations (e.g., 1000 or more recommended).
%   - checks        : 'Y'/'N' – run residual normality diagnostics and save
%                   figures for the true model (hist/Q–Q via helper).
%   - summary       : 'Y'/'N' – print PRESS/RMSE/R^2_pred, edge counts, and
%                   chosen-threshold frequencies (if tuning was used).
%   - k             : K-fold inner-CV threshold parameter tuning
%
% Inputs
%   - data/resultsROI_Condition001.mat : contains variable Z (size M×M×N), where
%       M = number of nodes in the chosen brain atlas
%       N = number of subjects
%       Each slice Z(:,:,k) is the subject-level connectivity matrix, with 
%       element (i,j,k) representing the correlation between the BOLD 
%       timecourses of nodes i and j for subject k.
%
%   - data/<sample>.csv : subject-level table containing behavioural scores and
%       Columns: bdr_raw, age_scan, age_bdr, gap, sex (1/2), etc.
%
% Outputs
%   - Console: true r; (optionally) PRESS, RMSE, prediction R^2, edge counts,
%              and per-threshold selection frequencies (nested-CV case).
%   - Figures (true run, if enabled): scatter (pred vs obs) and boxplot
%              (predicted vs observed distributions).
% 
% Evaluation metrics 
%   - Pearson r across LOOCV predictions vs. observations (primary).
%   - RMSE in BDR units; prediction R^2 relative to a per-fold mean 
%     baseline (can be negative if the model underperforms baseline).
%
% -------------------------------------------------------------------------
% clear;
% clc;
% close all;

% ---------- params ----------
checks        = 'N';            % normality diagnostics 'Y'/'N'
sample        = 'bdr_6months';
which_age     = 'scan';
covariates    = {'age','sex'};  % choose from: age, sex, gap; {} to run with no covariates
k             = 10;              % k-fold best performing threshold per outer loop

% don't override params if set in shell script
if ~exist('threshold','var'), threshold = [0.001 0.01 0.05]; end  % scalar or vector
if ~exist('no_iterations','var'), no_iterations = 1; end    % >1 for permutation testing

fprintf('Params: iterations=%d | thresh=%s | cov=%s\n', no_iterations, mat2str ...
    (threshold), strjoin(covariates,','));

% set global font sizes
set(groot, 'defaultAxesFontSize', 12); 
set(groot, 'defaultTextFontSize', 14);

% fix the seed once at the start for reproducible folds
rng(0,'twister');   % so local default will run on HPC for true run

% ---------- paths ----------
parent   = regexprep(pwd, [filesep 'code$'], '');
data_dir = fullfile(parent, 'data');

load(fullfile(data_dir, 'resultsROI_Condition001.mat'), 'Z');
T = readtable(fullfile(data_dir, [sample '.csv']));   

% ---------- prepare data ----------
bad_idx   = isnan(T.bdr_raw);
all_mats = single(Z(:,:,~bad_idx));   
all_behav= T.bdr_raw(~bad_idx);
no_sub   = size(all_mats,3);

if isempty(covariates)
    cov = [];
else
    age = T{~bad_idx, ['age_' which_age]};
    sex = T.sex(~bad_idx) - 1;
    gap = T.gap(~bad_idx);
    cov_tbl = table(age,gap,sex,'VariableNames',{'age','gap','sex'});
    cov = cov_tbl{:, covariates};   
end

% print summary stats and plots when running true model locally
if no_iterations == 1
    summary = 'Y';
else
    summary = 'N';
end


% ------- true statistic -------
fprintf('\n=== True model ===\n');

true_r = predict_behavior_nested(all_mats, all_behav, threshold, cov, checks, summary, k);
fprintf('True r = %.4f\n', true_r);


% ------- permutation testing -------
if no_iterations >1
    
    % --- thread-based pool (shared memory) ---

    if isempty(gcp('nocreate')), parpool("Threads"); end
    
    fprintf('\n=== Permutations (N=%d) ===\n', no_iterations);
    
    % --- precompute label permutations ---    
    perm_idx = zeros(no_sub, no_iterations);
    perm_idx(:,1) = 1:no_sub;
    
    for it = 2:no_iterations
        perm_idx(:,it) = randperm(no_sub).';
    end
    
    y_perm_mat = all_behav(perm_idx);   % N x I matrix of permuted labels
    
    % --- parallel permutations ---
    
    perm_r = nan(no_iterations,1);
    perm_r(1) = true_r;

    % for permutation progress prints   
    t0 = tic;
    total = no_iterations - 1;   % number of permutations
    dq = parallel.pool.DataQueue;
    afterEach(dq, @(~) prog_tick(total, t0));   

    % create estimate distribution of the test statistic
    % via random shuffles of data labels  
    parfor it = 2:no_iterations
        rng(it, 'threefry');        % reproducible random number generator
        y_perm = y_perm_mat(:, it);   
        perm_r(it) = predict_behavior_nested(all_mats, y_perm, threshold, cov, 'N','N', k);
        send(dq, 1);                % triggers a progress print 
    end
       
    % ---------- p-value (right-tailed) ----------
    p_perm = (1 + sum(perm_r >= true_r)) / (no_iterations + 1);
    
    fprintf('\nTrue r = %.4f | perm p = %.6f\n', true_r, p_perm);

    delete(gcp('nocreate'));
    
end


