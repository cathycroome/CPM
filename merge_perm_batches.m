function merge_perm_batches(results_root)
    % results_root = outputs/<JOBID>
    % Merges all perm_batch_*.mat files under job_* subfolders.

    files = dir(fullfile(results_root,'job_*','perm_batch_*.mat'));
    assert(~isempty(files), 'No batch files found in %s', results_root);

    % collect permutation results
    all_perm = [];
    for f = 1:numel(files)
        S = load(fullfile(files(f).folder, files(f).name));
        all_perm = [all_perm; S.perm_r(:)];
    end

    % load true stat + parameters from first batch
    S = load(fullfile(files(1).folder, files(1).name), ...
             'true_prediction_r','threshold','which_age','covariates','no_iterations','fs_option','sample');
    true_r = S.true_prediction_r;

    % compute p-value (one-sided, r_perm >= r_true)
    pval = (sum(all_perm >= true_r) + 1) / (numel(all_perm) + 1);

    % print results
    fprintf('\ntrue_r %.5f, p %.6f (Nperm=%d)\n', true_r, pval, numel(all_perm));

    % parameter summary
    fprintf('\n=== Run parameters ===\n');
    fprintf('Threshold   : %s\n', mat2str(S.threshold));
    fprintf('Which age   : %s\n', S.which_age);
    fprintf('Covariates  : %s\n', strjoin(S.covariates, ', '));
    fprintf('Iterations  : %d\n', S.no_iterations);
    fprintf('FS option   : %d\n', S.fs_option);
    fprintf('Sample      : %s\n', S.sample);

end
