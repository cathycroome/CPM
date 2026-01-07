function r = predict_behavior_nested(all_mats, all_behav, thresh, cov, checks, summary, k)
% CPM with LOOCV + optional inner-K-Fold tuning of p-threshold.
% Always uses Spearman's rank or partial Spearman for edge selection.
%
% Inputs
%   - all_mats  : M×M×N connectivity matrices (symmetric; one slice per subject)
%   - all_behav : N×1 behavioural scores
%   - thresh    : scalar or vector of candidate p-thresholds
%   - cov       : N×L covariate matrix (pass [] otherwise)
%   - checks    : 'Y' to print normality diagnostics (behaviour and residuals)
%   - summary   : 'Y' to print metrics and export figures
%   - k         : K-fold inner-CV threshold parameter tuning
%
% Outputs
%   - Variable  : r (pearson corr. between predicted and observed behaviour)
%   - Console   : true r, normality diagnostics, summary metrics/edge counts (if enabled)
%   - Figures   : scatter and boxplot comparing predicted vs observed (if enabled)

if nargin < 5, checks  = 'N'; end
if nargin < 6, summary = 'N'; end

no_sub  = size(all_mats,3);
no_node = size(all_mats,1);
behav_pred      = zeros(no_sub,1);
num_pos_edge    = zeros(no_sub,1);
num_neg_edge    = zeros(no_sub,1);
train_mean      = zeros(no_sub,1);
chosen_thresh   = nan(no_sub,1);

if isscalar(thresh)
    fprintf('\nLOOCV (outer) with partial Spearman; threshold =%.3g\n', thresh);
else
    fprintf('\nLOOCV (outer) with partial Spearman; threshold tuning\n');
end

for leftout = 1:no_sub
    train_mats = all_mats;  train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    train_behav  = all_behav; 
    train_behav(leftout)= [];
    train_mean(leftout) = mean(train_behav);

    if ~isempty(cov)
        cov_train = cov;
        cov_train(leftout,:) = [];
    else
        cov_train = [];
    end


    % ---- inner tuning ----

    if isscalar(thresh)
        best_t = thresh;
    else
        best_t = tune_inner(train_mats, train_behav, cov_train, thresh, k);
    end
    chosen_thresh(leftout) = best_t;

    fprintf(' Outer fold %d/%d | best threshold = %.3g\n', leftout, no_sub, best_t);

    % ---- feature selection on full training ----
    
    if isempty(cov_train)
        [r_mat, p_mat] = corr(train_vcts', train_behav, 'type','Spearman');
    else
        [r_mat, p_mat] = partialcorr(train_vcts', train_behav, cov_train, 'type','Spearman');
    end

    r_mat = reshape(r_mat,no_node,no_node);
    p_mat = reshape(p_mat,no_node,no_node);

    pos_mask = (r_mat>0) & (p_mat<best_t);
    neg_mask = (r_mat<0) & (p_mat<best_t);

    num_pos_edge(leftout) = nnz(triu(pos_mask,1));
    num_neg_edge(leftout) = nnz(triu(neg_mask,1));

    % sums over masks
    train_sumpos = zeros(no_sub-1,1); 
    train_sumneg = zeros(no_sub-1,1);

    for ss = 1:no_sub-1
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask, 'omitnan'))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask, 'omitnan'))/2;
    end

    b = regress(train_behav, [train_sumpos, train_sumneg, ones(no_sub-1,1)]);

    test_mat    = all_mats(:,:,leftout);
    test_sumpos    = sum(sum(test_mat.*pos_mask, 'omitnan'))/2;
    test_sumneg    = sum(sum(test_mat.*neg_mask, 'omitnan'))/2;
    behav_pred(leftout) = b(1)*test_sumpos + b(2)*test_sumneg + b(3);
end

[r, ~] = corr(behav_pred, all_behav);

if checks=='Y'
    res = all_behav - behav_pred;
    normality_checks(res, 'Residuals (Observed - Predicted)');
end

if summary=='Y'
    SSE = (all_behav - behav_pred).^2;
    PRESS = sum(SSE);
    RMSE  = sqrt(PRESS / no_sub);
    SST   = (all_behav - train_mean).^2;
    pred_R2 = 1 - sum(SSE)/sum(SST);
    fprintf('\nr=%.3f | PRESS=%.1f | RMSE=%.3f | Prediction R^2=%.3f\n', r, PRESS, RMSE, pred_R2);

    num_edge = num_pos_edge + num_neg_edge;
    fprintf('Total   : mean=%.0f sd=%.0f (min/max %d/%d)\n', mean ...
        (num_edge), std(num_edge), min(num_edge), max(num_edge));

    if ~isscalar(thresh)
        fprintf('Chosen thresholds (outer):\n');
        for t = thresh
            count = sum(chosen_thresh == t);
            fprintf('   %.3g : %d folds\n', t, count);
        end
    end

    % Scatter plot: predicted vs observed
    figure; scatter(all_behav, behav_pred, 30, 'filled'); hold on;
    plot([min(all_behav) max(all_behav)], [min(all_behav) max(all_behav)], ...
         'k--', 'LineWidth', 1); % y = x reference line
    
    xlabel('Observed BDR Score'); ylabel('Predicted BDR Score'); box on;

    set(gcf,'Units','inches','Position',[0 0 6 5]);
    exportgraphics(gcf,'scatter_predicted_vs_observed.png','Resolution',300);

    % Boxplot: predicted vs observed distributions
    allData = [behav_pred(:); all_behav(:)];
    group   = [repmat({'Predicted'}, length(behav_pred), 1); ...
               repmat({'Observed'},  length(all_behav),  1)];
    
    figure; boxplot(allData, group); ylabel('BDR Score');
    set(gca,'FontSize',14);

    set(gcf,'Units','inches','Position',[0 0 4 5]); 
    exportgraphics(gcf,'boxplot_predicted_vs_observed.png','Resolution',300);


end

