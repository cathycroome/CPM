function r = predict_behavior(all_mats, all_behav, thresh, fs_option, cov, checks, summary)
% CPM prediction with LOOCV and a p-threshold for edge selection
%
% Inputs
%   all_mats  : M×M×N connectivity matrices (symmetric; one slice per subject)
%   all_behav : N×1 behavioural scores
%   thresh    : scalar p-value threshold for edge selection
%   fs_option : feature-selection mode
%                 1 = Pearson corr
%                 2 = Spearman corr
%                 3 = Partial Pearson (controls for cov)
%                 4 = Partial Spearman (controls for cov)
%   cov       : N×L covariate matrix (required for fs_option 3 or 4; pass [] otherwise)
%   checks    : 'Y' to print normality diagnostics (behaviour and residuals)
%   summary   : 'Y' to print metrics and export figures
%
% Outputs
%   Variable: r (pearson corr. between predicted and observed behaviour across all N subjects)
%   Console: true r, permutation p-value, and summary metrics/edge counts (if enabled)
%   Figures (optional): scatter and boxplot comparing predicted vs observed (summary='Y')


if (fs_option==3 || fs_option==4) && (isempty(cov))
    error('Covariate matrix ''cov'' is required for fs_option=%d (partial correlation).', fs_option);
end

% Check BDR scores for normality
if checks == 'Y'
    normality_checks(all_behav, 'Backward Digit Recall Scores');
end

% initialise matrices
no_sub = size(all_mats,3);
no_node = size(all_mats,1);
behav_pred = zeros(no_sub,1);
num_pos_edge = zeros(no_sub,1);
num_neg_edge = zeros(no_sub,1);
train_mean = zeros(no_sub,1);   % training-fold mean y for each left-out

if( fs_option==1)
    disp('Edge selected based on Pearson correlation');
elseif( fs_option==2)
    disp('Edge selected based on Spearman correlation');
elseif( fs_option==3)
    disp('Edge selected based on partial correlation - Pearson');
elseif (fs_option==4)
    disp('Edge selected based on partial correlation - Spearman');
end

fprintf('\nLeave-one-out cross-validation\n');

for leftout = 1:no_sub
   % fprintf('\n Leaving out subj # %6d',leftout);
    
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    
    train_behav = all_behav;
    train_behav(leftout) = [];
    train_mean(leftout) = mean(train_behav);  
    
    if( fs_option==1)
        % correlate all edges with behavior using Pearson correlation
        [r_mat, p_mat] = corr(train_vcts', train_behav);
        
    elseif( fs_option==2)
        % correlate all edges with behavior using rank correlation
        [r_mat, p_mat] = corr(train_vcts', train_behav, 'type', 'Spearman');
        
    elseif(fs_option==3)
        % correlate all edges with behavior using partial corr - Pearsons
        cov_train = cov;
        cov_train(leftout,:) = [];
        [r_mat, p_mat] = partialcorr(train_vcts', train_behav, cov_train);
    elseif(fs_option==4)
        % correlate all edges with behavior using partial corr - Spearman
        cov_train = cov;
        cov_train(leftout,:) = [];
        [r_mat, p_mat] = partialcorr(train_vcts', train_behav, cov_train, 'type','Spearman');
      
    end
        
    r_mat = reshape(r_mat,no_node,no_node);
    p_mat = reshape(p_mat,no_node,no_node);
    
    % set threshold and define masks 
    pos_mask = zeros(no_node, no_node);
    neg_mask = zeros(no_node, no_node);
    
    pos_edge = find( r_mat >0 & p_mat < thresh);
    neg_edge = find( r_mat <0 & p_mat < thresh);
   
    pos_mask(pos_edge) = 1;
    neg_mask(neg_edge) = 1;

    % count unique edges (upper triangle, exclude diagonal)
    num_pos_edge(leftout) = nnz(triu(pos_mask,1));
    num_neg_edge(leftout) = nnz(triu(neg_mask,1));

    % get sum of all edges in TRAIN subs (divide by 2; matrices are symmetric)
    train_sumpos = zeros(no_sub-1,1);
    train_sumneg = zeros(no_sub-1,1);
    
    for ss = 1:no_sub-1
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask, 'omitnan'))/2;
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask, 'omitnan'))/2;
    end
    
    % build model on TRAIN subs
    b = regress(train_behav, [train_sumpos, train_sumneg, ones(no_sub-1,1)]);
    
    % run model on TEST sub
    test_mat = all_mats(:,:,leftout);
    test_sumpos = sum(sum(test_mat.*pos_mask, 'omitnan'))/2;
    test_sumneg = sum(sum(test_mat.*neg_mask, 'omitnan'))/2;

    behav_pred(leftout) = b(1)*test_sumpos + b(2)*test_sumneg + b(3);

end

% output correlation values
[r, ~]=corr(behav_pred, all_behav);

if checks == 'Y'
    % check residuals of observed vs predicted values for normality
    res = all_behav - behav_pred;
    normality_checks(res, 'Residuals (Observed - Predicted)'); 
end

%% Summary metrics and plots

% calculate validation metrics
SSE = (all_behav - behav_pred) .^2;  % sum of squares error
SST = (all_behav - train_mean).^2;   % sum of squares total
PRESS = sum(SSE);                    % predicted residual error sum of sqaures
RMSE  = sqrt(PRESS / no_sub);        % root mean squared error
pred_R2 = 1 - (sum(SSE) / sum(SST)); % prediction R^2
fprintf('\nr=%.3f | PRESS=%.1f | RMSE=%.3f | Prediction R^2=%.3f\n', r, PRESS, RMSE, pred_R2);

% calculate number of edges
num_edge = num_pos_edge + num_neg_edge;

if summary == 'Y'
    
    fprintf('Pos edges: mean=%.0f, sd=%.0f, median=%.0f, min/max=%d/%d\n', ...
        mean(num_pos_edge), std(num_pos_edge), median(num_pos_edge), min(num_pos_edge), max(num_pos_edge));
    fprintf('Neg edges: mean=%.0f, sd=%.0f, median=%.0f, min/max=%d/%d\n', ...
        mean(num_neg_edge), std(num_neg_edge), median(num_neg_edge), min(num_neg_edge), max(num_neg_edge));
    fprintf('Total edges: mean=%.0f, sd=%.0f, median=%.0f, min/max=%d/%d\n', ...
        mean(num_edge), std(num_edge), median(num_edge), min(num_edge), max(num_edge));

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



