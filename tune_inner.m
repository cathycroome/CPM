function best_t = tune_inner(train_mats, train_behav, cov_train, thresh_list, K)
% K-fold inner CV for threshold tuning
% train_mats  : M x M x N  connectivity matrices
% train_behav : N x 1      behaviour scores
% cov_train   : N x L      covariates (can be [])
% thresh_list : vector of candidate thresholds
% K           : number of folds (default 5)

if nargin < 5 || isempty(K), K = 10; end

no_sub  = size(train_mats,3);
no_node = size(train_mats,1);
scores  = nan(length(thresh_list),1);

% --- assign folds using cvpartition ---
cvp = cvpartition(no_sub, 'KFold', K);

% ---- iterate over thresholds ----
for option = 1:length(thresh_list)
    
    t = thresh_list(option);
    predicted_inner = nan(no_sub,1);

    for f = 1:K
        test_idx  = test(cvp, f);     % logical mask for test subjects
        train_idx = training(cvp, f); % logical mask for training subjects

        train_mats_fold  = train_mats(:,:,train_idx);
        train_behav_fold = train_behav(train_idx);

        if ~isempty(cov_train)
            cov_train_fold = cov_train(train_idx,:);
        else
            cov_train_fold = [];
        end

        vcts_train = reshape(train_mats_fold, [], size(train_mats_fold,3));

        % edge-behaviour associations
        if isempty(cov_train_fold)
            [r_mat, p_mat] = corr(vcts_train', train_behav_fold, 'type','Spearman');
        else
            [r_mat, p_mat] = partialcorr(vcts_train', train_behav_fold, cov_train_fold, 'type','Spearman');
        end

        r_mat = reshape(r_mat, no_node, no_node);
        p_mat = reshape(p_mat, no_node, no_node);

        pos_mask = (r_mat > 0) & (p_mat < t);
        neg_mask = (r_mat < 0) & (p_mat < t);

        % if no edges, fall back to mean predictor
        if ~any(pos_mask(:)) && ~any(neg_mask(:))
            predicted_inner(test_idx) = mean(train_behav_fold);
            continue
        end

        % build regression on TRAIN subjects
        n_train = size(train_mats_fold,3);
        train_sum_pos = zeros(n_train,1);
        train_sum_neg = zeros(n_train,1);
        for ss = 1:n_train
            train_sum_pos(ss) = sum(sum(train_mats_fold(:,:,ss).*pos_mask,'omitnan'))/2;
            train_sum_neg(ss) = sum(sum(train_mats_fold(:,:,ss).*neg_mask,'omitnan'))/2;
        end

        b = regress(train_behav_fold, [train_sum_pos, train_sum_neg, ones(n_train,1)]);

        % predict TEST subjects
        test_ids = find(test_idx);         
        for subj_id = 1:length(test_ids)
            test_mat = train_mats(:,:,subj_id);
            test_sum_pos = sum(sum(test_mat.*pos_mask,'omitnan'))/2;
            test_sum_neg = sum(sum(test_mat.*neg_mask,'omitnan'))/2;
            predicted_inner(subj_id) = b(1)*test_sum_pos + b(2)*test_sum_neg + b(3);
        end
    end

    scores(option) = corr(predicted_inner, train_behav, 'rows','complete');

end

[~, idx_best] = max(scores);
best_t = thresh_list(idx_best);
end
