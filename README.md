## Source and Attribution

Original CPM Matlab scripts were obtained from NITRC (Package: bioimagesuite, Release: CPM_OHBM, file: ConnectomeCodeShen_OHBM.zip, released 09 Dec 2024). 
Available at: https://www.nitrc.org/frs/?group_id=51

Shen, X., Finn, E. S., Scheinost, D., Rosenberg, M. D., Chun, M. M., Papademetris, X., & Constable, R. T. (2017). Using connectome-based predictive modeling to predict individual behavior from brain connectivity. Nature Protocols, 12(3), 506–518. https://doi.org/10.1038/nprot.2016.178

--------

## Original README (from NITRC package, unmodified)

The Matlab script files are examples used in the "How-to CPM" tutorial by Xilin Shen. The paper "Using connectome-based predictive modeling to predict individual behavior from brain connectivity" published in Nature Protocols describes the original CPM protocol.

codeshare_behavioralprediction.m implements the connectome based model to predict a behavioral measure, in a leave-one-subject-out scheme. 

permutation_test_example.m estimates the significance of the test statistics, which is the model performance. 1000 CPM models are built based on randomly shuffled pairing between the connectome and the dependent variable, to generate the distribution of the test statistics. The statistical significance of the true model is calculated as the percentage of cases exceeding the performance of the true model.

cpm_kfold_test_example.m demonstrates how to run CPM using K-fold cross validation. The code outputs the binary masks of the connections which can be visualized using tools available at https://bioimagesuiteweb.github.io/webapp/connviewer.html?species=human#

--------

## Modifications 

Compared to the original NITRC package, the following modifications have been made:

`permutation_test_example.m`
- Added functionality to split permutations across HPC job arrays.
- Renamed to permutation_test_batch.m

`predict_behavior.m`
- Pos/neg models removed.
- Added validation metrics (PRESS, RMSE, prediction R²).
- Reports number of selected edges per fold.
- Optional scatterplot and boxplot outputs.

New functions/scripts
- `merge_perm_batches.m` merges permutation results across array jobs, computes p-value, saves combined outputs.
- `normality_checks.m` computes skewness, kurtosis, and Lilliefors test; produces histogram + Q–Q plot and exports QC figure.
- `permutation_test_nested.m` runs CPM with outer LOOCV + optional inner-K-Fold tuning of p-threshold (parallelised permutation testing).
- `tune inner.m` runs K-fold inner CV for threshold tuning
- `prog.tick.m` helper function to print progress for permutation_test_nested.m
- `perm_batch.sh` runs batched permutation workflow (example SLURM submission script for running the batched permutation workflow on Stanage HPC)
- `perm_nest.sh` runs parallelised permutation workflow (example SLURM submission script for running the nested/parallelised permutation workflow on Stanage HPC)

--------

## Workflows (choose one)
 
1) `permutation_test_batch.m`

- Runs fixed-threshold CPM permutations serially; each array task processes a slice of iterations and writes a batch file.
- When to use: Large permutation counts on HPC with job arrays (runs faster than nested when not threshold tuning)
- CPU: 1 core per task (no parallel pool).
- An example SLURM script (perm_batch.sh) is provided for Stanage.
- After: Run merge_perm_batches.m to combine all batch files and compute the p-value.


2) `permutation_test_nested.m`

- Runs CPM with outer LOOCV and (optional) inner 5-fold CV to tune the p-threshold per outer fold (from {.001, .01, .05}); also performs permutation testing in-process.
- When to use: You need threshold tuning and/or want permutations handled within a single job.
- CPU: 32–64 cores recommended (thread-based pool) for ≥1,000 permutations.
- An example SLURM script (perm_nest.sh) is provided for Stanage.
