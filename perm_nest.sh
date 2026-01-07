#!/bin/bash
#SBATCH --job-name=perm_nest
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --output=outputs/%x_%A.out
#SBATCH --error=outputs/%x_%A.err

module purge
module load MATLAB/2023b

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

matlab -nodisplay -nosplash -r "no_iterations=1000; threshold=[0.001 0.01 0.05]; run('permutation_test_nested.m'); exit;"
