#!/bin/bash
#SBATCH --job-name=perm_batch
#SBATCH --time=03:00:00
#SBATCH --mem=6G
#SBATCH --array=1-50
#SBATCH --output=outputs/%A/job_%A_%a.out
#SBATCH --error=outputs/%A/job_%A_%a.err

module load MATLAB/2023b
 
CHUNK=20
START=$(( (SLURM_ARRAY_TASK_ID-1)*CHUNK + 1 ))
END=$(( SLURM_ARRAY_TASK_ID*CHUNK ))

RESULTS_DIR="$PWD/outputs/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$RESULTS_DIR"

export START_ITER=$START
export END_ITER=$END
export RESULTS_DIR

matlab -batch "addpath('/users/pip24cc/code'); run('permutation_test_batch.m');"

