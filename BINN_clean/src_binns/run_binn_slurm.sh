#!/bin/bash

# Runs BINN training on GPUs, as a Slurm job. Usage:
# sbatch run_slurm.sh
# (To change the number of GPUs, change --gpus and --num_CPU arguments to that number.)
# (To run on CPU, remove the --gpus line and set --num_CPU to the number of CPUs.)
# Output will appear in a file 'slurm-N.out' where N is the job ID.

# Request the full partition, which contains the GPU nodes.
#SBATCH -p full
# Name the job so it's meaningful in the job list
#SBATCH -J binn_train
# Request 4 GPUs 
#SBATCH --gpus 4
# Request 4 CPU cores (9 hyperthreads).
#SBATCH -c 8
# Specify the resources should be assigned to a single task on one node.
#SBATCH -N 1 -n 1
# Request a total of 50GB RAM
#SBATCH --mem=50GB
# Request a walltime limit of 72 hours
#SBATCH -t 72:00:00

# INFO: Print properties of job as submitted
echo "SLURM_JOB_ID = $SLURM_JOB_ID"
echo "SLURM_JOB_NAME = $SLURM_JOB_NAME"
echo "SLURM_JOB_PARTITION = $SLURM_JOB_PARTITION"
echo "SLURM_SUBMIT_HOST = $SLURM_SUBMIT_HOST"
echo "SLURM_NTASKS = $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE = $SLURM_MEM_PER_NODE"
echo "SLURM_JOB_NUM_NODES = $SLURM_JOB_NUM_NODES"
echo "SLURM_GPUS = $SLURM_GPUS"

# INFO: Print properties of job as scheduled by Slurm
echo "SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
echo "SLURMD_NODENAME = $SLURMD_NODENAME"
echo "SLURM_TASKS_PER_NODE = $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE = $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"


# Set number of threads
# export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/2))

# # Use scratch for temporary files to avoid space limits in /tmp
# export TMPDIR=/glade/scratch/$USER/temp
# mkdir -p $TMPDIR

# Load modules to match compile-time environment
# module purge
source ~/.bashrc
# module load cuda
# module load mkl

# Activate environment in conda
conda activate binn

# Basic BINN training: loop through learning rates, folds, seeds
for LR in 1e-2
do
    for FOLD in 1 2 3 4 5 6 7 8 9 10
    do
        for SEED in 0
        do
            python3 binns_DDP.py --data_seed 12345 --split random --cross_val_idx $FOLD --n_folds 10 \
                --optimizer AdamW --lr $LR --weight_decay 0 \
                --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
                --n_epochs 50 --patience 10 --model new_mlp --vertical_mixing original --vectorized yes \
                --activation leaky_relu --use_bn --embed_dim 5 --pos_enc early \
                --losses smooth_l1 param_reg --lambdas 1 100 \
                --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 71.5 --note "REPRO_BINN"
        done
    done
done
