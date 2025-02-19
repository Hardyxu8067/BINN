# Runs BINN training on GPUs, interactively.
# 1) Request job. On slurm, an example command to request 4 GPUs is:
# srun -p full -n 1 -c 8 --time=72:00:00 --mem-per-cpu=10G --gpus 4 --pty /bin/bash -l
# 2) Run the script. (--num_CPU should be set to number of GPUs if available, otherwise the number of CPUs.)
# ./run_binn.sh

# # Small run for debugging
# for LR in 1e-2
# do
#     for FOLD in 1
#     do
#         for SEED in 0
#         do
#             python3 binns_DDP.py --data_seed 12345 --split random --n_datapoints 500 --cross_val_idx $FOLD --n_folds 10 \
#                 --optimizer AdamW --lr $LR --weight_decay 0 \
#                 --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
#                 --n_epochs 50 --patience 10 --model new_mlp --vertical_mixing original --vectorized compare \
#                 --leaky_relu --use_bn --embed_dim 5 --pos_enc early \
#                 --losses l1 param_reg --lambdas 1 100 \
#                 --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "DEBUGGING500"
#         done
#     done
# done
# exit

# Full run
for LR in 1e-2
do
    for FOLD in 1 2 3
    do
        for SEED in 0
        do
            python3 binns_DDP.py --data_seed 12345 --split random --cross_val_idx $FOLD --n_folds 10 \
                --optimizer AdamW --lr $LR --weight_decay 0 \
                --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
                --n_epochs 200 --patience 100 --model new_mlp --vertical_mixing original \
                --leaky_relu --use_bn --embed_dim 5 --pos_enc none \
                --losses l1 param_reg --lambdas 1 100 \
                --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "BINN_REPRO_NOPOSENC"
                # --whether_resume 1 --previous_job_id "20250127-214804_BINN_REPRO_lr=1e-02_seed=0_fold=1"
        done
    done
done

