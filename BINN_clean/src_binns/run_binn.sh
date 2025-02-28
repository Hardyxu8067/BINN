# Runs BINN training on GPUs, interactively.
# 1) Request job. On slurm, an example command to request 4 GPUs is:
# srun -p full -n 1 -c 8 --time=72:00:00 --mem-per-cpu=10G --gpus 4 --pty /bin/bash -l
# 2) Run the script. From src_binns directory:
# ./run_binn.sh

# NOTES:
# For debugging, you can add --representative sample to restrict to 1000 representative sites. Otherwise we use the whole dataset (~26000 sites)
# --losses can be any number of losses, see its help. You can weight them manually using --lambdas.
# --num_CPU should be set to number of GPUs if GPUs are available, otherwise the number of CPUs.
# --note is a string that describes this run; it's used in the output directory name, and we append the final
# metrics to a result excel file with that name.
# See documentation in binns_DDP for info on other arguments.
for LR in 1e-2
do
    for FOLD in 1
    do
        for SEED in 0
        do
            python3 binns_DDP.py --data_seed 12345 --split random --representative_sample --cross_val_idx $FOLD --n_folds 10 \
                --optimizer AdamW --lr $LR --weight_decay 0 \
                --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
                --n_epochs 50 --patience 10 --model new_mlp --vertical_mixing original --vectorized yes \
                --activation leaky_relu --use_bn --embed_dim 5 --pos_enc early \
                --losses smooth_l1 param_reg --lambdas 1 100 \
                --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "REPRO_BINN_SAMPLE"
        done
    done
done
for LR in 1e-2
do
    for FOLD in 1
    do
        for SEED in 0
        do
            python3 binns_DDP.py --data_seed 12345 --split random --representative_sample --cross_val_idx $FOLD --n_folds 10 \
                --optimizer AdamW --lr $LR --weight_decay 0 \
                --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
                --n_epochs 50 --patience 10 --model new_mlp --vertical_mixing original --vectorized yes \
                --activation leaky_relu --use_bn --embed_dim 5 --pos_enc early \
                --losses smooth_l1 param_reg --lambdas 1 100 \
                --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "REPRO_BINN"
        done
    done
done

exit

# To resume from a previous partial run, run something like this.
# Note that "--whether_resume" should be set to 1, and --previous_job_id" should be set to the output folder name we load from.
# for LR in 1e-2
# do
#     for FOLD in 1
#     do
#         for SEED in 0
#         do
#             python3 binns_DDP.py --data_seed 12345 --split random --cross_val_idx $FOLD --n_folds 10 \
#                 --optimizer AdamW --lr $LR --weight_decay 0 \
#                 --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
#                 --n_epochs 200 --patience 100 --model new_mlp --vertical_mixing original \
#                 --activation leaky_relu --use_bn --embed_dim 5 --pos_enc early \
#                 --losses smooth_l1 param_reg --lambdas 1 100 \
#                 --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "REPRO_BINN" \
#                 --whether_resume 1 --previous_job_id "20250127-214804_BINN_REPRO_lr=1e-02_seed=0_fold=1"
#         done
#     done
# done

