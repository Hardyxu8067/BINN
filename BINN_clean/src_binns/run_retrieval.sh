# Reproducing the BINN paper: RETRIEVAL TEST

# Small sample first
for LR in 1e-2
do
    for FOLD in 1
    do
        for SEED in 0
        do
            python3 binns_DDP.py --data_seed 12345 --split random --representative_sample --cross_val_idx $FOLD --n_folds 10 \
                --para_to_predict four --synthetic_labels \
                --optimizer AdamW --lr $LR --weight_decay 0 \
                --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
                --n_epochs 50 --patience 10 --model new_mlp --vertical_mixing original --vectorized yes \
                --activation leaky_relu --use_bn --embed_dim 5 --pos_enc early \
                --losses smooth_l1 param_reg --lambdas 1 100 \
                --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "REPRO_BINN_RETRIEVAL_SAMPLE"
        done
    done
done

# Full dataset
for LR in 1e-2
do
    for FOLD in 1
    do
        for SEED in 0
        do
            python3 binns_DDP.py --data_seed 12345 --split random --cross_val_idx $FOLD --n_folds 10 \
                --para_to_predict four --synthetic_labels \
                --optimizer AdamW --lr $LR --weight_decay 0 \
                --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
                --n_epochs 50 --patience 10 --model new_mlp --vertical_mixing original --vectorized yes \
                --activation leaky_relu --use_bn --embed_dim 5 --pos_enc early \
                --losses smooth_l1 param_reg --lambdas 1 100 \
                --num_CPU 4 --use_ddp 1 --job_scheduler slurm --time_limit 23.5 --note "REPRO_BINN_RETRIEVAL"
        done
    done
done