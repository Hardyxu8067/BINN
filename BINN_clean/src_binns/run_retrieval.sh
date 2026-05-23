# Reproducing the BINN paper: RETRIEVAL TEST

# Small sample first
for LR in 1e-2
do
    for FOLD in {1..10}
    do
        for SEED in 111
        do
            python3 binns_DDP.py --data_seed -1 --split random --representative_sample --cross_val_idx $FOLD --n_folds 10 \
                --para_to_predict five --synthetic_labels \
                --optimizer AdamW --lr $LR --weight_decay 0 --dropout_prob 0.3 --MC_dropout \
                --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109  \
                --n_epochs 50 --patience 10 --model new_mlp --vertical_mixing original --vectorized yes \
                --activation leaky_relu --para_activation sigmoid --use_bn --embed_dim 5 --pos_enc early \
                --losses smooth_l1 param_reg --lambdas 1 100 \
                --num_CPU 4 --use_ddp 1 --job_scheduler pbs --time_limit 12 --note "REPRO_BINN_RETRIEVAL_SAMPLE"
        done
    done
done

# Full dataset
for LR in 1e-2
do
    for FOLD in {1..10}
    do
        for SEED in 111
        do
            python3 binns_DDP.py --data_seed -1 --split random --cross_val_idx $FOLD --n_folds 10 \
                --para_to_predict five --synthetic_labels \
                --optimizer AdamW --lr $LR --weight_decay 0 --dropout_prob 0.3 --MC_dropout \
                --seed $SEED --init xavier_uniform --min_temp 10 --max_temp 109 \
                --n_epochs 300 --patience 300 --model new_mlp --vertical_mixing original --vectorized yes \
                --activation leaky_relu --para_activation sigmoid --use_bn --embed_dim 5 --pos_enc early \
                --losses smooth_l1 param_reg --lambdas 1 100 \
                --num_CPU 128 --use_ddp 1 --job_scheduler pbs --time_limit 12 --note "REPRO_BINN_RETRIEVAL"
        done
    done
done