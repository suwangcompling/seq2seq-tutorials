#!/bin/bash

python seq2seq_lite.py \
    --data_path=mscoco/mscoco_formatted.p \
    --model_save_path=mscoco/seq2seq.ckpt \
    --result_save_path=mscoco/results.txt \
    --clear_prev_result=True \
    --batch_size=32 \
    --hidden_size=100 \
    --n_layers=2 \
    --dropout=0.3 \
    --residual=True \
    --lr=1e-5 \
    --enforce_ratio=0.5 \
    --clip=10.0 \
    --n_epochs=10 \
    --epoch_size=10000 \
    --print_every=100 \
    --n_eval_batches=625 \
    --max_decoding_length=20
