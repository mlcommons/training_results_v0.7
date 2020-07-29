#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=8 \
    -u /workspace/bert/run_pretraining.py \
    --seed=42 \
    --do_train \
    --target_accuracy=0.714 \
    --accuracy_score_averaging=1 \
    --config_file=/workspace/phase1/bert_config.json \
    --skip_checkpoint \
    --output_dir=/results \
    --fp16 \
    --allreduce_post_accumulation --allreduce_post_accumulation_fp16 \
    --gradient_accumulation_steps=1 \
    --log_freq=1 \
    --train_batch_size=4 \
    --learning_rate=4e-5 \
    --warmup_proportion=1.0 \
    --input_dir=/workspace/data_phase2 \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --max_steps=100 \
    --init_checkpoint=/workspace/phase1/model.ckpt-28252 \

