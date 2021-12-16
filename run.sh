#!/bin/bash
python main.py \
--d_model 64 \
--attn_heads 4 \
--bert_layers 3 \
--mask_prob 0.3 \
--eval_per_steps 3000 \
--num_epoch 15 \
--loss_type le \
--device cuda \
--le_share 0 \
--soft_taget mlp \
--le_res 0.1 \
--enable_sample 1 \
--data_path "processed_ml-20m.csv\"
