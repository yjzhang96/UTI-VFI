#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python train_deblur.py \
# --model_name='deblur-SEframe-stage1' \
# --train_stage=1

CUDA_VISIBLE_DEVICES=0 python train_deblur.py \
--model_name='deblur-SEframe' \
--train_stage=2