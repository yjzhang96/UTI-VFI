#!/bin/bash

result_dir='./results'
save_name="UTI-gopro-55"
interpolate_root="./results/deblur-400epoch-55"


CUDA_VISIBLE_DEVICES=0 python test.py \
--model_name='deblur-SEfrmae' \
--result=$result_dir \
--save=$save_name \
--deblur=1 \
--inter=1 \
--inter_root=${result_dir}/${save_name} \
--blurry_videos="/home/yjz/datasets/LFR_gopro_55/test" \
--test_type='validation'