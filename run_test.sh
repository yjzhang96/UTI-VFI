#!/bin/bash

model_name='deblur-2step-400epoch'
result_dir='./results'
save_name="deblur-gopro-55"
interpolate_root="${result_dir}/${save_name}"
gpu_id=1
inter=1
deblur=1
know_time=0
inter_type='UTI'


CUDA_VISIBLE_DEVICES=$gpu_id python test.py \
--model_name=$model_name \
--result=$result_dir \
--save=$save_name \
--deblur=$deblur \
--inter=$inter \
--know_time_interval=$know_time \
--inter_root=$interpolate_root \
--blurry_videos="/home/yjz/datasets/LFR_gopro_55/test" \
--dataset_mode="4frames" \
--inter_type=$inter_type
