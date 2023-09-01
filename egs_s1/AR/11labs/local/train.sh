#!/bin/bash

config_path=$1
train_output_path=$2
root_dir=$3
dump_dir=$4

omp_num=16

# 注意 *_dirs 参数后面不可以有 ''='
# no train_non_speech_dirs for  11labs only
OMP_NUM_THREADS=${omp_num} python3 ${BIN_DIR}/train_librilight_6k.py \
    --config_file=${config_path} \
    --train_semantic_dirs ''${root_dir}'/'${dump_dir}'/11labs/train/' \
    --train_phoneme_dirs ''${root_dir}'/'${dump_dir}'/11labs/train/' \
    --dev_semantic_dirs ''${root_dir}'/'${dump_dir}'/11labs/dev/' \
    --dev_phoneme_dirs ''${root_dir}'/'${dump_dir}'/11labs/dev/' \
    --output_dir=${root_dir}/${train_output_path}
