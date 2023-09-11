#!/bin/bash
# small + medium + 11labs
## 1. finetune from model pretrained with small + medium
## 2. train from scratch

config_path=$1
train_output_path=$2
root_dir=$3
dump_dir=$4

omp_num=24

# 注意 *_dirs 参数后面不可以有 ''='
OMP_NUM_THREADS=${omp_num} python3 ${BIN_DIR}/train_librilight_60k.py \
    --config_file=${config_path} \
    --train_semantic_dirs ''${root_dir}'/'${dump_dir}'/small/train/' ''${root_dir}'/'${dump_dir}'/medium/train/' \
    --train_phoneme_dirs ''${root_dir}'/'${dump_dir}'/small/train/' ''${root_dir}'/'${dump_dir}'/medium/train/' \
    --dev_semantic_dirs ''${root_dir}'/'${dump_dir}'/small/dev/' ''${root_dir}'/'${dump_dir}'/medium/dev/' \
    --dev_phoneme_dirs ''${root_dir}'/'${dump_dir}'/small/dev/' ''${root_dir}'/'${dump_dir}'/medium/dev/' \
    --train_non_speech_dirs ''${root_dir}'/'${dump_dir}'/small/train/' ''${root_dir}'/'${dump_dir}'/medium/train/' \
    --dev_non_speech_dirs ''${root_dir}'/'${dump_dir}'/small/dev/' ''${root_dir}'/'${dump_dir}'/medium/dev/' \
    --output_dir=${root_dir}/${train_output_path}
