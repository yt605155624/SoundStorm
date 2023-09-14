#!/bin/bash

config_path=$1
train_output_path=$2
root_dir=$3
dump_dir=$4

omp_num=16

OMP_NUM_THREADS=${omp_num} python3 ${BIN_DIR}/train.py \
    --config_file=${config_path} \
    --train_semantic_path=${root_dir}/${dump_dir}/train/semantic_token.npy \
    --train_phoneme_path=${root_dir}/${dump_dir}/train/phonemes.npy \
    --dev_semantic_path=${root_dir}/${dump_dir}/dev/semantic_token.npy \
    --dev_phoneme_path=${root_dir}/${dump_dir}/dev/phonemes.npy \
    --output_dir=${root_dir}/${train_output_path}