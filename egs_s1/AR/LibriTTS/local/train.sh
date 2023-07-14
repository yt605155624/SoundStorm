#!/bin/bash

config_path=$1
train_output_path=$2
root_dir=$3
dump_dir=$4

python3 ${BIN_DIR}/train.py \
    --config_file=${config_path} \
    --train_semantic_path=${root_dir}/${dump_dir}/test/semantic_token.tsv \
    --train_phoneme_path=${root_dir}/${dump_dir}/test/phonemes.npy \
    --dev_semantic_path=${root_dir}/${dump_dir}/dev/semantic_token.tsv \
    --dev_phoneme_path=${root_dir}/${dump_dir}/dev/phonemes.npy \
    --output_path=${root_dir}/${train_output_path}