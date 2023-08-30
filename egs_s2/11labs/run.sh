#!/bin/bash
# run_base_L7_km300
# train LibriLight 6k (small + medium) by default
set -e

source path.sh

gpus=0,1,2,3
stage=0
stop_stage=100
train_output_path='exp_11labs/default'
# dir to set part/all of dump dataset and experiment result
root_dir='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm'
# there should be *.wav 、*/*.wav or */*/*.wav in data_dir
data_dir='~/datasets/11labs_merge'
config_path='conf/default.yaml'
log_frequency=1
# 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)
dist_url='tcp://127.0.0.1:29505'
# use which checkpoint file to test
ckpt_name='last.pth'
# should be same with ${layer} in hubert_kms.sh
layer=7
# should be same with ${hubert_path} in hubert_kms.sh
hubert_path=pretrained_model/hubert/hubert_base_ls960.pt
quantizer_path=pretrained_model/hubert/train-clean-360_hubert_base_ls960_L7_km300.bin
dump_dir=dump_11labs
# for synthesize_e2e.sh
prompt_wav_path='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_libritts_base_L9_km500/test/synthesize_input/1006_135212_000060_000004.wav'
S1_config_file='../../egs_s1/AR/LibriLight/conf/default.yaml'
S1_ckpt_path='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/ar_s1/SoundStorm/exp_librilight/small_medium_filter_nonspeech/ckpt/epoch=19-step=37000.ckpt'
sil_token=4 # 4 for 300 bin

# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh ${root_dir} ${data_dir} ${hubert_path} ${quantizer_path} ${layer} ${dump_dir}|| exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${config_path} ${train_output_path} ${root_dir} ${log_frequency} ${dist_url} ${dump_dir}|| exit -1
fi
# test with test dataset, prompt and target should be the same audio
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/test.sh ${config_path} ${train_output_path} ${ckpt_name} ${root_dir} ${dump_dir}|| exit -1
fi

# synthesize input prompt/target_semantic/acoustic 4 files, which means prompt and target can from different audios
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize.sh \
    ${config_path} ${train_output_path} ${ckpt_name} ${root_dir} \
    ${hubert_path} ${quantizer_path} ${dump_dir}|| exit -1
fi

# synthesize_e2e with S1 (text -> semantic token) model
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/synthesize_e2e.sh \
    ${config_path} ${train_output_path} ${ckpt_name} ${root_dir} \
    ${hubert_path} ${quantizer_path} ${prompt_wav_path} \
    ${S1_config_file} ${S1_ckpt_path} ${sil_token}|| exit -1
fi