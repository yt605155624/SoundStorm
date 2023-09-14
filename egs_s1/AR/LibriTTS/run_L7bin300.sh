#!/bin/bash
set -e

source path.sh

gpus=0,1,2,3
stage=0
stop_stage=100
train_output_path=exp/base_L7_km300
# dir to set part/all of dump dataset and experiment result
root_dir='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/ar_s1/SoundStorm'
# there should be *.wav 、*/*.wav or */*/*.wav in data_dir
data_dir='~/datasets/LibriTTS-R'
config_path=conf/base_L7bin300.yaml
dump_dir=dump_libritts_base_L7_km300
ckpt_name='epoch=99-step=49000.ckpt'
hubert_path=pretrained_model/hubert/hubert_base_ls960.pt
quantizer_path=pretrained_model/hubert/train-clean-360_hubert_base_ls960_L7_km300.bin
prompt_wav_path='/nfs-speech-cpfs/dev/yuantian04/Vivid_TTS/SoundStorm/SoundStorm/SoundStorm/dump_libritts_universal_hificodec/test/synthesize_input/1001_134708_000013_000000.wav'

# with the following command, you can choose the stage range you want to run
# such as `./run.sh --stage 0 --stop-stage 0`
# this can not be mixed use with `$1`, `$2` ...
source ${MAIN_ROOT}/utils/parse_options.sh || exit 1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # prepare data
    CUDA_VISIBLE_DEVICES=${gpus} ./local/preprocess.sh ${root_dir} ${data_dir} ${dump_dir}|| exit -1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/train.sh ${config_path} ${train_output_path} ${root_dir} ${dump_dir}|| exit -1
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/test.sh ${config_path} ${train_output_path} ${ckpt_name} ${root_dir} ${dump_dir}|| exit -1
fi

# text-to-semantic 
# text -> frontend -> semantic
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    CUDA_VISIBLE_DEVICES=${gpus} ./local/t2s.sh ${config_path} ${train_output_path} ${ckpt_name} ${root_dir} ${hubert_path} ${quantizer_path} ${prompt_wav_path}|| exit -1
fi
