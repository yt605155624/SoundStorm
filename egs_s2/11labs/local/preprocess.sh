#!/bin/bash
stage=0
stop_stage=0
root_dir=$1
data_dir=$2
hubert_path=$3
quantizer_path=$4
layer=$5
dump_dir=$6

# please download VAD file for LibriLight first, see ../README.md
# sort spk_id list first by lexicographical order
# split sorted spk_id list into ${nshard}, ${rank} should be in 0 ~ ${nshard} - 1

# get semantic token, download Hubert to pretrained_model/hubert/
# get semantic for small (489 speakers)
# ${nshard} can be 1 for small
# --num-cpu=256 cost 50G GPU of A100 (这个是不设置 OMP_NUM_THREADS 是的数)
# OMP_NUM_THREADS=1 占用 9G 显存, OMP_NUM_THREADS=4 10G
# cost ~6 hours on one machine (可能存在 cpu 资源争抢)
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "get_semantic_token_11labs.py start!"
    for rank_id in {0..7}; do
        gpu_id=$((rank_id / 2))
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 ${BIN_DIR}/get_semantic_token_11labs.py \
            --data_dir=${data_dir} \
            --dump_dir=${root_dir}/${dump_dir} \
            --hubert_path=${hubert_path} \
            --quantizer_path=${quantizer_path} \
            --num-cpu=256 \
            --layer=${layer} \
            --nshard=8 \
            --rank=${rank_id} &
        eval pid${rank_id}="$!"
    done
    wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5" "$pid6" "$pid7"
    echo "get_semantic_token_11labs.py done!"
fi

# extract acoustic token by HiFi-Codec `.pth`
# download hificodec's param to pretrained_model/hificodec/
# softlink AcademiCodec/academicodec to ${MAIN_ROOT} first

# get acoustic for small
# num-cpu=30 for 80G GPU, cost 2 ~ 3 hours
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "get_acoustic_token_11labs.py start!"
    for rank_id in {0..7}; do
        gpu_id=$((rank_id))
        OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${gpu_id} python3 ${BIN_DIR}/get_acoustic_token_11labs.py \
            --data_dir=${data_dir} \
            --dump_dir=${root_dir}/${dump_dir} \
            --codec_name=hificodec \
            --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
            --config_path=pretrained_model/hificodec/config_16k_320d.json \
            --sr=16000 \
            --num-cpu=30 \
            --nshard=8 \
            --rank=${rank_id} &
        eval pid${rank_id}="$!"
    done
    wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5" "$pid6" "$pid7"
    echo "get_acoustic_token_11labs.py done!"
fi

# test the generated acoustic_token
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mkdir -p codebook2wav_output
    # HiFi-Codec
    python3 ${BIN_DIR}/codebook2wav.py \
        --codec_name=hificodec \
        --model_path=pretrained_model/hificodec/HiFi-Codec-16k-320d-large-universal \
        --config_path=pretrained_model/hificodec/config_16k_320d.json \
        --sr=16000 \
        --input_path=${root_dir}/${dump_dir}/test/acoustic_token/hificodec/'ZinedineZidane-ZinedineZidane_999-1.npy' \
        --output_dir=codebook2wav_output/ \
        # --num_quant=3 # NOT WORK HERE, default Nq of HiFi-Codec is 4 and cannot be reduced
fi
