#!/bin/bash
stage=0
stop_stage=0
root_dir=$1
data_dir=$2
dump_dir=$3

# use CPU only
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "get_phones_11labs.py start!"
    for rank_id in {0..7}; do
        OMP_NUM_THREADS=1 python3 ${BIN_DIR}/get_phones_11labs.py \
            --data_dir=${data_dir} \
            --dump_dir=${root_dir}/${dump_dir} \
            --num-cpu=30 \
            --nshard=8 \
            --rank=${rank_id} &
        eval pid${rank_id}="$!"
    done
    wait "$pid0" "$pid1" "$pid2" "$pid3" "$pid4" "$pid5" "$pid6" "$pid7"
    echo "get_phones_11labs.py done!"
fi

# generate semantic_token (hificodec.pth) in egs1
