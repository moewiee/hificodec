#!/bin/bash
source path.sh
set -e

input_validation_file="../../data/valid.lst"
input_hash_file="../../data/en_hash.json"

echo "Eval model..."
export CUDA_VISIBLE_DEVICES=0
python ${BIN_DIR}/evaluate.py \
--config config_24k_320d.json \
--input_validation_file ${input_validation_file} \
--input_hash_file ${input_hash_file} \
--pretrained_path /home/ubuntu/tuna/AcademiCodec/egs/HiFi-Codec-24k-320d/93m_logs_8cb/g_00230000
