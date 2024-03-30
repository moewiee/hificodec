#!/bin/bash
source path.sh
set -e

log_root="93m_logs_8cb"
# .lst save the wav path.
input_training_file="../../data/train_en_hash.lst" 
input_validation_file="../../data/valid_en_hash.lst"
input_hash_file="../../data/en_hash.json"


echo "Train model..."
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ${BIN_DIR}/train.py \
  --config config_24k_320d.json \
  --checkpoint_path ${log_root} \
  --input_training_file ${input_training_file} \
  --input_validation_file ${input_validation_file} \
  --input_hash_file ${input_hash_file} \
  --checkpoint_interval 10000 \
  --summary_interval 100 \
  --validation_interval 10000 \
  --training_epoch 1 \
  --stdout_interval 10 \
  --pretrain_path /home/ubuntu/tuna/AcademiCodec/egs/HiFi-Codec-24k-320d/HiFi-Codec-24k-320d
