#!/usr/bin/env bash

data_dir=voc1
EXP_DIR=exps/${data_dir}

for fewshot_seed in 01 02 03 04 05 06 07 08 09 10
do
  for num_shot in 01 02 03 05 10
  do
    FS_FT_DIR=${EXP_DIR}/seed${fewshot_seed}_${num_shot}shot
    FS_FT_INFER=${FS_FT_DIR}/inference
    mkdir ${FS_FT_INFER}

    if [ $num_shot -eq 1 ]
    then
      checkpoint='checkpoint0699.pth'
    elif [ $num_shot -eq 2 ]
    then
      checkpoint='checkpoint0599.pth'
    elif [ $num_shot -eq 3 ]
    then
      checkpoint='checkpoint0599.pth'
    elif [ $num_shot -eq 5 ]
    then
      checkpoint='checkpoint0499.pth'
    elif [ $num_shot -eq 10 ]
    then
      checkpoint='checkpoint0499.pth'
    else
      exit
    fi

    python -u main.py \
        --dataset_file ${data_dir} \
        --backbone resnet101 \
        --num_feature_levels 1 \
        --enc_layers 6 \
        --dec_layers 6 \
        --hidden_dim 256 \
        --num_queries 300 \
        --batch_size 2 \
        --class_prototypes_cls_loss \
        --resume ${FS_FT_DIR}/${checkpoint} \
        --fewshot_finetune \
        --fewshot_seed ${fewshot_seed} \
        --num_shots ${num_shot} \
        --eval \
    2>&1 | tee ${FS_FT_INFER}/log.txt
  done
done
