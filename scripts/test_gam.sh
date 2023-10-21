#!/usr/bin/env bash
set -e 
set -x 


GPU=$1
lambda1=$2
DATASET=$3
SEED=$4
lr=$5
alpha=$6
beta=$7
m3=$8

CKPT_NAME=${DATASET}'_'${SEED}'_'${lr}'_'${lambda1}'_'${alpha}'_'${beta}'_'${m3}

python src/test.py \
     --load_ckpt=checkpoints/${CKPT_NAME}/best_epoch.ckpt \
     --num_iterative_epochs 6 \
     --gpus ${GPU} \
     --dataset ${DATASET} \
     --data_file 'preprocessed/'${CKPT_NAME} \
     --seed ${SEED} \
     --lambda1 ${lambda1} \
     --alpha ${alpha} \
     --beta ${beta} \
     --m3 ${m3}