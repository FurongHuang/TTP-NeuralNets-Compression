#!/bin/bash

METHOD=$1
RATE=$2
DEVICE=$3
DATA_DIR=$4
PRETRAINED_MODEL=$5
PRETRAINED_MODEL_FILENAME=$6
OUTPUT=$6

PHASE0="${OUTPUT}/phase0"
PHASE1="${OUTPUT}/phase0"
PHASE2="${OUTPUT}/phase0"

#source ~/tensorflow/bin/activate
#cd /home/jingling/tensor-net/models-1.7.0/official/resnet_imagenet_2.0

export CUDA_VISIBLE_DEVICES=$DEVICE

python phase0.py --data_dir=$DATA_DIR --pretrained_model_dir=$PRETRAINED_MODEL --filename=$PRETRAINED_MODEL_FILENAME --method=$METHOD --scope=$METHOD --rate=$RATE --output_path=$PHASE0

python phase1.py --batch_size=128 --data_dir=$DATA_DIR --pretrained_model_dir=$PRETRAINED_MODEL --method=$METHOD --scope=$METHOD --rate=$RATE --phase_zero=$PHASE0 --output_path=$PHASE1 --train_epochs=100 --exp_growth=False

# the filename (e.g. "'model.ckpt-78201'" ) below should be the latest model in $PHASE1
python phase2.py --batch_size=128 --data_dir=$DATA_DIR —--method=$METHOD --scope=$METHOD --rate=$RATE --phase_one=$PHASE1 —filename='model.ckpt-78201' —output_path=$PHASE2 —rate_decay='flat'

python cifar10_main_phase2.py —batch_size=128 --data_dir=$DATA_DIR —--method=$METHOD --scope=$METHOD --rate=$RATE —model_dir="${PHASE2}/${METHOD}/rate${RATE}/" —rate_decay='flat'
