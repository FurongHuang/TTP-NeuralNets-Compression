#!/bin/bash

METHOD=$1
RATE=$2
DEVICE=$3
RATE_DECAY=$4

source ~/tensorflow/bin/activate
cd /home/jingling/tensor-net/models-1.7.0/official/resnet

export CUDA_VISIBLE_DEVICES=$3
python phase0.py --rate_decay=$RATE_DECAY --data_dir='/home/jingling/Data/cifar10_data/' --pretrained_model_dir='/home/jingling/models/cifar10/normal' --filename='normal_weights.ckpt.meta' --method=$METHOD --scope=$METHOD --rate=$RATE --output_path='/home/jingling/models/cifar10/dynamic_rate/phase0/'${RATE_DECAY}

export CUDA_VISIBLE_DEVICES=$3 
python phase1.py --rate_decay=$RATE_DECAY --batch_size=512 --data_dir='/home/jingling/Data/cifar10_data/' --pretrained_model_dir='/home/jingling/models/cifar10/normal' --method=$METHOD --scope=$METHOD --rate=$RATE --phase_zero='/home/jingling/models/cifar10/dynamic_rate/phase0/'${RATE_DECAY} --output_path='/home/jingling/models/cifar10/dynamic_rate/phase1/'${RATE_DECAY} --train_epochs=100 --exp_growth=False