#!/bin/sh

METHOD=$1
RATE=$2
WORK_DIR=$3
DATA_DIR=$4
PRETRAINED_MODEL_DIR=$5
OUTPUT=$6
BATCH_SIZE=$7
EPOCHS=$8

PHASE0="${OUTPUT}/phase0"
PHASE1="${OUTPUT}/phase1"
PHASE2="${OUTPUT}/phase2"
TMP_STORE="${OUTPUT}/intermediate"
PRETRAINED_MODEL="${OUTPUT}/pretrained"

cd $WORK_DIR

python load_pretrained_imagenet_model.py --data_dir=$DATA_DIR --model_dir=$PRETRAINED_MODEL_DIR --inter_store=$TMP_STORE --output_path=$PRETRAINED_MODEL

python phase0.py --resnet_size=50 --batch_size=$BATCH_SIZE --model_class='imagenet' --data_dir=$DATA_DIR --pretrained_model_dir=$PRETRAINED_MODEL --method=$METHOD --scope=$METHOD --rate=$RATE --output_path=$PHASE0

python phase1.py --resnet_size=50 --batch_size=$BATCH_SIZE --model_class='imagenet' --data_dir=$DATA_DIR --pretrained_model_dir=$PRETRAINED_MODEL --method=$METHOD --scope=$METHOD --rate=$RATE --phase_zero=$PHASE0 --output_path=$PHASE1 --train_epochs=$EPOCHS --exp_growth=False

# The results in our paper is obtained without phase2. Please uncomment the scripts below to run/play with phase2.

# python phase2.py --resnet_size=50 --batch_size=$BATCH_SIZE --model_class='imagenet' --data_dir=$DATA_DIR --method=$METHOD --scope=$METHOD --rate=$RATE --phase_one=$PHASE1  --output_path=$PHASE2

# python imagenet_main_phase2.py --batch_size=$BATCH_SIZE --data_dir=$DATA_DIR --method=$METHOD --scope=$METHOD --rate=$RATE --model_dir="${PHASE2}/${METHOD}/rate${RATE}/" --train_epochs=$EPOCHS
