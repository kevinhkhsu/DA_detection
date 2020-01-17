#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
ADAPT_MODE=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${ADAPT_MODE} in
  K2C)
    TRAIN_IMDB_S="KITTI_synthCity"
    TRAIN_IMDB_T="cityscapes_train"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    NET_FINAL='trained_weights/adapt_weight/vgg16_faster_rcnn_K2C_stage2.pth'
    ;;
  C2F)
    TRAIN_IMDB_S="cityscapes_synthFoggytrain"
    TRAIN_IMDB_T="cityscapes_foggytrain"
    TEST_IMDB="cityscapes_foggyval"
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    NET_FINAL='trained_weights/adapt_weight/vgg16_faster_rcnn_C2F_stage2.pth'
    ;;
  C2BDD)
    TRAIN_IMDB_S="cityscapes_synthBDDdaytrain+cityscapes_synthBDDdayval"
    TRAIN_IMDB_T="bdd100k_train"
    TEST_IMDB="bdd100k_dayval"
    STEPSIZE="[50000]"
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    NET_FINAL='trained_weights/adapt_weight/vgg16_faster_rcnn_C2BDD_stage2.pth'
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB_S}_adapt_${TEST_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
    --tag ${EXTRA_ARGS_SLUG}_adapt \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
    --tag ${EXTRA_ARGS_SLUG}_adapt \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
fi

