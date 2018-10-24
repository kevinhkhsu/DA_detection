#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
TEST_ITER=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  KITTI)
    TRAIN_IMDB="KITTI_train+KITTI_val"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[350000]"
    ITERS=${TEST_ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  cityscapes)
    TRAIN_IMDB="cityscapes_train"
    TEST_IMDB="KITTI_val"
    STEPSIZE="[350000]"
    ITERS=${TEST_ITER}
    ANCHORS="[4,8,16,32,64]"
    RATIOS="[0.5,1,2]"
    ;;
  bdd100k)
    TRAIN_IMDB="bdd100k_daytrain"
    TEST_IMDB="bdd100k_nightval"
    STEPSIZE="[350000]"
    ITERS=${TEST_ITER}
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac


LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_crossDomain_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_day_iter_${ITERS}.pth
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
fi

