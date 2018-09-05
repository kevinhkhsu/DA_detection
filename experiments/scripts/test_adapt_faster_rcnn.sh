#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  KITTI)
    TRAIN_IMDB_S="KITTI_train"
    TRAIN_IMDB_T="KITTI_fake"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[50000]"
    ITERS=140000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  cityscapes)
    TRAIN_IMDB_S="cityscapes_train"
    TRAIN_IMDB_T="KITTI_train"
    TEST_IMDB="KITTI_val"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB_S}_adapt_$(TEST_IMDB)_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB_S}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB_S}/_adapt/${NET}_faster_rcnn_img_K2C_490k_imnet_iter_${ITERS}.pth
fi

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG}_adapt \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG}_adapt \
    --net ${NET} \
    --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
          ${EXTRA_ARGS}
fi

