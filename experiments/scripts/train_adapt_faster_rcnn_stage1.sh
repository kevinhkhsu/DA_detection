#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
ADAPT_MODE=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${ADAPT_MODE} in
  K2C)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_KITTI_pretrained.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="KITTI_train+KITTI_val"
    TRAIN_IMDB_T="KITTI_synthCity"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2F)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_city_pretrained_8class.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="cityscapes_train"
    TRAIN_IMDB_T="cityscapes_synthFoggytrain"
    TEST_IMDB="cityscapes_foggyval"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2BDD)
    PRETRAINED_WEIGHT="${NET}_faster_rcnn_city_pretrained_10class.pth"
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage1"
    TRAIN_IMDB_S="cityscapes_train+cityscapes_val"
    TRAIN_IMDB_T="cityscapes_synthBDDdaytrain+cityscapes_synthBDDdayval"
    TEST_IMDB="bdd100k_dayval"
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

LOG="experiments/logs/${NET}_${TRAIN_IMDB_S}2${TRAIN_IMDB_T}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=output/${NET}/${TRAIN_IMDB_S}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.pth
else
  NET_FINAL=output/${NET}/${TRAIN_IMDB_S}/adapt/${NET}_faster_rcnn_iter_${ITERS}.pth
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net_adapt.py \
      --weight trained_weights/prerained_detector/${PRETRAINED_WEIGHT} \
      --imdb ${TRAIN_IMDB_S} \
      --imdbval ${TEST_IMDB} \
      --imdb_T ${TRAIN_IMDB_T} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
      --tag ${EXTRA_ARGS_SLUG}_adapt \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT_PREFIX} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net_adapt.py \
      --weight trained_weights/pretrained_detector/${PRETRAINED_WEIGHT} \
      --imdb ${TRAIN_IMDB_S} \
      --imdbval ${TEST_IMDB} \
      --imdb_T ${TRAIN_IMDB_T} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
      --tag ${EXTRA_ARGS_SLUG}_adapt \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT_PREFIX} ${EXTRA_ARGS}
  fi
fi

./experiments/scripts/test_adapt_faster_rcnn_stage1.sh $@ ${ITERS}
