#!/bin/bash

set -x
set -e:w

export PYTHONUNBUFFERED="True"

GPU_ID=$1
ADAPT_MODE=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  K2C)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="KITTI_train+KITTI_val"
    TRAIN_IMDB_S="KITTI_synthCity"
    TRAIN_IMDB_T="cityscapes_train"
    TEST_IMDB="cityscapes_val"
    STEPSIZE="[]"
    ITERS=10000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2F)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="cityscapes_train"
    TRAIN_IMDB_S="cityscapes_synthFoggytrain"
    TRAIN_IMDB_T="cityscapes_foggytrain"
    TEST_IMDB="cityscapes_foggyval"
    STEPSIZE="[]"
    ITERS=60000
    ANCHORS="[4,8,16,32]"
    RATIOS="[0.5,1,2]"
    ;;
  C2BDD)
    SNAPSHOT_PREFIX="${NET}_faster_rcnn_${ADAPT_MODE}_stage2"
    PREV_S="cityscapes_train+cityscapes_val"
    TRAIN_IMDB_S="cityscapes_synthBDDdaytrain+cityscapes_synthBDDdayval"
    TRAIN_IMDB_T="bdd100k_daytrain"
    TEST_IMDB="bdd100k_dayval"
    STEPSIZE="[10000]"
    ITERS=30000
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
      --weight output/${NET}/${PREV_S}/_adapt/${NET}_faster_rcnn_${ADAPT_MODE}_stage1_iter_70000.pth \
      --imdb ${TRAIN_IMDB_S} \
      --imdbval ${TEST_IMDB} \
      --imdb_T ${TRAIN_IMDB_T} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
      --tag ${EXTRA_ARGS_SLUG}_adapt \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ADAPT_MODE ${ADAPT_MODE} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT_PREFIX} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net_adapt.py \
      --weight output/${NET}/${PREV_S}/_adapt/${NET}_faster_rcnn_${ADAPT_MODE}_stage1_iter_70000.pth \
      --imdb ${TRAIN_IMDB_S} \
      --imdbval ${TEST_IMDB} \
      --imdb_T ${TRAIN_IMDB_T} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}_${ADAPT_MODE}.yml \
      --tag ${EXTRA_ARGS_SLUG}_adapt \
      --net ${NET} \
      --set ANCHOR_SCALES ${ANCHORS} ANCHOR_RATIOS ${RATIOS} ADAPT_MODE ${ADAPT_MODE} \
      TRAIN.STEPSIZE ${STEPSIZE} TRAIN.SNAPSHOT_PREFIX ${SNAPSHOT_PREFIX} ${EXTRA_ARGS}
  fi
fi

./experiments/scripts/test_adapt_faster_rcnn_stage2.sh $@ ${ITERS}
