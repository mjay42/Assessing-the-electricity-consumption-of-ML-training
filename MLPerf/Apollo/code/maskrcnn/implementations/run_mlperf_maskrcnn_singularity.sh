#!/bin/bash

export curDir=$PWD


# Copy dataset on local NVME
#mkdir -p /lvol/preprocess/
#rsync -aviP  /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017 /lvol/preprocess/

export CONT=/apps/gpu/docker/mlperfv20.maskrcnn.sif

source ${curDir}/pytorch/config_675D.sh

export DATADIR=/cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017 # /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017/
#export DATADIR=//lvol/preprocess # /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017/
export PKLDIR=$DATADIR/pkl_coco
export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

#Create overlay for writable needs
export OVERLAY="/dev/shm/${SLURM_JOB_ID}"
mkdir -p ${OVERLAY} 


cd pytorch
./run_with_singularity.sh
