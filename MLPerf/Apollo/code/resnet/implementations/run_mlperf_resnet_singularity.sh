#!/bin/bash

export curDir=$PWD

# Copy dataset on local NVME
mkdir -p /lvol/preprocess/
rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/resnet/preprocess/ /lvol/preprocess/

export CONT=/apps/gpu/docker/mlperfv20.resnet.sif

source ${curDir}/mxnet/config_675D.sh

export DATADIR=/lvol/preprocess 
export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

cd mxnet
./run_with_singularity.sh
