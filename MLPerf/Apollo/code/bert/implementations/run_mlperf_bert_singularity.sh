#!/bin/bash

export curDir=$PWD


# Copy dataset on local NVME
#mkdir -p /lvol/preprocess/
#rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/resnet/preprocess/ /lvol/preprocess/

export CONT=/apps/gpu/docker/mlperfv20.bert.sif


export RESULTSDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}


export SCRIPTDIR=${curDir}
export DATADIR="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-2048/hdf5_2048_shards_uncompressed" #<path/to/4320_shards_varlength/dir> 
export DATADIR_PHASE2="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-2048/hdf5_2048_shards_uncompressed" #<path/to/4320_shards_varlength/dir> 
export EVALDIR="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/eval_varlength" #<path/to/eval_varlength/dir> 
export CHECKPOINTDIR="${curDir}/results/chkpt-${SLURM_JOB_ID}" #<path/to/result/checkpointdir> 
mkdir -p ${CHECKPOINTDIR}
export CHECKPOINTDIR_PHASE1="/cstor/SHARED/datasets/MLPERF/training2.0/bert/phase1/" #<path/to/pytorch/ckpt/dir> 
export UNITTESTDIR=/lvol/unittest
mkdir -p ${UNITTESTDIR}


cd pytorch
source ${curDir}/pytorch/config_675D_singularity_1x8x56x1.sh
./run_with_singularity.sh
