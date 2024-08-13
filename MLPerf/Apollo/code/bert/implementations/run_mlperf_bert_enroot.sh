#!/bin/bash

#SBATCH --job-name=MLPerf21-bert
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=4
#SBATCH --cpus-per-task=8
##
#SBATCH --partition=champollion
#SBATCH --time=04:00:00
#SBATCH --exclusive

NBNODE=${1:-1}
echo "Running BERT on ${NBNODE} nodes"

export curDir=$PWD

# Copy dataset on local NVME
#mkdir -p /lvol/preprocess/
#rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/resnet/preprocess/ /lvol/preprocess/

#export CONT=/cstor/SHARED/containers/enroot/bert.sqsh
export CONT=/cstor/SHARED/containers/enroot/mlperfv21.bert.sqsh


export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

export SCRIPTDIR=${curDir}

# 4320 SHARDS
export DATADIR="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export DATADIR_PHASE2="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export EVALDIR="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/eval_varlength" #<path/to/eval_varlength/dir> 
#export UNITTESTDIR=/lvol/unittest
#export UNITTESTDIR=/cstor/bruno/MLPERF/
#export CHECKPOINTDIR="/lvol/chkpt-${SLURM_JOB_ID}" #<path/to/result/checkpointdir> 
#mkdir -p ${CHECKPOINTDIR}
export CHECKPOINTDIR_PHASE1="/cstor/SHARED/datasets/MLPERF/training2.0/bert/phase1/" #<path/to/pytorch/ckpt/dir> 
export NEXP=1 # 0

#cd pytorch
source ${curDir}/pytorch/config_675D_enroot_x${NBNODE}.sh
sbatch --comment="turbo ; sysctl file=${PWD}/systcl-bert " --export=ALL  -N ${NBNODE} ${curDir}/pytorch/run.hpe.sub
