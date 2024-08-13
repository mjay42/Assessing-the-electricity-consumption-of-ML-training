#!/bin/bash

export curDir=$PWD

# Check if the correct image is loaded
if [ "$(docker images | grep mlperfv20 | grep bert)" ]; then
	echo "Container ready"
else
	echo "Loading Image classification container"
	docker load  < /apps/gpu/docker/mlperfv20.bert.bz2
	# Wait a little
	sleep 10
fi


# Copy dataset on local NVME
mkdir -p /lvol/preprocess/
mkdir -p /lvol/bert/
mkdir -p /lvol/preprocess/eval_varlength/
mkdir -p /lvol/preprocess/phase1/
mkdir -p /lvol/preprocess/training-2048/hdf5_2048_shards_uncompressed/

rsync -aviP $curDir/ /lvol/bert/
cd /lvol/bert/

cp -ar /cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/eval_varlength/ /lvol/preprocess/ &
cp -ar /cstor/SHARED/datasets/MLPERF/training2.0/bert/phase1/ /lvol/preprocess/ &
cp -ar /cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-2048/hdf5_2048_shards_uncompressed/ /lvol/preprocess/training-2048/ 


export CONT=nvcr.io/nvdlfwea/mlperfv20/bert:20220509.pytorch

export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

export SCRIPTDIR=${curDir}
export DATADIR="/lvol/preprocess/training-2048/hdf5_2048_shards_uncompressed" #<path/to/4320_shards_varlength/dir> 
export DATADIR_PHASE2="/lvol/preprocess/training-2048/hdf5_2048_shards_uncompressed" #<path/to/4320_shards_varlength/dir> 
export EVALDIR="/lvol/preprocess/eval_varlength" #<path/to/eval_varlength/dir> 
export CHECKPOINTDIR="/lvol/results/chkpt-${SLURM_JOB_ID}" #<path/to/result/checkpointdir> 
mkdir -p ${CHECKPOINTDIR}
export CHECKPOINTDIR_PHASE1="/lvol/preprocess/phase1" #<path/to/pytorch/ckpt/dir> 


cd pytorch
source ${curDir}/pytorch/config_675D_1x8x56x1.sh
./run_with_docker_HPE.sh

