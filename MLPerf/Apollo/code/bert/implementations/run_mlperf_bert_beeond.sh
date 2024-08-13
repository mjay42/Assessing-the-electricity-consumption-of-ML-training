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
mkdir -p /beeond/preprocess/
mkdir -p /beeond/preprocess/eval_varlength/
mkdir -p /beeond/preprocess/phase1/
mkdir -p /beeond/preprocess/training-2048/hdf5_2048_shards_uncompressed/

#rsync -avi /cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/eval_varlength/ /beeond/preprocess/eval_varlength/ &
#rsync -avi /cstor/SHARED/datasets/MLPERF/training2.0/bert/phase1/ /beeond/preprocess/phase1/ &
#rsync -avi /cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-2048/hdf5_2048_shards_uncompressed/ /beeond/preprocess/training-2048/hdf5_2048_shards_uncompressed/ 
cp -a /cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/eval_varlength/ /beeond/preprocess/eval_varlength/ &
cp -a /cstor/SHARED/datasets/MLPERF/training2.0/bert/phase1/ /beeond/preprocess/phase1/ &
cp -a /cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-2048/hdf5_2048_shards_uncompressed/ /beeond/preprocess/training-2048/hdf5_2048_shards_uncompressed/ 

export CONT=nvcr.io/nvdlfwea/mlperfv20/bert:20220509.pytorch

export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/beeond/logs/shm2
mkdir -p ${LOGDIR}

export DATADIR="/beeond/preprocess/training-2048/hdf5_2048_shards_uncompressed" #<path/to/4320_shards_varlength/dir> 
export DATADIR_PHASE2="/beeond/preprocess/training-2048/hdf5_2048_shards_uncompressed" #<path/to/4320_shards_varlength/dir> 
export EVALDIR="/beeond/preprocess/eval_varlength" #<path/to/eval_varlength/dir> 
export CHECKPOINTDIR="${curDir}/results/chkpt-${SLURM_JOB_ID}" #<path/to/result/checkpointdir> 
mkdir -p ${CHECKPOINTDIR}
export CHECKPOINTDIR_PHASE1="/beeond/preprocess/phase1" #<path/to/pytorch/ckpt/dir> 


cd pytorch
source ${curDir}/pytorch/config_675D_1x8x56x1.sh
./run_with_docker_HPE.sh
