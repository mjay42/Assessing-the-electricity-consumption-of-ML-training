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


mkdir -p /lvol/bert/
cd /lvol/bert/
echo "untar bert 4320 SHARDS locally"
tar xf /cstor/SHARED/datasets/MLPERF/bert_4320_shards.tar
echo "untar done"


cd $curDir
export CONT=nvcr.io/nvdlfwea/mlperfv20/bert:20220509.pytorch

export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

export SCRIPTDIR=${curDir}
export DATADIR="/lvol/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export DATADIR_PHASE2="/lvol/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export EVALDIR="/lvol/bert/hdf5/eval_varlength" #<path/to/eval_varlength/dir> 
export CHECKPOINTDIR="/lvol/results/chkpt-${SLURM_JOB_ID}" #<path/to/result/checkpointdir> 
mkdir -p ${CHECKPOINTDIR}
#export CHECKPOINTDIR_PHASE1="/lvol/bert/phase1" #<path/to/pytorch/ckpt/dir> 
export CHECKPOINTDIR_PHASE1="/cstor/SHARED/datasets/MLPERF/training2.0/bert/phase1/"

cd pytorch
source ${curDir}/pytorch/config_675D_1x8x56x1.sh
./run_with_docker_HPE.sh

