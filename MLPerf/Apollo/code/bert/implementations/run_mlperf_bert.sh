#!/bin/bash


#SBATCH --job-name=MLPerf-bert-docker
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=4
#SBATCH --cpus-per-task=8
##
#SBATCH --partition=champollion
#SBATCH --time=01:00:00
#SBATCH --exclusive

export curDir=$PWD

# Check if the correct image is loaded
if [ "$(docker images | grep mlperfv20 | grep bert)" ]; then
	echo "Container ready"
else
	echo "Loading Image classification container"
	docker load  < /apps/gpu/docker/mlperfv20.bert.bz2
	# Wait a little
	sleep 30
fi


# Copy dataset on local NVME
#mkdir -p /lvol/preprocess/
#rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/resnet/preprocess/ /lvol/preprocess/

export CONT=nvcr.io/nvdlfwea/mlperfv20/bert:20220509.pytorch


export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}


#export DATADIR=<path/to/4320_shards_varlength/dir> 
#export DATADIR_PHASE2=<path/to/4320_shards_varlength/dir> 
#export EVALDIR=<path/to/eval_varlength/dir> 
#export CHECKPOINTDIR=<path/to/result/checkpointdir> 
#export CHECKPOINTDIR_PHASE1=<path/to/pytorch/ckpt/dir> 
export SCRIPTDIR=${curDir}

# 2048 SHARDS
#export DATADIR="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-2048/hdf5_2048_shards_uncompressed" #<path/to/4320_shards_varlength/dir> 
#export DATADIR_PHASE2="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-2048/hdf5_2048_shards_uncompressed" #<path/to/4320_shards_varlength/dir> 

# Manju's 
#export DATADIR="/cstor/manju/datasets/mlperf-training-v1.1/bert_new/hdf5_2048_shards_varlength" #<path/to/4320_shards_varlength/dir> 
#export DATADIR_PHASE2="/cstor/manju/datasets/mlperf-training-v1.1/bert_new/hdf5_2048_shards_varlength" #<path/to/4320_shards_varlength/dir> 
#export EVALDIR="/cstor/manju/datasets/mlperf-training-v1.1/bert_new/eval_varlength" #<path/to/eval_varlength/dir> 


# 4320 SHARDS
export DATADIR="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export DATADIR_PHASE2="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export EVALDIR="/cstor/SHARED/datasets/MLPERF/training2.0/bert/hdf5/eval_varlength" #<path/to/eval_varlength/dir> 

export CHECKPOINTDIR="/lvol/chkpt-${SLURM_JOB_ID}" #<path/to/result/checkpointdir> 
mkdir -p ${CHECKPOINTDIR}
export CHECKPOINTDIR_PHASE1="/cstor/SHARED/datasets/MLPERF/training2.0/bert/phase1/" #<path/to/pytorch/ckpt/dir> 
export NEXP=10

cd pytorch
source ${curDir}/pytorch/config_675D_1x8x56x1.sh
./run_with_docker_HPE.sh
