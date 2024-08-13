#!/bin/bash

export curDir=$PWD

# Check if the correct image is loaded

if [ "$(docker images | grep mlperfv20 | grep rnnt)" ]; then
	echo "Container ready"
else
	echo "Loading Image classification container"
	docker load  < /apps/gpu/docker/mlperfv20.rnnt.bz2
	# Wait a little
	sleep 30
fi


export CONT=nvcr.io/nvdlfwea/mlperfv20/rnnt:20220509.pytorch

#source ${curDir}/pytorch/config_675D.sh
source ./pytorch/config_675D.sh

export DATADIR="/cstor/SHARED/datasets/MLPERF/training2.0/librispeech"
export CHECKPOINT_DIR=""
export RESULTS_DIR=""
export METADATA_DIR="/lvol/metadata"
export SENTENCEPIECES_DIR="/lvol/sentencepieces"
export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}
mkdir -p ${METADATA_DIR}
mkdir -p ${SENTENCEPIECES_DIR}

cd pytorch
./run_with_docker_HPE.sh
