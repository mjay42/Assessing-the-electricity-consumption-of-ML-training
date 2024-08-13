#!/bin/bash

export curDir=$PWD

# Check if the correct image is loaded

if [ "$(docker images | grep mlperfv20 | grep resnet)" ]; then
	echo "Container ready"
else
	echo "Loading Image classification container"
	docker load  < /apps/gpu/docker/mlperfv20.dlrm.bz2
	# Wait a little
	sleep 10
fi


# Copy dataset on local NVME
mkdir -p /beeond/preprocess/
rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/dlrm/val_data.bin /beeond/preprocess/ &
rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/dlrm/test_data.bin /beeond/preprocess/ &
rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/dlrm/train_data.bin /beeond/preprocess/


export CONT=nvcr.io/nvdlfwea/mlperfv20/dlrm:20220509.hugectr


export DATADIR=/beeond/preprocess 
export LOGDIR="${curDir}/logs/${SLURM_JOB_ID}" #/lvol/logs/shm2
export SCRIPTS="${curDir}/hugectr"
mkdir -p ${LOGDIR}

export NEXP=5
cd hugectr
source ./config_675D.sh
./run_with_docker_HPE.sh
