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
mkdir -p /lvol/preprocess/
#cp -a /cstor/SHARED/datasets/MLPERF/training2.0/dlrm/day_*.bin /lvol/preprocess/
rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/dlrm/*_data.bin /lvol/preprocess/


export CONT=nvcr.io/nvdlfwea/mlperfv20/dlrm:20220509.hugectr


export DATADIR=/lvol/preprocess 
export LOGDIR="${curDir}/logs/${SLURM_JOB_ID}" #/lvol/logs/shm2
export SCRIPTS="${curDir}/hugectr"
mkdir -p ${LOGDIR}

export NEXP=5
cd hugectr
source ./config_675D.sh
./run_with_docker_HPE.sh
