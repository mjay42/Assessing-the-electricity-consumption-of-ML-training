#!/bin/bash

export curDir=$PWD

# Check if the correct image is loaded
if [ "$(docker images | grep mlperfv20 | grep maskrcnn)" ]; then
	echo "Container ready"
else
	echo "Loading Image classification container"
	docker load  < /apps/gpu/docker/mlperfv20.maskrcnn.bz2
	# Wait a little
	sleep 30
fi


# Copy dataset on local NVME
mkdir -p /lvol/preprocess/
#rsync -aviP  /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017 /lvol/preprocess/
cp -ar /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017.bz2  /lvol/preprocess/
cd /lvol/preprocess/
tar xf ./coco2017.bz2
cd $curDir

export CONT=nvcr.io/nvdlfwea/mlperfv20/maskrcnn:20220509.pytorch

source ${curDir}/pytorch/config_675D.sh

#export DATADIR=/cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn # /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017/
export DATADIR=/lvol/preprocess # /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017/
export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}


cd pytorch
./run_with_docker_HPE.sh
