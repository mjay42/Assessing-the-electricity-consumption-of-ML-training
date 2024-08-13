#!/bin/bash

export DATADIR=/cstor/SHARED/datasets/MLPERF/training2.0/dlrm

docker run --gpus=all -it --rm --ipc=host \
        --cap-add SYS_ADMIN --cap-add SYS_TIME \
        -e NVIDIA_VISIBLE_DEVICES= \
        --shm-size=32gb \
        -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
        --security-opt apparmor=unconfined --security-opt seccomp=unconfined \
        --net host --device /dev/fuse \
	-v ${DATADIR}:/raid/datasets/criteo/mlperf/40m.limit_preshuffled \
	-v /cstor/SHARED/datasets/MLPERF/training2.0/dlrm:/workspace/dlrm_data \
	-v /nfs/bruno/APPLICATIONS/MLCOMMONS/training2.0/HPE/benchmarks/dlrm/implementations/hugectr:/workspace/implementations \
        -e HOST_HOSTNAME="`hostname`" \
	nvcr.io/nvdlfwea/mlperfv20/dlrm:20220509.hugectr

# 
	#-v $PWD/hugectr:/workspace/dlrm \
