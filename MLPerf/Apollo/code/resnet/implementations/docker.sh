#!/bin/bash


docker run --gpus=all -i --rm --ipc=host \
        --cap-add SYS_ADMIN --cap-add SYS_TIME \
        -e NVIDIA_VISIBLE_DEVICES= \
        --shm-size=32gb \
        -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
        --security-opt apparmor=unconfined --security-opt seccomp=unconfined \
        --net host --device /dev/fuse \
	-v /cstor/SHARED/datasets/MLPERF/training2.0/resnet:/workspace/imagenet \
	-v /nfs/bruno/APPLICATIONS/MLCOMMONS/training2.0/HPE/benchmarks/resnet/implementations/mxnet:/workspace/implementations \
        -e HOST_HOSTNAME="`hostname`" \
	nvcr.io/nvdlfwea/mlperfv20/resnet:20220509.mxnet 
	#-v $PWD/mxnet:/workspace/image_classification \
