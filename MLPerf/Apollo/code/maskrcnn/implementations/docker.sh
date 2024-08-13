#!/bin/bash


docker run --gpus=all -it --rm --ipc=host \
        --cap-add SYS_ADMIN --cap-add SYS_TIME \
        -e NVIDIA_VISIBLE_DEVICES= \
        --shm-size=32gb \
        -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
        --security-opt apparmor=unconfined --security-opt seccomp=unconfined \
        --net host --device /dev/fuse \
	-v /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017:/workspace/dataset \
	-v /nfs/bruno/APPLICATIONS/MLCOMMONS/training2.0/HPE/benchmarks/maskrcnn//implementations/pytorch:/workspace/implementations \
        -e HOST_HOSTNAME="`hostname`" \
	nvcr.io/nvdlfwea/mlperfv20/maskrcnn:20220509.pytorch
