#!/bin/bash


if [ "$(docker images | grep mlperfv20 | grep rnnt)" ]; then
        echo "Container ready"
else
        echo "Loading Image classification container"
        docker load  < /apps/gpu/docker/mlperfv20.rnnt.bz2
        # Wait a little
        sleep 5
fi

export DATADIR="/cstor/SHARED/datasets/MLPERF/training2.0/librispeech"
export CHECKPOINT_DIR="/lvol/checkpoint"
export RESULTS_DIR=""
export METADATA_DIR="/lvol/metadata"
export SENTENCEPIECES_DIR="/lvol/sentencepieces"
export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}
mkdir -p ${METADATA_DIR}
mkdir -p ${SENTENCEPIECES_DIR}
mkdir -p ${CHECKPOINT_DIR}

docker run --gpus=all -i --rm --ipc=host \
        --cap-add SYS_ADMIN --cap-add SYS_TIME \
        -e NVIDIA_VISIBLE_DEVICES= \
        --shm-size=32gb \
        -v /etc/timezone:/etc/timezone:ro -v /etc/localtime:/etc/localtime:ro \
	-v $DATADIR:/datasets \
	-v $METADATA_DIR:/metadata \
	-v $SENTENCEPIECES_DIR:/sentencepieces \
        --security-opt apparmor=unconfined --security-opt seccomp=unconfined \
        --net host --device /dev/fuse \
	-v /nfs/bruno/APPLICATIONS/MLCOMMONS/training2.0/HPE/benchmarks/rnnt/implementations/pytorch:/workspace/rnnt \
	-w /workspace/rnnt \
	-v /lvol:/lvol \
	-v /cstor/SHARED/datasets/MLPERF/training2.0/librispeech:/cstor/SHARED/datasets/MLPERF/training2.0/librispeech \
        -e HOST_HOSTNAME="`hostname`" \
	nvcr.io/nvdlfwea/mlperfv20/rnnt:20220509.pytorch ./download_and_process_dataset.sh

