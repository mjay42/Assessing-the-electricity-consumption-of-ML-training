#!/bin/bash

export curDir=$PWD

# Check if the correct image is loaded
if [ "$(docker images | grep dlrm_reference )" ]; then
        echo "Container ready"
else
        echo "Loading Image classification container"
        docker load  < /apps/gpu/docker/dlrm_reference.latest.bz2
        # Wait a little
        sleep 10
fi


export CONT=dlrm_reference:latest
export DATA="/cstor/SHARED/datasets/MLPERF/training2.0/dlrm"

#docker run -it --rm --network=host --ipc=host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all  -v ${DATA}:/data  ${CONT} 


docker run  -i --rm --network=host --ipc=host --shm-size=1g --ulimit memlock=-1 \
 --ulimit stack=67108864 --gpus=all -v $PWD:/work -v ${DATA}:/data ${CONT} /work/convert_criteo.sh 

