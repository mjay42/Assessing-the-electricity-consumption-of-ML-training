#!/bin/bash

export curDir=$PWD

# Copy dataset on local NVME
#mkdir -p /beeond/preprocess/
#rsync -aviP /cstor/SHARED/datasets/MLPERF/training2.0/resnet/preprocess/ /beeond/preprocess/

export CONT=/apps/gpu/docker/mlperfv20.resnet.sif

source ${curDir}/mxnet/config_675D.sh

export MPI_ROOT="/apps/gpu/openmpi/openmpi-4.1.3-gcc8.5.0-cuda11.6"
export PATH="${MPI_ROOT}/bin:$PATH"
export LD_LIBRARY_PATH="${MPI_ROOT}/lib:$LD_LIBRARY_PATH"

export OMPI_DIR=$MPI_ROOT
export SINGULARITY_OMPI_DIR=$OMPI_DIR
export SINGULARITYENV_APPEND_PATH=$OMPI_DIR/bin
export SINGULAIRTYENV_APPEND_LD_LIBRARY_PATH=$OMPI_DIR/lib

export DATADIR=/beeond/preprocess 
export DATADIR=/cstor/SHARED/datasets/MLPERF/training2.0/resnet/preprocess/
export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

cd mxnet
./run_2N_with_singularity.sh
