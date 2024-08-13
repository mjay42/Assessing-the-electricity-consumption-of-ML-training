#!/bin/bash

NBNODE=${1:-1}
echo "Running Unet3D on ${NBNODE} nodes"

export curDir=$PWD

export TGT_DIR=/pfss/hddfs1/bruno/MLCOMMONS/training2.1/

export CONT=/pfss/hddfs1/bruno/MLCOMMONS/containers/enroot/mlperfv20.unet3d.sqsh


export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

export SCRIPTDIR=${curDir}

# 4320 SHARDS
export DATADIR="${TGT_DIR}/kits19"
export NEXP=1 # 0

#cd pytorch
source ${curDir}/mxnet/config_675D_enroot_x${NBNODE}.sh
export MELLANOX_VISIBLE_DEVICES=all
sbatch --comment="turbo ; sysctl file=${PWD}/systcl-bert " --export=ALL  -N ${NBNODE} ${curDir}/mxnet/run.hpe.sub
