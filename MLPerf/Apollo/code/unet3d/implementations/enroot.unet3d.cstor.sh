#!/bin/bash

NBNODE=${1:-1}
echo "Running Unet3D on ${NBNODE} nodes"

export curDir=$PWD

export TGT_DIR=/cstor/SHARED/datasets/MLPERF/training2.1

export CONT=/cstor/SHARED/containers/enroot/mlperfv20.unet3d.sqsh


export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

export SCRIPTDIR=${curDir}

# 4320 SHARDS
export DATADIR="${TGT_DIR}/kits19/results"
export NEXP=1 # 0
export CLEAR_CACHES=0
export ranks=$(( 8*NBNODE ))

#cd pytorch
source ${curDir}/mxnet/config_675D_enroot_x${NBNODE}.sh
export MELLANOX_VISIBLE_DEVICES=all
# sbatch -p champollion  --export=ALL  -N ${NBNODE} ${curDir}/mxnet/run.hpe.sub

for rep in 1 2 3
do
	for seed in 51 6 42 7 
	do
        export SEED=$seed
        sbatch --begin="19:00 11/07/23" --comment="turbo ; disable cpus = ht ; gpu freq=1593,1410 " --export=ALL  -N ${NBNODE} -n ${ranks} ${curDir}/mxnet/run.hpe.sub
    done
done