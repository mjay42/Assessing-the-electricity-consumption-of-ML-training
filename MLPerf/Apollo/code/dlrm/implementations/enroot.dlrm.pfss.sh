#!/bin/bash

NBNODE=${1:-1}
echo "Running DLRM on ${NBNODE} nodes"

#export TGT_DIR=/pfss/hddfs1/bruno/MLCOMMONS/training2.1
#export TGT_DIR=/pfss/nvmefs1/bruno/MLCOMMONS/training2.1
export TGT_DIR=/cstor/SHARED/datasets/MLPERF/training2.1

export curDir=$PWD

export CONT=/cstor/SHARED/containers/enroot/mlperfv20.dlrm.sqsh

#export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} 
#mkdir -p ${LOGDIR}

export SCRIPTDIR=${curDir}
export DATADIR="${TGT_DIR}/dlrm"
export SCRIPTS="${curDir}/hugectr"
export NEXP=1

export _config_file=${curDir}/hugectr/config_675D_x${NBNODE}.sh
source ${_config_file}
export MELLANOX_VISIBLE_DEVICES=all
#export MELLANOX_VISIBLE_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_4
#export MELLANOX_VISIBLE_DEVICES=0,1,2,4
export CLEAR_CACHES=0
export ranks=$(( 8*NBNODE ))

for rep in 1 2 3 # 4 
do
    for seed in 30 4 55  #26 888
    do
        export SEED=$seed
        sbatch ${SBATCH_OPTION} --comment="turbo ; disable cpus = ht ; gpu freq=1593,1410 " --export=ALL  -N ${NBNODE} -n ${ranks} ${curDir}/hugectr/run.hpe.sub
    done
done

