#!/bin/bash

NBNODE=${1:-1}

FS=${2:-cstor}

SBATCH_FS=''
case $FS in
        beeond )
                SBATCH_FS=' ; beeond ; '
                export TGT_DIR=$FS
                ;;
        lvol )
                SBATCH_FS=''
                export TGT_DIR=$FS
                ;;
        nfsond )
                SBATCH_FS=''
                export TGT_DIR=$FS
                SBATCH_FS=' ; nfsond=ib-rdma ; '
                ;;
        daos )
                SBATCH_FS=' ; daos ; '
                export TGT_DIR=$FS
                ;;
        pfss)
                export TGT_DIR=/pfss/nvmefs1/bruno/MLCOMMONS/training2.1/
                SBATCH_FS=''
                ;;
	cstor)
                export TGT_DIR=/cstor/SHARED/datasets/MLPERF/training2.1/
                export TGT_DIR=$FS
                ;;
esac

export FS=$FS

echo "Running RNNT on ${NBNODE} nodes, FS = $FS"

export curDir=$PWD

export CONT=/cstor/SHARED/containers/enroot/mlperfv20.rnnt.sqsh


export SCRIPTDIR=${curDir}

# 4320 SHARDS
export DATADIR="${TGT_DIR}/LibriSpeech"
export METADATA_DIR="${TGT_DIR}/LibriSpeech/tokenized"
export SENTENCEPIECES_DIR="${TGT_DIR}/LibriSpeech/sentencepieces"
export CHECKPOINT_DIR=""
export RESULTS_DIR=""
export NEXP=1 # 0
export ranks=$(( 8*NBNODE ))
#cd pytorch
source ${curDir}/pytorch/config_675D_enroot_x${NBNODE}.sh
export MELLANOX_VISIBLE_DEVICES=all

for seed in 51 6 42 7
do
        export SEED=$seed
        for rep in 1 2 3 4
        do
                sbatch ${SBATCH_OPTION} --comment="turbo ; disable cpus = ht ; gpu freq=1593,1410  ${SBATCH_FS} " --export=ALL  -N ${NBNODE} -n ${ranks} ${curDir}/pytorch/run.hpe.sub #--begin="14:17 07/11/23"
        done
done