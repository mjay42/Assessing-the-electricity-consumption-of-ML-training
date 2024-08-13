#!/bin/bash

NBNODE=${1:-1}

export curDir=$PWD

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
		export TGT_DIR="/pfss/nvmefs1/bruno/MLCOMMONS/training2.1/mask-rcnn"
                SBATCH_FS=''
                ;;
	cstor)
		export TGT_DIR="/cstor/SHARED/datasets/MLPERF/training2.1/mask-rcnn"
                SBATCH_FS=''
                ;;
esac

echo "Running MaskRCNN on ${NBNODE} nodes on FS=$TGT_DIR"

export FS=$FS

export CONT=/cstor/SHARED/containers/enroot/mlperfv21.maskrcnn.sqsh


export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

export SCRIPTDIR=${curDir}


#export DATADIR=/cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017 # /cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn/coco2017/
export DATADIR=/cstor/SHARED/datasets/MLPERF/training2.0/mask-rcnn
export PKLDIR=$DATADIR/coco2017/pkl_coco
#export DATADIR="${TGT_DIR}" 
#export PKLDIR="${TGT_DIR}/coco2017/pkl_coco"

export NEXP=1 # 0

#cd pytorch
export CLEAR_CACHES=0
source ${curDir}/pytorch/config_675D_enroot_x${NBNODE}.sh
export MELLANOX_VISIBLE_DEVICES=all
export NBGPU=$(( 8 * NBNODE ))
export WORLD_SIZE=${NBGPU}

export GPUS_PER_NODE=8
export MASTER_ADDR=${SLURMD_NODENAME}
export MASTER_PORT=9901

for rep in 1 2 3
do
        for seed in 51 4 7 42
        do
                sbatch ${SBATCH_OPTION} --comment="turbo ; disable cpus = ht ; gpu freq=1593,1410 ${SBATCH_FS}" --export=ALL  -N ${NBNODE} -n ${NBGPU}  ${curDir}/pytorch/run.hpe.sub
        done
done

