#!/bin/bash

NBNODE=${1:-1}

export curDir=$PWD

FS=${2:-pfss}


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
		export TGT_DIR=/cstor/SHARED/datasets/MLPERF/training2.1/
                SBATCH_FS=''
                ;;
esac

echo "Running Resnet50 on ${NBNODE} nodes on FS=$TGT_DIR"

export FS=$FS

export CONT=/cstor/SHARED/containers/enroot/mlperfv20.resnet.sqsh


export LOGDIR=${curDir}/logs/${SLURM_JOB_ID} #/lvol/logs/shm2
mkdir -p ${LOGDIR}

export SCRIPTDIR=${curDir}

# 4320 SHARDS
export DATADIR="${TGT_DIR}/resnet/preprocess" 
export NEXP=1 # 0

#cd pytorch
source ${curDir}/mxnet/config_675D_enroot_x${NBNODE}.sh
export MELLANOX_VISIBLE_DEVICES=all
export ranks=$(( 8*NBNODE ))


# for seed in 51  #42 6 9998 7
# do
#         export SEED=$seed
#         for acc in "0.3"
#         do
#                 export TARGET_ACCURACY=$acc
#                 for size in 256000
#                 do  
#                         export NUMEXAMPLES=$size
#                         sbatch ${SCHEDULE_RUN} --comment="turbo ;  gpu freq=1593,1410 ${SBATCH_FS}" --export=ALL  -N ${NBNODE} -n ${ranks} ${curDir}/mxnet/run.hpe.sub #--begin="07:00 07/15/23" 
#                 done
#         done
# done   

for rep in 1 
do
	for seed in 51 # 6 42 # 7 9998  
	do
        	export SEED=$seed
                sbatch ${SCHEDULE_RUN} --comment="turbo ; disable cpus = ht ; gpu freq=1593,1410 " --export=ALL  -N ${NBNODE} -n ${ranks} ${curDir}/mxnet/run.hpe.sub
                
        done
done