#!/bin/bash

NBNODE=${1:-1}
echo "Running BERT on ${NBNODE} nodes"

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

echo "Running Bert on ${NBNODE} nodes on FS=$TGT_DIR"

export FS=$FS



export CONT=/cstor/SHARED/containers/enroot/mlperfv21.bert.sqsh


export SCRIPTDIR=${curDir}

# 4320 SHARDS
export DATADIR="${TGT_DIR}/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export DATADIR_PHASE2="${TGT_DIR}/bert/hdf5/training-4320/hdf5_4320_shards_varlength" #<path/to/4320_shards_varlength/dir> 
export EVALDIR="${TGT_DIR}/bert/hdf5/eval_varlength" #<path/to/eval_varlength/dir> 
export CHECKPOINTDIR_PHASE1="${TGT_DIR}/bert/phase1/" #<path/to/pytorch/ckpt/dir> 
export NEXP=1 # 0

#cd pytorch
source ${curDir}/pytorch/config_675D_enroot_x${NBNODE}.sh
export MELLANOX_VISIBLE_DEVICES=all
export ranks=$(( 8*NBNODE ))

export MAX_SAMPLES_TERMINATION=14000000
export TARGET_MLM_ACCURACY=0.72

for rep in 1 2
do
	for seed in 51 6 # 42 # 7 9998 
	do
        	export SEED=$seed
                for grad in 1
                do
                        # export BATCHSIZE=$batch
                        #  --begin="19:00 11/06/23" 
                        export GRADIENT_STEPS=$grad
                        sbatch ${SCHEDULE_RUN} --comment="turbo ; disable cpus = ht ; gpu freq=1593,1410 " --export=ALL  -N ${NBNODE} -n ${ranks} ${curDir}/pytorch/run.hpe.sub
                done
        done
done

# job_id=$(/apps/slurm/bin/sbatch ${SCHEDULE_RUN} --parsable --comment="turbo ; gpu freq=1593,1410 " --export=ALL  -N ${NBNODE} -n ${ranks} ${curDir}/pytorch/run.hpe.sub)

# # Loop until the job is running
# while true; do
#     # Get the job state using scontrol and extract the job status
#     job_status=$(scontrol show job $job_id | grep "JobState=" | awk -F= '{print $2}')
    
#     # Check if the job status is "RUNNING"
#     if [[ $job_status == "RUNNING Reason" ]]; then
#         break  # Exit the loop if the job is running
#     fi
    
#     sleep 1  # Wait for 1 second before checking the status again
# done

# echo "Job $job_id is now running!"
