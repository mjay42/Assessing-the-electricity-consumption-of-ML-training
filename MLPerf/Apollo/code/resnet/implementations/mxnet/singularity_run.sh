#!/bin/bash

export CMD="$@" # --env OMPI_COMM_WORLD_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK} --env LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}"
#echo "DUMP -------------->"
#for v in ${CMD[@]}; 
#do
#	if [[ $v == *"LOCAL_RANK"* ]]; then
#		$v="${v/=0/=$OMPI_COMM_WORLD_LOCAL_RANK}"
#		#export $v=`sed "s/=0/=${OMPI_COMM_WORLD_LOCAL_RANK}/g)" <<< $v`
#		echo $v
#	fi
#done
#echo "DUMP <--------------"

#mapfile -t _config_env < <(for v in "${CMD[@]}"; if [ "${v}" == "LOCAL_RANK" ]; then
#do echo "$v"; done)

export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}
export OMPI_COMM_WORLD_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK}

#echo "Running : ${CMD} $PWD/run_and_time.sh for local rank : ${LOCAL_RANK} local slurm ID : ${SLURM_LOCALID} openmpi local rank : ${OMPI_COMM_WORLD_LOCAL_RANK}  on host `hostname`" 
${CMD} ./run_and_time.sh
