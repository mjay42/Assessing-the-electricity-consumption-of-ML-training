#!/bin/bash
#Before running this script, make sure that the docker container is not running on interactive mode (-it) and that the machine runs on its maxmimum setting
# sudo ./run.sh

TRAIN_REPO="/home/mjay/jetson-inference"
TRAINING_DIR="${TRAIN_REPO}/python/training/classification"
REPO="/home/mjay/ai-energy-consumption/Jetson"
TEGRA_SCRIPT="${REPO}/utils/jetson_monitoring.py"
JETSON_DATA="/root/energyfl/imagenet-1k" 
# JETSON_DATA="/root/energyfl/cat_dog"
BASH_LOG="run.log"

SLEEP_TIME=60
BATCH_SIZE=64
SEED=42

#TODO Set fan to its highest setting

for i in {1..1}
do
	echo Batch size: $BATCH_SIZE - Experiment: $i >> ${LOG_DIR}/$BASH_LOG
	current_time=$(date +%Y%m%d_%H%M%S)
	echo $current_time >> ${LOG_DIR}/$BASH_LOG
	LOG_DIR="${REPO}/logs/${current_time}/"
	mkdir -p ${LOG_DIR}

	USER_COMMAND="python3 ${REPO}/train.py \
		$JETSON_DATA\
		-a=resnet50\
		--seed=${SEED}\
		-b=${BATCH_SIZE}\
		--epochs=2 -p=1000\
		--model-dir=${LOG_DIR}\
		--log-dir=${LOG_DIR}\
		" 
		
	echo Freeing cache  >> ${LOG_DIR}/$BASH_LOG
	sync && echo 3 | tee /proc/sys/vm/drop_caches
	python3 ${TEGRA_SCRIPT} --log-dir=${LOG_DIR} &
	TEGRA_PID=$!
	echo tegra-stats is running with PID $TEGRA_PID >> ${LOG_DIR}/$BASH_LOG

	sleep ${SLEEP_TIME}

	cd ${TRAIN_REPO}
	echo Training container spinning up >> ${LOG_DIR}/$BASH_LOG

	${USER_COMMAND} >> ${LOG_DIR}/$BASH_LOG
	
	echo Training ended >> ${LOG_DIR}/$BASH_LOG
	#cp -R ${TRAINING_DIR}/models/imagenet/tensorboard/${current_time}/ $LOG_DIR

	#echo Copied tensorboard files to logs
	
	echo Sleeping for cooling >> ${LOG_DIR}/$BASH_LOG
	sleep ${SLEEP_TIME}

	kill $TEGRA_PID

	echo Killed tegra-stats script >> ${LOG_DIR}/$BASH_LOG
done