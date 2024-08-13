#!/bin/bash
#Before running this script, make sure that the docker container is not running on interactive mode (-it) and that the machine runs on its maxmimum setting
# sudo ./run.sh

TRAIN_REPO="/home/mjay/jetson-inference"
TRAINING_DIR="${TRAIN_REPO}/python/training/classification"
DOCKER_TRAIN_REPO="/jetson-inference/python/training/classification"
REPO="/home/mjay/ai-energy-consumption/Jetson"
TEGRA_SCRIPT="${REPO}/utils/jetson_monitoring.py"
JETSON_DATA="/root/energyfl/imagenet-1k" # "/root/energyfl/cat_dog"
DOCKER_DATA="/data"
BASH_LOG="run.log"

SLEEP_TIME=0
BATCH_SIZE=52
SEED=42

#TODO Set fan to its highest setting

for i in {1..1}
do
	echo Batch size: 32 - Experiment: $i >> ${LOG_DIR}/$BASH_LOG
	current_time=$(date +%Y%m%d_%H%M%S)
	echo $current_time >> ${LOG_DIR}/$BASH_LOG
	LOG_DIR="${REPO}/logs/${current_time}/"
	DOCKER_LOG_DIR="$DOCKER_TRAIN_REPO/logs/${current_time}/"
	CHECKPOINT_DIR="$DOCKER_TRAIN_REPO/models/model_b32_03072023.pth.tar"
	mkdir -p ${LOG_DIR}

	USER_COMMAND="pip3 install --no-cache-dir --verbose jetson-stats; pip3 install --no-cache-dir --verbose datasets[vision]; python3 ${DOCKER_TRAIN_REPO}/train.py \
		$DOCKER_DATA\
		-a=resnet50\
		--seed=${SEED}\
		-b=${BATCH_SIZE}\
		--epochs=2 -p=1000\
		--model-dir=${DOCKER_LOG_DIR}\
		--log-dir=${DOCKER_LOG_DIR}\
		" 
		
	echo Freeing cache  >> ${LOG_DIR}/$BASH_LOG
	sync && echo 3 | tee /proc/sys/vm/drop_caches
	python3 ${TEGRA_SCRIPT} --log-dir=${LOG_DIR} &
	TEGRA_PID=$!
	echo tegra-stats is running with PID $TEGRA_PID >> ${LOG_DIR}/$BASH_LOG

	sleep ${SLEEP_TIME}

	cd ${TRAIN_REPO}
	echo Training container spinning up >> ${LOG_DIR}/$BASH_LOG

	bash docker/run.sh --volume $TRAINING_DIR/:/jetson-inference/python/training/classification --volume $JETSON_DATA/:$DOCKER_DATA/ --run "${USER_COMMAND}" >> ${LOG_DIR}/$BASH_LOG
	
	echo Training ended >> ${LOG_DIR}/$BASH_LOG
	#cp -R ${TRAINING_DIR}/models/imagenet/tensorboard/${current_time}/ $LOG_DIR

	#echo Copied tensorboard files to logs
	
	echo Sleeping for cooling >> ${LOG_DIR}/$BASH_LOG
	sleep ${SLEEP_TIME}

	kill $TEGRA_PID

	echo Killed tegra-stats script >> ${LOG_DIR}/$BASH_LOG
done