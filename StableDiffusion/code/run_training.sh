#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_training.sh <result path> <random seed 1-5>
# run_training.sh /home/${USERNAME}/laion/pokemon/tests/

# oarsub -t exotic -p gemini-1 -l walltime=14:00 "2023-09-04 19:00:00" -I 

RESULT_DIR=$1 #"/home/${USERNAME}/laion/pokemon/results_31_07/"
SEED=${2:--1}

USERNAME="mjay"
DIR="/home/${USERNAME}/ai-energy-consumption-framework/"

TRAIN_FILE="${DIR}/stable-diffusion/code/train_text_to_image.py"
# TRAIN_FILE="${DIR}/stable-diffusion/train_text_to_image_2.py"
#CACHE_DIR="${DIR}/stable-diffusion/dataset/laion400M/"
CACHE_DIR="/home/${USERNAME}/laion/pokemon/cache"
DATASET_DIR="lambdalabs/pokemon-blip-captions"
# DATASET_DIR="laion/laion2B-en"
# DATASET_DIR="/home/${USERNAME}/laion/img2dataset/test/" #laion2B-en-webdataset/"
MODEL_NAME="CompVis/stable-diffusion-v1-1"
LR_SCHEDULER="constant"
DATASET_SIZE=600
# SEED=43
MAX_EPOCH=5
TRAIN_BATCH_SIZE=4
GRADIENT_ACC=2
IMG_SIZE=256

TEMPLATE="${DIR}/utils/script_template.sh"
SENSOR_DIR="/home/${USERNAME}/sensors/nvml-sensor/"
SLEEP_BEFORE=90
SLEEP_AFTER=90
SAMPLING_PERIOD=0.5

sudo-g5k
module load conda
conda activate stable_diffusion
cp /home/${USERNAME}/ai-energy-consumption-framework/stable-diffusion/config/default_config.yaml /home/${USERNAME}/.cache/huggingface/accelerate
cp /home/${USERNAME}/ai-energy-consumption-framework/stable-diffusion/config/token /home/${USERNAME}/.cache/huggingface/

# for MODEL_NAME in "CompVis/stable-diffusion-v1-1" "CompVis/stable-diffusion-v1-2" "CompVis/stable-diffusion-v1-3" "CompVis/stable-diffusion-v1-4" "runwayml/stable-diffusion-v1-5" "stabilityai/stable-diffusion-2-1"
for rep in 1 2 3
do
  for SEED in 42 3 67
  do
    for IMG_SIZE in 256 512 
    do 
      for DATASET_SIZE in 400
        do
        for MAX_EPOCH in 100
        do
          declare -i y=0
          folder="$RANDOM"
          while [ -d "${RESULT_DIR}/${folder}_$y/" ]
          do 
              y=${y}+1
          done
          LOG_DIR="${RESULT_DIR}/${folder}_$y/"
          mkdir ${LOG_DIR}
          chmod 777 "${LOG_DIR}"
          echo "Log dir defined and create at ${LOG_DIR}" &>> "${LOG_DIR}/stdout.txt"

            # --train_data_dir=${CACHE_DIR} \
            # --max_train_steps=${MAX_STEP} \
            # --cache_dir=${CACHE_DIR} \
            # --mixed_precision="fp16" \

          COMMAND="accelerate launch ${TRAIN_FILE} \
            --pretrained_model_name_or_path=${MODEL_NAME} \
            --dataset_name=${DATASET_DIR} \
            --use_ema \
            --resolution=${IMG_SIZE} --center_crop --random_flip \
            --train_batch_size=${TRAIN_BATCH_SIZE} \
            --gradient_accumulation_steps=${GRADIENT_ACC} \
            --gradient_checkpointing \
            --num_train_epochs=${MAX_EPOCH}\
            --learning_rate=1e-05 \
            --max_grad_norm=1 \
            --lr_scheduler="constant" --lr_warmup_steps=0 \
            --seed=${SEED} \
            --dataloader_num_workers=2 \
            --output_dir=${LOG_DIR} \
            --max_train_samples=${DATASET_SIZE} \
            --validation_prompts="yoda" \
            --validation_epochs 100 \
            --use_8bit_adam=False
            "

          start_fmt=$(date +%Y-%m-%d\ %r)
          echo "STARTING EXPERIMENT AT $start_fmt" &>> "${LOG_DIR}/stdout.txt"
          python ${DIR}/start_exp.py \
              --template_script "${TEMPLATE}" \
              --result_folder "${RESULT_DIR}" \
              --log_dir "${LOG_DIR}" \
              --tool_folder "${SENSOR_DIR}" \
              --sampling_period ${SAMPLING_PERIOD} \
              --sleep_before ${SLEEP_BEFORE} \
              --sleep_after ${SLEEP_AFTER} \
              --benchmark_execution_command "${COMMAND}" &>> "${LOG_DIR}/stdout.txt"

          # end timing
          end_fmt=$(date +%Y-%m-%d\ %r)
          echo "ENDING EXPERIMENT AT $end_fmt" &>> "${LOG_DIR}/stdout.txt"

          if [ -d "${LOG_DIR}/logs/" ];
          then
            echo "Getting data from tensorboard..." &>> "${LOG_DIR}/stdout.txt"
            python ${DIR}/get_data_from_tensorboard.py \
              --tb_log_dir="${LOG_DIR}/logs/"
            echo "... Done." &>> "${LOG_DIR}/stdout.txt"
          fi
        done
      done
    done
  done
done