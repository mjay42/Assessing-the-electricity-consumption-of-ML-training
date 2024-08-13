#!/bin/bash
set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_inference.sh <result path> <random seed 1-5>

RESULT_DIR=$1 #"/home/${USERNAME}/laion/pokemon/results_31_07/"
SEED=${2:--1}

USERNAME="mjay"
DIR="/home/${USERNAME}/ai-energy-consumption-framework/"
TRAIN_FILE="${DIR}/stable-diffusion/code/diffusion_pipeline.py"
CACHE_DIR="/home/${USERNAME}/laion/pokemon/cache"
# RESULT_DIR="/home/${USERNAME}/laion/pokemon/inference_5_09/"

SEED=43
# NUM_STEPS=50
# https://stablediffusion.fr/prompts many examples with lots of tokens
PROMPT_FILE="${DIR}/stable-diffusion/prompts/processed_prompts.csv"
MODEL_NAME="CompVis/stable-diffusion-v1-1"
TEMPLATE="${DIR}/utils/script_template.sh"
SENSOR_DIR="/home/${USERNAME}/sensors/nvml-sensor/"
SLEEP_BEFORE=10
SLEEP_AFTER=10
SAMPLING_PERIOD=0.5

sudo-g5k
module load conda
conda activate stable_diffusion
cp /home/${USERNAME}/ai-energy-consumption-framework/stable-diffusion/config/default_config.yaml /home/${USERNAME}/.cache/huggingface/accelerate
cp /home/${USERNAME}/ai-energy-consumption-framework/stable-diffusion/config/token /home/${USERNAME}/.cache/huggingface/

for rep in 1
do
  for SEED in 41 42
  do
    for NUM_STEPS in 25
    do
      for IMG_SIZE in 512 
      do 
        for line in {400..499}
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

          COMMAND="python ${TRAIN_FILE} \
            --pretrained_model_name_or_path=${MODEL_NAME} \
            --cache_dir=${CACHE_DIR} \
            --seed=${SEED} \
            --negative_prompt="green" \
            --num_steps=${NUM_STEPS} \
            --img_size=${IMG_SIZE} \
            --prompt_csv=${PROMPT_FILE} \
            --prompt_line=$line
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
          done
      done
    done
  done
done