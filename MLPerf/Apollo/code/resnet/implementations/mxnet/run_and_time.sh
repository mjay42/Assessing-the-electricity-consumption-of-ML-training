#!/bin/bash

# Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
readonly global_rank=${SLURM_NODEID:-}
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"

SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}
OPTIMIZER=${OPTIMIZER:-"sgd"}
BATCHSIZE=${BATCHSIZE:-1664}
INPUT_BATCH_MULTIPLIER=${INPUT_BATCH_MULTIPLIER:-1}
KVSTORE=${KVSTORE:-"device"}
LR=${LR:-"0.6"}
MOM=${MOM:-"0.9"}
LRSCHED=${LRSCHED:-"30,60,80"}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-5}
LARSETA=${LARSETA:-'0.001'}
DALI_HW_DECODER_LOAD=${DALI_HW_DECODER_LOAD:-'0.0'}
WD=${WD:-'0.0001'}
LABELSMOOTHING=${LABELSMOOTHING:-'0.0'}
SEED=${SEED:-1}
EVAL_OFFSET=${EVAL_OFFSET:-2}
EVAL_PERIOD=${EVAL_PERIOD:-4}
DALI_PREFETCH_QUEUE=${DALI_PREFETCH_QUEUE:-2}
DALI_NVJPEG_MEMPADDING=${DALI_NVJPEG_MEMPADDING:-64}
DALI_THREADS=${DALI_THREADS:-3}
DALI_CACHE_SIZE=${DALI_CACHE_SIZE:-0}
DALI_ROI_DECODE=${DALI_ROI_DECODE:-0}
DALI_PREALLOCATE_WIDTH=${DALI_PREALLOCATE_WIDTH:-0}
DALI_PREALLOCATE_HEIGHT=${DALI_PREALLOCATE_HEIGHT:-0}
DALI_TMP_BUFFER_HINT=${DALI_TMP_BUFFER_HINT:-25273239}
DALI_DECODER_BUFFER_HINT=${DALI_DECODER_BUFFER_HINT:-1315942}
DALI_CROP_BUFFER_HINT=${DALI_CROP_BUFFER_HINT:-165581}
DALI_NORMALIZE_BUFFER_HINT=${DALI_NORMALIZE_BUFFER_HINT:-441549}
DALI_DONT_USE_MMAP=${DALI_DONT_USE_MMAP:-0}
NUMEPOCHS=${NUMEPOCHS:-30}
NETWORK=${NETWORK:-"resnet-v1b-fl"}
BN_GROUP=${BN_GROUP:-1}
PROFILE=${PROFILE:-0}
PROFILE_EXCEL=${PROFILE_EXCEL:-0}
NUMEXAMPLES=${NUMEXAMPLES:-}
NUMVALEXAMPLES=${NUMVALEXAMPLES:-}
PROFILE_ALL_LOCAL_RANKS=${PROFILE_ALL_LOCAL_RANKS:-0}
THR=${TARGET_ACCURACY:-"0.759"}
E2E_CUDA_GRAPHS=${E2E_CUDA_GRAPHS:-0}

DATAROOT="/data"
TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-0}
EPOCH_PROF=${EPOCH_PROF:-0}
SYNTH_DATA=${SYNTH_DATA:-0}

if [ ${NVTX_FLAG} -gt 0 ]; then
 NSYSCMD=" nsys profile --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --output /results/image_classification_mxnet_${DGXNNODES}x${DGXNGPU}x${BATCH_SIZE}_${DATESTAMP}_${SLURM_PROCID}_${SYNTH_DATA}.nsys-rep "
else
 NSYSCMD=""
fi

#if [ ${E2E_CUDA_GRAPHS} -gt 0 ]; then
# if [ ${TIME_TAGS} -gt 0 ]; then
#  TIME_TAGS=0
#  NVTX_FLAG=1
#  echo "Unset TIME_TAGS and set NVTX_FLAG because cuda graph is on"
# fi
#fi

if [ ${NVTX_FLAG--1} -gt 0 ] ||  [ ${TIME_TAGS--1} -gt 0 ]; then
export NUMEPOCHS=4
export WARMUP_EPOCHS=2
fi

echo "running benchmark"
export NGPUS=$SLURM_NTASKS_PER_NODE
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}

if [[ ${PROFILE} -ge 1 ]]; then
    export TMPDIR="/result/"
fi

GPUS=$(seq 0 $(($NGPUS - 1)) | tr "\n" "," | sed 's/,$//')
PARAMS=(
      --gpus               "${GPUS}"
      --batch-size         "${BATCHSIZE}"
      --kv-store           "${KVSTORE}"
      --lr                 "${LR}"
      --mom                "${MOM}"
      --lr-step-epochs     "${LRSCHED}"
      --lars-eta           "${LARSETA}"
      --label-smoothing    "${LABELSMOOTHING}"
      --wd                 "${WD}"
      --warmup-epochs      "${WARMUP_EPOCHS}"
      --eval-period        "${EVAL_PERIOD}"
      --eval-offset        "${EVAL_OFFSET}"
      --optimizer          "${OPTIMIZER}"
      --network            "${NETWORK}"
      --num-layers         "50"
      --num-epochs         "${NUMEPOCHS}"
      --accuracy-threshold "${THR}"
      --seed               "${SEED}"
      --dtype              "float16"
      --disp-batches       "20"
      --image-shape        "4,224,224"
      --fuse-bn-relu       "1"
      --fuse-bn-add-relu   "1"
      --bn-group           "${BN_GROUP}"
      --min-random-area    "0.05"
      --max-random-area    "1.0"
      --conv-algo          "1"
      --force-tensor-core  "1"
      --input-layout       "NHWC"
      --conv-layout        "NHWC"
      --batchnorm-layout   "NHWC"
      --pooling-layout     "NHWC"
      --batchnorm-mom      "0.9"
      --batchnorm-eps      "1e-5"
      --data-train         "${DATAROOT}/train.rec"
      --data-train-idx     "${DATAROOT}/train.idx"
      --data-val           "${DATAROOT}/val.rec"
      --data-val-idx       "${DATAROOT}/val.idx"
      --dali-dont-use-mmap "${DALI_DONT_USE_MMAP}"
      --dali-hw-decoder-load "${DALI_HW_DECODER_LOAD}"
      --dali-prefetch-queue        "${DALI_PREFETCH_QUEUE}"
      --dali-nvjpeg-memory-padding "${DALI_NVJPEG_MEMPADDING}"
      --input-batch-multiplier     "${INPUT_BATCH_MULTIPLIER}"
      --dali-threads       "${DALI_THREADS}"
      --dali-cache-size    "${DALI_CACHE_SIZE}"
      --dali-roi-decode    "${DALI_ROI_DECODE}"
      --dali-preallocate-width "${DALI_PREALLOCATE_WIDTH}"
      --dali-preallocate-height "${DALI_PREALLOCATE_HEIGHT}"
      --dali-tmp-buffer-hint "${DALI_TMP_BUFFER_HINT}"
      --dali-decoder-buffer-hint "${DALI_DECODER_BUFFER_HINT}"
      --dali-crop-buffer-hint "${DALI_CROP_BUFFER_HINT}"
      --dali-normalize-buffer-hint "${DALI_NORMALIZE_BUFFER_HINT}"
      --profile            "${PROFILE}"
      --e2e-cuda-graphs    "${E2E_CUDA_GRAPHS}"
)
if [[ ${SYNTH_DATA} -lt 1 ]]; then
    PARAMS+=(
    --use-dali
    )
fi

# If numexamples is set then we will override the numexamples
if [[ ${NUMEXAMPLES} -ge 1 ]]; then
        PARAMS+=(
        --num-examples "${NUMEXAMPLES}"
        )
fi

# If numvalexamples is set then we will override the numexamples
if [[ ${NUMVALEXAMPLES} -ge 1 ]]; then
        PARAMS+=(
        --num-val-examples "${NUMVALEXAMPLES}"
        )
fi

PROFILE_COMMAND=""
if [[ ${PROFILE} -ge 1 ]]; then
    if [[ ${global_rank} == 0 ]]; then
        if [[ ${local_rank} == 0 || ${PROFILE_ALL_LOCAL_RANKS} == 1 ]]; then
        PROFILE_COMMAND="nsys profile --capture-range cudaProfilerApi --stop-on-range-end true --sample cpu --trace=cuda,nvtx --force-overwrite true --export=sqlite --output /results/${NETWORK}_b${BATCHSIZE}_%h_${local_rank}_${global_rank}.nsys-rep "
        fi
    fi
fi

BIND=''
if [[ "${KVSTORE}" == "horovod" ]]; then
    cluster=''
    if [[ "${DGXSYSTEM}" == DGX2* ]]; then
        cluster='circe'
    fi
    if [[ "${DGXSYSTEM}" == 675D* ]]; then
        cluster='champollion'
    fi
    if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
        cluster='selene'
    fi
    #echo "HPE : binding parameters with cluster : ${cluster} for system : ${DGXSYSTEM} my rank is ${LOCAL_RANK} : ${OMPI_COMM_WORLD_LOCAL_RANK} "
    #BIND="./bind.sh --cpu=exclusive --ib=single --cluster=${cluster} --"
    BIND="./bind.sh --mem=./675D_topology.sh --ib=single --cluster=${cluster} --"
    #BIND=""
fi

if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  # TODO(ahmadki): track the apiLog.sh bug and remove the workaround
  # there is a bug in apiLog.sh preventing it from collecting
  # NCCL logs, the workaround is to log a single rank only
  # LOCAL_RANK is set with an enroot hook for Pytorch containers
  # SLURM_LOCALID is set by Slurm
  # OMPI_COMM_WORLD_LOCAL_RANK is set by mpirun
  readonly node_rank="${SLURM_NODEID:-0}"
  readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
  else
    LOGGER=""
  fi
fi

echo "[run_and_time.sh] Hyper-parameters: seed=$SEED batch-size=$BATCHSIZE num-sample=$NUMEXAMPLES"

echo "PARAMS: $PARAMS"

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING train_imagenet.py AT $start_fmt"

if [[ ${PROFILE} -ge 1 ]]; then
    TMPDIR=/results ${DISTRIBUTED} ${BIND} ${PROFILE_COMMAND} python train_imagenet.py "${PARAMS[@]}"; ret_code=$?
else
    ${LOGGER:-} ${DISTRIBUTED} ${BIND} ${NSYSCMD} python train_imagenet.py "${PARAMS[@]}"; ret_code=$?
fi

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING train_imagenet.py AT $start_fmt"

if [[ ${PROFILE} -ge 1 ]]; then
    if [[ ${PROFILE_EXCEL} -ge 1 ]]; then
        for i in $( find /results -name '*.sqlite' );
        do  
            echo "$(dpkg -l | grep cudnn)"
            echo "python qa/plot_nsight.py --in_files ${i} --out_file "${i/sqlite/txt}" --framework FW_MXNET "
            python qa/plot_nsight.py --in_files ${i} --out_file "${i/sqlite/txt}" --framework FW_MXNET
            echo "python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch ${BATCHSIZE}"
            python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch "${BATCHSIZE}" --mxnet 21.07 --network ${NETWORK}
            python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch "${BATCHSIZE}" --mxnet 21.05 --network ${NETWORK}
            python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch "${BATCHSIZE}" --mxnet 21.03 --network ${NETWORK}
            python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch "${BATCHSIZE}" --mxnet 20.10 --network ${NETWORK}
            #python qa/compare_perf.py --ifile ${i/sqlite/csvcombined_tbl.csv_ave.xlsx} --batch "${BATCHSIZE}" --filter_type BatchNorm
            python qa/draw_iter_time.py --ifile ${i/sqlite/csv} --max_width 40 --start_index 2 --scale 5 --end_index 90 --batch "${BATCHSIZE}"
        done 
    fi
fi

sleep 3

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( $end - $start ))
result_name="IMAGE_CLASSIFICATION"
echo "RESULT,$result_name,,$result,$USER,$start_fmt"
export PROFILE=0
