#!/bin/bash

# Copyright (c) 2018-2021, NVIDIA CORPORATION. All rights reserved.
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

set -e

ulimit -Sn 100000 || true

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# run benchmark
set -x

echo "running benchmark"

export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb=256'

DATASET_DIR='/data'
## ln -sTf "${DATASET_DIR}/coco2017" /coco
echo `ls /data`
echo `ls /coco`
echo `ls /pkl_coco`

TIME_TAGS=${TIME_TAGS:-0}
NVTX_FLAG=${NVTX_FLAG:-0}
NCCL_TEST=${NCCL_TEST:-0}
EPOCH_PROF=${EPOCH_PROF:-0}
SYNTH_DATA=${SYNTH_DATA:-0}
DISABLE_CG=${DISABLE_CG:-0}

#disable nsight when synth is on because it segment faults
#if [ ${SYNTH_DATA} -gt 0 ]; then
#NVTX_FLAG=0
#fi

if [ ${NVTX_FLAG} -gt 0 ]; then
 NSYSCMD=" /nsight/bin/nsys profile --capture-range cudaProfilerApi --capture-range-end stop --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --output /results/object_detection_pytorch_${DGXNNODES}x${DGXNGPU}x${BATCHSIZE}_${DATESTAMP}_${SLURM_PROCID}_${SYNTH_DATA}_${DISABLE_CG}.nsys-rep "
else
 NSYSCMD=""
fi
if [ ${TIME_TAGS} -gt 0 ]; then
WALLTIME=${WALLTIME}+10
fi

declare -a CMD
# if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]]; then
#     # Mode 1: Slurm launched a task for each GPU and set some envvars
#     if [[ "${DGXSYSTEM}" == DGXA100* ]]; then
#         CMD=( 'bindpcie' '--cpu=dgxa100_topology.sh' '--mem=dgxa100_topology.sh' '--' ${NSYSCMD} 'python' '-u' )
#     else
#         CMD=( 'bindpcie' '--cpu=exclusive' '--' ${NSYSCMD} 'python' '-u' )
#     fi
# else
#     # docker or single gpu, no need to bind
#     CMD=( ${NSYSCMD} 'python' '-u' )
# fi

if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]]; then
    # Mode 1: Slurm launched a task for each GPU and set some envvars
    #CMD=( 'bindpcie' '--cpu=exclusive' '--ib=single' '--' ${NSYSCMD} 'python' '-u')
    echo "Enable binding"
#    CMD=( '/workspace/object_detection/bindpcie'  '--cpu=/workspace/object_detection/675D_cpu_topology.sh' '--ib=single' '--mem=/workspace/object_detection/675D_mem_topology.sh' '--' ${NSYSCMD} 'python' '-u')
    CMD=( '/workspace/object_detection/bindpcie'  '--cpu=/workspace/object_detection/675D_cpu_topology.sh' '--' ${NSYSCMD} 'python' '-u')
else
    echo "No binding !"
    # docker or single gpu, no need to bind
    CMD=( ${NSYSCMD} 'python' '-u' )
fi


if [ ${SYNTH_DATA} -gt 0 ]; then
EXTRA_CONFIG=$(echo $EXTRA_CONFIG | sed 's/DATALOADER.HYBRID\sTrue/DATALOADER.HYBRID False/')
EXTRA_CONFIG+=" DATALOADER.USE_SYNTHETIC_INPUT True "
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

echo "[run_and_time.sh] Hyper-parameters: seed=$SEED"
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING train_mlperf.py AT $start_fmt"

${LOGGER:-} "${CMD[@]}" maskrcnn/tools/train_mlperf.py \
  ${EXTRA_PARAMS} \
  --config-file 'maskrcnn/configs/e2e_mask_rcnn_R_50_FPN_1x.yaml' \
  DTYPE 'float16' \
  PATHS_CATALOG 'maskrcnn/maskrcnn_benchmark/config/paths_catalog_dbcluster.py' \
  MODEL.WEIGHT '/coco/models/R-50.pkl' \
  DISABLE_REDUCED_LOGGING True \
  ${EXTRA_CONFIG} ; ret_code=$?

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING train_mlperf.py AT $start_fmt"


set +x

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

# report result
result=$(( $end - $start ))
result_name="OBJECT_DETECTION"

echo "RESULT,$result_name,,$result,nvidia,$start_fmt"

