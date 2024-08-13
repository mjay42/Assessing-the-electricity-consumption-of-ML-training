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

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DATADIR:=/raid/datasets/coco/coco-2017}"
: "${PKLDIR:=/lustre/fsw/mlperf/mlperft-mrcnn/pkl_coco}"
: "${LOGDIR:=$(pwd)/results}"

# Other vars
readonly _config_file="./config_${DGXSYSTEM}singularity.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=object_detection
_cont_mounts=("--volume=${PWD}:/workspace/object_detection" "--volume=${DATADIR}:/data" "--volume=${LOGDIR}:/results" "--volume=${PKLDIR}:/pkl_coco")
export MOUNTS="-e --pwd /workspace/object_detection -B ${PWD}:/workspace/object_detection -B ${DATADIR}:/data -B ${DATADIR}:/coco -B ${LOGDIR}:/results -B ${PKLDIR}:/pkl_coco"

# MLPerf vars
MLPERF_HOST_OS=$(
    source /etc/os-release
    source /etc/dgx-release || true
    echo "${PRETTY_NAME}"
)
#export MLPERF_HOST_OS

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
#_config_env+=(MLPERF_HOST_OS)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env $v=${!v}"; done)

#export SING="singularity exec --overlay ${OVERLAY} ${_config_env[@]} ${MOUNTS} --nv ${CONT}  "
export SING="singularity exec ${_config_env[@]} ${MOUNTS} --nv ${CONT}  "

echo "Singularity command : $SING"

readonly TORCH_RUN="python -m torch.distributed.run --standalone --no_python"

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Print system info
        ${SING} python -c "
from mlperf_logging.mllog import constants
from maskrcnn_benchmark.utils.mlperf_logger import mlperf_submission_log
mlperf_submission_log(constants.MASKRCNN)"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            ${SING} python -c "
from mlperf_logging.mllog import constants
from maskrcnn_benchmark.utils.mlperf_logger import log_event
log_event(key=constants.CACHE_CLEAR, value=True, stack_offset=0)"
        fi

        # Run experiment
        ${SING} ${TORCH_RUN} --nproc_per_node=${DGXNGPU} ./run_and_time_singularity.sh
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done
