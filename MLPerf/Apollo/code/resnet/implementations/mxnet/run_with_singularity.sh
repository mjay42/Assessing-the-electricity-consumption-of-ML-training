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

set -euxo pipefail

# Vars without defaults
: "${DGXSYSTEM:?DGXSYSTEM not set}"
: "${CONT:?CONT not set}"

# Vars with defaults
: "${NEXP:=5}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CLEAR_CACHES:=1}"
: "${DATADIR:=/raid/datasets/train-val-recordio-passthrough}"
: "${LOGDIR:=$(pwd)/results}"
: "${COPY_DATASET:=}"

echo $COPY_DATASET

if [ ! -z $COPY_DATASET ]; then
    readonly copy_datadir=$COPY_DATASET
    mkdir -p "${DATADIR}"
    ${CODEDIR}/copy-data.sh "${copy_datadir}" "${DATADIR}"
    ls ${DATADIR}
fi

# Other vars
readonly _seed_override=${SEED:-}
readonly _config_file="./config_${DGXSYSTEM}.sh"
readonly _logfile_base="${LOGDIR}/${DATESTAMP}"
readonly _cont_name=image_classification
_cont_mounts=("--B ${PWD}:/workspace/image_classification" "--B ${DATADIR}:/data" "--B ${LOGDIR}:/results")
export MOUNTS="-e --pwd /workspace/image_classification -B ${PWD}:/workspace/image_classification -B ${DATADIR}:/data -B ${LOGDIR}:/results"

# MLPerf vars
MLPERF_HOST_OS=$(
    source /etc/os-release
    source /etc/dgx-release || true
    echo "${PRETTY_NAME}"
)

# Setup directories
mkdir -p "${LOGDIR}"

# Get list of envvars to pass to docker
mapfile -t _config_env < <(env -i bash -c ". ${_config_file} && compgen -e" | grep -E -v '^(PWD|SHLVL)')
#_config_env+=(MLPERF_HOST_OS)
#_config_env+=(SEED)
mapfile -t _config_env < <(for v in "${_config_env[@]}"; do echo "--env $v=${!v}"; done)

#echo "${_config_env[@]}"

export SING="singularity exec ${_config_env[@]} ${MOUNTS} --nv ${CONT}"

echo "Singularity command : $SING"

# Run experiments
for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Print system info
        ${SING} python -c "
import mlperf_log_utils
from mlperf_logging.mllog import constants

mlperf_log_utils.mlperf_submission_log(constants.RESNET)"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            ${SING} python -c "
import mlperf_log_utils
from mlperf_logging.mllog import constants

mlperf_log_utils.mx_resnet_print_event(key=constants.CACHE_CLEAR, val=True)"
        fi

        # Run experiment
        export SEED=${_seed_override:-$RANDOM}
        ${SING} mpirun -v -display-map --bind-to none --np ${DGXNGPU} ./run_and_time.sh 
    ) |& tee "${_logfile_base}_${_experiment_index}.log"
done
