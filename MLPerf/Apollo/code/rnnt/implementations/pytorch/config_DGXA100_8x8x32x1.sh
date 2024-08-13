# v2.1 efficient scale config (replacing 16x8x16x1)

## System config params
export DGXNNODES=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXNGPU=8
export DGXSOCKETCORES=64
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

## Run specific params
export DATADIR="/raid/datasets/rnnt/"
export METADATA_DIR="/lustre/fsw/mlperf-ci/tokenized/"
export SENTENCEPIECES_DIR="/lustre/fsw/mlperf-ci/sentpiece"
export BATCHSIZE=32
export EVAL_BATCHSIZE=43
export GRAD_ACCUMULATION_STEPS=1
WALLTIME_MINUTES=15
export WALLTIME=$(( 15 + ${NEXP:-1} * ${WALLTIME_MINUTES} ))
export VAL_FREQUENCY=1
export MAX_SYMBOL=300
export DATA_CPU_THREADS=8

source $(dirname ${BASH_SOURCE[0]})/hyperparameters_2048.sh

## Opt flag
export FUSE_RELU_DROPOUT=true
export MULTI_TENSOR_EMA=true
export BATCH_EVAL_MODE=cg_unroll_pipeline
export APEX_LOSS=fp16
export APEX_JOINT=pack_w_relu_dropout
export AMP_LVL=2
export BUFFER_PREALLOC=true
export VECTORIZED_SA=true
export EMA_UPDATE_TYPE=fp16
export DIST_LAMB=true
export MULTILAYER_LSTM=false
export ENABLE_PREFETCH=true
export VECTORIZED_SAMPLER=true
export DIST_SAMPLER=true
export TOKENIZED_TRANSCRIPT=true
export FC_IMPL=apex_fused_dense
export LOG_FREQUENCY=1000

## network flag
export SBATCH_NETWORK=sharp
export NCCL_COLLNET_ENABLE=1
