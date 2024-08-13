# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
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

import os
import math
import numpy as np

import mxnet as mx
from mxnet.contrib import amp
import horovod.mxnet as hvd
from mpi4py import MPI

from mlperf_logging import mllog
from mlperf_logging.mllog import constants
from mlperf_logger import get_logger, mllog_start, mllog_end, mllog_event, mlperf_run_param_log, mlperf_submission_log

from model.unet3d import Unet3D
from model.losses import DiceScore
from data_loading.data_loader import get_data_loaders
from runtime.arguments import PARSER
from runtime.training import train
from runtime.warmup import train as train_init
from runtime.distributed import assign_mpiranks, get_group_comm
from runtime.inference import evaluate, SlidingWindow
from runtime.setup import seed_everything, get_seed, check_flags
from runtime.callbacks import get_callbacks
from scaleoutbridge import init_bridge, ScaleoutBridge as SBridge


def main():
    mllog.config(filename=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet3d.log'))
    mllog.config(filename=os.path.join("/results", 'unet3d.log'))
    mllogger = mllog.get_mllogger()
    mllogger.logger.propagate = False
    mllog_event(key=constants.CACHE_CLEAR, value=True)
    mllog_start(key=constants.INIT_START)

    flags = PARSER.parse_args()
    if flags.nodes_for_eval:
        os.environ["NCCL_SHARP_GROUP_SIZE_THRESH"] = str(flags.nodes_for_eval + 1)
    comm = MPI.COMM_WORLD
    global_rank = comm.Get_rank()
    world_size = comm.Get_size()
    check_flags(flags, world_size)
    local_rank = global_rank % flags.gpu_per_node

    train_ranks, eval_ranks, transfer_ranks = assign_mpiranks(local_rank, world_size,
                                                              flags.nodes_for_eval, flags.gpu_per_node)
    mlperf_submission_log(num_nodes=max(1, world_size // flags.gpu_per_node))

    train_comm = get_group_comm(comm, train_ranks)
    eval_comm = get_group_comm(comm, eval_ranks)
    transfer_comm = get_group_comm(comm, transfer_ranks)

    worker_seed = flags.seed
    if global_rank in train_ranks:
        hvd.init(train_comm)
        worker_seed = get_seed(flags.seed, flags.spatial_group_size, flags.use_cached_loader, flags.stick_to_shard)
        seed_everything(worker_seed)
        mllog_event(key=constants.SEED, value=flags.seed if flags.seed != -1 else worker_seed, sync=False)

    if (global_rank in train_ranks) and flags.use_nvshmem:
        mx.cuda_utils.nvshmem_init(train_comm)

    ctx = mx.gpu(local_rank)
    if global_rank == 0:
        mlperf_run_param_log(flags)

    if flags.amp:
        amp.init()

    dllogger = get_logger(flags, eval_ranks, global_rank)
    callbacks = get_callbacks(flags, dllogger, eval_ranks, global_rank, world_size)
    score_fn = DiceScore(to_onehot_y=True, use_argmax=True, include_background=flags.include_background,
                         spatial_group_size=flags.spatial_group_size if not flags.shard_eval else 1)
    sw_inference = SlidingWindow(batch_size=flags.val_batch_size,
                                 mode="gaussian",
                                 roi_shape=flags.val_input_shape,
                                 precision=np.float16 if flags.static_cast or flags.amp else np.float32,
                                 data_precision=np.float16 if flags.static_cast else np.float32,
                                 ctx=ctx,
                                 local_rank=local_rank,
                                 spatial_group_size=flags.spatial_group_size,
                                 cache_dataset=True,
                                 shard_eval=flags.shard_eval,
                                 shard_eval_size=min(len(eval_ranks), flags.spatial_group_size))

    current_comm = train_comm if global_rank in train_ranks else eval_comm
    model = Unet3D(n_classes=3, spatial_group_size=flags.spatial_group_size, local_rank=local_rank, comm=current_comm,
                   is_eval=global_rank in eval_ranks, shard_eval=flags.shard_eval)

    global_batch_size = flags.batch_size * (len(train_ranks) // flags.spatial_group_size)
    steps_per_epoch = math.ceil(168/global_batch_size)
    world_size = world_size if global_rank in train_ranks else 1
    model.init(flags, ctx=ctx, world_size=world_size, steps_per_epoch=steps_per_epoch,
               is_training_rank=global_rank in train_ranks, cold_init=True, warmup_iters=flags.warmup_iters)
    if flags.static_cast:
        model.cast('float16')
        if flags.fp16in:
            model.cast_in()

    if flags.exec_mode == 'train' and flags.warmup and global_rank in train_ranks:

        train_init(flags, model, comm, train_comm, eval_comm, transfer_comm,
                   train_ranks, eval_ranks, ctx=ctx)

        model.init(flags, ctx=ctx, world_size=world_size, steps_per_epoch=steps_per_epoch,
                   is_training_rank=True, cold_init=False)
    
    mllog_end(key=constants.INIT_STOP)
    comm.Barrier()
    mllog_start(key=constants.RUN_START)
    comm.Barrier()

    sbridge = init_bridge(global_rank)
    sbridge.start_prof(SBridge.LOAD_TIME)

    train_loader, val_loader = get_data_loaders(flags, data_dir=flags.data_dir, seed=worker_seed,
                                                local_rank=local_rank, global_rank=global_rank,
                                                train_ranks=train_ranks, eval_ranks=eval_ranks,
                                                spatial_group_size=flags.spatial_group_size,
                                                shard_eval=flags.shard_eval, ctx=ctx, world_size=world_size)
    sbridge.stop_prof(SBridge.LOAD_TIME)

    mllog_event(key=constants.GLOBAL_BATCH_SIZE, sync=False, value=global_batch_size)
    mllog_event(key=constants.GRADIENT_ACCUMULATION_STEPS, sync=False, value=1)

    if flags.exec_mode == 'train':
        train(flags, model, train_loader, val_loader, score_fn, sw_inference,
              comm, train_comm, eval_comm, transfer_comm,
              train_ranks, eval_ranks, transfer_ranks, ctx=ctx, callbacks=callbacks)

    elif flags.exec_mode == 'evaluate':
        eval_metrics = evaluate(flags, model, val_loader, sw_inference, score_fn, ctx=ctx, eval_comm=eval_comm)
        eval_metrics = evaluate(flags, model, val_loader, sw_inference, score_fn, ctx=ctx, eval_comm=eval_comm)
        if global_rank == 0:
            for key in eval_metrics.keys():
                print(key, eval_metrics[key])


if __name__ == '__main__':
    main()
