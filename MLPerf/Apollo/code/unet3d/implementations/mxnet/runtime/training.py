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

import math
from time import time
from tqdm import tqdm

import horovod.mxnet as hvd
from mxnet.contrib import amp
from mxnet import nd, autograd
from runtime.inference import evaluate
from runtime.distributed import sync_training_and_evaluation
from mlperf_logger import mllog_event, mllog_start, mllog_end, CONSTANTS
import mxnet as mx
from scaleoutbridge import init_bridge, ScaleoutBridge as SBridge


def train(flags, model, train_loader, val_loader, score_fn, sw_inference, comm, train_comm, eval_comm,
          transfer_comm, train_ranks, eval_ranks, transfer_ranks, ctx, callbacks):
    rank = comm.Get_rank()
    stop_training = False
    diverged = False
    eval_warmup = flags.nodes_for_eval > 0

    if rank in train_ranks:
        train_size = hvd.size()
        samples_per_epoch = math.ceil(168 / ((train_size // flags.spatial_group_size) * flags.batch_size))
        samples_per_epoch = samples_per_epoch * flags.batch_size * (train_size // flags.spatial_group_size)
        mllog_event(key='samples_per_epoch', value=samples_per_epoch, sync=False)

    for callback in callbacks:
        callback.on_fit_start()

    global_epoch = 1
    max_cycles = flags.epochs if flags.epochs < flags.evaluate_every else (flags.epochs // flags.evaluate_every + 1)

    sbridge = init_bridge(rank)
    sbridge.start_epoch_prof()

    for cycle in range(1, max_cycles):
        mllog_start(key=CONSTANTS.BLOCK_START, sync=False,
                    metadata={CONSTANTS.FIRST_EPOCH_NUM: global_epoch, CONSTANTS.EPOCH_COUNT: flags.evaluate_every})
        for callback in callbacks:
            callback.on_cycle_start()

        if rank in train_ranks:
            cycle_start_time = time()

            for training_epoch in range(0, flags.evaluate_every):
                for i, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
                    image, label = batch
                    if flags.static_cast:
                        image = image.astype(dtype='float16')

                    with autograd.record():
                        sbridge.start_prof(SBridge.FWD_TIME)

                        loss_value = model(image, label)

                        sbridge.stop_start_prof(SBridge.FWD_TIME, SBridge.BWD_TIME)

                        if flags.amp:
                            with amp.scale_loss(loss_value, model.trainer) as scaled_loss:
                                autograd.backward(scaled_loss)
                        elif flags.static_cast:
                            scaled_loss = loss_value * flags.static_loss_scale
                            autograd.backward(scaled_loss)
                        else:
                            loss_value.backward()

                        sbridge.stop_prof(SBridge.BWD_TIME)

                    sbridge.start_prof(SBridge.OPT_TIME)

                    model.trainer.step(image.shape[0] / flags.spatial_group_size)
                    loss_value.asnumpy()  # to prevent hang

                    sbridge.stop_prof(SBridge.OPT_TIME)


            throughput = samples_per_epoch * flags.evaluate_every / (time() - cycle_start_time)
            loss_scale = flags.static_loss_scale if flags.static_cast else model.trainer._amp_loss_scaler.loss_scale
            mllog_event(key='throughput', value=throughput, sync=False)
            mllog_event(key='current_lr', value=model.trainer.learning_rate, sync=False)
            mllog_event(key='tracked_stats', metadata={'step': global_epoch}, sync=False,
                        value={"throughput": throughput, "iterations": i + 1,
                               "loss_scale": loss_scale})

            if cycle in flags.loss_scale_inc_cycles and flags.static_cast:
                flags.static_loss_scale *= 2.0
                model.trainer._scale /= 2.0
        # Sync training and eval nodes
        global_epoch = cycle * flags.evaluate_every
        if (global_epoch >= flags.start_eval_at) and flags.nodes_for_eval:
            sbridge.start_eval_prof()
            stop_training, model = sync_training_and_evaluation(comm, transfer_comm, rank, model, eval_ranks,
                                                                transfer_ranks, stop_training)
            sbridge.stop_eval_prof()

        if stop_training:
            break

        if rank in eval_ranks and eval_warmup:
            eval_metrics = evaluate(flags, model, val_loader, sw_inference, score_fn, ctx, eval_comm, global_epoch)
            eval_warmup = False

        if rank in eval_ranks and (global_epoch >= flags.start_eval_at):
            sbridge.start_eval_prof()
            mllog_start(key=CONSTANTS.EVAL_START, value=global_epoch, sync=False, force=rank == eval_ranks[0],
                        metadata={CONSTANTS.EPOCH_NUM: global_epoch})
            eval_metrics = evaluate(flags, model, val_loader, sw_inference, score_fn, ctx, eval_comm, global_epoch)
            mllog_event(key=CONSTANTS.EVAL_ACCURACY,
                        value=eval_metrics["mean_dice"],
                        metadata={CONSTANTS.EPOCH_NUM: global_epoch},
                        sync=False,
                        force=rank == eval_ranks[0])
            mllog_end(key=CONSTANTS.EVAL_STOP, value=global_epoch, sync=False, force=rank == eval_ranks[0],
                      metadata={CONSTANTS.EPOCH_NUM: global_epoch})
            sbridge.stop_eval_prof()

            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                stop_training = True
                mllog_end(key=CONSTANTS.RUN_STOP, sync=False, force=rank == eval_ranks[0],
                          metadata={CONSTANTS.STATUS: CONSTANTS.SUCCESS, "epoch": global_epoch})
            elif eval_metrics["mean_dice"] < 1e-4:
                stop_training = True
                diverged = True
                mllog_end(key=CONSTANTS.RUN_STOP, sync=False, force=rank == eval_ranks[0],
                          metadata={CONSTANTS.STATUS: CONSTANTS.ABORTED})

            for callback in callbacks:
                callback.on_cycle_end(epoch=global_epoch, metrics=eval_metrics, model=model)

        mllog_end(key=CONSTANTS.BLOCK_STOP, sync=False,
                  metadata={CONSTANTS.FIRST_EPOCH_NUM: global_epoch, CONSTANTS.EPOCH_COUNT: flags.evaluate_every})
    sbridge.stop_epoch_prof()
    if rank == eval_ranks[0] and (not diverged and not stop_training):
        mllog_end(key=CONSTANTS.RUN_STOP, sync=False, force=rank == eval_ranks[0],
                  metadata={CONSTANTS.STATUS: CONSTANTS.ABORTED})

    for callback in callbacks:
        callback.on_fit_end(model=model)

    nd.waitall()

