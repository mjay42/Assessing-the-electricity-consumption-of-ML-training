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

import mxnet
# Monkey patching. Code will be available in the 22.05 container
from model.conv_layers import _SpatialParallelHelper
mxnet.gluon.contrib.nn.conv_layers._SpatialParallelHelper = _SpatialParallelHelper

import numpy as np
from mxnet import gluon, init
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import SpatialParallelConv3D, SpatialParallelAllgather, SpatialParallelSplit

from mlperf_logging.mllog import constants
from mlperf_logger import mllog_event

from model.gbn import GroupInstanceNorm

normalizations = {"instancenorm": nn.InstanceNormV2, "batchnorm": nn.BatchNorm}
activations = {"relu": nn.Activation("relu"), "leaky_relu": nn.LeakyReLU(0.01)}
CONV_WORKSPACE_LIMIT = 4096


def weight_and_bias_init(channels_in, kernel_size):
    w_init = init.Uniform(np.sqrt(6.0 / (6 * channels_in * kernel_size ** 3)))  # kaiming
    b_init = init.Uniform(np.sqrt(1.0 / (channels_in * kernel_size ** 3)))  # 1 / math.sqrt(fan_in)
    return w_init, b_init


class InstanceNorm(gluon.HybridBlock):
    def __init__(self, in_channels, axis, scale, center, act_type):
        super(InstanceNorm, self).__init__()
        with self.name_scope():
            self.instance_norm = nn.InstanceNormV2(in_channels=in_channels, axis=axis, scale=scale,
                                                   center=center, act_type=act_type)

    def hybrid_forward(self, F, x):
        x = self.instance_norm(x)
        return x


def conv_block(channels_in, channels_out, kernel_size=3, strides=1, padding=1, name="conv",
               spatial=False, spatial_group_size=1, local_rank=0, comm=None):
    w_init, b_init = weight_and_bias_init(channels_in, kernel_size)
    Conv = SpatialParallelConv3D if spatial else nn.Conv3D
    Norm = GroupInstanceNorm if spatial else InstanceNorm
    conv_kwargs = dict(channels=channels_out,
                       in_channels=channels_in,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding=padding,
                       use_bias=False,
                       layout='NDHWC',
                       weight_initializer=w_init,
                       bias_initializer=b_init,
                       workspace=CONV_WORKSPACE_LIMIT)
    norm_kwargs = dict(in_channels=channels_out,
                       axis=-1,
                       scale=True,
                       center=True,
                       act_type='relu')
    if spatial:
        conv_kwargs["num_gpus"] = spatial_group_size
        norm_kwargs["spatial_group_size"] = spatial_group_size
        norm_kwargs["local_rank"] = local_rank
        norm_kwargs["comm"] = comm

    block = nn.HybridSequential()
    with block.name_scope():
        block.add(Conv(**conv_kwargs))
        mllog_event(key=constants.WEIGHTS_INITIALIZATION, sync=False, metadata=dict(tensor=name + "_conv"))
        block.add(Norm(**norm_kwargs))
        mllog_event(key=constants.WEIGHTS_INITIALIZATION, sync=False, metadata=dict(tensor=name + "_instancenorm"))
        if not spatial:
            block.collect_params().setattr('lr_mult', 1.0 / spatial_group_size)

    return block


class InputBlock(gluon.HybridBlock):
    def __init__(self, channels_in, channels_out, spatial_group_size=1, local_rank=0, spatial=False, comm=None):
        super(InputBlock, self).__init__()
        with self.name_scope():
            self.conv1 = conv_block(channels_in, channels_out, name="input_block_0",
                                    spatial=spatial, spatial_group_size=spatial_group_size,
                                    local_rank=local_rank, comm=comm)
            self.conv2 = conv_block(channels_out, channels_out, name="input_block_1",
                                    spatial=spatial, spatial_group_size=spatial_group_size,
                                    local_rank=local_rank, comm=comm)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DownsampleBlock(gluon.HybridBlock):
    def __init__(self, channels_in, channels_out, index, spatial_group_size=1, local_rank=0, spatial=False, comm=None):
        super(DownsampleBlock, self).__init__()
        with self.name_scope():
            self.conv1 = conv_block(channels_in, channels_out, strides=2, name=f"down{index}_block_0",
                                    spatial=spatial, spatial_group_size=spatial_group_size,
                                    local_rank=local_rank, comm=comm)
            self.conv2 = conv_block(channels_out, channels_out, name=f"down{index}_block_1",
                                    spatial=spatial, spatial_group_size=spatial_group_size,
                                    local_rank=local_rank, comm=comm)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpsampleBlock(gluon.HybridBlock):
    def __init__(self, channels_in, channels_out, index, spatial_group_size=1, local_rank=0, spatial=False, comm=None):
        super(UpsampleBlock, self).__init__()
        self.concat_axis = -1
        with self.name_scope():
            w_init, b_init = weight_and_bias_init(channels_in=channels_out, kernel_size=2)
            self.upsample_conv = nn.Conv3DTranspose(channels=channels_out,
                                                    in_channels=channels_in,
                                                    kernel_size=2,
                                                    strides=2,
                                                    padding=0,
                                                    use_bias=True,
                                                    layout='NDHWC',
                                                    weight_initializer=w_init,
                                                    bias_initializer=b_init,
                                                    workspace=CONV_WORKSPACE_LIMIT
                                                    )
            mllog_event(key=constants.WEIGHTS_INITIALIZATION, sync=False, metadata=dict(tensor=f"up{index}_deconv"))
            self.conv1 = conv_block(2 * channels_out, channels_out, name=f"up{index}_block_0",
                                    spatial=spatial, spatial_group_size=spatial_group_size,
                                    local_rank=local_rank, comm=comm)
            self.conv2 = conv_block(channels_out, channels_out, name=f"up{index}_block_1",
                                    spatial=spatial, spatial_group_size=spatial_group_size,
                                    local_rank=local_rank, comm=comm)

            if not spatial:
                self.upsample_conv.collect_params().setattr('lr_mult', 1.0 / spatial_group_size)

    def hybrid_forward(self, F, x, skip):
        x = self.upsample_conv(x)
        x = F.concat(x, skip, dim=self.concat_axis, num_args=2)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class OutputBlock(gluon.HybridBlock):
    def __init__(self, channels_in, channels_out, spatial_group_size=1, local_rank=0, spatial=False, comm=None):
        super(OutputBlock, self).__init__()
        with self.name_scope():
            w_init, b_init = weight_and_bias_init(channels_in, 1)
            Conv = nn.Conv3D
            kwargs = dict(channels=channels_out,
                          in_channels=channels_in,
                          kernel_size=1,
                          strides=1,
                          padding=0,
                          use_bias=True,
                          layout='NDHWC',
                          weight_initializer=w_init,
                          bias_initializer=b_init,
                          workspace=CONV_WORKSPACE_LIMIT)
            self.conv = Conv(**kwargs)
            mllog_event(key=constants.WEIGHTS_INITIALIZATION, sync=False, metadata=dict(tensor=f"output_conv"))

            if not spatial:
                self.conv.collect_params().setattr('lr_mult', 1.0 / spatial_group_size)

    def hybrid_forward(self, F, x):
        return self.conv(x)


class SplitBlock(gluon.HybridBlock):
    def __init__(self, spatial_group_size=16):
        super(SplitBlock, self).__init__()
        with self.name_scope():
            self.split = SpatialParallelSplit(num_gpus=spatial_group_size)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.split(x)


class GatherBlock(gluon.HybridBlock):
    def __init__(self, spatial_group_size=16):
        super(GatherBlock, self).__init__()
        with self.name_scope():
            self.gather = SpatialParallelAllgather(num_gpus=spatial_group_size)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.gather(x)
