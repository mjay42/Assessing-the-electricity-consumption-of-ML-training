# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
import torch
import torch.nn.functional as F
from torch import nn
from maskrcnn_benchmark.layers.nhwc import Conv2d_NHWC, nhwc_to_nchw_transform, nchw_to_nhwc_transform, interpolate_nhwc
from maskrcnn_benchmark.layers.nhwc import MaxPool2d_NHWC 
from maskrcnn_benchmark.layers.nhwc import init
from maskrcnn_benchmark import _C
from maskrcnn_benchmark.utils.mlperf_logger import mllogger
from maskrcnn_benchmark.utils.fuse_conv import ConvBias_

import itertools

ConvBias = ConvBias_.apply

class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    def __init__(
        self, in_channels_list, out_channels, conv_block, top_blocks=None, nhwc=False, use_fusion=False
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        super(FPN, self).__init__()
        self.inner_blocks = []
        self.layer_blocks = []
        self.nhwc = nhwc
        self.use_fusion = use_fusion
        for idx, in_channels in enumerate(in_channels_list, 1):
            inner_block = "fpn_inner{}".format(idx)
            layer_block = "fpn_layer{}".format(idx)
            if in_channels == 0:
                continue
            inner_block_module = conv_block(in_channels, out_channels, 1, nhwc=nhwc)
            mllogger.event(mllogger.constants.WEIGHTS_INITIALIZATION,
                metadata=dict(tensor="FPN_inner_block"+str(idx)))
            layer_block_module = conv_block(out_channels, out_channels, 3, 1, nhwc=nhwc)
            mllogger.event(mllogger.constants.WEIGHTS_INITIALIZATION,
                metadata=dict(tensor="FPN_layer_block"+str(idx)))
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        
        results, streams = [], []
        
        if self.use_fusion and self.nhwc:
            last_inner = ConvBias(x[-1], torch.permute(getattr(self, self.inner_blocks[-1]).weight.half(), (0, 3, 1, 2)), \
                                  getattr(self, self.inner_blocks[-1]).bias.reshape(1, -1, 1, 1).half(), 0, 1)
            
            results.append(ConvBias(last_inner, torch.permute(getattr(self, self.layer_blocks[-1]).weight.half(), (0, 3, 1, 2)), \
                     getattr(self, self.layer_blocks[-1]).bias.reshape(1, -1, 1, 1).half(), 1, 1))
    
        else:
            last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
            results.append(getattr(self, self.layer_blocks[-1])(last_inner))
        for feature, inner_block, layer_block in zip(
            x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            if not inner_block:
                continue
            interpolate_func = F.interpolate if not self.nhwc else interpolate_nhwc
            inner_top_down = interpolate_func(last_inner, scale_factor=2, mode="nearest")

            if self.nhwc and self.use_fusion:
                inner_lateral = ConvBias(feature, torch.permute(getattr(self, inner_block).weight.half(), (0, 3, 1, 2)), \
                                                                getattr(self, inner_block).bias.reshape(1, -1, 1, 1).half(), \
                                         0, 1) 
                last_inner = inner_lateral + inner_top_down
                s1 = torch.cuda.Stream()
                streams.append(s1)
                s1.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s1):
                    results.insert(0, ConvBias(last_inner, torch.permute(getattr(self, layer_block).weight.half(), (0, 3, 1, 2)), \
                                           getattr(self, layer_block).bias.reshape(1, -1, 1, 1).half(), 1, 1))

            else:
                inner_lateral = getattr(self, inner_block)(feature)
                # TODO use size instead of scale to make it robust to different sizes
                last_inner = inner_lateral + inner_top_down
                s1 = torch.cuda.Stream()
                streams.append(s1)
                s1.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s1):
                    results.insert(0, getattr(self, layer_block)(last_inner))
        if isinstance(self.top_blocks, LastLevelP6P7):
            s1 = torch.cuda.Stream()
            streams.append(s1)
            s1.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s1):
                last_results = self.top_blocks(x[-1], results[-1])
                results.extend(last_results)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            s1 = torch.cuda.Stream()
            streams.append(s1)
            s1.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s1):
                last_results = self.top_blocks(results[-1], self.nhwc)
                results.extend(last_results)
        for s1 in streams:
            torch.cuda.current_stream().wait_stream(s1)
        if self.nhwc:
            return tuple(results)
        assert False, "code path not tested with cuda graphs"
        return tuple(results)


class LastLevelMaxPool(nn.Module):
    def forward(self, x, nhwc):
        op = MaxPool2d_NHWC(1,2,0) if nhwc else nn.MaxPool2d(1,2,0)
        return [op(x)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """
    def __init__(self, in_channels, out_channels, nhwc):
        super(LastLevelP6P7, self).__init__()
        conv = conv2d_NHWC if nhwc else nn.Conv2d
        self.p6 = conv(in_channels, out_channels, 3, 2, 1)
        self.p7 = conv(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            init.kaiming_uniform_(module.weight, a=1, nhwc=nhwc)
            init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]
