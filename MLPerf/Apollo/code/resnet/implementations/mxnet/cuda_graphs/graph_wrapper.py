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

from ctypes import c_void_p, c_int, c_size_t, byref, CDLL
graph_lib = CDLL('/workspace/image_classification/cuda_graphs/graph_lib.so')

def start_capture(graph_id, rank, inputs):
    inputs = [arr.handle for arr in inputs]
    if len(inputs) > 0:
        inputs_arr = c_void_p * len(inputs)
        inputs_arr = inputs_arr(*inputs)
    else:
        inputs_arr = byref(c_void_p())
    print("Start Graph Capture")
    graph_lib.start_capture(c_int(graph_id), c_int(rank), inputs_arr, c_size_t(len(inputs)))

def end_capture(graph_id, rank, outputs):
    outputs = [arr.handle for arr in outputs]
    if len(outputs) > 0:
        outputs_arr = c_void_p * len(outputs)
        outputs_arr = outputs_arr(*outputs)
    else:
        outputs_arr = byref(c_void_p())
    graph_lib.end_capture(c_int(graph_id), c_int(rank), outputs_arr, c_size_t(len(outputs)))

def graph_replay(graph_id, rank, inputs, outputs):
    inputs = [arr.handle for arr in inputs]
    outputs = [arr.handle for arr in outputs]
    if len(inputs) > 0:
        inputs_arr = c_void_p * len(inputs)
        inputs_arr = inputs_arr(*inputs)
    else:
        inputs_arr = byref(c_void_p())
    if len(outputs) > 0:
        outputs_arr = c_void_p * len(outputs)
        outputs_arr = outputs_arr(*outputs)
    else:
        outputs_arr = byref(c_void_p())
    graph_lib.graph_replay(c_int(graph_id), c_int(rank), inputs_arr, c_size_t(len(inputs)),
                           outputs_arr, c_size_t(len(outputs)))

def finalize(rank=0):
    import mxnet as mx
    mx.nd.waitall()
    dummy = mx.nd.zeros((10,10), ctx=mx.gpu(rank))
    dummy1 = mx.nd.zeros((10,10), ctx=mx.gpu(rank))
    for _ in range(100):
        dummy[:] += 1
    for _ in range(100):
        dummy1[:].copyto(dummy[:])
    mx.nd.waitall()
