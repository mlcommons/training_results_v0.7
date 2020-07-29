/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2019 by Contributors
 * \file c_api_gpuipc.cc
 * \brief auxilary functions to initialize GPU Inter-Process-Communication
 * \author Evgeni Krimer
*/


#include <cuda.h>
#include <stdio.h>
#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <nnvm/c_api.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include "./c_api_common.h"



#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess && __err != cudaErrorPeerAccessAlreadyEnabled) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


namespace gpuipc {
// from: src/operator/nn/cudnn/nhwc_batch_norm_kernel.h
// The number of threads per pixel.
const int THREADS_PER_PIXEL = 16;
// The number of elements per ldg.
const int ELEMENTS_PER_LDG = 4;
// The number of reducing ops, each uses its own space : mean, var, dscale, dbias
const int REDUCE_OPS = 4;
// Maximum block.y supported - limited due to buffer allocation
const int MAX_BLOCK_Y = 256;
const int MAX_OFFSET = REDUCE_OPS*MAX_BLOCK_Y;
const int BYTES_PER_ELEM = 4;
// Buffer size per sync step
const int SINGLE_SYNC_BUFFER_BYTES =
              MAX_OFFSET*THREADS_PER_PIXEL*(1+ELEMENTS_PER_LDG)*BYTES_PER_ELEM;
};   // namespace gpuipc

// Initialize IPC exchange buffers on GPUs 0..gpus. Turn on peer access for GPUs expected
// to communicate. This function would be used to initialize exchange buffers for group BN
// in case of a single process driving multiple GPUs.
//
// input :
//         gpus - number of gpus used by the process (not the group size)
//         sync_steps - number of group synchronization steps. Effectively = log2(group_size)
//
// output :
//         xbuf_ptr - a preallocated array of void_ptr is filled in with buffer pointers
//                    such that xbuf_ptr[i] would contain the pointer for buffer allocated on GPU i
//
// FIXME: pass gpus as a list? Currently will assume GPUs 0..gpus participate
//
int MXInitXBuf(int gpus, int sync_steps, void** xbuf_ptr) {
    const int buffer_size = sync_steps*gpuipc::SINGLE_SYNC_BUFFER_BYTES;
    for (int i=0; i < gpus; ++i) {
        uint8_t *data;
        cudaCheckErrors("MXInitXBuf");

        cudaSetDevice(i);
        cudaCheckErrors("MXInitXBuf: set device");

        for (int j=0; j < sync_steps; ++j) {
            cudaDeviceEnablePeerAccess(i^(1 << j), 0);
            cudaCheckErrors("MXInitXBuf: enable p2p");
        }

        cudaMalloc(&data, buffer_size);
        cudaMemset(data, 0, buffer_size);
        cudaCheckErrors("MXInitXBuf: malloc/memset");

        cudaIpcMemHandle_t my_handle;
        cudaIpcGetMemHandle(&my_handle, data);

        xbuf_ptr[i] = data;
    }

    return 0;
}


// Initialize IPC exchange buffers on a single GPU. Turn on peer access for GPUs expected
// to communicate. This function would be used to initialize exchange buffers for group BN
// in case of a GPU per process.
//
// input :
//         gpu_id - id of gpu used by the process
//         sync_steps - number of group synchronization steps. Effectively = log2(group_size)
//
// output :
//         xbuf_ptr - a preallocated array of void_ptr is filled in with buffer pointers
//                    such that xbuf_ptr[i] would contain the pointer for buffer allocated on GPU i.
//                    All other entries remain unchanged
//         hndl_ptr - a pointer to cudaIpcMemHandle_t buffer. Will be filled in with
//                    handle information of allocated buffer
//
int MXInitXBufSingle(int gpu_id, int sync_steps, void** xbuf_ptr, void* hndl_ptr) {
    const int buffer_size = sync_steps*gpuipc::SINGLE_SYNC_BUFFER_BYTES;
    uint8_t *data;
    cudaCheckErrors("MXInitXBufSingle: init");

    cudaSetDevice(gpu_id);
    cudaCheckErrors("MXInitXBufSingle: set device");

    for (int j=0; j < sync_steps; ++j) {
        cudaDeviceEnablePeerAccess(gpu_id^(1 << j), 0);
        cudaCheckErrors("MXInitXBufSingle: enable p2p");
    }

    cudaMalloc(&data, buffer_size);
    cudaMemset(data, 0, buffer_size);
    cudaCheckErrors("MXInitXBufSingle: malloc/memset");

    cudaIpcMemHandle_t my_handle;
    cudaIpcGetMemHandle(&my_handle, data);
    cudaCheckErrors("MXInitXBufSingle: get handle");

    xbuf_ptr[gpu_id] = data;
    memcpy(hndl_ptr, (unsigned char *)(&my_handle), sizeof(my_handle));

    return 0;
}


// Initialize IPC exchange buffers from IPC handles list.
// This function would be used to initialize exchange buffers for group BN in case of a
// GPU per process. For any GPU [i] that is expected to communicate with current GPU, will
// open IPC handle from hndl_ptr[i] and store a pointer to local reflected buffer to xbuf_ptr[i].
// This function concludes the process of establishing exchange buffers in process per GPU
// environment after this function xbuf_ptr will obtain buffers for exchange with other GPUs
// similarly to the condition after calling MXInitXBuf in case of a single process.
//
// input :
//         gpu_id - id of gpu used by the process
//         gpus - number of gpus used by the process (not the group size)
//         sync_steps - number of group syncronization steps. Effectively = log2(group_size)
//         hndl_ptr - array of pointer to cudaIpcMemHandle_t buffers received from other GPUs
//
// output :
//         xbuf_ptr - a preallocated array of void_ptr is filled in with buffer pointers
//                    such that xbuf_ptr[i] would contain the pointer for buffer allocated on GPU i.
//                    All other entries remain unchanged
//
int MXOpenIpcHandles(int gpu_id, int gpus, int sync_steps, void** xbuf_ptr, void** hndl_ptr) {
    for (int i=0; i < gpus; ++i) {
        for (int j=0; j < sync_steps; ++j) {
            if (i == (gpu_id^(1 << j))) {
                uint8_t *data;
                cudaIpcMemHandle_t* my_handle = (reinterpret_cast<cudaIpcMemHandle_t*>(hndl_ptr))+i;

                cudaIpcOpenMemHandle(reinterpret_cast<void **>(&data),
                                     *my_handle, cudaIpcMemLazyEnablePeerAccess);
                cudaCheckErrors("MXOpenIpcHandles: ipc open failed");
                xbuf_ptr[i] = data;
            }
        }
    }
    return 0;
}
