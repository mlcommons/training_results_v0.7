#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/AccumulateType.h>
#include <THC/THC.h>

#define ASSERT_INT4_ALIGNED(PTR) \
    AT_ASSERTM(is_aligned<int4>(PTR), "Tensor is not int4 aligned")

template<class T>
bool
is_aligned(const void * ptr) noexcept {
    auto iptr = reinterpret_cast<std::uintptr_t>(ptr);
    return !(iptr % alignof(T));
}

/** Each block process TILE_Q*TILE_K*hidden volumn. */
template <int TILE, typename scalar_t, typename accscalar_t, typename outscalar_t>
__global__ void
cunn_AttnScoreForward(
    outscalar_t *output,
    const scalar_t* __restrict__ attn_query,
    const scalar_t* __restrict__ attn_keys,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ linear_attn,
    int t_q,
    int t_k,
    int hidden) {
    
    extern __shared__ unsigned char smem[];
    auto tmp_q = reinterpret_cast<scalar_t*>(smem);
    auto tmp_k = tmp_q + TILE * blockDim.x;
    auto tmp_b = tmp_k + TILE * blockDim.x;
    auto tmp_l = tmp_b + blockDim.x;
    auto tmp_o = reinterpret_cast<accscalar_t*>(tmp_l + blockDim.x);

    int batch_id = blockIdx.x;
    int q_start = blockIdx.y * TILE;
    int k_start = blockIdx.z * TILE;
    
    attn_query += batch_id*t_q*hidden + q_start*hidden;
    attn_keys += batch_id*t_k*hidden + k_start*hidden;
    output += batch_id*t_q*t_k;

    // initialize intermediate result
    #pragma unroll
    for (int i = 0; i < TILE; i++)
        #pragma unroll
        for (int j = 0; j < TILE; j++)
            tmp_o[i*TILE*blockDim.x+j*blockDim.x+threadIdx.x] = 0;

    // ilpReduce
    int offset = threadIdx.x;
    int last = hidden % blockDim.x;

    // ilpReduce on regular data
    for (; offset < hidden - last; offset += blockDim.x) {
        // prolog: load query slices to shared memory
        for (int i = 0; i < t_q - q_start && i < TILE; i++)
            tmp_q[i*blockDim.x+threadIdx.x] = attn_query[i*hidden+offset];

        // prolog: load key slices to shared memory
        for (int i = 0; i < t_k - k_start && i < TILE; i++)
            tmp_k[i*blockDim.x+threadIdx.x] = attn_keys[i*hidden+offset];

        // prolog: load bias and linear_attn slices to shared memory
        tmp_b[threadIdx.x] = bias[offset];
        tmp_l[threadIdx.x] = linear_attn[offset];

        // main loop
        for (int i = 0; i < t_q - q_start && i < TILE; i++) {
            for (int j = 0; j < t_k - k_start && j < TILE; j++) {
                accscalar_t s = static_cast<accscalar_t>(
                    tmp_q[i*blockDim.x+threadIdx.x] +
                    tmp_k[j*blockDim.x+threadIdx.x] +
                    tmp_b[threadIdx.x]);
                tmp_o[i*TILE*blockDim.x+j*blockDim.x+threadIdx.x] += tanhf(s) * tmp_l[threadIdx.x];
            }
        }
    }

    // ilpReduce on boundary
    for (; offset < hidden; offset += blockDim.x) {
        // prolog: load query slices to shared memory
        for (int i = 0; i < t_q - q_start && i < TILE; i++)
            tmp_q[i*blockDim.x+threadIdx.x] = attn_query[i*hidden+offset];

        // prolog: load key slices to shared memory
        for (int i = 0; i < t_k - k_start && i < TILE; i++)
            tmp_k[i*blockDim.x+threadIdx.x] = attn_keys[i*hidden+offset];

        // prolog: load bias and linear_attn slices to shared memory
        tmp_b[threadIdx.x] = bias[offset];
        tmp_l[threadIdx.x] = linear_attn[offset];

        // main loop
        for (int i = 0; i < t_q - q_start && i < TILE; i++) {
            for (int j = 0; j < t_k - k_start && j < TILE; j++) {
                accscalar_t s = static_cast<accscalar_t>(
                    tmp_q[i*blockDim.x+threadIdx.x] +
                    tmp_k[j*blockDim.x+threadIdx.x] +
                    tmp_b[threadIdx.x]);
                tmp_o[i*TILE*blockDim.x+j*blockDim.x+threadIdx.x] += tanhf(s) * tmp_l[threadIdx.x];
            }
        }
    }

    // blockReduce
    __syncthreads();

    // First warp will perform per-warp reductions for the remaining warps
    uint32_t mask = (((uint64_t)1) << (blockDim.x / 32)) - 1;
    if (threadIdx.x < 32) {
        int lane = threadIdx.x % 32;
        if (lane < blockDim.x / 32) {
            for (int i = 0; i < t_q - q_start && i < TILE; i++) {
                for (int j = 0; j < t_k - k_start && j < TILE; j++) {
                    accscalar_t warpVal = static_cast<accscalar_t>(0);
                    #pragma unroll
                    for (int k = 0; k < 32; ++k) {
                        warpVal += tmp_o[i*TILE*blockDim.x+j*blockDim.x+lane*32+k];
                    }
                    __syncwarp(mask);
                    tmp_o[i*TILE*blockDim.x+j*blockDim.x+lane] = warpVal;
                }
            }
        }
    }

    __syncthreads();

    // First thread will perform a reduction of the above per-warp reductions
    if (threadIdx.x == 0) {
        for (int i = 0; i < t_q - q_start && i < TILE; i++) {
            for (int j = 0; j < t_k - k_start && j < TILE; j++) {
                accscalar_t blockVal = static_cast<accscalar_t>(0);
                for (int k = 0; k < blockDim.x / 32; ++k) {
                    blockVal += tmp_o[i*TILE*blockDim.x+j*blockDim.x+k];
                }
                output[(i+q_start)*t_k+(j+k_start)] = static_cast<outscalar_t>(blockVal);
            }
        }
    }

    // Sync and broadcast
    __syncthreads();
}

at::Tensor attn_score_forward_cuda(
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &bias,
    const at::Tensor &linear_attn) {
    int batch_sz = attn_query.size(0);
    int t_q = attn_query.size(1);
    int t_k = attn_keys.size(1);
    int hidden = attn_query.size(2);

    at::Tensor output = at::empty({batch_sz, t_q, t_k}, attn_query.options());

    const int TILE = 4;
    int grid_x = batch_sz;
    int grid_y = (t_q + TILE - 1) / TILE;
    int grid_z = (t_k + TILE - 1) / TILE;

    // Each block process TILE_Q*TILE_K*hidden volumn. 
    dim3 block(128);
    dim3 grid(grid_x, grid_y, grid_z);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Each block load (TILE_Q+TILE_K)*block.x volumn each time
    // Each block load block.x volumn bias and linear_attn
    // Each thread reserve its local results for intra block reduction
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn_query.scalar_type(), "attn_score_fprop", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        cunn_AttnScoreForward<TILE, scalar_t, accscalar_t, scalar_t>
        <<<grid, block, (2*TILE+2)*block.x * sizeof(scalar_t)+
            block.x * TILE * TILE * sizeof(accscalar_t), stream>>>(
            output.data_ptr<scalar_t>(), attn_query.data_ptr<scalar_t>(),
            attn_keys.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
            linear_attn.data_ptr<scalar_t>(), t_q, t_k, hidden
        );
    });

    THCudaCheck(cudaGetLastError());
	return output;
}

// Extends cuda/include/vector_types.h
struct __builtin_align__(16) float8 {
    float x0, x1, x2, x3, x4, x5, x6, x7;
};
typedef struct float8 float8;

// Extends torch/include/ATen/AccumulateType.h
template <typename T, typename U>
struct VectorType {};

#if defined(__CUDACC__) || defined(__HIPCC__)
template <> struct VectorType<half, float> { using type = float8;  };
#endif
template <> struct VectorType<at::Half, float> { using type = float8;  };
template <> struct VectorType<float, float>    { using type = float4;  };
template <> struct VectorType<double, double>  { using type = double2; };

template<typename T, typename U>
using vec_type = typename VectorType<T, U>::type;

// Convert int4 data to corresponding to vector type
void __device__ __inline__ int4ToVector(float8 *dst, int4 *src) {
    at::Half *src_t = reinterpret_cast<at::Half *>(src);
    dst->x0 = static_cast<float>(src_t[0]);
    dst->x1 = static_cast<float>(src_t[1]);
    dst->x2 = static_cast<float>(src_t[2]);
    dst->x3 = static_cast<float>(src_t[3]);
    dst->x4 = static_cast<float>(src_t[4]);
    dst->x5 = static_cast<float>(src_t[5]);
    dst->x6 = static_cast<float>(src_t[6]);
    dst->x7 = static_cast<float>(src_t[7]);
}
void __device__ __inline__ int4ToVector(float4 *dst, int4 *src) {
    float4 *src_t = reinterpret_cast<float4 *>(src);
    *dst = *src_t;
}
void __device__ __inline__ int4ToVector(double2 *dst, int4 *src) {
    double2 *src_t = reinterpret_cast<double2 *>(src);
    *dst = *src_t;
}

// Convert vector type to int4
void __device__ __inline__ vectorToInt4(int4 *dst, float8 *src) {
    at::Half *dst_t = reinterpret_cast<at::Half *>(dst);
    dst_t[0] = static_cast<at::Half>(src->x0);
    dst_t[1] = static_cast<at::Half>(src->x1);
    dst_t[2] = static_cast<at::Half>(src->x2);
    dst_t[3] = static_cast<at::Half>(src->x3);
    dst_t[4] = static_cast<at::Half>(src->x4);
    dst_t[5] = static_cast<at::Half>(src->x5);
    dst_t[6] = static_cast<at::Half>(src->x6);
    dst_t[7] = static_cast<at::Half>(src->x7);
}
void __device__ __inline__ vectorToInt4(int4 *dst, float4 *src) {
    int4 *src_t = reinterpret_cast<int4 *>(src);
    *dst = *src_t;
}
void __device__ __inline__ vectorToInt4(int4 *dst, double2 *src) {
    int4 *src_t = reinterpret_cast<int4 *>(src);
    *dst = *src_t;
}

/**
 * Each block process BZ*t_q*t_k*LEN volumn.
 */
template <int THREADS, int ILP, int LEN, int TILE, int BZ, typename scalar_t, typename accscalar_t, typename vector_t, typename outscalar_t>
__global__ void
cunn_AttnScoreBackward(
    outscalar_t *grad_query,
    outscalar_t *grad_key,
    outscalar_t *grad_biases,
    outscalar_t *grad_lins,
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ attn_query,
    const scalar_t* __restrict__ attn_key,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ linear_attn,
    int batch_sz,
    int t_q,
    int t_k,
    int hidden) {

    // common parameter check
    static_assert((LEN > 1) & !(LEN & (LEN - 1)), "LEN should be power of 2 for faster mod.");
    static_assert((TILE > 1) & !(TILE & (TILE - 1)), "TILE should be power of 2 for faster round down.");
    static_assert((LEN/ILP > 1) & !(LEN/ILP & (LEN/ILP - 1)), "LEN/ILP should be power of 2 for faster mod.");
    static_assert(TILE*TILE*LEN/ILP%THREADS == 0, "Tailing of tile is not expected.");
    static_assert(TILE*LEN == ILP*THREADS, "Expect threads process a 2D slice of one TILE each time for better performance.");
    static_assert(TILE % ILP == 0, "Expect gradients w.r.t. output can use int4.");

    // calculate rounded up/down bounday
    int t_kd = t_k & ~(TILE - 1);
    int t_qu = (t_q + TILE - 1) / TILE * TILE;
    int t_ku = (t_k + TILE - 1) / TILE * TILE;

    // assign shared memory address
    // keep input key as scalar_t to reduce shared memory usage
    extern __shared__ unsigned char smem[];
    auto tmp_qk = reinterpret_cast<accscalar_t*>(smem);
    auto tmp_gk = tmp_qk + TILE * LEN;
    auto tmp_k = reinterpret_cast<scalar_t*>(tmp_gk + t_ku * LEN);

    // calculate hidden start and batch start
    int tid = threadIdx.x;
    int h_start = blockIdx.x % (hidden / LEN) * LEN;
    int n_start = blockIdx.x / (hidden / LEN) * BZ;
    int h_offset = (tid & (LEN / ILP - 1)) * ILP;

    // update pointers with offset
    grad_output += n_start * t_q * t_k;
    attn_query  += h_start + n_start * t_q * hidden;
    attn_key    += h_start + n_start * t_k * hidden;
    bias        += h_start;
    linear_attn += h_start;
    grad_query  += h_start + n_start * t_q * hidden;
    grad_key    += h_start + n_start * t_k * hidden;
    grad_biases += blockIdx.x * LEN;
    grad_lins   += blockIdx.x * LEN;

    // load bias and linear_attn volume to registers
    // assume one thread process the same hidden id
    static_assert(THREADS % (LEN / ILP) == 0, "Expect one thread process the same hidden index.");
    vector_t tmp_b, tmp_l;
    int4ToVector(&tmp_b, (int4*)(&bias[h_offset]));
    int4ToVector(&tmp_l, (int4*)(&linear_attn[h_offset]));

    // initialize bias and linear_attn gradients to zero
    vector_t tmp_gb = {0}, tmp_gl = {0};

    for (int n=0; n<BZ && n<(batch_sz-n_start); n++) {
        // initialize gradients of key to zero
        // load batch specific key to shared memory
        for (int i=tid*ILP; i<t_kd*LEN; i+=THREADS*ILP) {
            *(int4*)&tmp_k[i] = *(int4*)&attn_key[i/LEN*hidden + (i&(LEN-1))];
            *(vector_t*)&tmp_gk[i] = {0};
        }
        for (int i=t_kd*LEN+tid*ILP; i<t_ku*LEN; i+=THREADS*ILP) {
            if (i/LEN >= t_k)
                *(int4*)&tmp_k[i] = {0};
            else
                *(int4*)&tmp_k[i] = *(int4*)&attn_key[i/LEN*hidden + (i&(LEN-1))];
            *(vector_t*)&tmp_gk[i] = {0};
        }
        __syncthreads();
         
        // loop each tile along query dimension
        for (int tile_q=0; tile_q<t_qu; tile_q+=TILE) {
            // load per thread query of shape ILP to registers
            // initialize gradients of query to zero
            int q_id = tile_q + tid / (LEN / ILP);
            vector_t tmp_q = {0}, tmp_gq = {0};
            if (q_id < t_q)
                int4ToVector(&tmp_q, (int4*)&attn_query[q_id*hidden + h_offset]);

            // loop each tile along key dimension
            for (int tile_k=0; tile_k<t_ku; tile_k+=TILE) {
                // load per thread g_o of shape TILE to registers
                accscalar_t tmp_go[TILE] = {0};
                if (q_id < t_q) {
                    const scalar_t *grad_o = grad_output + q_id * t_k + tile_k;
                    if (tile_k < t_kd) {
                        #pragma unroll
                        for (int i=0; i<TILE/ILP; i++) {
                            int4ToVector(&((vector_t *)tmp_go)[i],
                                (int4*)&grad_o[i*ILP]);
                        }
                    } else {
                        for (int i=0; i<t_k-t_kd; i++) {
                            tmp_go[i] = static_cast<accscalar_t>(grad_o[i]);
                        }
                    }
                }
                __syncthreads();

                // loop each TILE_Q * LEN slice along key dimension
                for (int k=tile_k; k<tile_k+TILE; k++) {
                    // load per thread k and g_k to registers
                    vector_t tmp_k_r;
                    int idx = k * LEN + h_offset;
                    int4ToVector(&tmp_k_r, (int4*)&tmp_k[idx]);
                 
                    accscalar_t t;
                    vector_t g_qk = {0};
                    #pragma unroll
                    for (int i=0; i<ILP; i++) {
                        t = *((accscalar_t *)(&tmp_q)+i) +
                            *((accscalar_t *)(&tmp_k_r)+i) +
                            *((accscalar_t *)(&tmp_b)+i);
                        t = tanhf(t);
                        *((accscalar_t *)(&tmp_gl)+i) += t * tmp_go[k - tile_k];
                        t = *((accscalar_t *)(&tmp_l)+i) * tmp_go[k - tile_k] *
                            (1.f - t * t);
                        *((accscalar_t *)(&tmp_gq)+i) += t;
                        *((accscalar_t *)(&g_qk)+i) = t;
                    }

                    ((vector_t*)tmp_qk)[tid] = g_qk;
                    __syncthreads();

                    // reduce gradients of key, TILE*LEN == THREADS*ILP
                    t = 0;
                    #pragma unroll
                    for (int i=0; i<ILP; i++) {
                        t += tmp_qk[tid + THREADS*i];
                    }
                    tmp_qk[tid] = t;
                    __syncthreads();
                    if (LEN <= 512 && THREADS >= 1024 && tid < 512)
                        tmp_qk[tid] += tmp_qk[tid + 512];
                    __syncthreads();
                    if (LEN <= 256 && THREADS >= 512 && tid < 256)
                        tmp_qk[tid] += tmp_qk[tid + 256];
                    __syncthreads();
                    if (LEN <= 128 && THREADS >= 256 && tid < 128)
                        tmp_qk[tid] += tmp_qk[tid + 128];
                    __syncthreads();
                    if (LEN <= 64 && THREADS >= 128 && tid < 64)
                        tmp_qk[tid] += tmp_qk[tid + 64];
                    __syncthreads();
                    if (LEN <= 32 && tid < 32) {
                        accscalar_t t;
                        #pragma unroll
                        for (int m=32; m>=LEN; m>>=1) {
                            t = tmp_qk[tid] + tmp_qk[tid + m];
                            __syncwarp();
                            tmp_qk[tid] = t;
                        }
                    }
                    __syncthreads();
                    if (tid < LEN) {
                        tmp_gk[k * LEN + tid] += tmp_qk[tid];
                    }
                    __syncthreads();
                }
            }

            // store g_q to global memory
            // accumulate partial g_b using g_q
            if (q_id < t_q) {
                vectorToInt4((int4*)&grad_query[q_id*hidden + h_offset], &tmp_gq);
                #pragma unroll
                for (int i=0; i<ILP; i++) {
                    *((accscalar_t *)(&tmp_gb)+i) += *((accscalar_t *)(&tmp_gq)+i);
                }
            }
        }

        // store g_k to global memory
        for (int i=tid*ILP; i<t_k*LEN; i+=THREADS*ILP) {
            vectorToInt4((int4*)&grad_key[i/LEN*hidden + (i&(LEN-1))],
                (vector_t*)&tmp_gk[i]);
        }

        // update pointer for next batch
        grad_output += t_q * t_k;
        grad_query  += t_q * hidden;
        grad_key    += t_k * hidden;
        attn_query  += t_q * hidden;
        attn_key    += t_k * hidden;
    }

    // reduce partial g_b, g_l
    auto smem_gb = reinterpret_cast<accscalar_t*>(smem);
    auto smem_gl = smem_gb + THREADS * ILP;

    *(vector_t*)&smem_gb[tid * ILP] = tmp_gb;
    *(vector_t*)&smem_gl[tid * ILP] = tmp_gl;
    __syncthreads();

    accscalar_t s = 0, t = 0;
    #pragma unroll
    for (int i=0; i<ILP; i++) {
        s += smem_gb[tid + THREADS*i];
        t += smem_gl[tid + THREADS*i];
    }
    smem_gb[tid] = s;
    smem_gl[tid] = t;
    __syncthreads();
    if (LEN <= 512 && THREADS >= 1024 && tid < 512) {
        smem_gb[tid] += smem_gb[tid + 512];
        smem_gl[tid] += smem_gl[tid + 512];
    }
    __syncthreads();
    if (LEN <= 256 && THREADS >= 512 && tid < 256) {
        smem_gb[tid] += smem_gb[tid + 256];
        smem_gl[tid] += smem_gl[tid + 256];
    }
    __syncthreads();
    if (LEN <= 128 && THREADS >= 256 && tid < 128) {
        smem_gb[tid] += smem_gb[tid + 128];
        smem_gl[tid] += smem_gl[tid + 128];
    }
    __syncthreads();
    if (LEN <= 64 && THREADS >= 128 && tid < 64) {
        smem_gb[tid] += smem_gb[tid + 64];
        smem_gl[tid] += smem_gl[tid + 64];
    }
    __syncthreads();
    if (LEN <= 32 && tid < 32) {
        #pragma unroll
        for (int m=32; m>=LEN; m>>=1) {
            t = smem_gb[tid] + smem_gb[tid + m];
            s = smem_gl[tid] + smem_gl[tid + m];
            __syncwarp();
            smem_gb[tid] = t;
            smem_gl[tid] = s;
        }
    }
    __syncthreads();

    // store per CTA g_b, g_l to global memory
    if (tid < LEN / ILP) {
        vectorToInt4((int4*)&grad_biases[h_offset], (vector_t*)&smem_gb[h_offset]);
        vectorToInt4((int4*)&grad_lins[h_offset], (vector_t*)&smem_gl[h_offset]);
    }
    __syncthreads();
}

std::vector<at::Tensor> attn_score_backward_cuda(
    const at::Tensor &grad_output,
    const at::Tensor &attn_query,
    const at::Tensor &attn_keys,
    const at::Tensor &bias,
    const at::Tensor &linear_attn) {

    int batch_sz = attn_query.size(0);
    int t_q = attn_query.size(1);
    int t_k = attn_keys.size(1);
    int hidden = attn_query.size(2);

    at::Tensor grad_query = at::empty_like(attn_query);
    at::Tensor grad_keys = at::empty_like(attn_keys);

    const int BZ = 2;
    const int THREADS = 128;
    const int ILP = sizeof(int4) / attn_query.element_size();
    const int len = (t_k <= 80) ? 8 * ILP : 4 * ILP;

    assert(hidden % len == 0);

    // Each CTA process BZ*t_q*t_k*len volume
    // Each thread process 1*1*1*int4 a time
    dim3 block(THREADS);
    dim3 grid(((batch_sz+BZ-1)/BZ) * (hidden/len));

    // Allocate per-CTA buffer for future reduction on bias and linear_attn
    at::Tensor grad_biases = at::empty({grid.x, len}, bias.options());
    at::Tensor grad_lins = at::empty({grid.x, len}, linear_attn.options());

    // Check alignment
    ASSERT_INT4_ALIGNED(grad_query.data_ptr());
    ASSERT_INT4_ALIGNED(grad_keys.data_ptr());
    ASSERT_INT4_ALIGNED(grad_biases.data_ptr());
    ASSERT_INT4_ALIGNED(grad_lins.data_ptr());
    ASSERT_INT4_ALIGNED(grad_output.data_ptr());
    ASSERT_INT4_ALIGNED(attn_query.data_ptr());
    ASSERT_INT4_ALIGNED(attn_keys.data_ptr());
    ASSERT_INT4_ALIGNED(bias.data_ptr());
    ASSERT_INT4_ALIGNED(linear_attn.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (t_k <= 80) {
        const int TILE = 16;
        const int THREADS_PER_LEN = 8;
        const int LEN = THREADS_PER_LEN * ILP;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn_query.scalar_type(), "attn_score_bprop", [&] {
            using accscalar_t = at::acc_type<scalar_t, true>;
            using vector_t = vec_type<scalar_t, accscalar_t>;
            cunn_AttnScoreBackward<THREADS, sizeof(int4) / sizeof(scalar_t),
                THREADS_PER_LEN * sizeof(int4) / sizeof(scalar_t), TILE, BZ,
                scalar_t, accscalar_t, vector_t, scalar_t>
            <<<grid, block, (TILE + (t_k + TILE - 1) / TILE * TILE) * LEN *
                sizeof(accscalar_t) + (t_k + TILE - 1) / TILE * TILE * LEN *
                sizeof(scalar_t), stream>>>(
                grad_query.data_ptr<scalar_t>(), grad_keys.data_ptr<scalar_t>(),
                grad_biases.data_ptr<scalar_t>(), grad_lins.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(), attn_query.data_ptr<scalar_t>(),
                attn_keys.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                linear_attn.data_ptr<scalar_t>(), batch_sz, t_q, t_k, hidden
            );
        });
    } else {
        const int TILE = 32;
        const int THREADS_PER_LEN = 4;
        const int LEN = THREADS_PER_LEN * ILP;
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn_query.scalar_type(), "attn_score_bprop", [&] {
            using accscalar_t = at::acc_type<scalar_t, true>;
            using vector_t = vec_type<scalar_t, accscalar_t>;
            cunn_AttnScoreBackward<THREADS, sizeof(int4) / sizeof(scalar_t),
                THREADS_PER_LEN * sizeof(int4) / sizeof(scalar_t), TILE, BZ,
                scalar_t, accscalar_t, vector_t, scalar_t>
            <<<grid, block, (TILE + (t_k + TILE - 1) / TILE * TILE) * LEN *
                sizeof(accscalar_t) + (t_k + TILE - 1) / TILE * TILE * LEN *
                sizeof(scalar_t), stream>>>(
                grad_query.data_ptr<scalar_t>(), grad_keys.data_ptr<scalar_t>(),
                grad_biases.data_ptr<scalar_t>(), grad_lins.data_ptr<scalar_t>(),
                grad_output.data_ptr<scalar_t>(), attn_query.data_ptr<scalar_t>(),
                attn_keys.data_ptr<scalar_t>(), bias.data_ptr<scalar_t>(),
                linear_attn.data_ptr<scalar_t>(), batch_sz, t_q, t_k, hidden
            );
        });
    }

    // Reduce bias and linear_attn gradients
    at::Tensor grad_bias = at::sum(grad_biases.view({-1, hidden}), 0);
    at::Tensor grad_lin = at::sum(grad_lins.view({-1, hidden}), 0);

    THCudaCheck(cudaGetLastError());
	std::vector<at::Tensor> ret = {grad_query, grad_keys, grad_bias, grad_lin};
	return ret;	
}

