#include"kernel.h"
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_cuda_mul_plain(cvalue_ptr a, cvalue_ptr b, value_ptr res, size_t M, size_t N, size_t K){
    size_t x = threadIdx.y + blockDim.x * blockIdx.x;
    size_t y = threadIdx.x + blockDim.y * blockIdx.y;

    if(x < M && y < N){
        value_type val = 0;
        for(int i = 0; i < K; i++)
            val += a[x * K + i] * b[i * N + y];
        res[x * N + y] = val;
    }
}

__global__ void kernel_cuda_mul_trans(cvalue_ptr a, cvalue_ptr b, value_ptr res, size_t M, size_t N, size_t K){
    size_t x = threadIdx.y + blockDim.x * blockIdx.x;
    size_t y = threadIdx.x + blockDim.y * blockIdx.y;

    if(x < M && y < N){
        value_type val = 0;
        for(int i = 0; i < K; i++)
            val += a[x * K + i] * b[y * K + i];
        res[x * N + y] = val;
    }
}

__global__ void kernel_cuda_mul_shared(cvalue_ptr a, cvalue_ptr b, value_ptr res, size_t M, size_t N, size_t K){
    const size_t SIZ = 32;
    __shared__ value_type shared_a[SIZ][SIZ];
    __shared__ value_type shared_b[SIZ][SIZ];

    size_t px = threadIdx.y;
    size_t py = threadIdx.x;
    size_t x = px + blockIdx.x * SIZ;
    size_t y = py + blockIdx.y * SIZ;
    
    value_type val = 0;;
    for(size_t bk = 0; bk < K; bk += SIZ){
        __syncthreads();

        shared_a[px][py] = a[x * K + bk + py];
        shared_b[px][py] = b[(bk + px) * N + y];

        __syncthreads();
        if(x < M && y < N){
            size_t k_ed = min(SIZ, K - bk);
            for(size_t k = 0; k < k_ed; ++k)
                val += shared_a[px][k] * shared_b[k][py];
        }
    }
    if(x < M && y < N){
        res[x * N + y] = val;
    }
}

__global__ void kernel_cuda_mul_reg(cvalue_ptr a, cvalue_ptr b, value_ptr res, size_t M, size_t N, size_t K){
    const size_t shared_len = 32, SIZ = 32, BS = 4;
    __shared__ value_type shared_a[shared_len][SIZ * BS];
    __shared__ value_type shared_b[shared_len][SIZ * BS];

    size_t py = threadIdx.x * BS;
    size_t px = threadIdx.y * BS;
    size_t x = px + blockIdx.x * SIZ * BS;
    size_t y = py + blockIdx.y * SIZ * BS;

    value_type res_buf[BS][BS] = {{0}};
    value_type a_buf[BS], b_buf[BS];
    size_t bk, k_ed, k_id, km;
    for(bk = 0; bk < K; bk += shared_len){
        k_ed = min(bk + shared_len, K);
        km = k_ed - bk;
        //load
        k_id = threadIdx.x;
        if(k_id < km){
            shared_a[k_id][px] = a[(x)* K + bk + k_id];
            shared_a[k_id][px + 1] = a[(x + 1) * K + bk + k_id];
            shared_a[k_id][px + 2] = a[(x + 2) * K + bk + k_id];
            shared_a[k_id][px + 3] = a[(x + 3) * K + bk + k_id];
        }

        k_id = threadIdx.y;
        if(k_id < km){
            shared_b[k_id][py] = b[(bk + k_id) * N + y];
            shared_b[k_id][py + 1] = b[(bk + k_id) * N + y + 1];
            shared_b[k_id][py + 2] = b[(bk + k_id) * N + y + 2];
            shared_b[k_id][py + 3] = b[(bk + k_id) * N + y + 3];
        }

        //calc
        __syncthreads();
        for(size_t k = 0; k < km; ++k){
            a_buf[0] = shared_a[k][px];
            a_buf[1] = shared_a[k][px + 1];
            a_buf[2] = shared_a[k][px + 2];
            a_buf[3] = shared_a[k][px + 3];

            b_buf[0] = shared_b[k][py];
            b_buf[1] = shared_b[k][py + 1];
            b_buf[2] = shared_b[k][py + 2];
            b_buf[3] = shared_b[k][py + 3];

            for(int i = 0; i < BS; ++i){
                for(int j = 0; j < BS; ++j){
                    res_buf[i][j] += a_buf[i] * b_buf[j];
                }
            }
        }
        __syncthreads();
    }

    size_t mx = min(x + BS, M), my = min(y + BS, N);
    for(size_t i = x; i < mx; ++i){
        for(size_t j = y; j < my; ++j){
            res[i * N + j] = res_buf[i - x][j - y]; 
        }
    }
    __syncthreads();
}

__global__ void kernel_cuda_mul_tile(cvalue_ptr __restrict__ a, cvalue_ptr __restrict__ b, value_ptr __restrict__ res, 
                                    size_t M, size_t N, size_t K){
    const size_t shared_len = 32, SIZ = 32, BS = 4;
    __shared__ value_type shared_a[shared_len][SIZ * BS];
    __shared__ value_type shared_b[shared_len][SIZ * BS];

    size_t py = threadIdx.x * BS;
    size_t px = threadIdx.y * BS;
    size_t x = px + blockIdx.x * SIZ * BS;
    size_t y = py + blockIdx.y * SIZ * BS;

    value_type res_buf[BS][BS] = {{0}};
    value_type a_buf[BS], b_buf[BS];
    size_t bk, k_ed, k_id, km;
    for(bk = 0; bk < K; bk += shared_len){
        k_ed = min(bk + shared_len, K);
        km = k_ed - bk;
        //load
        k_id = threadIdx.x;
        if(k_id < km){
            shared_a[k_id][px] = a[(x)* K + bk + k_id];
            shared_a[k_id][px + 1] = a[(x + 1) * K + bk + k_id];
            shared_a[k_id][px + 2] = a[(x + 2) * K + bk + k_id];
            shared_a[k_id][px + 3] = a[(x + 3) * K + bk + k_id];
        }

        k_id = threadIdx.y;
        if(k_id < km){
            shared_b[k_id][py] = b[y * K + (bk + k_id)];
            shared_b[k_id][py + 1] = b[(y + 1) * K + (bk + k_id)];
            shared_b[k_id][py + 2] = b[(y + 2) * K + bk + k_id];
            shared_b[k_id][py + 3] = b[(y + 3) * K + bk + k_id];
        }

        //calc
        __syncthreads();
        for(size_t k = 0; k < km; ++k){
            a_buf[0] = shared_a[k][px];
            a_buf[1] = shared_a[k][px + 1];
            a_buf[2] = shared_a[k][px + 2];
            a_buf[3] = shared_a[k][px + 3];

            b_buf[0] = shared_b[k][py];
            b_buf[1] = shared_b[k][py + 1];
            b_buf[2] = shared_b[k][py + 2];
            b_buf[3] = shared_b[k][py + 3];

            for(int i = 0; i < BS; ++i){
                for(int j = 0; j < BS; ++j){
                    res_buf[i][j] += a_buf[i] * b_buf[j];
                }
            }
        }
        __syncthreads();
    }

    size_t mx = min(x + BS, M), my = min(y + BS, N);
    for(size_t i = x; i < mx; ++i){
        for(size_t j = y; j < my; ++j){
            res[i * N + j] = res_buf[i - x][j - y]; 
        }
    }
    // __syncthreads();
}

__global__ void kernel_cuda_mul_tile4(cvalue_ptr __restrict__ a, cvalue_ptr __restrict__ b, float4* __restrict__ res, 
                                    size_t M, size_t N, size_t K){
    const size_t shared_len = 32, SIZ = 32, BS = 4;
    __shared__ float4 shared_a[shared_len][SIZ];
    __shared__ float4 shared_b[shared_len][SIZ];

    size_t py = threadIdx.x;
    size_t px = threadIdx.y;
    size_t x = px * BS + blockIdx.x * SIZ * BS;
    size_t y = py * BS + blockIdx.y * SIZ * BS;

    float4 zero = make_float4(0, 0, 0, 0);
    float4 res_buf[BS] = {zero, zero, zero, zero};
    float4 a_buf, b_buf;
    for(size_t bk = 0; bk < K; bk += shared_len){
        shared_a[py][px] = 
            make_float4(a[(x)* K + bk + py], 
                        a[(x + 1)* K + bk + py],
                        a[(x + 2)* K + bk + py],
                        a[(x + 3)* K + bk + py]);

        shared_b[px][py] = 
            make_float4(b[(y)* K + bk + px], 
                        b[(y + 1)* K + bk + px],
                        b[(y + 2)* K + bk + px],
                        b[(y + 3)* K + bk + px]);

        __syncthreads();
        for(size_t k = 0; k < shared_len; ++k){
            a_buf = shared_a[k][px];
            b_buf = shared_b[k][py];

#define _float4_mul(A, B)\
            res_buf[A].x += a_buf.B * b_buf.x;\
            res_buf[A].y += a_buf.B * b_buf.y;\
            res_buf[A].z += a_buf.B * b_buf.z;\
            res_buf[A].w += a_buf.B * b_buf.w;

            _float4_mul(0, x);
            _float4_mul(1, y);
            _float4_mul(2, z);
            _float4_mul(3, w);
            
#undef _float4_mul

        }
        __syncthreads();
    }

    for(size_t i = 0; i < BS; ++i){
        res[((i + x) * N + y) / BS] = res_buf[i];
    }
    // __syncthreads();
}

__global__ void kernel_cuda_mul_prefetch(cvalue_ptr a, cvalue_ptr b, float4* res, 
                                    size_t M, size_t N, size_t K){
    const size_t shared_len = 32, SIZ = 16, BS = 4;
    __shared__ float4 shared_a[2][shared_len][SIZ];
    __shared__ float4 shared_b[2][shared_len][SIZ];

    size_t py = threadIdx.x;
    size_t px = threadIdx.y;
    size_t x = px * BS + blockIdx.x * SIZ * BS;
    size_t y = py * BS + blockIdx.y * SIZ * BS;

    int page = 0, idy = py * 2, idx = px * 2;

    float4 zero = make_float4(0, 0, 0, 0);
    float4 res_buf[BS] = {zero, zero, zero, zero};
    float4 a_buf, b_buf;

    shared_a[page][idy][px] = 
            make_float4(a[(x)* K + 0 + idy], 
                        a[(x + 1)* K + 0 + idy],
                        a[(x + 2)* K + 0 + idy],
                        a[(x + 3)* K + 0 + idy]);
    shared_a[page][idy + 1][px] = 
            make_float4(a[(x)* K + 0 + idy + 1], 
                        a[(x + 1)* K + 0 + idy + 1],
                        a[(x + 2)* K + 0 + idy + 1],
                        a[(x + 3)* K + 0 + idy + 1]);

    shared_b[page][idx][py] = 
        make_float4(b[(y)* K + 0 + idx], 
                    b[(y + 1)* K + 0 + idx],
                    b[(y + 2)* K + 0 + idx],
                    b[(y + 3)* K + 0 + idx]);
    
    shared_b[page][idx + 1][py] = 
        make_float4(b[(y)* K + 0 + idx + 1], 
                    b[(y + 1)* K + 0 + idx + 1],
                    b[(y + 2)* K + 0 + idx + 1],
                    b[(y + 3)* K + 0 + idx + 1]);

    for(size_t bk = 0; bk < K;){
        __syncthreads();
        for(size_t k = 0; k < shared_len; ++k){
            a_buf = shared_a[page][k][px];
            b_buf = shared_b[page][k][py];

#define _float4_mul(A, B)\
            res_buf[A].x += a_buf.B * b_buf.x;\
            res_buf[A].y += a_buf.B * b_buf.y;\
            res_buf[A].z += a_buf.B * b_buf.z;\
            res_buf[A].w += a_buf.B * b_buf.w;

            _float4_mul(0, x);
            _float4_mul(1, y);
            _float4_mul(2, z);
            _float4_mul(3, w);
            
#undef _float4_mul

        }

        bk += shared_len;
        page ^= 1;

        if(bk < K){
            shared_a[page][idy][px] = 
                make_float4(a[(x)* K + bk + idy], 
                            a[(x + 1)* K + bk + idy],
                            a[(x + 2)* K + bk + idy],
                            a[(x + 3)* K + bk + idy]);
            shared_a[page][idy + 1][px] = 
                make_float4(a[(x)* K + bk + idy + 1], 
                            a[(x + 1)* K + bk + idy + 1],
                            a[(x + 2)* K + bk + idy + 1],
                            a[(x + 3)* K + bk + idy + 1]);

            shared_b[page][idx][py] = 
                make_float4(b[(y)* K + bk + idx], 
                            b[(y + 1)* K + bk + idx],
                            b[(y + 2)* K + bk + idx],
                            b[(y + 3)* K + bk + idx]);
            
            shared_b[page][idx + 1][py] = 
                make_float4(b[(y)* K + bk + idx + 1], 
                            b[(y + 1)* K + bk + idx + 1],
                            b[(y + 2)* K + bk + idx + 1],
                            b[(y + 3)* K + bk + idx + 1]);
        }
    }

    for(size_t i = 0; i < BS; ++i){
        res[((i + x) * N + y) / BS] = res_buf[i];
    }
}

C_FIELD_BEGIN

int cuda_device_cnt = 0;

#define _CUDA_MATMUL_PRE\
    value_ptr device_a, device_b, device_res;\
    cudaMalloc(&device_a, M * K * sizeof(value_type));\
    cudaMalloc(&device_b, K * N * sizeof(value_type));\
    cudaMalloc(&device_res, M * N * sizeof(value_type));\
    cudaMemcpy(device_a, a->data, M * K * sizeof(value_type), cudaMemcpyHostToDevice);\
    cudaMemcpy(device_b, b->data, K * N * sizeof(value_type), cudaMemcpyHostToDevice)

#define _CUDA_MATMUL_FIN\
    cudaMemcpy(res->data, device_res, M * N * sizeof(value_type), cudaMemcpyDeviceToHost);\
    cudaFree(device_a);\
    cudaFree(device_b);\
    cudaFree(device_res);

mat_ptr mat_mul_cuda_plain(const mat_ptr a, const mat_ptr b){
    _MAT_MUL_PRE;

    size_t M = a->rows;
    size_t K = a->cols;
    size_t N = b->cols;

    _REQUIRE(cuda_device_cnt > 0, NULL);
    _CUDA_MATMUL_PRE;

    const size_t SIZ = 32;

    dim3 block((M + SIZ - 1) / SIZ, (N + SIZ - 1) / SIZ);
    dim3 thread(SIZ, SIZ);

    kernel_cuda_mul_plain<<<block, thread>>>
        ((value_ptr)device_a, (value_ptr)device_b, (value_ptr)device_res, M, N, K);

    _CUDA_MATMUL_FIN;

    return res;
}

mat_ptr mat_mul_cuda_shared(const mat_ptr a, const mat_ptr b){
    _MAT_MUL_PRE;

    size_t M = a->rows;
    size_t K = a->cols;
    size_t N = b->cols;

    _REQUIRE(cuda_device_cnt > 0, NULL);
    _CUDA_MATMUL_PRE;

    const size_t SIZ = 32;

    dim3 block((M + SIZ - 1) / SIZ, (N + SIZ - 1) / SIZ);
    dim3 thread(SIZ, SIZ);

    kernel_cuda_mul_shared<<<block, thread>>>
        (device_a, device_b, device_res, M, N, K);

    _CUDA_MATMUL_FIN;

    return res;
}

mat_ptr mat_mul_cuda_reg(const mat_ptr a, const mat_ptr b){
    _MAT_MUL_PRE;

    size_t M = a->rows;
    size_t K = a->cols;
    size_t N = b->cols;

    _REQUIRE(cuda_device_cnt > 0, NULL);
    _CUDA_MATMUL_PRE;

    const size_t SIZ = 32, TSIZ = 128;

    dim3 block((M + TSIZ - 1) / TSIZ, (N + TSIZ - 1) / TSIZ);
    dim3 threads(SIZ, SIZ);

    kernel_cuda_mul_reg<<<block, threads>>>
        ((value_ptr)device_a, (value_ptr)device_b, (value_ptr)device_res, M, N, K);
    _CUDA_MATMUL_FIN;

    return res;
}

mat_ptr mat_mul_cuda_multi(const mat_ptr a, const mat_ptr b){
    _MAT_MUL_PRE;
    if(cuda_device_cnt < 4){
        mat_free(res);
        ERR_INFO("Require no less than 4 GPU.");
        return NULL;
    }

    size_t M = a->rows;
    size_t K = a->cols;
    size_t N = b->cols;

    const int DEVICE_CNT = 4;

    mat_ptr b_trans = mat_trans_copy(b);
    value_ptr mat_a = a->data, mat_b = b_trans->data, mat_res = res->data;
    
    const size_t A_TILE_LEN = M / DEVICE_CNT, B_TILE_LEN = N / DEVICE_CNT;
    const size_t RES_TILE_SIZ = A_TILE_LEN * B_TILE_LEN;

    #pragma omp parallel for num_threads(DEVICE_CNT)
    for(int dev_id = 0; dev_id < DEVICE_CNT; ++dev_id){
        void *a_buf, *b_buf, *res_buf;
        value_ptr res_host;
        cudaSetDevice(dev_id);

        cudaMalloc(&a_buf, A_TILE_LEN * K * sizeof(value_type));
        cudaMalloc(&b_buf, B_TILE_LEN * K * sizeof(value_type));
        cudaMalloc(&res_buf, RES_TILE_SIZ * sizeof(value_type));

        res_host = mat_aligned_alloc(RES_TILE_SIZ * sizeof(value_type));

        value_ptr b_tile = mat_b + B_TILE_LEN * K * dev_id;
        cudaMemcpy(b_buf, b_tile, B_TILE_LEN * K * sizeof(value_type), cudaMemcpyHostToDevice);

        for(int row_id = 0; row_id < DEVICE_CNT; ++row_id){
            int act_id = (row_id + dev_id) % DEVICE_CNT;

            value_ptr a_tile = mat_a + A_TILE_LEN * K * act_id;
            cudaMemcpy(a_buf, a_tile, A_TILE_LEN * K * sizeof(value_type), cudaMemcpyHostToDevice);

            const int THREADS_CNT = 32, DIV_LEN = THREADS_CNT * DEVICE_CNT * 4;
            dim3 threads(THREADS_CNT, THREADS_CNT);
            dim3 block(M / DIV_LEN, N / DIV_LEN);

            kernel_cuda_mul_tile4<<<block, threads>>>
                ((cvalue_ptr)a_buf, (cvalue_ptr)b_buf, (float4*)res_buf, A_TILE_LEN, B_TILE_LEN, K);

            cudaMemcpy(res_host, res_buf, RES_TILE_SIZ * sizeof(value_type), cudaMemcpyDeviceToHost);
            
            value_ptr res_ptr = mat_res + A_TILE_LEN * N * act_id + B_TILE_LEN * dev_id;

            for(int i = 0; i < A_TILE_LEN; ++i){
                #pragma omp simd
                for(int j = 0; j < B_TILE_LEN; ++j){
                    res_ptr[i * N + j] = res_host[i * B_TILE_LEN + j];
                }
            }
        }

        cudaFree(a_buf);
        cudaFree(b_buf);
        cudaFree(res_buf);
        mat_aligned_free(res_host);
    }

    mat_free(b_trans);

    return res;
}

mat_ptr mat_mul_cuda_prefetch(const mat_ptr a, const mat_ptr b){
    _MAT_MUL_PRE;
    if(cuda_device_cnt < 4){
        mat_free(res);
        ERR_INFO("Require no less than 4 GPU.");
        return NULL;
    }

    size_t M = a->rows;
    size_t K = a->cols;
    size_t N = b->cols;

    const int DEVICE_CNT = 4;

    mat_ptr b_trans = mat_trans_copy(b);
    value_ptr mat_a = a->data, mat_b = b_trans->data, mat_res = res->data;
    
    const size_t A_TILE_LEN = M / DEVICE_CNT, B_TILE_LEN = N / DEVICE_CNT;
    const size_t RES_TILE_SIZ = A_TILE_LEN * B_TILE_LEN;

    #pragma omp parallel for num_threads(DEVICE_CNT)
    for(int dev_id = 0; dev_id < DEVICE_CNT; ++dev_id){
        void *a_buf, *b_buf, *res_buf;
        value_ptr res_host;
        cudaSetDevice(dev_id);

        cudaMalloc(&a_buf, A_TILE_LEN * K * sizeof(value_type));
        cudaMalloc(&b_buf, B_TILE_LEN * K * sizeof(value_type));
        cudaMalloc(&res_buf, RES_TILE_SIZ * sizeof(value_type));

        res_host = mat_aligned_alloc(RES_TILE_SIZ * sizeof(value_type));

        value_ptr b_tile = mat_b + B_TILE_LEN * K * dev_id;
        cudaMemcpy(b_buf, b_tile, B_TILE_LEN * K * sizeof(value_type), cudaMemcpyHostToDevice);

        for(int row_id = 0; row_id < DEVICE_CNT; ++row_id){
            int act_id = (row_id + dev_id) % DEVICE_CNT;

            value_ptr a_tile = mat_a + A_TILE_LEN * K * act_id;
            cudaMemcpy(a_buf, a_tile, A_TILE_LEN * K * sizeof(value_type), cudaMemcpyHostToDevice);

            const int THREADS_CNT = 16, DIV_LEN = THREADS_CNT * DEVICE_CNT * 4;
            dim3 threads(THREADS_CNT, THREADS_CNT);
            dim3 block(M / DIV_LEN, N / DIV_LEN);

            kernel_cuda_mul_prefetch<<<block, threads>>>
                ((cvalue_ptr)a_buf, (cvalue_ptr)b_buf, (float4*)res_buf, A_TILE_LEN, B_TILE_LEN, K);

            cudaMemcpy(res_host, res_buf, RES_TILE_SIZ * sizeof(value_type), cudaMemcpyDeviceToHost);
            
            value_ptr res_ptr = mat_res + A_TILE_LEN * N * act_id + B_TILE_LEN * dev_id;

            for(int i = 0; i < A_TILE_LEN; ++i){
                #pragma omp simd
                for(int j = 0; j < B_TILE_LEN; ++j){
                    res_ptr[i * N + j] = res_host[i * B_TILE_LEN + j];
                }
            }
        }

        cudaFree(a_buf);
        cudaFree(b_buf);
        cudaFree(res_buf);
        mat_aligned_free(res_host);
    }

    mat_free(b_trans);

    return res;
}

void cuda_init(){
    cudaError_t err = cudaGetDeviceCount(&cuda_device_cnt);
    err = cudaSetDevice(3);
    printf("CUDA device detected: %d\n", cuda_device_cnt);
    cudaDeviceProp cuda_device_prop;
    err = cudaGetDeviceProperties(&cuda_device_prop, 0);
    int * grid_siz = cuda_device_prop.maxGridSize;
    printf("block cnt = (%d, %d, %d), thread cnt = %d\n", 
        grid_siz[0], grid_siz[1], grid_siz[2], cuda_device_prop.maxThreadsPerBlock);
}

C_FIELD_END