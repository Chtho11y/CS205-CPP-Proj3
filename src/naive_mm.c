#include "kernel.h"
#include "include/cblas.h"
#include<stdlib.h>
#include<string.h>
#include<stdalign.h>
#include<stdio.h>

#include<immintrin.h>
#include<omp.h>

mat_ptr mat_mul_plain(mat_ptr a, mat_ptr b){
    _MAT_MUL_PRE;

    size_t M = a->rows, K = a->cols, N = b->cols;

    for(size_t i = 0; i < M; ++i)
        for(size_t j = 0; j < N; ++j){
            value_type val = 0;
            for(size_t k = 0; k < K; ++k)
                val += a->data[i * K + k] * b->data[k * N + j];
            res->data[i * N + j] = val;
        }
    return res;
}

mat_ptr mat_mul_trans(mat_ptr a, mat_ptr b){
    _MAT_MUL_PRE;

    mat_ptr bt = mat_trans_copy(b);

    if(bt == NULL)
        return NULL;

    size_t M = a->rows, K = a->cols, N = b->cols;

    for(size_t i = 0; i < M; ++i)
        for(size_t j = 0; j < N; ++j){
            value_type val = 0;
            #pragma omp simd reduction(+: val)
            for(size_t k = 0; k < K; ++k)
                val += a->data[i * K + k] * bt->data[j * K + k];
            res->data[i * N + j] = val;
        }
    mat_free(bt);
    
    return res;
}

mat_ptr mat_mul_reorder(mat_ptr a, mat_ptr b){
    _MAT_MUL_PRE;

    size_t M = a->rows, K = a->cols, N = b->cols;

    mat_set(res, 0);

    #pragma omp parallel for
    for(size_t i = 0; i < M; ++i)
        for(size_t k = 0; k < K; ++k)
            for(size_t j = 0; j < N; ++j)
                res->data[i * N + j] += a->data[i * K + k] * b->data[k * N + j];
    
    return res;
}

//kernel 2x4xBS
void mat_mul_store_2x4xN(cvalue_ptr a, const size_t a_step, 
                cvalue_ptr b, const size_t b_step,
                value_ptr res,  const size_t res_step, 
                size_t M, size_t N, size_t K){

#define BS 64

    alignas(_MALLOC_ALIGN) float A[BS][BS];
    alignas(_MALLOC_ALIGN) float B[BS][BS];
    alignas(_MALLOC_ALIGN) float C[BS][BS] = {};

#ifdef _MAT_TIMING

    double reset_cl = 0;
    double kernel_cl = 0;
    double icpy_cl = 0;
    double ocpy_cl = 0;
    double clk = 0;

#endif

    CLK_START;

    #pragma omp parallel for
    for(int i = 0; i < M; ++i){
        value_ptr ptr = res + i * res_step;
        #pragma omp simd
        for(int j = 0; j < N; ++j)
            ptr[j] = 0;
    }

    CLK_END(reset_cl);

    #pragma omp parallel for private(A, B, C)
    for(size_t bi = 0; bi < M; bi += BS){
        for(size_t bk = 0; bk < K; bk += BS){

            CLK_START;
            
            for(int i = 0; i < BS; ++i){
                cvalue_ptr a_ptr  = a + (i + bi) * a_step + bk;
                value_ptr abuf = A[i];
                #pragma omp simd
                for(int k = 0; k < BS; ++k)
                    abuf[k] = a_ptr[k];
            }

            CLK_END(icpy_cl);            

            for(size_t bj = 0; bj < N; bj += BS){

                CLK_START;

                for(int k = 0; k < BS; ++k){
                    cvalue_ptr b_ptr = b + (k + bk) * b_step + bj;
                    value_ptr bbuf = B[k];
                    #pragma omp simd
                    for(size_t j = 0; j < BS; ++j)
                        bbuf[j] = b_ptr[j];
                }

                CLK_END(icpy_cl);

                CLK_START;
                for(int i = 0; i < BS; i += 2){
                    
                    for(int k = 0; k < BS; k += 4){
                        value_type v01 = A[i][k], v02 = A[i][k + 1], 
                                    v03 = A[i][k + 2], v04 = A[i][k + 3];
                        value_type v11 = A[i + 1][k], v12 = A[i + 1][k + 1],
                                    v13 = A[i + 1][k + 2], v14 = A[i + 1][k + 3];
                        #pragma omp simd
                        for(int j = 0; j < BS; ++j){
                            value_type t1 = 0 , t2 = 0;

                            t1 += v01 * B[k][j];
                            t2 += v11 * B[k][j];
                            
                            t1 += v02 * B[k + 1][j];
                            t2 += v12 * B[k + 1][j];

                            t1 += v03 * B[k + 2][j];
                            t2 += v13 * B[k + 2][j];

                            t1 += v04 * B[k + 3][j];
                            t2 += v14 * B[k + 3][j];

                            C[i][j] += t1;
                            C[i + 1][j] += t2;
                        }
                    }
                }
                CLK_END(kernel_cl);

                CLK_START;
                
                for(int i = 0; i < BS; ++i){
                    value_ptr res_ptr = res + (i + bi) * res_step + bj;
                    value_ptr cbuf = C[i];
                    #pragma omp simd
                    for(int j = 0; j < BS; ++j){
                        res_ptr[j] += cbuf[j];
                        cbuf[j] = 0;
                    }
                }

                CLK_END(ocpy_cl);
            }
        }
    }

#ifdef _MAT_TIMING

    printf("reset_cl:%lf\n", reset_cl);
    printf("icpy_cl: %lf\n", icpy_cl);
    printf("ocpy_cl: %lf\n", ocpy_cl);
    printf("kernel_cl: %lf\n", kernel_cl);

#endif
    
#undef BS
}

//3xNx32 for avx2
//3xNx64 for avx512
void kernel_matmul_3xNx4(const value_ptr A, const value_ptr B, value_ptr C, const size_t N){

    const int KN_BS = N - N % 3;
    const int KN_REM = N % 3;
    const int KSIZ = VEC_SIZ * 4;

    //16 ymm registers
    MM_REG c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23, a0, a1, a2, bv;

    //3x32 slices of C
    for(int si = 0; si < KN_BS; si += 3){
        for(int sj = 0; sj < N; sj += KSIZ){
            c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = c20 = c21 = c22 = c23 = MM_ZERO();

            //3xN and Nx32 (32 is unrolled using avx2) slices for A and B
            for(int k = 0; k < N; ++k){
                a0 = MM_SETALL(A[si * N + k]);
                a1 = MM_SETALL(A[(si + 1) * N + k]);
                a2 = MM_SETALL(A[(si + 2) * N + k]);
                bv = MM_LOAD(B + k * N + sj);

                c00 = MM_FMADD(a0, bv, c00);
                c10 = MM_FMADD(a1, bv, c10);
                c20 = MM_FMADD(a2, bv, c20);

                bv = MM_LOAD(B + k * N + sj + VEC_SIZ);

                c01 = MM_FMADD(a0, bv, c01);
                c11 = MM_FMADD(a1, bv, c11);
                c21 = MM_FMADD(a2, bv, c21);
                
                bv = MM_LOAD(B + k * N + sj + 2 * VEC_SIZ);

                c02 = MM_FMADD(a0, bv, c02);
                c12 = MM_FMADD(a1, bv, c12);
                c22 = MM_FMADD(a2, bv, c22);

                bv = MM_LOAD(B + k * N + sj + 3 * VEC_SIZ);

                c03 = MM_FMADD(a0, bv, c03);
                c13 = MM_FMADD(a1, bv, c13);
                c23 = MM_FMADD(a2, bv, c23);
            }

            MM_STORE(C + si * N + sj, c00);
            MM_STORE(C + si * N + sj + VEC_SIZ, c01);
            MM_STORE(C + si * N + sj + 2 * VEC_SIZ, c02);
            MM_STORE(C + si * N + sj + 3 * VEC_SIZ, c03);
            
            MM_STORE(C + (si + 1) * N + sj, c10);
            MM_STORE(C + (si + 1) * N + sj + VEC_SIZ, c11);
            MM_STORE(C + (si + 1) * N + sj + 2 * VEC_SIZ, c12);
            MM_STORE(C + (si + 1) * N + sj + 3 * VEC_SIZ, c13);
            
            MM_STORE(C + (si + 2) * N + sj, c20);
            MM_STORE(C + (si + 2) * N + sj + VEC_SIZ, c21);
            MM_STORE(C + (si + 2) * N + sj + 2 * VEC_SIZ, c22);
            MM_STORE(C + (si + 2) * N + sj + 3 * VEC_SIZ, c23);
        }
    }

    for(int si = KN_BS; si < N; ++si){
        for(int sj = 0; sj < N; sj += KSIZ){
            c00 = c01 = c02 = c03 = MM_ZERO();
            for(int k = 0; k < N; ++k){
                a0 = MM_SETALL(A[si * N + k]);

                bv = MM_LOAD(B + k * N + sj);
                c00 = MM_FMADD(a0, bv, c00);

                bv = MM_LOAD(B + k * N + sj + VEC_SIZ);
                c01 = MM_FMADD(a0, bv, c01);
                
                bv = MM_LOAD(B + k * N + sj + 2 * VEC_SIZ);
                c02 = MM_FMADD(a0, bv, c02);

                bv = MM_LOAD(B + k * N + sj + 3 * VEC_SIZ);
                c03 = MM_FMADD(a0, bv, c03);
            }

            MM_STORE(C + si * N + sj, c00);
            MM_STORE(C + si * N + sj + VEC_SIZ, c01);
            MM_STORE(C + si * N + sj + 2 * VEC_SIZ, c02);
            MM_STORE(C + si * N + sj + 3 * VEC_SIZ, c03);
        }
    }
}

//3xNx32 for avx2
//3xNx64 for avx512
void kernel_matmul_6xNx4(const value_ptr A, const value_ptr B, value_ptr C, const size_t N){

    const int KN_BS = N - N % 6;
    const int KN_REM = N % 6;
    const int KSIZ = VEC_SIZ * 4;

    //32 zmm registers
    MM_REG c00, c01, c02, c03, c10, c11, c12, c13, c20, c21, c22, c23;
    MM_REG c30, c31, c32, c33, c40, c41, c42, c43, c50, c51, c52, c53;
    MM_REG a0, a1, a2, a3, a4, a5;
    MM_REG bv1, bv2;

    //3x32 slices of C.
    for(int si = 0; si < KN_BS; si += 6){
        for(int sj = 0; sj < N; sj += KSIZ){
            c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = c20 = c21 = c22 = c23 = MM_ZERO();
            c30 = c31 = c32 = c33 = c40 = c41 = c42 = c43 = c50 = c51 = c52 = c53 = MM_ZERO();

            //3xN and Nx32 (32 is unrolled using avx2) slices for A and B
            for(int k = 0; k < N; ++k){
                a0 = MM_SETALL(A[si * N + k]);
                a1 = MM_SETALL(A[(si + 1) * N + k]);
                a2 = MM_SETALL(A[(si + 2) * N + k]);
                a3 = MM_SETALL(A[(si + 3) * N + k]);
                a4 = MM_SETALL(A[(si + 4) * N + k]);
                a5 = MM_SETALL(A[(si + 5) * N + k]);

                bv1 = MM_LOAD(B + k * N + sj);

                c00 = MM_FMADD(a0, bv1, c00);
                c10 = MM_FMADD(a1, bv1, c10);
                c20 = MM_FMADD(a2, bv1, c20);
                c30 = MM_FMADD(a3, bv1, c30);
                c40 = MM_FMADD(a4, bv1, c40);
                c50 = MM_FMADD(a5, bv1, c50);

                bv2 = MM_LOAD(B + k * N + sj + VEC_SIZ);

                c01 = MM_FMADD(a0, bv2, c01);
                c11 = MM_FMADD(a1, bv2, c11);
                c21 = MM_FMADD(a2, bv2, c21);
                c31 = MM_FMADD(a3, bv2, c31);
                c41 = MM_FMADD(a4, bv2, c41);
                c51 = MM_FMADD(a5, bv2, c51);
                
                bv1 = MM_LOAD(B + k * N + sj + 2 * VEC_SIZ);

                c02 = MM_FMADD(a0, bv1, c02);
                c12 = MM_FMADD(a1, bv1, c12);
                c22 = MM_FMADD(a2, bv1, c22);
                c32 = MM_FMADD(a3, bv1, c32);
                c42 = MM_FMADD(a4, bv1, c42);
                c52 = MM_FMADD(a5, bv1, c52);

                bv2 = MM_LOAD(B + k * N + sj + 3 * VEC_SIZ);

                c03 = MM_FMADD(a0, bv2, c03);
                c13 = MM_FMADD(a1, bv2, c13);
                c23 = MM_FMADD(a2, bv2, c23);
                c33 = MM_FMADD(a3, bv2, c33);
                c43 = MM_FMADD(a4, bv2, c43);
                c53 = MM_FMADD(a5, bv2, c53);
            }

            MM_STORE(C + si * N + sj, c00);
            MM_STORE(C + si * N + sj + VEC_SIZ, c01);
            MM_STORE(C + si * N + sj + 2 * VEC_SIZ, c02);
            MM_STORE(C + si * N + sj + 3 * VEC_SIZ, c03);
            
            MM_STORE(C + (si + 1) * N + sj, c10);
            MM_STORE(C + (si + 1) * N + sj + VEC_SIZ, c11);
            MM_STORE(C + (si + 1) * N + sj + 2 * VEC_SIZ, c12);
            MM_STORE(C + (si + 1) * N + sj + 3 * VEC_SIZ, c13);
            
            MM_STORE(C + (si + 2) * N + sj, c20);
            MM_STORE(C + (si + 2) * N + sj + VEC_SIZ, c21);
            MM_STORE(C + (si + 2) * N + sj + 2 * VEC_SIZ, c22);
            MM_STORE(C + (si + 2) * N + sj + 3 * VEC_SIZ, c23);

            MM_STORE(C + (si + 3) * N + sj, c30);
            MM_STORE(C + (si + 3) * N + sj + VEC_SIZ, c31);
            MM_STORE(C + (si + 3) * N + sj + 2 * VEC_SIZ, c32);
            MM_STORE(C + (si + 3) * N + sj + 3 * VEC_SIZ, c33);

            MM_STORE(C + (si + 4) * N + sj, c40);
            MM_STORE(C + (si + 4) * N + sj + VEC_SIZ, c41);
            MM_STORE(C + (si + 4) * N + sj + 2 * VEC_SIZ, c42);
            MM_STORE(C + (si + 4) * N + sj + 3 * VEC_SIZ, c43);

            MM_STORE(C + (si + 5) * N + sj, c50);
            MM_STORE(C + (si + 5) * N + sj + VEC_SIZ, c51);
            MM_STORE(C + (si + 5) * N + sj + 2 * VEC_SIZ, c52);
            MM_STORE(C + (si + 5) * N + sj + 3 * VEC_SIZ, c53);
        }
    }

    int si = KN_BS;
    if(KN_REM >= 3){
        for(int sj = 0; sj < N; sj += KSIZ){
            c00 = c01 = c02 = c03 = c10 = c11 = c12 = c13 = c20 = c21 = c22 = c23 = MM_ZERO();

            //3xN and Nx32 (32 is unrolled using avx2) slices for A and B
            for(int k = 0; k < N; ++k){
                a0 = MM_SETALL(A[si * N + k]);
                a1 = MM_SETALL(A[(si + 1) * N + k]);
                a2 = MM_SETALL(A[(si + 2) * N + k]);
                bv1 = MM_LOAD(B + k * N + sj);

                c00 = MM_FMADD(a0, bv1, c00);
                c10 = MM_FMADD(a1, bv1, c10);
                c20 = MM_FMADD(a2, bv1, c20);

                bv2 = MM_LOAD(B + k * N + sj + VEC_SIZ);

                c01 = MM_FMADD(a0, bv2, c01);
                c11 = MM_FMADD(a1, bv2, c11);
                c21 = MM_FMADD(a2, bv2, c21);
                
                bv1 = MM_LOAD(B + k * N + sj + 2 * VEC_SIZ);

                c02 = MM_FMADD(a0, bv1, c02);
                c12 = MM_FMADD(a1, bv1, c12);
                c22 = MM_FMADD(a2, bv1, c22);

                bv2 = MM_LOAD(B + k * N + sj + 3 * VEC_SIZ);

                c03 = MM_FMADD(a0, bv2, c03);
                c13 = MM_FMADD(a1, bv2, c13);
                c23 = MM_FMADD(a2, bv2, c23);
            }

            MM_STORE(C + si * N + sj, c00);
            MM_STORE(C + si * N + sj + VEC_SIZ, c01);
            MM_STORE(C + si * N + sj + 2 * VEC_SIZ, c02);
            MM_STORE(C + si * N + sj + 3 * VEC_SIZ, c03);
            
            MM_STORE(C + (si + 1) * N + sj, c10);
            MM_STORE(C + (si + 1) * N + sj + VEC_SIZ, c11);
            MM_STORE(C + (si + 1) * N + sj + 2 * VEC_SIZ, c12);
            MM_STORE(C + (si + 1) * N + sj + 3 * VEC_SIZ, c13);
            
            MM_STORE(C + (si + 2) * N + sj, c20);
            MM_STORE(C + (si + 2) * N + sj + VEC_SIZ, c21);
            MM_STORE(C + (si + 2) * N + sj + 2 * VEC_SIZ, c22);
            MM_STORE(C + (si + 2) * N + sj + 3 * VEC_SIZ, c23);
        }
        si += 3;
    }

    for(; si < N; ++si){
        for(int sj = 0; sj < N; sj += KSIZ){
            c00 = c01 = c02 = c03 = MM_ZERO();
            for(int k = 0; k < N; ++k){
                a0 = MM_SETALL(A[si * N + k]);

                bv1 = MM_LOAD(B + k * N + sj);
                c00 = MM_FMADD(a0, bv1, c00);

                bv2 = MM_LOAD(B + k * N + sj + VEC_SIZ);
                c01 = MM_FMADD(a0, bv2, c01);
                
                bv1 = MM_LOAD(B + k * N + sj + 2 * VEC_SIZ);
                c02 = MM_FMADD(a0, bv1, c02);

                bv2 = MM_LOAD(B + k * N + sj + 3 * VEC_SIZ);
                c03 = MM_FMADD(a0, bv2, c03);
            }

            MM_STORE(C + si * N + sj, c00);
            MM_STORE(C + si * N + sj + VEC_SIZ, c01);
            MM_STORE(C + si * N + sj + 2 * VEC_SIZ, c02);
            MM_STORE(C + si * N + sj + 3 * VEC_SIZ, c03);
        }
    }
}

#ifdef _MAT_AVX512_ON
#define kernel_matmul_4unroll kernel_matmul_6xNx4
#else
#define kernel_matmul_4unroll kernel_matmul_3xNx4
#endif

//kernel 256x256->3xBSx32
void kernel_mat_mul_store_256x256(cvalue_ptr a, const size_t a_step, 
                cvalue_ptr b, const size_t b_step,
                value_ptr res,  const size_t res_step, 
                size_t M, size_t N, size_t K){
#define BS 256

    alignas(_MALLOC_ALIGN) value_type A[BS * BS];
    alignas(_MALLOC_ALIGN) value_type B[BS * BS];
    alignas(_MALLOC_ALIGN) value_type C[BS * BS];

#ifdef _MAT_TIMING

    double reset_cl = 0;
    double kernel_cl = 0;
    double icpy_cl = 0;
    double ocpy_cl = 0;
    double clk = 0;

#endif

    CLK_START;

    if(M <= 2048){
        memset(res, 0, M * N * sizeof(value_type));
    }else{
        #pragma omp parallel for
        for(size_t i = 0; i < M; ++i){
            value_ptr ptr = res + i * res_step;
            #pragma omp simd
            for(size_t j = 0; j < N; ++j)
                ptr[j] = 0;
        }
    }

    CLK_END(reset_cl);

    #pragma omp parallel for private(A, B, C)
    for(size_t bi = 0; bi < M; bi += BS){
        for(size_t bk = 0; bk < K; bk += BS){

            CLK_START;

            for(int i = 0; i < BS; ++i){
                cvalue_ptr a_ptr  = a + (i + bi) * a_step + bk;
                #pragma omp simd
                for(int k = 0; k < BS; ++k)
                    A[i * BS + k] = a_ptr[k];
            }

            CLK_END(icpy_cl);            

            for(size_t bj = 0; bj < N; bj += BS){

                CLK_START;

                #pragma omp simd
                for(int k = 0; k < BS; ++k){
                    cvalue_ptr b_ptr = b + (k + bk) * b_step + bj;
                    
                    for(size_t j = 0; j < BS; ++j)
                        B[k * BS + j] = b_ptr[j];
                }

                CLK_END(icpy_cl);

                CLK_START;

                kernel_matmul_4unroll(A, B, C, BS);

                CLK_END(kernel_cl);

                CLK_START;

                #pragma omp simd
                for(int i = 0; i < BS; ++i){
                    value_ptr res_ptr = res + (i + bi) * res_step + bj;
                    for(int j = 0; j < BS; ++j){
                        res_ptr[j] += C[i * BS + j];
                    }
                }

                CLK_END(ocpy_cl);
            }
        }
    }

#ifdef _MAT_TIMING

    printf("reset_cl:%lf\n", reset_cl);
    printf("icpy_cl: %lf\n", icpy_cl);
    printf("ocpy_cl: %lf\n", ocpy_cl);
    printf("kernel_cl: %lf\n", kernel_cl);

#endif
#undef BS
}

void kernel_mat_mul_store_64x64(cvalue_ptr a, const size_t a_step, 
                cvalue_ptr b, const size_t b_step,
                value_ptr res,  const size_t res_step, 
                size_t M, size_t N, size_t K){

#define BS 64

    alignas(_MALLOC_ALIGN) value_type A[BS * BS];
    alignas(_MALLOC_ALIGN) value_type B[BS * BS];
    alignas(_MALLOC_ALIGN) value_type C[BS * BS];

    if(M <= 2048){
        memset(res, 0, M * N * sizeof(value_type));
    }else{
        #pragma omp parallel for
        for(size_t i = 0; i < M; ++i){
            value_ptr ptr = res + i * res_step;
            #pragma omp simd
            for(size_t j = 0; j < N; ++j)
                ptr[j] = 0;
        }
    }

    #pragma omp parallel for private(A, B, C)
    for(size_t bi = 0; bi < M; bi += BS){
        for(size_t bk = 0; bk < K; bk += BS){

            #pragma omp simd
            for(int i = 0; i < BS; ++i){
                cvalue_ptr a_ptr  = a + (i + bi) * a_step + bk;
                for(int k = 0; k < BS; ++k)
                    A[i * BS + k] = a_ptr[k];
            }         

            for(size_t bj = 0; bj < N; bj += BS){

                #pragma omp simd
                for(int k = 0; k < BS; ++k){
                    cvalue_ptr b_ptr = b + (k + bk) * b_step + bj;
                    
                    for(size_t j = 0; j < BS; ++j)
                        B[k * BS + j] = b_ptr[j];
                }

                kernel_matmul_4unroll(A, B, C, BS);

                #pragma omp simd
                for(int i = 0; i < BS; ++i){
                    value_ptr res_ptr = res + (i + bi) * res_step + bj;
                    for(int j = 0; j < BS; ++j){
                        res_ptr[j] += C[i * BS + j];
                    }
                }
            }
        }
    }

#undef BS
}

mat_ptr mat_mul_unroll(mat_ptr a, mat_ptr b){
    _MAT_MUL_PRE;
    size_t M = a->rows, N = b->cols, K = a->cols;
    mat_mul_store_2x4xN(a->data, a->cols, b->data, b->cols, res->data, res->cols, M, N, K);
    return res;
}

mat_ptr mat_mul_inst(mat_ptr a, mat_ptr b){
    _MAT_MUL_PRE;
    size_t M = a->rows, N = b->cols, K = a->cols;
    if(M >= 6000){
        kernel_mat_mul_store_256x256(a->data, a->cols, b->data, b->cols, res->data, res->cols, M, N, K);
    }else if(M >=512){
        kernel_mat_mul_store_64x64(a->data, a->cols, b->data, b->cols, res->data, res->cols, M, N, K);
    }else{
        kernel_matmul_4unroll(a->data, b->data, res->data, N);
    }
    // kernel_mat_mul_store_256x256(a->data, a->cols, b->data, b->cols, res->data, res->cols, M, N, K);
    return res;
}

mat_ptr mat_mul_openblas(const mat_ptr a, const mat_ptr b){
    _MAT_MUL_PRE;

    size_t M = a->rows, N = b->cols, K = a->cols;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K, 
        1.0, a->data, K, b->data, N, 
        0.0, res->data, N
    );

    return res;
}