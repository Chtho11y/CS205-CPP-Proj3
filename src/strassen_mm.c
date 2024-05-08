#include "kernel.h"
#include<stdlib.h>
#include<stdalign.h>

/**
 * kernel_sqmat functions only calculate a siz*siz part of a matrix with 'step' columns.
*/

//dst = src
void kernel_sqmat_load_par(cvalue_ptr src, const size_t src_step, 
                        value_ptr dst, const size_t dst_step, const size_t siz){
    #pragma omp parallel for simd
    for(size_t i = 0; i < siz; ++i)
        for(size_t j = 0; j < siz; ++j)
            dst[i * dst_step + j] = src[i * src_step + j];
}

//dst1 = dst2 = src
void kernel_sqmat_2load_par(cvalue_ptr src, const size_t src_step, 
                        value_ptr dst1, const size_t dst1_step,
                        value_ptr dst2, const size_t dst2_step, const size_t siz){
    #pragma omp parallel for simd
    for(size_t i = 0; i < siz; ++i)
        for(size_t j = 0; j < siz; ++j){
            dst1[i * dst1_step + j] = src[i * src_step + j];
            dst2[i * dst2_step + j] = src[i * src_step + j];
        }
}

//dst = a + b
void kernel_sqmat_add_par(cvalue_ptr a, const size_t a_step, 
                        cvalue_ptr b, const size_t b_step, 
                        value_ptr dst, const size_t dst_step, const size_t siz){
    #pragma omp parallel for simd
    for(size_t i = 0; i < siz; ++i)
        for(size_t j = 0; j < siz; ++j)
            dst[i * dst_step + j] = a[i * a_step + j] + b[i * b_step + j];
}

//a += b
void kernel_sqmat_addeq_par(value_ptr a, const size_t a_step, cvalue_ptr b, const size_t b_step,  const size_t siz){
    #pragma omp parallel for simd
    for(size_t i = 0; i < siz; ++i)
        for(size_t j = 0; j < siz; ++j)
            a[i * a_step + j] += b[i * b_step + j];
}

//dst1, dst2 += b
void kernel_sqmat_2addeq_par(value_ptr dst1, const size_t dst1_step,
                             value_ptr dst2, const size_t dst2_step,
                             cvalue_ptr b, const size_t b_step,
                             const size_t siz){
    #pragma omp parallel for simd
    for(size_t i = 0; i < siz; ++i)
        for(size_t j = 0; j < siz; ++j){
            dst1[i * dst1_step + j] += b[i * b_step + j];
            dst2[i * dst2_step + j] += b[i * b_step + j];
        }
}

//dst1 += b, dst2 -= b
void kernel_sqmat_addsubeq_par(value_ptr dst1, const size_t dst1_step,
                             value_ptr dst2, const size_t dst2_step,
                             cvalue_ptr b, const size_t b_step,
                             const size_t siz){
    #pragma omp parallel for simd
    for(size_t i = 0; i < siz; ++i)
        for(size_t j = 0; j < siz; ++j){
            dst1[i * dst1_step + j] += b[i * b_step + j];
            dst2[i * dst2_step + j] -= b[i * b_step + j];
        }
}

//dst = a - b
void kernel_sqmat_sub_par(cvalue_ptr a, const size_t a_step, 
                        cvalue_ptr b, const size_t b_step, 
                        value_ptr dst, const size_t dst_step, const size_t siz){
    #pragma omp parallel for simd
    for(size_t i = 0; i < siz; ++i)
        for(size_t j = 0; j < siz; ++j)
            dst[i * dst_step + j] = a[i * a_step + j] - b[i * b_step + j];
}

//a -= b
void kernel_sqmat_subeq_par(value_ptr a, const size_t a_step, cvalue_ptr b, const size_t b_step,  const size_t siz){
    #pragma omp parallel for simd
    for(size_t i = 0; i < siz; ++i)
        for(size_t j = 0; j < siz; ++j)
            a[i * a_step + j] -= b[i * b_step + j];
}

//a = b - a
void kernel_sqmat_nsubeq_par(value_ptr a, const size_t a_step, cvalue_ptr b, const size_t b_step,  const size_t siz){
    #pragma omp parallel for simd
    for(size_t i = 0; i < siz; ++i)
        for(size_t j = 0; j < siz; ++j)
            a[i * a_step + j] = b[i * b_step + j] - a[i * a_step + j];
}

void kernel_sqmat_mul_par(cvalue_ptr a, const size_t a_step, 
                    cvalue_ptr b, const size_t b_step, 
                    value_ptr res, const size_t res_step, value_ptr buf, const size_t N){
    if(N > 6000)
        kernel_mat_mul_store_256x256(a, a_step, b, b_step, res, res_step, N, N, N);
    else
        kernel_mat_mul_store_64x64(a, a_step, b, b_step, res, res_step, N, N, N);
}

#define STRASSEN_LIMIT 14000

#ifdef _MAT_TIMING
static double _strassen_kernel_clk = 0;
#endif

void kernel_strassen_plain(cvalue_ptr a, const size_t a_step, 
                    cvalue_ptr b, const size_t b_step, 
                    value_ptr res, const size_t res_step, const size_t N){
#define kernel_sqmat_mul_par kernel_strassen_plain
    if(N <= STRASSEN_LIMIT){

#ifdef _MAT_TIMING
        double clk = 0;
#endif
        CLK_START;

        kernel_mat_mul_store_64x64(a, a_step, b, b_step, res, res_step, N, N, N);

        CLK_END(_strassen_kernel_clk);
        
#ifdef _MAT_TIMING
        printf("kernel call: %.2f\n", _strassen_kernel_clk);
#endif
        return;
    }
#define A00 a, a_step
#define A01 a + H, a_step
#define A10 a + H * a_step, a_step
#define A11 a + H * a_step + H, a_step
#define B00 b, b_step
#define B01 b + H, b_step
#define B10 b + H * b_step, b_step
#define B11 b + H * b_step + H, b_step
#define C00 res, res_step
#define C01 res + H, res_step
#define C10 res + H * res_step, res_step
#define C11 res + H * res_step + H, res_step
#define reg(x) x, H
#define inst(x) kernel_sqmat_##x##_par(
#define $ , H);

    const size_t H = N >> 1;
    const size_t buf_siz = H * H;

    value_ptr mat0 = mat_aligned_alloc(buf_siz * sizeof(value_type));
    value_ptr mat1 = mat_aligned_alloc(buf_siz * sizeof(value_type));
    value_ptr mat2 = mat_aligned_alloc(buf_siz * sizeof(value_type));

    value_ptr s1 = mat0, p1 = mat1;

    inst(sub)   B01, B11, reg(s1)     $
    inst(mul)   A00, reg(s1), reg(p1) $
    inst(2load) reg(p1), C11, C01     $

    value_ptr s4 = mat0, p4 = mat1;

    inst(sub)   B10, B00, reg(s4)      $
    inst(mul)   A11, reg(s4), reg(p4)  $
    inst(2load) reg(p4), C10, C00      $

    value_ptr s2 = mat0, p2 = mat1;

    inst(add)       A00, A01, reg(s2)     $
    inst(mul)       reg(s2), B11, reg(p2) $
    inst(addsubeq)  C01, C00, reg(p2)     $

    value_ptr s3 = mat0, p3 = mat1;

    inst(add)       A10, A11, reg(s3)      $
    inst(mul)       reg(s3), B00, reg(p3)  $
    inst(addsubeq)  C10, C11, reg(p3)      $

    value_ptr s5 = mat0, p5 = mat1, s6 = mat2;

    inst(add)    A11, A00, reg(s5)              $
    inst(add)    B00, B11, reg(s6)              $
    inst(mul)    reg(s5), reg(s6), reg(p5)      $
    inst(2addeq) C11, C00, reg(p5)              $

    value_ptr s7 = mat0, p6 = mat1, s8 = mat2;

    inst(sub)    A01, A11, reg(s7)              $
    inst(add)    B10, B11, reg(s8)              $
    inst(mul)    reg(s7), reg(s8), reg(p6)      $
    inst(addeq)  C00, reg(p6)                   $

    value_ptr s9 = mat0, p7 = mat1, s10 = mat2;

    inst(sub)    A00, A10, reg(s9)               $
    inst(add)    B00, B01, reg(s10)              $
    inst(mul)    reg(s9), reg(s10), reg(p7)      $
    inst(subeq)  C11, reg(p7)                    $

    mat_aligned_free(mat0);
    mat_aligned_free(mat1);
    mat_aligned_free(mat2);

#undef A00
#undef A01
#undef A10
#undef A11
#undef B00
#undef B01
#undef B10
#undef B11
#undef C00
#undef C01
#undef C10
#undef C11
#undef reg
#undef inst
#undef $

#undef kernel_sqmat_mul_par
}

void kernel_strassen(cvalue_ptr a, const size_t a_step, 
                    cvalue_ptr b, const size_t b_step, 
                    value_ptr res, const size_t res_step, value_ptr buf, const size_t N){
#define kernel_sqmat_mul_par kernel_strassen
    if(N <= STRASSEN_LIMIT){

#ifdef _MAT_TIMING
        double clk = 0;
#endif
        CLK_START;

        if(N > 6000)
            kernel_mat_mul_store_256x256(a, a_step, b, b_step, res, res_step, N, N, N);
        else
            kernel_mat_mul_store_64x64(a, a_step, b, b_step, res, res_step, N, N, N);

        CLK_END(_strassen_kernel_clk);
        
#ifdef _MAT_TIMING
        printf("kernel call: %.2f\n", _strassen_kernel_clk);
#endif
        return;
    }
#define A00 a, a_step
#define A01 a + H, a_step
#define A10 a + H * a_step, a_step
#define A11 a + H * a_step + H, a_step
#define B00 b, b_step
#define B01 b + H, b_step
#define B10 b + H * b_step, b_step
#define B11 b + H * b_step + H, b_step
#define C00 res, res_step
#define C01 res + H, res_step
#define C10 res + H * res_step, res_step
#define C11 res + H * res_step + H, res_step
#define reg(x) x, H
#define inst(x) kernel_sqmat_##x##_par(
#define $ , H);

    const size_t H = N >> 1;
    const size_t buf_siz = H * H;

    value_ptr mat0 = buf + buf_siz;
    value_ptr mat1 = mat0 + buf_siz;
    value_ptr mat2 = mat1 + buf_siz;

    value_ptr s1 = mat0, p1 = mat1;

    inst(sub)   B01, B11, reg(s1)          $
    inst(mul)   A00, reg(s1), reg(p1), buf $
    inst(2load) reg(p1), C11, C01          $

    value_ptr s4 = mat0, p4 = mat1;

    inst(sub)   B10, B00, reg(s4)          $
    inst(mul)   A11, reg(s4), reg(p4), buf $
    inst(2load) reg(p4), C10, C00          $

    value_ptr s2 = mat0, p2 = mat1;

    inst(add)       A00, A01, reg(s2)           $
    inst(mul)       reg(s2), B11, reg(p2), buf  $
    inst(addsubeq)  C01, C00, reg(p2)           $

    value_ptr s3 = mat0, p3 = mat1;

    inst(add)       A10, A11, reg(s3)           $
    inst(mul)       reg(s3), B00, reg(p3), buf  $
    inst(addsubeq)  C10, C11, reg(p3)           $

    value_ptr s5 = mat0, p5 = mat1, s6 = mat2;

    inst(add)    A11, A00, reg(s5)              $
    inst(add)    B00, B11, reg(s6)              $
    inst(mul)    reg(s5), reg(s6), reg(p5), buf $
    inst(2addeq) C11, C00, reg(p5)              $

    value_ptr s7 = mat0, p6 = mat1, s8 = mat2;

    inst(sub)    A01, A11, reg(s7)              $
    inst(add)    B10, B11, reg(s8)              $
    inst(mul)    reg(s7), reg(s8), reg(p6), buf $
    inst(addeq)  C00, reg(p6)                   $

    value_ptr s9 = mat0, p7 = mat1, s10 = mat2;

    inst(sub)    A00, A10, reg(s9)               $
    inst(add)    B00, B01, reg(s10)              $
    inst(mul)    reg(s9), reg(s10), reg(p7), buf $
    inst(subeq)  C11, reg(p7)                    $

#undef A00
#undef A01
#undef A10
#undef A11
#undef B00
#undef B01
#undef B10
#undef B11
#undef C00
#undef C01
#undef C10
#undef C11
#undef reg
#undef inst
#undef $

#undef kernel_sqmat_mul_par
}

//require square matrix with 2^n size.
mat_ptr mat_mul_strassen(const mat_ptr a, const mat_ptr b){
    _REQUIRE_MAT_VALID(a, NULL);
    _REQUIRE_MAT_VALID(b, NULL);
    _REQUIRE(a->cols == b->rows && a->cols == a->rows && b->cols == b->rows, NULL);
    
    size_t N = a->cols;
    
    // if(N & (N - 1) > 0){
    //     ERR_INFO("N must be power of 2.\n");
    //     return NULL;
    // }

    mat_ptr res = mat_create(N, N);

    value_ptr buffer = mat_aligned_alloc(N * N * sizeof(value_type));

    kernel_strassen(a->data, N, b->data, N, res->data, N, buffer, N);

    mat_aligned_free(buffer);

    return res;
}
