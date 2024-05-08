#pragma once

#include"matrix.h"

#define INIT_FLAG 1919810

#if defined(__cpluscplus) || defined(__NVCC__)
//for cuda
#define C_FIELD_BEGIN extern "C" {
#define C_FIELD_END }

#else

#define C_FIELD_BEGIN
#define C_FIELD_END

#endif

#ifdef __linux__

//alias for linux
#define mat_aligned_alloc(size) ((value_ptr)aligned_alloc(_MALLOC_ALIGN, size))
#define mat_aligned_free(ptr) free(ptr)

#else

//alias for windows
#define mat_aligned_alloc(size) ((value_ptr)_aligned_malloc(size, _MALLOC_ALIGN))
#define mat_aligned_free(ptr) _aligned_free(ptr)

#endif

#define ERR_INFO(...)\
    do{\
        fprintf(stderr, "[ERROR] %s: line %d, in function '%s':\n", __FILE__, __LINE__, __FUNCTION__);\
        fprintf(stderr, __VA_ARGS__);\
    }while(0)

#define _REQUIRE(expr, ret)\
    if(!(expr)){\
        ERR_INFO("assertion failed: require '%s'\n", #expr);\
        return ret;\
    }

#define _REQUIRE_MAT_VALID(m, ret)\
    if(m == NULL || m->init_flag != INIT_FLAG){\
        ERR_INFO("Using matrix without initialization\n");\
        return ret;\
    }

#define _MAT_MUL_PRE\
    _REQUIRE_MAT_VALID(a, NULL);\
    _REQUIRE_MAT_VALID(b, NULL);\
    _REQUIRE(a->cols == b->rows, NULL);\
    mat_ptr res = mat_create(a->rows, b->cols);\
    if(res == NULL)\
        return NULL;

#ifdef __AVX512F__

#define _MALLOC_ALIGN 512
#define _MAT_AVX512_ON

#elif defined(__AVX2__)

#define _MALLOC_ALIGN 256
#define _MAT_AVX256_ON

#elif defined(__SSE2__)

#define _MALLOC_ALIGN 128

#else

#define _MALLOC_ALIGN 64

#endif

#ifdef __AVX512F__

//alias for avx512
#define MM_REG __m512
#define MM_ZERO _mm512_setzero_ps
#define MM_SETALL _mm512_set1_ps
#define MM_LOAD _mm512_load_ps
#define MM_STORE _mm512_store_ps
#define MM_FMADD _mm512_fmadd_ps
#define VEC_SIZ 16

#else

//alias for avx256
#define MM_REG __m256
#define MM_ZERO _mm256_setzero_ps
#define MM_SETALL _mm256_set1_ps
#define MM_LOAD _mm256_load_ps
#define MM_STORE _mm256_store_ps
#define MM_FMADD _mm256_fmadd_ps
#define VEC_SIZ 8

#endif

// #define _MAT_TIMING

#ifdef _MAT_TIMING

#define CLK_START clk = get_clock()
#define CLK_END(x) x += get_clock() - clk

#else 

#define CLK_START
#define CLK_END(x)

#endif

void kernel_mat_mul_store_64x64(cvalue_ptr a, const size_t a_step, 
                cvalue_ptr b, const size_t b_step,
                value_ptr res,  const size_t res_step, 
                size_t M, size_t N, size_t K);
void kernel_mat_mul_store_256x256(cvalue_ptr a, const size_t a_step, 
                cvalue_ptr b, const size_t b_step,
                value_ptr res,  const size_t res_step, 
                size_t M, size_t N, size_t K);
