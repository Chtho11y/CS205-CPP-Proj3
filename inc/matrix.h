#pragma once

#include<stdbool.h> //bool
#include<stdio.h>   //FILE*, size_t

typedef float value_type;
typedef value_type* value_ptr;
typedef const value_type* cvalue_ptr;

struct matrix{
    size_t cols;
    size_t rows;

    value_ptr data;

    int init_flag;      //flag to check if matrix is initialized
};

typedef struct matrix mat_t[1];
typedef struct matrix *mat_ptr;

#if defined(__cpluscplus) || defined(__NVCC__)
extern "C"{
#endif

bool mat_init(mat_ptr m, size_t rows, size_t cols);
mat_ptr mat_create(size_t rows, size_t cols);
void mat_init_from(size_t rows, size_t cols, mat_ptr ptr, value_ptr data);
bool mat_clear(mat_ptr m);
bool mat_free(mat_ptr m);

bool mat_random(mat_ptr m);
bool mat_set(mat_ptr m, value_type val);
mat_ptr mat_trans_copy(const mat_ptr m);

bool mat_print(FILE *target, const char *fmt, const char *sp, const char *end, const mat_ptr m);

value_type mat_diff(const mat_ptr mat1, const mat_ptr mat2);

mat_ptr mat_mul_plain(const mat_ptr mat1, const mat_ptr mat2);
mat_ptr mat_mul_trans(mat_ptr a, mat_ptr b);
mat_ptr mat_mul_reorder(mat_ptr a, mat_ptr b);
mat_ptr mat_mul_unroll(const mat_ptr mat1, const mat_ptr mat2);
mat_ptr mat_mul_inst(const mat_ptr mat1, const mat_ptr mat2);
mat_ptr mat_mul_openblas(const mat_ptr mat1, const mat_ptr mat2);
mat_ptr mat_mul_strassen(const mat_ptr a, const mat_ptr b);
mat_ptr mat_mul_cuda_plain(const mat_ptr a, const mat_ptr b);
mat_ptr mat_mul_cuda_shared(const mat_ptr a, const mat_ptr b);
mat_ptr mat_mul_cuda_reg(const mat_ptr a, const mat_ptr b);
mat_ptr mat_mul_cuda_multi(const mat_ptr a, const mat_ptr b);
mat_ptr mat_mul_cuda_prefetch(const mat_ptr a, const mat_ptr b);
void cuda_init();

#if defined(__cpluscplus) || defined(__NVCC__)
}
#endif

double get_clock();