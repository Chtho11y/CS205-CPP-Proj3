#include "kernel.h"
#include<stdlib.h>
#include<string.h>
#include<stdalign.h>
#include<stdio.h>

#include<omp.h>

bool mat_print(FILE *target, const char *fmt, const char *sp, const char *end, const mat_ptr m){
    _REQUIRE_MAT_VALID(m, false);
    _REQUIRE(target != NULL, NULL);

    for(size_t i = 0; i < m->rows; ++i){
        for(size_t j = 0; j < m->cols; ++j){
            fprintf(target, fmt, m->data[i * m->cols + j]);
            fputs(sp, target);
        }
        fputs(end, target);
    }

    fflush(target);
}

bool mat_init(mat_ptr m, size_t rows, size_t cols){
    _REQUIRE(rows > 0 && cols > 0, NULL);

    m->init_flag = INIT_FLAG;

    m->rows = rows;
    m->cols = cols;

    m->data = mat_aligned_alloc(sizeof(value_type) * rows * cols);

    if(m->data == NULL){
        ERR_INFO("bad alloc!");
        m->init_flag = 0;
        return false;
    }
    return true;
}

mat_ptr mat_create(size_t rows, size_t cols){
    mat_ptr m = (mat_ptr)malloc(sizeof(mat_t));
    if(m == NULL){
        ERR_INFO("bad alloc!");
        return NULL;
    }

    bool res = mat_init(m, rows, cols);

    if(!res){
        free(m);
        return NULL;
    }
    return m;
}

//initialize a matrix with given data field.
void mat_init_from(size_t rows, size_t cols, mat_ptr ptr, value_ptr data){
    ptr->init_flag = INIT_FLAG;
    ptr->data = data;
    ptr->cols = cols;
    ptr->rows = rows;
}

bool mat_clear(mat_ptr m){
    _REQUIRE_MAT_VALID(m, false);

    mat_aligned_free(m->data);

    m->init_flag = 0;
    return true;
}

bool mat_free(mat_ptr m){
    _REQUIRE(m != NULL, NULL);

    if(!mat_clear(m))
        return false;
    free(m);
    return true;
}

bool mat_random(mat_ptr m){
    _REQUIRE_MAT_VALID(m, false);

    const size_t N = m->rows * m->cols;

    for(size_t i = 0; i < N; ++i){
        m->data[i] = rand()/(value_type)RAND_MAX * 2 - 1;
    }
    return true;
}

value_type val_abs(value_type a){
    return a > 0 ? a : -a;
}

value_type val_max(value_type a, value_type b){
    return a > b ? a : b;
}

value_type mat_diff(const mat_ptr mat1, const mat_ptr mat2){
    _REQUIRE_MAT_VALID(mat1, -1);
    _REQUIRE_MAT_VALID(mat2, -1);
    value_type val = 0;
    const size_t siz = mat1->rows * mat1->cols;
    for(size_t i = 0; i < siz; ++i)
        val = val_max(val, val_abs(mat1->data[i] - mat2->data[i]));
    return val;
}

bool mat_set(mat_ptr m, value_type val){
    _REQUIRE_MAT_VALID(m, false);

    const size_t N = m->rows * m->cols;

    if(val == 0){
        memset(m->data, 0, sizeof(value_type) * N);
        return true;
    }

    #pragma omp simd
    for(size_t i = 0; i < N; ++i)
        m->data[i] = val;
    
    return true;
}

mat_ptr mat_trans_copy(const mat_ptr m){
    _REQUIRE_MAT_VALID(m, NULL);

    mat_ptr res = mat_create(m->cols, m->rows);
    if(res == NULL)
        return NULL;

    #pragma omp parallel for
    for(size_t i = 0; i < m->rows; ++i)
        for(size_t j = 0; j < m->cols; ++j)
            res->data[j * res->cols + i] = m->data[i * m->cols + j];
    
    return res;
}