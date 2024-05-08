#define CUDA_ON

#include"matrix.h"
#include<stdio.h>

#ifdef __linux__

#include<sys/time.h>

double get_clock(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1e3 + t.tv_usec / 1e3;
}

#else
#include<time.h>

double get_clock(){
    struct timespec tim;
    clock_gettime(CLOCK_MONOTONIC, &tim);
    return tim.tv_sec * 1e3 + tim.tv_nsec / 1e6;
}

#endif

typedef mat_ptr (*matmul_func)(const mat_ptr, const mat_ptr);

#define TEST_CNT 4

#define print_info(...) printf(__VA_ARGS__)
#define print_log(...)  fprintf(result_file, __VA_ARGS__)

const char *result_filename = "../result/result.txt";
int siz[100], n;
FILE *result_file;

// #define JUDGE

// #define LINER

/*
Ncuda_prefetch = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384, 20480, 24576, 28672, 32768, 49152]
Tcuda_prefetch = [23.29541, 49.20532, 74.26644, 104.38265, 151.56698, 234.30005, 299.16227, 394.71265, 509.62598, 625.71305, 772.54289, 1188.66577, 1117.38973, 1294.51839, 1474.82861, 1785.04956, 2782.23657, 4758.01058, 6175.21826, 8414.91968, 24217.45328]

Ncuda_multi = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 16384, 20480, 24576, 28672, 32768]
Tcuda_multi = [24.64795, 45.85400, 84.09399, 104.29688, 152.09009, 253.85205, 288.22900, 385.45410, 508.07910, 687.02295, 890.41089, 1889.19397, 2303.04993, 4329.43750, 7311.90662, 6793.87000, 8677.59802]
*/

#ifdef LINER
#define STEP 4096
const int TEST_L = 20 * 1024;
const int TEST_R = 32 * 1024;
#else
#define STEP 2
const int TEST_L = 1<<4;
const int TEST_R = 1024;
#endif

void print_barket(double x){
    char buf[20];
    sprintf(buf, "(%.2f)", x);
    print_info("%12s", buf);
    fflush(stdout);
}

void benchmark(matmul_func matmul, const char *label, const int* siz_list, const int N){
    print_info("Testing %s: Range = [%d, %d]\n", label, TEST_L, TEST_R);
    print_log("N = [");
    for(int i = 0; i < N; ++i){
        print_log("%d%c%c", siz_list[i], ",]"[i == N - 1], " \n"[i == N - 1]);
    }
    print_log("%s = [", label);

    for(int i = 0; i < N; ++i){
        int siz = siz_list[i];
        mat_ptr a = mat_create(siz, siz);
        mat_ptr b = mat_create(siz, siz);
        mat_ptr res, ans;

        printf("N = %5d:", siz);

        double tot_time = 0, diff;

        for(int j = 0; j < TEST_CNT; ++j){
            mat_random(a);
            mat_random(b);
#ifdef JUDGE
            if(j == 0)
                ans = mat_mul_openblas(a, b);
#endif
            
            double st = get_clock();
            res = matmul(a, b);
            double ed = get_clock();

            if(j == 0){
            #ifdef JUDGE
                diff = mat_diff(res, ans);
                mat_free(ans);
            #endif    
                print_barket(ed - st);
            }else{
                print_info("%12.2f", ed - st);
                tot_time += ed - st;
                fflush(stdout);
            }
            
            mat_free(res);
        }

        tot_time /= (TEST_CNT - 1);

        double gflops = 2.0 * siz * siz * siz / (tot_time * 1e6);
        print_info(" => %8.2f, GFLOPS = %8.2f, diff = %.6f\n", tot_time, gflops , diff);
        print_log("%.5f%c%c", tot_time, ",]"[i == N - 1], " \n"[i == N - 1]);

        mat_free(a);
        mat_free(b);
        fflush(result_file);
    }
}

#define BENCHMARK(name) benchmark(mat_mul_##name, #name, siz, n)

void init(){
#ifdef LINER
    for(int i = TEST_L; i <= TEST_R; i += STEP)
        siz[n++] = i;
#else
    for(int i = TEST_L; i <= TEST_R; i *= STEP)
        siz[n++] = i;
#endif
    result_file = fopen(result_filename, "w");
}

int main(){
    init();

#ifdef CUDA_ON
    cuda_init();
#endif

    // BENCHMARK(reorder);
    // BENCHMARK(strassen);
    // BENCHMARK(openblas);

    const int N = 128;
    mat_ptr a = mat_create(N, N), b = mat_create(N, N);
    mat_random(a);
    mat_random(b);

    // // mat_ptr ans = mat_mul_openblas(a, b);

    double st = get_clock();
    mat_ptr res = mat_mul_cuda_plain(a, b);
    double ed = get_clock();
    // mat_free(res);
    // double st3 = get_clock();
    // res = mat_mul_cuda_prefetch(a, b);
    // double ed3 = get_clock();

    // printf("%.2f, diff = %.6f\n", ed - st, mat_diff(ans, res));
    printf("%.5f",ed - st);
}
// mat_print(result_file, "%.2f", ",", "\n", res);