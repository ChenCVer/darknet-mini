#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define TWO_PI 6.2831853071795864769252866f


float sum_array(float *a, int n);
char *basecfg(char *cfgfile);
void calloc_error();
char *copy_string(char *s);
void *xcalloc(size_t nmemb, size_t size);
void file_error(char *s);
void malloc_error();
char *fgetl(FILE *fp);
void error(const char *s);
void strip(char *s);
float rand_normal();
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void scal_cpu(int N, float ALPHA, float *X, int INCX);
float mag_array(float *a, int n);
float rand_uniform(float min, float max);
float rand_scale(float s);
int rand_int(int min, int max);
float dist_array(float *a, float *b, int n, int sub);
int constrain_int(int a, int min, int max);
double what_time_is_it_now();
void free_ptrs(void **ptrs, int n);

#endif