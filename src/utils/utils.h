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

#endif