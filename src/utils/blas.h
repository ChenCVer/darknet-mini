#ifndef BLAS_H
#define BLAS_H

#include<math.h>
#include <float.h>

void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void fill_cpu(int N, float ALPHA, float *X, int INCX);
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);

#endif
