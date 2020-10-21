#ifndef BLAS_H
#define BLAS_H

void copy_cpu(int N, float *X, int INCX, float *Y, int INCY);
void fill_cpu(int N, float ALPHA, float *X, int INCX);

#endif
