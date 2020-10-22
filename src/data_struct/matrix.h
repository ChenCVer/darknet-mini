#ifndef MATRIX_H
#define MATRIX_H

#include "darknet.h"

matrix make_matrix(int rows, int cols);
void free_matrix(matrix m);
matrix concat_matrix(matrix m1, matrix m2);


#endif
