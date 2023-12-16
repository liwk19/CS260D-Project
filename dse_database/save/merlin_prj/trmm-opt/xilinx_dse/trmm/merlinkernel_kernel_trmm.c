#include "merlin_type_define.h"
#pragma ACCEL kernel

void kernel_trmm(int m,int n,double alpha,double A[60][60],double B[60][80])
{
//BLAS parameters
//SIDE   = 'L'
//UPLO   = 'L'
//TRANSA = 'T'
//DIAG   = 'U'
// => Form  B := alpha*A**T*B.
// A is MxM
// B is MxN
  for (int i = 0; i < 60; i++) {
    for (int j = 0; j < 80; j++) {
      for (int k = 0; k < 60; k++) {
        if (k > i) {
          B[i][j] += A[k][i] * B[k][j];
        }
      }
      B[i][j] = alpha * B[i][j];
    }
  }
}
