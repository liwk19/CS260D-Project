#include "merlin_type_define.h"
#pragma ACCEL kernel

void kernel_syrk(int n,int m,double alpha,double beta,double C[80][80],double A[80][60])
{
  int i;
  int j;
  int k;
//BLAS PARAMS
//TRANS = 'N'
//UPLO  = 'L'
// =>  Form  C := alpha*A*A**T + beta*C.
//A is NxM
//C is NxN
  for (i = 0; i < 80; i++) {
    for (j = 0; j < 80; j++) {
      if (j <= i) {
        C[i][j] *= beta;
      }
    }
    for (k = 0; k < 60; k++) {
      for (j = 0; j < 80; j++) {
        if (j <= i) {
          C[i][j] += alpha * A[i][k] * A[j][k];
        }
      }
    }
  }
}
