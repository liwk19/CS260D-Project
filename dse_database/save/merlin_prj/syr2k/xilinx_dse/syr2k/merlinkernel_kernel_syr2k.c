#include "merlin_type_define.h"
#pragma ACCEL kernel

void kernel_syr2k(int n,int m,double alpha,double beta,double C[80][80],double A[80][60],double B[80][60])
{
  int i;
  int j;
  int k;
//BLAS PARAMS
//UPLO  = 'L'
//TRANS = 'N'
//A is NxM
//B is NxM
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
          C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];
        }
      }
    }
  }
}
