#include "merlin_type_define.h"
#pragma ACCEL kernel

void kernel_gesummv(int n,double alpha,double beta,double A[250][250],double B[250][250],double tmp[250],double x[250],double y[250])
{
  int i;
  int j;
  for (i = 0; i < 250; i++) {
    tmp[i] = 0.0;
    y[i] = 0.0;
    for (j = 0; j < 250; j++) {
      tmp[i] += A[i][j] * x[j];
      y[i] += B[i][j] * x[j];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
}
