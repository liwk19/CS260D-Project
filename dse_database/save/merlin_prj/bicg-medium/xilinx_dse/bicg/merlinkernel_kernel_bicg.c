#include "merlin_type_define.h"
#pragma ACCEL kernel

void kernel_bicg(int m,int n,double A[410][390],double s[390],double q[410],double p[390],double r[410])
{
  int i;
  int j;
  for (i = 0; i < 390; i++) {
    s[i] = ((double )0);
  }
  for (i = 0; i < 410; i++) {
    q[i] = 0.0;
    for (j = 0; j < 390; j++) {
      s[j] += r[i] * A[i][j];
      q[i] += A[i][j] * p[j];
    }
  }
}
