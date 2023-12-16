#include "merlin_type_define.h"
#pragma ACCEL kernel

void s341(double a[2048],double b[2048])
{
//    packing
//    pack positive values
//    not vectorizable, value of j in unknown at each iteration
  int j;
  j = - 1;
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 2048; i++) {
    if (b[i] > ((double )0.)) {
      j++;
      a[j] = b[i];
    }
  }
}
