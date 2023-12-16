#include "merlin_type_define.h"
#pragma ACCEL kernel

void s341(long a[1024],long b[1024])
{
//    packing
//    pack positive values
//    not vectorizable, value of j in unknown at each iteration
  int j;
  j = - 1;
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 1024; i++) {
    if (b[i] > ((long )0.)) {
      j++;
      a[j] = b[i];
    }
  }
}
