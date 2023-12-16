#include "merlin_type_define.h"
#pragma ACCEL kernel

void vtvtv(double a[512],double b[512],double c[512])
{
//    control loops
//    vector times vector times vector
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 512; i++) {
    a[i] = a[i] * b[i] * c[i];
  }
}
