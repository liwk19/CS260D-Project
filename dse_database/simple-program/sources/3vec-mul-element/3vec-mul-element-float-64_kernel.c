#include "merlin_type_define.h"
#pragma ACCEL kernel

void vtvtv(float a[64],float b[64],float c[64])
{
//    control loops
//    vector times vector times vector
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 64; i++) {
    a[i] = a[i] * b[i] * c[i];
  }
}
