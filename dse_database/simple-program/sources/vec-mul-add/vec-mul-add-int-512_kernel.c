#include "merlin_type_define.h"
#pragma ACCEL kernel

void vpvtv(int a[512],int b[512],int c[512])
{
//    control loops
//    vector plus vector times vector
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 0; i < 512; i++) {
    a[i] += b[i] * c[i];
  }
}
