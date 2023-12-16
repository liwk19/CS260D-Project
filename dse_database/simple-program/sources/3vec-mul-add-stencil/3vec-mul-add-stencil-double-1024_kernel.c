#include "merlin_type_define.h"
#pragma ACCEL kernel

void s322(double a[1024],double b[1024],double c[1024])
{
//    recurrences
//    second order linear recurrence
  
#pragma ACCEL PARALLEL FACTOR=auto{__PARA__L0}
  for (int i = 2; i < 1024; i++) {
    a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i];
  }
}
