#include "merlin_type_define.h"
#pragma ACCEL kernel

void stencil(int orig[8192],int sol[8192],int filter[9])
{
  int r;
  int c;
  int k1;
  int k2;
  int temp;
  int mul;
  
#pragma ACCEL PIPELINE off
  
#pragma ACCEL TILE FACTOR=1
  
#pragma ACCEL PARALLEL FACTOR=1
  stencil_label1:
  for (r = 0; r < 128 - 2; r++) {
    
#pragma ACCEL PIPELINE flatten
    
#pragma ACCEL TILE FACTOR=2
    
#pragma ACCEL PARALLEL FACTOR=2
    stencil_label2:
    for (c = 0; c < 64 - 2; c++) {
      temp = ((int )0);
      
#pragma ACCEL PIPELINE off
      stencil_label3:
      for (k1 = 0; k1 < 3; k1++) {
        stencil_label4:
        for (k2 = 0; k2 < 3; k2++) {
          mul = filter[k1 * 3 + k2] * orig[(r + k1) * 64 + c + k2];
          temp += mul;
        }
      }
      sol[r * 64 + c] = temp;
    }
  }
}
