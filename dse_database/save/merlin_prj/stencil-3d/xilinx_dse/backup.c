#include "merlin_type_define.h"
#pragma ACCEL kernel

void stencil3d(int C[2],int orig[16384],int sol[16384])
{
  int i;
  int j;
  int k;
  int sum0;
  int sum1;
  int mul0;
  int mul1;
// Handle boundary conditions by filling with original values
  
#pragma ACCEL PIPELINE auto{__PIPE__L0}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L0}
  height_bound_col:
  for (j = 0; j < 32; j++) {
    height_bound_row:
    for (k = 0; k < 16; k++) {
      sol[k + 16 * (j + 32 * 0)] = orig[k + 16 * (j + 32 * 0)];
      sol[k + 16 * (j + 32 * (32 - 1))] = orig[k + 16 * (j + 32 * (32 - 1))];
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L1}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L1}
  col_bound_height:
  for (i = 1; i < 32 - 1; i++) {
    col_bound_row:
    for (k = 0; k < 16; k++) {
      sol[k + 16 * (0 + 32 * i)] = orig[k + 16 * (0 + 32 * i)];
      sol[k + 16 * (32 - 1 + 32 * i)] = orig[k + 16 * (32 - 1 + 32 * i)];
    }
  }
  
#pragma ACCEL PIPELINE auto{__PIPE__L2}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L2}
  row_bound_height:
  for (i = 1; i < 32 - 1; i++) {
    row_bound_col:
    for (j = 1; j < 32 - 1; j++) {
      sol[0 + 16 * (j + 32 * i)] = orig[0 + 16 * (j + 32 * i)];
      sol[16 - 1 + 16 * (j + 32 * i)] = orig[16 - 1 + 16 * (j + 32 * i)];
    }
  }
// Stencil computation
  
#pragma ACCEL PIPELINE auto{__PIPE__L3}
  
#pragma ACCEL TILE FACTOR=auto{__TILE__L3}
  loop_height:
  for (i = 1; i < 32 - 1; i++) {
    
#pragma ACCEL PIPELINE auto{__PIPE__L7}
    
#pragma ACCEL TILE FACTOR=auto{__TILE__L7}
    loop_col:
    for (j = 1; j < 32 - 1; j++) {
      loop_row:
      for (k = 1; k < 16 - 1; k++) {
        sum0 = orig[k + 16 * (j + 32 * i)];
        sum1 = orig[k + 16 * (j + 32 * (i + 1))] + orig[k + 16 * (j + 32 * (i - 1))] + orig[k + 16 * (j + 1 + 32 * i)] + orig[k + 16 * (j - 1 + 32 * i)] + orig[k + 1 + 16 * (j + 32 * i)] + orig[k - 1 + 16 * (j + 32 * i)];
        mul0 = sum0 * C[0];
        mul1 = sum1 * C[1];
        sol[k + 16 * (j + 32 * i)] = mul0 + mul1;
      }
    }
  }
}
